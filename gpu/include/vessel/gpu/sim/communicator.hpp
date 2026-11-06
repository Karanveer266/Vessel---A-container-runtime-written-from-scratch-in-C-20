// SPDX-License-Identifier: MIT
// Simulated NCCL: a ring all-reduce between CPU threads.
//
// This is a faithful functional + structural model of what NCCL does on a
// ring topology, scaled down to threads and memcpy:
//
//   NCCL                                   | here
//   ---------------------------------------+----------------------------------
//   ncclComm_t (one per GPU)               | Communicator (one per rank thread)
//   ncclCommInitAll(comms, n, devs)        | World(n) + World::comm(r)
//   per-peer connection FIFO in peer memory| Link: bounded SPSC ring of slots
//   NCCL_STEPS (=8) slots per connection   | CommConfig::slots
//   chunkSize                              | CommConfig::chunk_elems
//   ncclAllReduce(send, recv, n, f32, op)  | Communicator::allReduce(...)
//   ring prims: send / recvReduceSend /    | identical step sequence, see
//   recvReduceCopySend / recvCopySend /    | communicator.cpp
//   recv                                   |
//
// Because links are bounded, a slow rank exerts back-pressure on its upstream
// neighbour exactly like a full NCCL FIFO does, and successive chunks pipeline
// around the ring. Each rank sends exactly 2(R-1) segments of ~N/R elements,
// i.e. 2(R-1)/R * N * sizeof(float) bytes — the bandwidth-optimal lower bound
// for all-reduce, which CommStats lets the tests verify.
#pragma once

#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#include "vessel/gpu/common.hpp"

namespace vessel::gpu::sim {

/// Reduction operators, mirroring ncclRedOp_t (ncclSum, ncclProd, ncclMax,
/// ncclMin, ncclAvg).
enum class RedOp { Sum, Prod, Max, Min, Avg };

struct CommConfig {
  /// Elements per pipeline slice ("chunk"). 32 Ki floats = 128 KiB: large
  /// enough to amortize the per-message synchronization, small enough that a
  /// slice being reduced stays resident in L2.
  std::size_t chunk_elems = 32 * 1024;
  /// Slots per link FIFO (NCCL_STEPS = 8). Clamped to >= 4. As in NCCL
  /// (chunkSteps = NCCL_STEPS/2), one chunk occupies at most half the FIFO:
  /// slot size = ceil(chunk_elems / (slots/2)). That slack is what makes the
  /// bounded ring deadlock-free (see the proof sketch in communicator.cpp).
  std::size_t slots = 8;
};

/// Per-rank traffic counters (since construction or the last reset()).
struct CommStats {
  std::uint64_t bytes_sent = 0;
  std::uint64_t bytes_received = 0;
  std::uint64_t messages_sent = 0;
  std::uint64_t send_stalls = 0;  // times the outgoing FIFO was full (back-pressure)
  std::uint64_t recv_stalls = 0;  // times the incoming FIFO was empty (starvation)
};

/// Thrown from blocking operations on every rank once any rank has failed,
/// so a single exception cannot deadlock the remaining ranks.
class CommAborted : public std::runtime_error {
 public:
  CommAborted() : std::runtime_error("communicator aborted by a failing peer") {}
};

/// Element range of segment `seg` (0..R-1) inside pipeline loop `loop`.
///
/// Like NCCL's runRing, the buffer is processed in loops of R * chunk_elems
/// elements; within a loop the (possibly short, final) span is divided into R
/// balanced contiguous segments. Rank r "owns" (finishes reducing) segment
/// r of every loop (NCCL convention: ring->userRanks[0]). Exposed so tests and byte accounting use the
/// exact same partition as the data path.
[[nodiscard]] Range ring_segment(std::size_t count, std::size_t nranks, std::size_t chunk_elems,
                                 std::size_t loop, std::size_t seg) noexcept;

/// Exact number of bytes rank `rank` sends in one ring all-reduce of `count`
/// elements over `nranks` ranks with the given chunking.
///
/// Per loop, in reduce-scatter rank r sends every segment except its own (r),
/// and in all-gather every segment except (r+1) mod R (the last one it
/// receives):
///     bytes = sum_loops (2*len - |seg_r| - |seg_{r+1}|) * elem
/// When R divides every loop span this equals the textbook 2(R-1)/R * N * elem,
/// and *summed over ranks* it equals 2(R-1) * N * elem for any N.
[[nodiscard]] std::uint64_t ring_allreduce_bytes_per_rank(std::size_t count, std::size_t nranks,
                                                          std::size_t rank,
                                                          std::size_t chunk_elems = CommConfig{}.chunk_elems,
                                                          std::size_t elem_size = sizeof(float)) noexcept;

/// Theoretical (continuous) per-rank traffic 2(R-1)/R * N * elem: the lower
/// bound for any all-reduce algorithm on R ranks (Patarasuk & Yuan 2009).
[[nodiscard]] double ring_allreduce_theoretical_bytes(std::size_t count, std::size_t nranks,
                                                      std::size_t elem_size = sizeof(float)) noexcept;

class World;

/// Bounded single-producer/single-consumer FIFO of fixed-size float slots
/// modelling one directed ring edge (rank r -> rank r+1).
///
/// Lock-free: the producer owns `head_`, the consumer owns `tail_`; each side
/// keeps a cached copy of the other's index so the shared cache line is only
/// touched when the FIFO looks full/empty (Rigtorp-style SPSC). Blocking is a
/// short spin followed by an *eventcount* built on C++20 atomic wait/notify
/// (futex on Linux), so oversubscribed ranks sleep instead of burning a core.
/// An eventcount (rather than waiting on head_/tail_ directly) is needed so
/// that abort() can wake a waiter without changing the indices.
class Link {
 public:
  Link(std::size_t slots, std::size_t slot_elems, const std::atomic<bool>& abort_flag);
  ~Link() = default;
  Link(const Link&) = delete;
  Link& operator=(const Link&) = delete;

  /// Producer: returns a writable slot, blocking while the FIFO is full.
  [[nodiscard]] std::span<float> begin_send(CommStats& st);
  /// Producer: publishes `n` elements written into the slot from begin_send().
  void end_send(std::size_t n) noexcept;
  /// Consumer: returns the next filled slot, blocking while the FIFO is empty.
  [[nodiscard]] std::span<const float> begin_recv(CommStats& st);
  /// Consumer: releases the slot obtained from begin_recv().
  void end_recv() noexcept;
  /// Wakes every waiter (used on abort; waiters then observe the abort flag).
  void wake_all() noexcept;

  [[nodiscard]] std::size_t slot_elems() const noexcept { return slot_elems_; }

 private:
  const std::size_t slots_;
  const std::size_t slot_elems_;
  const std::atomic<bool>& abort_;
  std::unique_ptr<float[]> storage_;        // slots_ * slot_elems_
  std::unique_ptr<std::size_t[]> sizes_;    // valid elements per slot

  // Producer and consumer indices on separate cache lines to avoid false
  // sharing; the cached copies live on the owner's line.
  //   line 1: written by producer   (head_, its private cache of tail_)
  //   line 2: written by consumer   (tail_, its private cache of head_)
  //   line 3/4: eventcounts, only touched on the slow (blocking) path and by
  //             the notifier; kept apart so a sleeper's futex word does not
  //             share a line with the hot indices.
  alignas(64) std::atomic<std::uint64_t> head_{0};  // slots published
  std::uint64_t cached_tail_ = 0;                     // producer-private
  alignas(64) std::atomic<std::uint64_t> tail_{0};   // slots released
  std::uint64_t cached_head_ = 0;                     // consumer-private
  alignas(64) std::atomic<std::uint32_t> data_ev_{0};   // bumped by producer on publish
  std::atomic<std::uint32_t> data_waiters_{0};
  alignas(64) std::atomic<std::uint32_t> space_ev_{0};  // bumped by consumer on release
  std::atomic<std::uint32_t> space_waiters_{0};
};

/// One rank's handle (ncclComm_t analogue). Not thread-safe: exactly one
/// thread (the rank's worker) may use a given Communicator, like a CUDA stream.
class Communicator {
 public:
  [[nodiscard]] int rank() const noexcept { return rank_; }
  [[nodiscard]] int size() const noexcept;

  /// Ring all-reduce. Collective: every rank must call it with the same
  /// count/op. `sendbuf == recvbuf` (in-place) is supported, as in NCCL;
  /// partially overlapping buffers are rejected.
  void allReduce(const float* sendbuf, float* recvbuf, std::size_t count, RedOp op = RedOp::Sum);

  /// std::barrier across all ranks.
  void barrier();

  [[nodiscard]] const CommStats& stats() const noexcept { return stats_; }
  void reset_stats() noexcept { stats_ = {}; }

 private:
  friend class World;
  Communicator(World& w, int rank, Link& to_next, Link& from_prev)
      : world_(w), rank_(rank), next_(to_next), prev_(from_prev) {}

  // NCCL ring primitives (names match src/device/prims_*.h in NCCL).
  void send(const float* src, std::size_t n);
  void recv_reduce_send(const float* own, float* dst, std::size_t n, RedOp op);
  void recv_reduce_copy_send(const float* own, float* dst, std::size_t n, RedOp op, float avg_scale);
  void recv_copy_send(float* dst, std::size_t n);
  void recv(float* dst, std::size_t n);

  World& world_;
  int rank_;
  Link& next_;  // outgoing edge  rank -> rank+1
  Link& prev_;  // incoming edge  rank-1 -> rank
  CommStats stats_;
};

/// A clique of R ranks wired as a ring (ncclCommInitAll analogue).
class World {
 public:
  explicit World(int nranks, CommConfig cfg = {});
  World(const World&) = delete;
  World& operator=(const World&) = delete;
  ~World();

  [[nodiscard]] int size() const noexcept { return nranks_; }
  [[nodiscard]] const CommConfig& config() const noexcept { return cfg_; }
  [[nodiscard]] Communicator& comm(int rank) { return *comms_.at(static_cast<std::size_t>(rank)); }

  /// Runs f(Communicator&) on R std::jthreads (one per rank), joins them and
  /// rethrows the first exception. If any rank throws, the world is aborted:
  /// peers blocked on a link get CommAborted, peers in barrier() are released
  /// via arrive_and_drop(), so the call never deadlocks.
  template <class F>
  void run(F&& f) {
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(nranks_));
    {
      std::vector<std::jthread> threads;
      threads.reserve(static_cast<std::size_t>(nranks_));
      for (int r = 0; r < nranks_; ++r) {
        threads.emplace_back([this, r, &f, &errors] {
          try {
            f(comm(r));
          } catch (...) {
            errors[static_cast<std::size_t>(r)] = std::current_exception();
            abort();
            barrier_.arrive_and_drop();
          }
        });
      }
    }
    // Prefer the root cause over the CommAborted echoes it triggered.
    std::exception_ptr first_abort;
    for (auto& e : errors) {
      if (!e) continue;
      try {
        std::rethrow_exception(e);
      } catch (const CommAborted&) {
        if (!first_abort) first_abort = e;
      } catch (...) {
        throw;
      }
    }
    if (first_abort) std::rethrow_exception(first_abort);
  }

  /// Marks the world failed and wakes every blocked rank. Irreversible.
  void abort() noexcept;
  [[nodiscard]] bool aborted() const noexcept { return aborted_.load(std::memory_order_acquire); }

 private:
  friend class Communicator;
  int nranks_;
  CommConfig cfg_;
  std::atomic<bool> aborted_{false};
  std::barrier<> barrier_;
  std::vector<std::unique_ptr<Link>> links_;  // links_[r]: r -> (r+1) % R
  std::vector<std::unique_ptr<Communicator>> comms_;
};

}  // namespace vessel::gpu::sim
