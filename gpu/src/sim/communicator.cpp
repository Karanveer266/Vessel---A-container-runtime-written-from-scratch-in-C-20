// SPDX-License-Identifier: MIT
// Ring all-reduce over bounded SPSC links — see communicator.hpp for the
// NCCL <-> simulator mapping.
//
// ---------------------------------------------------------------------------
// Algorithm (identical step order to NCCL's runRing<ncclFuncAllReduce>)
// ---------------------------------------------------------------------------
// The buffer is processed in "loops" of R * chunk elements. Inside a loop the
// span is cut into R segments; with ring position r and p(j) = (r - j) mod R:
//
//   phase           step     primitive           segment
//   reduce-scatter  0        send                p(1)
//   reduce-scatter  1..R-2   recvReduceSend      p(2) .. p(R-1)
//   reduce-scatter  R-1      recvReduceCopySend  p(0) = r   (now final)
//   all-gather      R..2R-3  recvCopySend        p(1) .. p(R-2)
//   all-gather      2R-2     recv                p(R-1) = r+1
//
// After reduce-scatter rank r holds the fully reduced segment r; all-gather
// circulates the finished segments once around the ring. Every rank sends
// exactly 2(R-1) segments per loop (the bandwidth-optimal 2(R-1)/R * N).
//
// Each segment is further split into slices of `slot_elems` (one FIFO slot
// each) and the recv->reduce->send of a slice is done before the next slice
// is touched, so successive slices are in flight on different links at the
// same time: that is the ring *pipeline*.
//
// ---------------------------------------------------------------------------
// Why the bounded ring cannot deadlock
// ---------------------------------------------------------------------------
// Let fill(r) be the number of occupied slots on link r -> r+1, and
// d(r) = sent(r) - received(r) for rank r. Then sum_r fill(r) = sum_r d(r).
// Within a loop, a rank only sends a slice after receiving one, except for
// the step-0 segment, so d(r) <= slices(seg p(1)) <= slots/2 (slot_elems is
// chosen so that a chunk is at most slots/2 slices — NCCL's chunkSteps =
// NCCL_STEPS/2). Completed loops contribute 0 because all segments of a
// non-final loop are equal-sized. Hence:
//  * "every rank blocked on a full outgoing link" needs sum fill = R*slots,
//    but sum d <= R*slots/2  -> impossible;
//  * a rank blocked on an empty incoming link implies its upstream neighbour
//    is not blocked on send, so it is itself blocked on recv (or finished,
//    in which case it already sent everything we need). Inductively all
//    ranks wait on empty links, i.e. sum d = 0 with everyone past step 0,
//    contradicting d(r) >= slices(seg p(1)) for non-empty segments. Empty
//    segments send/receive nothing on both sides, so they cannot block.
// ---------------------------------------------------------------------------
#include "vessel/gpu/sim/communicator.hpp"

#include <algorithm>
#include <cstring>
#include <functional>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace vessel::gpu::sim {

namespace {

inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(__i386__)
  _mm_pause();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#endif
}

// Spin this many times before sleeping on the futex. Short: on an
// oversubscribed host (8 ranks on 2 cores) spinning steals the core from the
// very peer we are waiting for.
constexpr int kSpinIterations = 128;

/// Blocks until pred() is true or `abort` is set. `ev` is an eventcount the
/// other side bumps (after making pred() true) iff `waiters` is non-zero.
template <class Pred>
bool wait_until(Pred pred, std::atomic<std::uint32_t>& ev, std::atomic<std::uint32_t>& waiters,
                const std::atomic<bool>& abort) {
  for (int i = 0; i < kSpinIterations; ++i) {
    if (pred()) return true;
    if (abort.load(std::memory_order_relaxed)) return false;
    cpu_relax();
  }
  for (;;) {
    // Lost-wake-up freedom without standalone fences (which TSan cannot
    // model): both sides perform an acq_rel RMW on the same `waiters` word,
    // and RMWs on one atomic are totally ordered.
    //  * waiter's RMW first  -> the notifier's RMW reads >= 1 and notifies;
    //  * notifier's RMW first -> our RMW reads its value, so its earlier
    //    head_/tail_ store happens-before our re-check of pred() below.
    // `seen` is read before the re-check, so a notify that lands between the
    // re-check and ev.wait() changes ev and wait() returns immediately.
    const std::uint32_t seen = ev.load(std::memory_order_acquire);
    waiters.fetch_add(1, std::memory_order_acq_rel);
    if (pred() || abort.load(std::memory_order_acquire)) {
      waiters.fetch_sub(1, std::memory_order_relaxed);
      return pred();
    }
    ev.wait(seen, std::memory_order_acquire);
    waiters.fetch_sub(1, std::memory_order_relaxed);
  }
}

inline void notify_side(std::atomic<std::uint32_t>& ev, std::atomic<std::uint32_t>& waiters) noexcept {
  // RMW (not a plain load): see wait_until. One uncontended-in-practice
  // `lock xadd` per 64 KiB slot is noise next to the slot's memcpy.
  if (waiters.fetch_add(0, std::memory_order_acq_rel) != 0) {
    ev.fetch_add(1, std::memory_order_release);
    ev.notify_all();
  }
}

// ---- element-wise reduction kernels --------------------------------------
// Templated on the operator so each instantiation is a straight-line loop
// the compiler auto-vectorizes (one switch per slice, not per element).
struct OpSum  { static float apply(float a, float b) noexcept { return a + b; } };
struct OpProd { static float apply(float a, float b) noexcept { return a * b; } };
struct OpMax  { static float apply(float a, float b) noexcept { return std::max(a, b); } };
struct OpMin  { static float apply(float a, float b) noexcept { return std::min(a, b); } };

// `own` may alias `out` element-for-element (in-place all-reduce, final
// reduce-scatter step) so only the FIFO slot pointer is __restrict; GCC/Clang
// still vectorize (identical-index aliasing is resolved by a runtime check).
template <class Op>
void reduce_into(float* out, const float* __restrict a, const float* b, std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) out[i] = Op::apply(a[i], b[i]);
}

template <class Op>
void reduce_scale_into(float* out, const float* __restrict a, const float* b, std::size_t n,
                       float scale) noexcept {
  for (std::size_t i = 0; i < n; ++i) out[i] = Op::apply(a[i], b[i]) * scale;
}

void reduce_dispatch(RedOp op, float* out, const float* slot, const float* own, std::size_t n,
                     float scale) noexcept {
  const bool scaled = scale != 1.0F;
  switch (op) {
    case RedOp::Avg:  // Avg == Sum followed by * 1/R on the owning rank (like NCCL's ncclAvg for floats).
    case RedOp::Sum:
      scaled ? reduce_scale_into<OpSum>(out, slot, own, n, scale) : reduce_into<OpSum>(out, slot, own, n);
      return;
    case RedOp::Prod: reduce_into<OpProd>(out, slot, own, n); return;
    case RedOp::Max:  reduce_into<OpMax>(out, slot, own, n); return;
    case RedOp::Min:  reduce_into<OpMin>(out, slot, own, n); return;
  }
}

// Sender and receiver slice a segment identically (same slot size on every
// link), so a received slice can never exceed what we still expect. If it
// does, ranks disagree about count/op: fail loudly instead of corrupting.
std::size_t checked_slice(std::size_t got, std::size_t remaining) {
  if (got == 0 || got > remaining)
    throw std::logic_error("allReduce: peer slice mismatch (ranks called with different counts?)");
  return got;
}

std::size_t slot_elems_for(const CommConfig& cfg) noexcept {
  const std::size_t chunk = std::max<std::size_t>(cfg.chunk_elems, 1);
  const std::size_t half = std::max<std::size_t>(cfg.slots, 4) / 2;
  return (chunk + half - 1) / half;
}

}  // namespace

// ---------------------------------------------------------------------------
// Partition / accounting helpers
// ---------------------------------------------------------------------------
Range ring_segment(std::size_t count, std::size_t nranks, std::size_t chunk_elems, std::size_t loop,
                   std::size_t seg) noexcept {
  chunk_elems = std::max<std::size_t>(chunk_elems, 1);
  const std::size_t loop_span = nranks * chunk_elems;
  const std::size_t base = loop * loop_span;
  if (base >= count) return {count, count};
  const std::size_t len = std::min(loop_span, count - base);
  const Range r = split_range(len, nranks, seg);
  return {base + r.begin, base + r.end};
}

std::uint64_t ring_allreduce_bytes_per_rank(std::size_t count, std::size_t nranks, std::size_t rank,
                                            std::size_t chunk_elems, std::size_t elem_size) noexcept {
  if (nranks <= 1 || count == 0) return 0;
  chunk_elems = std::max<std::size_t>(chunk_elems, 1);
  const std::size_t loop_span = nranks * chunk_elems;
  const std::size_t loops = (count + loop_span - 1) / loop_span;
  std::uint64_t elems = 0;
  for (std::size_t l = 0; l < loops; ++l) {
    const std::size_t len = std::min(loop_span, count - l * loop_span);
    const std::size_t own = ring_segment(count, nranks, chunk_elems, l, rank).size();
    const std::size_t last = ring_segment(count, nranks, chunk_elems, l, (rank + 1) % nranks).size();
    elems += 2 * len - own - last;
  }
  return elems * elem_size;
}

double ring_allreduce_theoretical_bytes(std::size_t count, std::size_t nranks, std::size_t elem_size) noexcept {
  if (nranks == 0) return 0.0;
  const double r = static_cast<double>(nranks);
  return 2.0 * (r - 1.0) / r * static_cast<double>(count) * static_cast<double>(elem_size);
}

// ---------------------------------------------------------------------------
// Link
// ---------------------------------------------------------------------------
Link::Link(std::size_t slots, std::size_t slot_elems, const std::atomic<bool>& abort_flag)
    : slots_(slots),
      slot_elems_(slot_elems),
      abort_(abort_flag),
      storage_(std::make_unique<float[]>(slots * slot_elems)),
      sizes_(std::make_unique<std::size_t[]>(slots)) {}

std::span<float> Link::begin_send(CommStats& st) {
  const std::uint64_t h = head_.load(std::memory_order_relaxed);
  if (h - cached_tail_ >= slots_) {
    cached_tail_ = tail_.load(std::memory_order_acquire);
    if (h - cached_tail_ >= slots_) {
      ++st.send_stalls;
      const bool ok = wait_until(
          [&] {
            cached_tail_ = tail_.load(std::memory_order_acquire);
            return h - cached_tail_ < slots_;
          },
          space_ev_, space_waiters_, abort_);
      if (!ok) throw CommAborted();
    }
  }
  return {storage_.get() + (h % slots_) * slot_elems_, slot_elems_};
}

void Link::end_send(std::size_t n) noexcept {
  const std::uint64_t h = head_.load(std::memory_order_relaxed);
  sizes_[h % slots_] = n;
  head_.store(h + 1, std::memory_order_release);  // publishes slot payload + size
  notify_side(data_ev_, data_waiters_);
}

std::span<const float> Link::begin_recv(CommStats& st) {
  const std::uint64_t t = tail_.load(std::memory_order_relaxed);
  if (cached_head_ == t) {
    cached_head_ = head_.load(std::memory_order_acquire);
    if (cached_head_ == t) {
      ++st.recv_stalls;
      const bool ok = wait_until(
          [&] {
            cached_head_ = head_.load(std::memory_order_acquire);
            return cached_head_ != t;
          },
          data_ev_, data_waiters_, abort_);
      if (!ok) throw CommAborted();
    }
  }
  const std::size_t idx = t % slots_;
  return {storage_.get() + idx * slot_elems_, sizes_[idx]};
}

void Link::end_recv() noexcept {
  const std::uint64_t t = tail_.load(std::memory_order_relaxed);
  tail_.store(t + 1, std::memory_order_release);  // slot may now be overwritten
  notify_side(space_ev_, space_waiters_);
}

void Link::wake_all() noexcept {
  data_ev_.fetch_add(1, std::memory_order_release);
  data_ev_.notify_all();
  space_ev_.fetch_add(1, std::memory_order_release);
  space_ev_.notify_all();
}

// ---------------------------------------------------------------------------
// World
// ---------------------------------------------------------------------------
World::World(int nranks, CommConfig cfg)
    : nranks_(nranks > 0 ? nranks : throw std::invalid_argument("World: nranks must be >= 1")),
      cfg_(cfg),
      barrier_(nranks) {
  cfg_.chunk_elems = std::max<std::size_t>(cfg_.chunk_elems, 1);
  cfg_.slots = std::max<std::size_t>(cfg_.slots, 4);
  const std::size_t slot_elems = slot_elems_for(cfg_);
  const auto n = static_cast<std::size_t>(nranks_);
  links_.reserve(n);
  for (std::size_t r = 0; r < n; ++r) links_.push_back(std::make_unique<Link>(cfg_.slots, slot_elems, aborted_));
  comms_.reserve(n);
  for (std::size_t r = 0; r < n; ++r) {
    Link& out = *links_[r];
    Link& in = *links_[(r + n - 1) % n];
    comms_.push_back(std::unique_ptr<Communicator>(new Communicator(*this, static_cast<int>(r), out, in)));
  }
}

World::~World() = default;

void World::abort() noexcept {
  aborted_.store(true, std::memory_order_release);
  for (auto& l : links_) l->wake_all();
}

// ---------------------------------------------------------------------------
// Communicator
// ---------------------------------------------------------------------------
int Communicator::size() const noexcept { return world_.size(); }

void Communicator::barrier() {
  if (world_.aborted()) throw CommAborted();
  world_.barrier_.arrive_and_wait();
  if (world_.aborted()) throw CommAborted();
}

void Communicator::send(const float* src, std::size_t n) {
  while (n > 0) {
    auto slot = next_.begin_send(stats_);
    const std::size_t m = std::min(n, slot.size());
    std::memcpy(slot.data(), src, m * sizeof(float));
    next_.end_send(m);
    stats_.bytes_sent += m * sizeof(float);
    ++stats_.messages_sent;
    src += m;
    n -= m;
  }
}

void Communicator::recv_reduce_send(const float* own, float* /*dst*/, std::size_t n, RedOp op) {
  // Intermediate reduce-scatter step: partial = recv (+) own, forwarded
  // downstream. The result is *not* written to the user buffer (NCCL does the
  // same — only the owning rank's final value is). The partial is reduced
  // directly into the outgoing slot: zero extra copies.
  while (n > 0) {
    auto in = prev_.begin_recv(stats_);
    auto out = next_.begin_send(stats_);
    const std::size_t m = checked_slice(in.size(), n);
    reduce_dispatch(op, out.data(), in.data(), own, m, 1.0F);
    prev_.end_recv();
    stats_.bytes_received += m * sizeof(float);
    next_.end_send(m);
    stats_.bytes_sent += m * sizeof(float);
    ++stats_.messages_sent;
    own += m;
    n -= m;
  }
}

void Communicator::recv_reduce_copy_send(const float* own, float* dst, std::size_t n, RedOp op, float avg_scale) {
  // Last reduce-scatter step: this rank completes the segment, stores it and
  // starts its trip around the ring (first all-gather hop) in the same pass.
  while (n > 0) {
    auto in = prev_.begin_recv(stats_);
    auto out = next_.begin_send(stats_);
    const std::size_t m = checked_slice(in.size(), n);
    reduce_dispatch(op, dst, in.data(), own, m, avg_scale);
    prev_.end_recv();
    stats_.bytes_received += m * sizeof(float);
    std::memcpy(out.data(), dst, m * sizeof(float));
    next_.end_send(m);
    stats_.bytes_sent += m * sizeof(float);
    ++stats_.messages_sent;
    own += m;
    dst += m;
    n -= m;
  }
}

void Communicator::recv_copy_send(float* dst, std::size_t n) {
  while (n > 0) {
    auto in = prev_.begin_recv(stats_);
    auto out = next_.begin_send(stats_);
    const std::size_t m = checked_slice(in.size(), n);
    std::memcpy(dst, in.data(), m * sizeof(float));
    std::memcpy(out.data(), in.data(), m * sizeof(float));
    prev_.end_recv();
    stats_.bytes_received += m * sizeof(float);
    next_.end_send(m);
    stats_.bytes_sent += m * sizeof(float);
    ++stats_.messages_sent;
    dst += m;
    n -= m;
  }
}

void Communicator::recv(float* dst, std::size_t n) {
  while (n > 0) {
    auto in = prev_.begin_recv(stats_);
    const std::size_t m = checked_slice(in.size(), n);
    std::memcpy(dst, in.data(), m * sizeof(float));
    prev_.end_recv();
    stats_.bytes_received += m * sizeof(float);
    dst += m;
    n -= m;
  }
}

void Communicator::allReduce(const float* sendbuf, float* recvbuf, std::size_t count, RedOp op) {
  if (world_.aborted()) throw CommAborted();
  if (count == 0) return;
  if (sendbuf == nullptr || recvbuf == nullptr) throw std::invalid_argument("allReduce: null buffer");
  if (sendbuf != recvbuf) {
    const auto s = reinterpret_cast<std::uintptr_t>(sendbuf);
    const auto d = reinterpret_cast<std::uintptr_t>(recvbuf);
    const std::uintptr_t bytes = count * sizeof(float);
    if (s < d + bytes && d < s + bytes)
      throw std::invalid_argument("allReduce: send and recv buffers partially overlap");
  }

  const auto R = static_cast<std::size_t>(world_.size());
  if (R == 1) {  // NCCL also short-circuits a 1-rank ring into a local copy.
    if (sendbuf != recvbuf) std::memcpy(recvbuf, sendbuf, count * sizeof(float));
    return;
  }

  const float avg_scale = op == RedOp::Avg ? 1.0F / static_cast<float>(R) : 1.0F;
  const std::size_t chunk = world_.config().chunk_elems;
  const std::size_t loop_span = R * chunk;
  const std::size_t loops = (count + loop_span - 1) / loop_span;
  const auto r = static_cast<std::size_t>(rank_);
  auto seg = [&](std::size_t loop, std::size_t j) {  // segment index p(j) = (r - j) mod R
    return ring_segment(count, R, chunk, loop, (r + R - (j % R)) % R);
  };

  for (std::size_t l = 0; l < loops; ++l) {
    // --- reduce-scatter ---
    Range s = seg(l, 1);
    send(sendbuf + s.begin, s.size());
    for (std::size_t j = 2; j < R; ++j) {
      s = seg(l, j);
      recv_reduce_send(sendbuf + s.begin, recvbuf + s.begin, s.size(), op);
    }
    s = seg(l, 0);
    recv_reduce_copy_send(sendbuf + s.begin, recvbuf + s.begin, s.size(), op, avg_scale);
    // --- all-gather ---
    for (std::size_t j = 1; j + 1 < R; ++j) {
      s = seg(l, j);
      recv_copy_send(recvbuf + s.begin, s.size());
    }
    s = seg(l, R - 1);
    recv(recvbuf + s.begin, s.size());
  }
}

}  // namespace vessel::gpu::sim
