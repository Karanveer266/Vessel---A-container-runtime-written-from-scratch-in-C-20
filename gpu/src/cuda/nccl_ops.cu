// SPDX-License-Identifier: MIT
// NCCL-based multi-GPU ops: split-K GEMM and an all-reduce bandwidth test.
//
// Process/communicator model
// --------------------------
// Single process, single host thread, R GPUs: ncclCommInitAll() creates R
// communicators in one call (it internally does the unique-id exchange). Each
// collective is then issued for every rank between ncclGroupStart() and
// ncclGroupEnd(). The group is *mandatory* here: ncclAllReduce on a comm
// blocks until all ranks of the clique have launched, so issuing rank 0's call
// alone from one thread would deadlock; inside a group NCCL fuses the R calls
// into one launch step.
//
// Split-K GEMM
// ------------
//   rank r computes C_r = A[:, K_r] * B[K_r, :] where K_r = split_range(K, R, r, 4)
//   then ncclAllReduce(C_r, C_r, M*N, ncclFloat, ncclSum)   (in place)
// so every GPU ends up with the full C. K-slice boundaries are aligned to 4
// so the A sub-matrix pointer (A + k0, lda = K) stays 16 B aligned and the
// float4 GEMM path is used whenever K % 4 == 0.
//
// Trade-off vs. a 2D (M/N) partition: split-K needs no data exchange of A/B
// tiles and gives each GPU an equal, contiguous share of FLOPs, but adds an
// all-reduce of M*N floats. Compute is 2MNK/R FLOP per GPU, communication is
// 2(R-1)/R * 4MN bytes per GPU, so the compute:communication ratio is
// ~K/(4R) FLOP/B: favourable for K >> M, N (e.g. weight-gradient GEMMs),
// unfavourable for square GEMMs on PCIe. The worker reports compute_ms and
// allreduce_ms separately so that crossover is directly observable.
//
// Timing
// ------
// cudaEvents are recorded on each device's stream: start before the GEMM,
// mid after it, stop after the all-reduce. The iteration time is the max over
// devices of (stop - start) — a collective is only done when its slowest
// participant is done. Events never synchronize the host inside the loop.
#include <algorithm>
#include <memory>
#include <cmath>
#include <string>
#include <vector>

#include <nccl.h>

#include "cuda/cuda_check.cuh"
#include "cuda/kernels.hpp"
#include "cuda/nccl_ops.hpp"
#include "vessel/gpu/common.hpp"
#include "vessel/gpu/cpu_kernels.hpp"

namespace vessel::gpu::cuda {

namespace {

/// Owns R NCCL communicators (one per device); destroyed in reverse order.
class NcclClique {
 public:
  explicit NcclClique(const std::vector<int>& devs) : comms_(devs.size(), nullptr) {
    VESSEL_NCCL_CHECK(ncclCommInitAll(comms_.data(), static_cast<int>(devs.size()), devs.data()));
  }
  ~NcclClique() {
    for (auto it = comms_.rbegin(); it != comms_.rend(); ++it)
      if (*it) ncclCommDestroy(*it);
  }
  NcclClique(const NcclClique&) = delete;
  NcclClique& operator=(const NcclClique&) = delete;
  [[nodiscard]] ncclComm_t operator[](std::size_t i) const { return comms_[i]; }

 private:
  std::vector<ncclComm_t> comms_;
};

/// Per-rank device resources. Members are destroyed with the rank's device
/// current (DeviceGuard in the owner) — streams/events/buffers are
/// device-affine.
struct Rank {
  int device = 0;
  Stream stream;
  Event start, mid, stop;
  DeviceBuffer<float> a, b, c, send, recv;
};

std::vector<int> pick_devices(std::size_t ranks, std::vector<std::string>& names) {
  int count = 0;
  VESSEL_CUDA_CHECK(cudaGetDeviceCount(&count));
  if (static_cast<std::size_t>(count) < ranks)
    throw CudaError("requested " + std::to_string(ranks) + " ranks but only " + std::to_string(count) +
                    " CUDA device(s) are visible");
  std::vector<int> devs(ranks);
  for (std::size_t r = 0; r < ranks; ++r) {
    devs[r] = static_cast<int>(r);
    cudaDeviceProp p{};
    VESSEL_CUDA_CHECK(cudaGetDeviceProperties(&p, devs[r]));
    names.emplace_back(p.name);
  }
  return devs;
}

/// Creates per-rank state with the right device current for each allocation.
std::vector<std::unique_ptr<Rank>> make_ranks(const std::vector<int>& devs) {
  std::vector<std::unique_ptr<Rank>> ranks;
  for (int d : devs) {
    DeviceGuard g(d);
    auto r = std::make_unique<Rank>();
    r->device = d;
    ranks.push_back(std::move(r));
  }
  return ranks;
}

/// Destroys rank state, each with its own device current.
void destroy_ranks(std::vector<std::unique_ptr<Rank>>& ranks) noexcept {
  for (auto& r : ranks) {
    if (!r) continue;
    cudaSetDevice(r->device);
    r.reset();
  }
}

struct RankGuard {
  std::vector<std::unique_ptr<Rank>>& ranks;
  ~RankGuard() { destroy_ranks(ranks); }
};

void sync_all(const std::vector<std::unique_ptr<Rank>>& ranks) {
  for (const auto& r : ranks) {
    VESSEL_CUDA_CHECK(cudaSetDevice(r->device));
    VESSEL_CUDA_CHECK(cudaStreamSynchronize(r->stream.get()));
  }
}

}  // namespace

OpResult run_splitk_nccl(const WorkerOptions& o) {
  const std::size_t R = o.ranks;
  const int M = static_cast<int>(o.m), N = static_cast<int>(o.n), K = static_cast<int>(o.k);
  const std::size_t MN = o.m * o.n;

  OpResult res;
  const std::vector<int> devs = pick_devices(R, res.devices);

  std::vector<float> hA(o.m * o.k), hB(o.k * o.n);
  fill_uniform(hA, o.seed);
  fill_uniform(hB, o.seed + 1);

  auto ranks = make_ranks(devs);
  RankGuard guard{ranks};
  // Communicators must be destroyed before the streams they ran on: declare after `ranks`.
  NcclClique comms(devs);

  for (std::size_t r = 0; r < R; ++r) {
    Rank& rk = *ranks[r];
    VESSEL_CUDA_CHECK(cudaSetDevice(rk.device));
    // Every rank holds full A and B for simplicity of the demo; only its K
    // slice is ever read. (A production split-K would scatter the slices.)
    rk.a = DeviceBuffer<float>(hA.size());
    rk.b = DeviceBuffer<float>(hB.size());
    rk.c = DeviceBuffer<float>(MN);
    VESSEL_CUDA_CHECK(cudaMemcpyAsync(rk.a.get(), hA.data(), rk.a.bytes(), cudaMemcpyHostToDevice, rk.stream.get()));
    VESSEL_CUDA_CHECK(cudaMemcpyAsync(rk.b.get(), hB.data(), rk.b.bytes(), cudaMemcpyHostToDevice, rk.stream.get()));
  }
  sync_all(ranks);

  auto issue_iteration = [&](bool record) {
    // 1) local partial GEMMs (one per device, all asynchronous)
    for (std::size_t r = 0; r < R; ++r) {
      Rank& rk = *ranks[r];
      VESSEL_CUDA_CHECK(cudaSetDevice(rk.device));
      if (record) VESSEL_CUDA_CHECK(cudaEventRecord(rk.start.get(), rk.stream.get()));
      const Range ks = split_range(o.k, R, r, 4);
      if (ks.size() == 0) {
        // Empty slice (K < 4R): contribute zeros to the sum.
        VESSEL_CUDA_CHECK(cudaMemsetAsync(rk.c.get(), 0, rk.c.bytes(), rk.stream.get()));
      } else {
        sgemm(M, N, static_cast<int>(ks.size()), 1.f, rk.a.get() + ks.begin, K, rk.b.get() + ks.begin * o.n, N, 0.f,
              rk.c.get(), N, rk.stream.get());
      }
      if (record) VESSEL_CUDA_CHECK(cudaEventRecord(rk.mid.get(), rk.stream.get()));
    }
    // 2) in-place sum all-reduce, all ranks inside one group (see header)
    VESSEL_NCCL_CHECK(ncclGroupStart());
    for (std::size_t r = 0; r < R; ++r) {
      Rank& rk = *ranks[r];
      VESSEL_NCCL_CHECK(ncclAllReduce(rk.c.get(), rk.c.get(), MN, ncclFloat, ncclSum, comms[r], rk.stream.get()));
    }
    VESSEL_NCCL_CHECK(ncclGroupEnd());
    if (record) {
      for (std::size_t r = 0; r < R; ++r) {
        VESSEL_CUDA_CHECK(cudaSetDevice(ranks[r]->device));
        VESSEL_CUDA_CHECK(cudaEventRecord(ranks[r]->stop.get(), ranks[r]->stream.get()));
      }
    }
  };

  for (std::size_t i = 0; i < o.warmup; ++i) issue_iteration(false);
  sync_all(ranks);

  double total_ms = 0.0, compute_ms = 0.0, comm_ms = 0.0;
  for (std::size_t i = 0; i < o.iters; ++i) {
    issue_iteration(true);
    double it = 0.0, comp = 0.0, cm = 0.0;
    for (const auto& rk : ranks) {
      it = std::max<double>(it, Event::elapsed_ms(rk->start, rk->stop));
      comp = std::max<double>(comp, Event::elapsed_ms(rk->start, rk->mid));
      cm = std::max<double>(cm, Event::elapsed_ms(rk->mid, rk->stop));
    }
    total_ms += it;
    compute_ms += comp;
    comm_ms += cm;
  }
  const double iters = static_cast<double>(o.iters);
  res.time_ms = total_ms / iters;
  res.gflops = 2.0 * static_cast<double>(o.m) * static_cast<double>(o.n) * static_cast<double>(o.k) / (res.time_ms * 1e6);
  const auto bw = allreduce_bandwidth(MN * sizeof(float), R, comm_ms / iters);
  res.bus_bw_gbps = bw.busbw_gbps;

  if (o.verify) {
    std::vector<float> ref(MN), got(MN);
    cpu::gemm_tiled(hA.data(), hB.data(), ref.data(), o.m, o.n, o.k, o.k, o.n, o.n, false, 0);
    for (const auto& rk : ranks) {
      VESSEL_CUDA_CHECK(cudaSetDevice(rk->device));
      VESSEL_CUDA_CHECK(cudaMemcpy(got.data(), rk->c.get(), rk->c.bytes(), cudaMemcpyDeviceToHost));
      res.max_abs_error = std::max(res.max_abs_error, max_abs_diff(got, ref));
    }
    res.add("tolerance", gemm_tolerance(o.k));
    if (!(res.max_abs_error <= gemm_tolerance(o.k))) res.error = "verification failed: max_abs_error above tolerance";
  }
  res.add("ranks", R);
  res.add("tile", std::string(sgemm_tile_string()));
  res.add("compute_ms", compute_ms / iters);
  res.add("allreduce_ms", comm_ms / iters);
  res.add("algbw_gbps", bw.algbw_gbps);
  res.add("bytes_per_rank_theory", 2.0 * (static_cast<double>(R) - 1.0) / static_cast<double>(R) *
                                       static_cast<double>(MN * sizeof(float)));
  res.add("nccl_version", std::to_string(NCCL_VERSION_CODE));
  return res;
}

OpResult run_allreduce_nccl(const WorkerOptions& o) {
  const std::size_t R = o.ranks;
  const std::size_t count = o.allreduce_count();
  OpResult res;
  const std::vector<int> devs = pick_devices(R, res.devices);
  auto ranks = make_ranks(devs);
  RankGuard guard{ranks};
  NcclClique comms(devs);

  // Rank r's input is a deterministic pattern; out-of-place so iterations
  // are idempotent and the result can be checked after timing.
  std::vector<std::vector<float>> host(R, std::vector<float>(count));
  for (std::size_t r = 0; r < R; ++r) {
    fill_uniform(host[r], o.seed + r);
    Rank& rk = *ranks[r];
    VESSEL_CUDA_CHECK(cudaSetDevice(rk.device));
    rk.send = DeviceBuffer<float>(count);
    rk.recv = DeviceBuffer<float>(count);
    VESSEL_CUDA_CHECK(cudaMemcpyAsync(rk.send.get(), host[r].data(), rk.send.bytes(), cudaMemcpyHostToDevice,
                                      rk.stream.get()));
  }
  sync_all(ranks);

  auto issue = [&] {
    VESSEL_NCCL_CHECK(ncclGroupStart());
    for (std::size_t r = 0; r < R; ++r)
      VESSEL_NCCL_CHECK(ncclAllReduce(ranks[r]->send.get(), ranks[r]->recv.get(), count, ncclFloat, ncclSum, comms[r],
                                      ranks[r]->stream.get()));
    VESSEL_NCCL_CHECK(ncclGroupEnd());
  };

  for (std::size_t i = 0; i < std::max<std::size_t>(o.warmup, 1); ++i) issue();  // >= 1: NCCL connects lazily
  sync_all(ranks);

  // nccl-tests style: one event pair around the whole batch of iterations
  // per device (back-to-back collectives, no host sync in between).
  for (const auto& rk : ranks) {
    VESSEL_CUDA_CHECK(cudaSetDevice(rk->device));
    VESSEL_CUDA_CHECK(cudaEventRecord(rk->start.get(), rk->stream.get()));
  }
  for (std::size_t i = 0; i < o.iters; ++i) issue();
  double ms = 0.0;
  for (const auto& rk : ranks) {
    VESSEL_CUDA_CHECK(cudaSetDevice(rk->device));
    VESSEL_CUDA_CHECK(cudaEventRecord(rk->stop.get(), rk->stream.get()));
  }
  for (const auto& rk : ranks) ms = std::max<double>(ms, Event::elapsed_ms(rk->start, rk->stop));

  res.time_ms = ms / static_cast<double>(o.iters);
  const auto bw = allreduce_bandwidth(count * sizeof(float), R, res.time_ms);
  res.bus_bw_gbps = bw.busbw_gbps;

  if (o.verify) {
    std::vector<float> got(count);
    double tol = 1e-6 * static_cast<double>(R) * static_cast<double>(R);
    for (const auto& rk : ranks) {
      VESSEL_CUDA_CHECK(cudaSetDevice(rk->device));
      VESSEL_CUDA_CHECK(cudaMemcpy(got.data(), rk->recv.get(), rk->recv.bytes(), cudaMemcpyDeviceToHost));
      for (std::size_t i = 0; i < count; ++i) {
        double ref = 0.0;
        for (std::size_t q = 0; q < R; ++q) ref += host[q][i];
        res.max_abs_error = std::max(res.max_abs_error, std::fabs(static_cast<double>(got[i]) - ref));
      }
    }
    res.add("tolerance", tol);
    if (!(res.max_abs_error <= tol)) res.error = "verification failed: max_abs_error above tolerance";
  }
  res.add("ranks", R);
  res.add("count", count);
  res.add("size_bytes", count * sizeof(float));
  res.add("algbw_gbps", bw.algbw_gbps);
  res.add("bytes_per_rank_theory", 2.0 * (static_cast<double>(R) - 1.0) / static_cast<double>(R) *
                                       static_cast<double>(count * sizeof(float)));
  res.add("nccl_version", std::to_string(NCCL_VERSION_CODE));
  return res;
}

}  // namespace vessel::gpu::cuda
