// SPDX-License-Identifier: MIT
// Simulated backend: runs every worker op on CPU threads with the same
// algorithms (and the same metric definitions) as the CUDA backend.
#include "vessel/gpu/sim/backend.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "vessel/gpu/common.hpp"
#include "vessel/gpu/cpu_kernels.hpp"
#include "vessel/gpu/sim/communicator.hpp"

namespace vessel::gpu::sim {

namespace {

// Refuse requests that would allocate absurd amounts of host memory rather
// than let the OOM killer take the worker (the daemon prefers ok=0).
constexpr std::size_t kMaxBytes = std::size_t{8} << 30;

std::size_t checked_mul(std::size_t a, std::size_t b) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) throw std::invalid_argument("problem size overflows size_t");
  return a * b;
}

void check_budget(std::size_t floats) {
  if (checked_mul(floats, sizeof(float)) > kMaxBytes)
    throw std::invalid_argument("problem needs " + std::to_string(floats * sizeof(float) >> 20) +
                                " MiB of host memory (sim limit 8192 MiB)");
}

std::vector<std::string> rank_names(std::size_t n) {
  std::vector<std::string> v;
  for (std::size_t r = 0; r < n; ++r) v.push_back("cpu-sim-rank" + std::to_string(r));
  return v;
}

/// Runs `body` warmup + iters times and returns the mean wall time per
/// measured iteration in ms.
template <class F>
double time_loop(const WorkerOptions& o, F&& body) {
  for (std::size_t i = 0; i < o.warmup; ++i) body();
  Stopwatch sw;
  for (std::size_t i = 0; i < o.iters; ++i) body();
  return sw.elapsed_ms() / static_cast<double>(o.iters);
}

OpResult run_gemm(const WorkerOptions& o) {
  const std::size_t M = o.m, N = o.n, K = o.k;
  check_budget(checked_mul(M, K) + checked_mul(K, N) + 2 * checked_mul(M, N));
  std::vector<float> A(M * K), B(K * N), C(M * N);
  fill_uniform(A, o.seed);
  fill_uniform(B, o.seed + 1);

  OpResult r;
  r.time_ms = time_loop(o, [&] { cpu::gemm_tiled(A.data(), B.data(), C.data(), M, N, K, K, N, N, false, o.threads); });
  r.gflops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K) / (r.time_ms * 1e6);
  if (o.verify) {
    std::vector<float> ref(M * N);
    cpu::gemm_naive(A.data(), B.data(), ref.data(), M, N, K, K, N, N);
    r.max_abs_error = max_abs_diff(C, ref);
    r.add("tolerance", gemm_tolerance(K));
    if (!(r.max_abs_error <= gemm_tolerance(K))) r.error = "verification failed: max_abs_error above tolerance";
  }
  r.devices = rank_names(1);
  r.add("kernel", std::string("cpu_blocked_sgemm"));
  r.add("tile", cpu::gemm_tiled_config());
  r.add("threads", static_cast<std::size_t>(o.threads ? o.threads : default_thread_count()));
  // Compulsory traffic (each matrix touched once) -> lower-bound intensity.
  const double bytes = 4.0 * (static_cast<double>(M * K) + static_cast<double>(K * N) + static_cast<double>(M * N));
  r.add("arith_intensity_flop_per_byte", 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K) / bytes);
  return r;
}

OpResult run_splitk(const WorkerOptions& o) {
  const std::size_t M = o.m, N = o.n, K = o.k, R = o.ranks;
  check_budget(checked_mul(M, K) + checked_mul(K, N) + checked_mul(R + 1, checked_mul(M, N)));
  std::vector<float> A(M * K), B(K * N);
  fill_uniform(A, o.seed);
  fill_uniform(B, o.seed + 1);
  std::vector<std::vector<float>> C(R, std::vector<float>(M * N));

  const unsigned total_threads = o.threads ? o.threads : default_thread_count();
  // Each simulated GPU gets an equal share of the host's cores (>= 1).
  const unsigned per_rank = std::max(1U, total_threads / static_cast<unsigned>(R));

  World world(static_cast<int>(R));
  std::vector<double> compute_ms(R, 0.0), comm_ms(R, 0.0), iter_ms(R, 0.0);
  std::vector<CommStats> stats(R);

  world.run([&](Communicator& comm) {
    const auto rk = static_cast<std::size_t>(comm.rank());
    // K-slice boundaries aligned to 4 (see split_range) — same split as CUDA.
    const Range ks = split_range(K, R, rk, 4);
    float* c = C[rk].data();
    auto one_iter = [&](bool measure) {
      comm.barrier();  // all "GPUs" start together, like a synchronized launch
      Stopwatch sw;
      // C_partial = A[:, k0:k1] * B[k0:k1, :]   (A slice via pointer + lda, zero copy)
      cpu::gemm_tiled(A.data() + ks.begin, B.data() + ks.begin * N, c, M, N, ks.size(), K, N, N, false, per_rank);
      const double t_comp = sw.elapsed_ms();
      comm.barrier();  // separate compute from communication for the breakdown
      Stopwatch sc;
      comm.allReduce(c, c, M * N, RedOp::Sum);  // in place, like ncclAllReduce(sendbuff == recvbuff)
      const double t_comm = sc.elapsed_ms();
      if (measure) {
        compute_ms[rk] += t_comp;
        comm_ms[rk] += t_comm;
        iter_ms[rk] += sw.elapsed_ms();
      }
    };
    for (std::size_t i = 0; i < o.warmup; ++i) one_iter(false);
    comm.reset_stats();
    one_iter(true);  // capture byte counters for exactly one all-reduce
    stats[rk] = comm.stats();
    for (std::size_t i = 1; i < o.iters; ++i) one_iter(true);
  });

  const double it = static_cast<double>(o.iters);
  OpResult r;
  // An iteration is as slow as its slowest rank.
  r.time_ms = *std::max_element(iter_ms.begin(), iter_ms.end()) / it;
  r.gflops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K) / (r.time_ms * 1e6);
  const double comm = *std::max_element(comm_ms.begin(), comm_ms.end()) / it;
  const auto bw = allreduce_bandwidth(M * N * sizeof(float), R, comm);
  r.bus_bw_gbps = bw.busbw_gbps;

  if (o.verify) {
    std::vector<float> ref(M * N);
    cpu::gemm_naive(A.data(), B.data(), ref.data(), M, N, K, K, N, N);
    for (const auto& c : C) r.max_abs_error = std::max(r.max_abs_error, max_abs_diff(c, ref));
    r.add("tolerance", gemm_tolerance(K));
    if (!(r.max_abs_error <= gemm_tolerance(K))) r.error = "verification failed: max_abs_error above tolerance";
  }
  r.devices = rank_names(R);
  r.add("ranks", R);
  r.add("threads_per_rank", static_cast<std::size_t>(per_rank));
  r.add("tile", cpu::gemm_tiled_config());
  r.add("compute_ms", *std::max_element(compute_ms.begin(), compute_ms.end()) / it);
  r.add("allreduce_ms", comm);
  r.add("algbw_gbps", bw.algbw_gbps);
  r.add("bytes_per_rank", static_cast<std::size_t>(stats[0].bytes_sent));
  r.add("bytes_per_rank_theory", ring_allreduce_theoretical_bytes(M * N, R));
  return r;
}

OpResult run_flash(const WorkerOptions& o) {
  const cpu::AttentionShape s{o.bh, o.seq, o.dim, o.causal};
  const std::size_t elems = checked_mul(checked_mul(s.bh, s.seq), s.dim);
  check_budget(checked_mul(elems, o.verify ? 5 : 4));
  std::vector<float> Q(elems), Kt(elems), V(elems), O(elems);
  fill_uniform(Q, o.seed);
  fill_uniform(Kt, o.seed + 1);
  fill_uniform(V, o.seed + 2);
  const float scale = 1.0F / std::sqrt(static_cast<float>(s.dim));

  OpResult r;
  r.time_ms = time_loop(o, [&] { cpu::flash_attention(Q.data(), Kt.data(), V.data(), O.data(), s, scale, o.threads); });
  r.gflops = cpu::attention_flops(s) / (r.time_ms * 1e6);
  if (o.verify) {
    std::vector<float> ref(elems);
    cpu::attention_naive(Q.data(), Kt.data(), V.data(), ref.data(), s, scale);
    r.max_abs_error = max_abs_diff(O, ref);
    r.add("tolerance", kAttentionTolerance);
    if (!(r.max_abs_error <= kAttentionTolerance)) r.error = "verification failed: max_abs_error above tolerance";
  }
  r.devices = rank_names(1);
  r.add("kernel", std::string("cpu_flash_attention_fwd"));
  r.add("tile", "Br" + std::to_string(cpu::FlashBlocking::Br) + "xBc" + std::to_string(cpu::FlashBlocking::Bc));
  r.add("causal", static_cast<std::size_t>(s.causal ? 1 : 0));
  // Memory the naive algorithm would need for the score matrix vs. what the
  // tiled one actually holds per worker — the whole point of the algorithm.
  r.add("naive_scores_bytes", checked_mul(checked_mul(s.bh, s.seq), s.seq) * sizeof(float));
  r.add("flash_tile_bytes", (cpu::FlashBlocking::Br * cpu::FlashBlocking::Bc + s.dim * cpu::FlashBlocking::Bc +
                             cpu::FlashBlocking::Br * s.dim) * sizeof(float));
  return r;
}

OpResult run_allreduce(const WorkerOptions& o) {
  const std::size_t count = o.allreduce_count();
  const std::size_t R = o.ranks;
  check_budget(checked_mul(2 * R, count));
  // Out-of-place (send != recv) so repeated iterations are idempotent and
  // the final result can be verified exactly like nccl-tests does.
  std::vector<std::vector<float>> send(R, std::vector<float>(count)), recv(R, std::vector<float>(count));
  for (std::size_t r = 0; r < R; ++r) fill_uniform(send[r], o.seed + r);

  World world(static_cast<int>(R));
  std::vector<double> ms(R, 0.0);
  std::vector<CommStats> stats(R);
  world.run([&](Communicator& comm) {
    const auto rk = static_cast<std::size_t>(comm.rank());
    for (std::size_t i = 0; i < o.warmup; ++i) comm.allReduce(send[rk].data(), recv[rk].data(), count);
    comm.reset_stats();
    comm.barrier();
    Stopwatch sw;
    for (std::size_t i = 0; i < o.iters; ++i) comm.allReduce(send[rk].data(), recv[rk].data(), count);
    // Time until the *last* rank finishes: barrier, then read the clock.
    comm.barrier();
    ms[rk] = sw.elapsed_ms();
    stats[rk] = comm.stats();
  });

  OpResult r;
  r.time_ms = *std::max_element(ms.begin(), ms.end()) / static_cast<double>(o.iters);
  const auto bw = allreduce_bandwidth(count * sizeof(float), R, r.time_ms);
  r.bus_bw_gbps = bw.busbw_gbps;
  r.gflops = 0.0;  // communication benchmark; (R-1)*count adds are negligible

  if (o.verify) {
    // Reference in double; the ring sums in a rotated order per segment, so
    // allow fp32 reassociation error ~ R * eps * |sum|.
    double tol = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
      double ref = 0.0;
      for (std::size_t q = 0; q < R; ++q) ref += send[q][i];
      for (std::size_t q = 0; q < R; ++q)
        r.max_abs_error = std::max(r.max_abs_error, std::fabs(static_cast<double>(recv[q][i]) - ref));
    }
    tol = 1e-6 * static_cast<double>(R) * static_cast<double>(R);
    r.add("tolerance", tol);
    if (!(r.max_abs_error <= tol)) r.error = "verification failed: max_abs_error above tolerance";
  }

  const std::uint64_t per_iter = stats[0].bytes_sent / std::max<std::size_t>(o.iters, 1);
  std::uint64_t max_rank = 0, total = 0, stalls = 0;
  for (const auto& st : stats) {
    max_rank = std::max<std::uint64_t>(max_rank, st.bytes_sent / o.iters);
    total += st.bytes_sent / o.iters;
    stalls += st.send_stalls;
  }
  r.devices = rank_names(R);
  r.add("ranks", R);
  r.add("count", count);
  r.add("size_bytes", count * sizeof(float));
  r.add("algbw_gbps", bw.algbw_gbps);
  r.add("bytes_per_rank", static_cast<std::size_t>(per_iter));
  r.add("bytes_per_rank_max", static_cast<std::size_t>(max_rank));
  r.add("bytes_per_rank_theory", ring_allreduce_theoretical_bytes(count, R));
  r.add("bytes_total", static_cast<std::size_t>(total));
  r.add("chunk_elems", world.config().chunk_elems);
  r.add("fifo_slots", world.config().slots);
  r.add("send_stalls_per_iter", static_cast<double>(stalls) / static_cast<double>(o.iters));
  return r;
}

}  // namespace

OpResult run(const WorkerOptions& o) {
  if (o.iters == 0) throw std::invalid_argument("--iters must be >= 1");
  OpResult r;
  switch (o.op) {
    case Op::Gemm: r = run_gemm(o); break;
    case Op::SplitK: r = run_splitk(o); break;
    case Op::Flash: r = run_flash(o); break;
    case Op::AllReduce: r = run_allreduce(o); break;
  }
  r.backend = "sim";
  r.ok = r.error.empty();
  return r;
}

}  // namespace vessel::gpu::sim
