// SPDX-License-Identifier: MIT
// Backend-neutral request/response types for vessel-gpu-worker.
//
// These are deliberately plain structs with no CUDA types so the same
// definitions are shared by the simulated backend (vessel_gpu_core), the CUDA
// backend (vessel_gpu_cuda, compiled by nvcc) and the worker front-end.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace vessel::gpu {

enum class Op { Gemm, SplitK, Flash, AllReduce };
enum class Backend { Auto, Cuda, Sim };

/// Everything the worker can be asked to do. Defaults are the documented CLI
/// defaults (see worker.hpp / README).
struct WorkerOptions {
  Op op = Op::Gemm;
  Backend backend = Backend::Auto;
  std::size_t m = 1024, n = 1024, k = 1024;  // GEMM / split-K shape
  std::size_t ranks = 2;                     // split-K / all-reduce participants
  std::size_t bh = 8, seq = 512, dim = 64;   // attention: batch*heads, seq len, head dim
  std::size_t count = 0;                     // all-reduce element count (0 => m*n)
  bool causal = false;
  std::size_t iters = 10;
  std::size_t warmup = 1;
  unsigned threads = 0;  // sim backend CPU threads (0 => hardware_concurrency)
  bool verify = false;
  std::uint64_t seed = 42;

  [[nodiscard]] std::size_t allreduce_count() const noexcept { return count ? count : m * n; }
};

/// Result of one benchmark run. `extra` holds free-form metrics that are
/// appended after the contract keys in insertion order.
struct OpResult {
  bool ok = false;
  std::string error;
  std::string backend;  // "cuda" | "sim"
  double time_ms = 0.0;
  double gflops = 0.0;
  double bus_bw_gbps = 0.0;
  double max_abs_error = 0.0;
  std::vector<std::string> devices;
  std::vector<std::pair<std::string, std::string>> extra;

  void add(std::string key, std::string value) { extra.emplace_back(std::move(key), std::move(value)); }
  void add(std::string key, double value);
  void add(std::string key, std::size_t value) { add(std::move(key), std::to_string(value)); }
};

/// Standard all-reduce bandwidth accounting, identical to nccl-tests:
///   algbw = bytes / time
///   busbw = algbw * 2(n-1)/n
/// busbw is the per-link rate a ring implementation actually needs, which makes
/// numbers comparable across rank counts (and against the link's peak).
struct AllReduceBandwidth {
  double algbw_gbps = 0.0;
  double busbw_gbps = 0.0;
};
[[nodiscard]] AllReduceBandwidth allreduce_bandwidth(std::size_t bytes, std::size_t ranks,
                                                     double time_ms) noexcept;

/// Absolute-error tolerances used by --verify. Inputs are U(-1, 1).
/// fp32 GEMM: the error of a K-term dot product grows ~ sqrt(K) * eps * |terms|
/// for random-sign data; we use a deliberately loose linear bound so that the
/// check flags *bugs* (errors O(1)), not reassociation noise.
[[nodiscard]] constexpr double gemm_tolerance(std::size_t k) noexcept {
  return 1e-4 + 1e-5 * static_cast<double>(k);
}
/// Attention outputs are convex combinations of V rows in [-1, 1].
inline constexpr double kAttentionTolerance = 1e-3;

}  // namespace vessel::gpu
