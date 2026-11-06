// SPDX-License-Identifier: MIT
// Vessel GPU module — small host-side utilities shared by every backend.
//
// Nothing in here may depend on CUDA: this header is compiled into the
// pure-C++ `vessel_gpu_core` library that must build on a laptop without a GPU.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace vessel::gpu {

/// Monotonic wall-clock stopwatch (steady_clock: immune to NTP slews).
class Stopwatch {
 public:
  Stopwatch() noexcept : start_(Clock::now()) {}
  void reset() noexcept { start_ = Clock::now(); }
  [[nodiscard]] double elapsed_ms() const noexcept {
    return std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
  }

 private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_;
};

/// Fills `out` with uniform values in [lo, hi) from a SplitMix64 stream.
///
/// Why not <random>? `std::uniform_real_distribution` is implementation
/// defined, so libstdc++ and libc++ (or a CUDA host build) would generate
/// different matrices for the same seed. SplitMix64 is a 3-line, bit-exact
/// generator, so CPU references and GPU runs see identical inputs everywhere.
void fill_uniform(std::span<float> out, std::uint64_t seed, float lo = -1.0F,
                  float hi = 1.0F) noexcept;

/// max_i |a_i - b_i|. Returns +inf on size mismatch, NaN-propagating.
[[nodiscard]] double max_abs_diff(std::span<const float> a, std::span<const float> b) noexcept;

/// Half-open index range [begin, end).
struct Range {
  std::size_t begin = 0;
  std::size_t end = 0;
  [[nodiscard]] constexpr std::size_t size() const noexcept { return end - begin; }
  friend constexpr bool operator==(const Range&, const Range&) = default;
};

/// Splits [0, n) into `parts` contiguous, non-overlapping ranges whose
/// interior boundaries are multiples of `align` (the last range absorbs the
/// remainder). Used for split-K slicing: aligning K-slice boundaries to 4
/// keeps every rank's A sub-matrix eligible for the float4 GEMM fast path.
/// Ranges may be empty when n < parts * align; callers must tolerate that.
[[nodiscard]] Range split_range(std::size_t n, std::size_t parts, std::size_t index,
                                std::size_t align = 1) noexcept;

/// Number of worker threads to use when the caller passes 0.
[[nodiscard]] unsigned default_thread_count() noexcept;

/// Static-partition parallel loop: splits [0, n) into `threads` contiguous
/// ranges and calls f(begin, end, thread_index) on each, the last range on the
/// calling thread. Static partitioning (instead of a work-stealing pool) is
/// deliberate: every user in this module has perfectly uniform work items, so
/// a dynamic scheduler would only add contention. Exceptions from workers are
/// propagated (the first one wins).
template <class F>
void parallel_for(std::size_t n, unsigned threads, F&& f) {
  if (n == 0) return;
  if (threads == 0) threads = default_thread_count();
  const std::size_t t = std::min<std::size_t>(threads, n);
  if (t <= 1) {
    f(std::size_t{0}, n, 0U);
    return;
  }
  const std::size_t base = n / t;
  const std::size_t rem = n % t;
  auto range = [&](std::size_t i) {
    const std::size_t b = i * base + std::min(i, rem);
    return std::pair{b, b + base + (i < rem ? 1 : 0)};
  };
  std::vector<std::exception_ptr> errors(t);
  {
    std::vector<std::jthread> pool;
    pool.reserve(t - 1);
    for (std::size_t i = 0; i + 1 < t; ++i) {
      pool.emplace_back([&, i] {
        try {
          auto [b, e] = range(i);
          f(b, e, static_cast<unsigned>(i));
        } catch (...) {
          errors[i] = std::current_exception();
        }
      });
    }
    try {
      auto [b, e] = range(t - 1);
      f(b, e, static_cast<unsigned>(t - 1));
    } catch (...) {
      errors[t - 1] = std::current_exception();
    }
  }  // jthreads join here
  for (auto& e : errors)
    if (e) std::rethrow_exception(e);
}

/// Dynamic (self-scheduling) parallel loop: workers pull indices from a shared
/// atomic counter and call f(i, thread_index). Used where work items are
/// *non*-uniform (causal attention: later query blocks see more keys). One
/// relaxed fetch_add per item is negligible next to the item cost (>= 10^4
/// FLOPs), and avoids the tail imbalance of a static split.
template <class F>
void parallel_for_dynamic(std::size_t n, unsigned threads, F&& f) {
  if (n == 0) return;
  std::atomic<std::size_t> next{0};
  if (threads == 0) threads = default_thread_count();
  const auto t = static_cast<unsigned>(std::min<std::size_t>(threads, n));
  parallel_for(t, t, [&](std::size_t, std::size_t, unsigned tid) {
    for (std::size_t i = next.fetch_add(1, std::memory_order_relaxed); i < n;
         i = next.fetch_add(1, std::memory_order_relaxed)) {
      f(i, tid);
    }
  });
}

}  // namespace vessel::gpu
