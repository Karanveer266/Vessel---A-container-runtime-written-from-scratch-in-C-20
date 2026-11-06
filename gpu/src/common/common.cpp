// SPDX-License-Identifier: MIT
#include "vessel/gpu/common.hpp"

#include <cmath>
#include <cstdio>
#include <limits>

#include "vessel/gpu/result.hpp"

namespace vessel::gpu {

namespace {
// SplitMix64 (Steele, Lea, Flood 2014). Passes BigCrush, one multiply-xorshift
// chain per output, trivially reproducible in any language.
constexpr std::uint64_t splitmix64(std::uint64_t& state) noexcept {
  std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
}  // namespace

void fill_uniform(std::span<float> out, std::uint64_t seed, float lo, float hi) noexcept {
  std::uint64_t state = seed;
  const float span = hi - lo;
  for (float& x : out) {
    // Top 24 bits -> exactly representable float in [0, 1).
    const auto bits = static_cast<std::uint32_t>(splitmix64(state) >> 40);
    x = lo + span * (static_cast<float>(bits) * 0x1.0p-24F);
  }
}

double max_abs_diff(std::span<const float> a, std::span<const float> b) noexcept {
  if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    if (std::isnan(d)) return d;  // NaN must never compare as "small"
    m = std::max(m, d);
  }
  return m;
}

Range split_range(std::size_t n, std::size_t parts, std::size_t index, std::size_t align) noexcept {
  if (parts == 0 || index >= parts) return {n, n};
  if (align == 0) align = 1;
  // Distribute whole `align`-sized units as evenly as possible; the final
  // part additionally takes the n % align tail.
  const std::size_t units = n / align;
  const std::size_t base = units / parts;
  const std::size_t rem = units % parts;
  const std::size_t b = (index * base + std::min(index, rem)) * align;
  std::size_t e = b + (base + (index < rem ? 1 : 0)) * align;
  if (index + 1 == parts) e = n;
  return {b, e};
}

unsigned default_thread_count() noexcept {
  const unsigned hc = std::thread::hardware_concurrency();
  return hc == 0 ? 1U : hc;
}

void OpResult::add(std::string key, double value) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.6g", value);
  extra.emplace_back(std::move(key), buf);
}

AllReduceBandwidth allreduce_bandwidth(std::size_t bytes, std::size_t ranks, double time_ms) noexcept {
  AllReduceBandwidth bw;
  if (time_ms <= 0.0 || ranks == 0) return bw;
  // bytes / (ms * 1e-3) / 1e9  ==  bytes / ms / 1e6   (GB/s, decimal like nccl-tests)
  bw.algbw_gbps = static_cast<double>(bytes) / time_ms / 1e6;
  const double n = static_cast<double>(ranks);
  bw.busbw_gbps = bw.algbw_gbps * (2.0 * (n - 1.0) / n);
  return bw;
}

}  // namespace vessel::gpu
