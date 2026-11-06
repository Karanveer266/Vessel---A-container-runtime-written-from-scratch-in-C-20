// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <time.h>

#include <cstdint>

namespace vessel::clk {

inline int64_t nowNs(clockid_t id) noexcept {
  timespec ts{};
  ::clock_gettime(id, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000LL + ts.tv_nsec;
}
// Wall clock, for timestamps shown to humans / sent over the API.
inline int64_t realtimeNs() noexcept { return nowNs(CLOCK_REALTIME); }
// Monotonic (includes suspend on Android: CLOCK_BOOTTIME) for intervals.
// bpf_ktime_get_ns() uses CLOCK_MONOTONIC, so kernel timestamps are converted
// with monoToRealtimeOffsetNs().
inline int64_t monotonicNs() noexcept { return nowNs(CLOCK_MONOTONIC); }
inline int64_t monoToRealtimeOffsetNs() noexcept { return realtimeNs() - monotonicNs(); }

}  // namespace vessel::clk
