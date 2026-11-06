// SPDX-License-Identifier: Apache-2.0
#include <algorithm>

#include "vessel/trace/trace.h"

namespace vessel::trace {

const char* backendName(Backend b) noexcept {
  switch (b) {
    case Backend::kEbpf: return "ebpf";
    case Backend::kPtrace: return "ptrace";
    case Backend::kAuto: return "auto";
    default: return "none";
  }
}

void SyscallAggregator::record(int nr, int64_t ret, uint64_t latency_ns) noexcept {
  if (nr < 0 || nr >= kMaxNr) return;
  total_.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard lk(mu_);
  auto& s = by_nr_[static_cast<size_t>(nr)];
  s.nr = nr;
  ++s.count;
  s.total_ns += latency_ns;
  if (ret < 0 && ret > -4096) ++s.errors;  // -4095..-1 are errno values
  s.max_ns = std::max(s.max_ns, latency_ns);
}

std::vector<SyscallStat> SyscallAggregator::snapshot() const {
  std::vector<SyscallStat> out;
  {
    std::lock_guard lk(mu_);
    for (const auto& s : by_nr_)
      if (s.count) out.push_back(s);
  }
  std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.count > b.count; });
  return out;
}

}  // namespace vessel::trace
