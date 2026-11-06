// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <vector>

namespace vessel::runtime::caps {

// Docker/OCI default capability set (14 caps). Everything else — notably
// CAP_SYS_ADMIN, CAP_SYS_PTRACE, CAP_SYS_MODULE, CAP_BPF, CAP_PERFMON,
// CAP_NET_ADMIN — is removed from all five sets.
std::vector<int> defaultSet();

struct Plan {
  uint64_t keep_mask = 0;   // bit N = keep CAP_N
  int last_cap = 40;        // highest cap known to the running kernel
};
Plan makePlan(const std::vector<int>& keep);

// Async-signal-safe: drops the bounding set, ambient set, and sets
// effective/permitted/inheritable to keep_mask. Returns 0 or -errno.
int apply(const Plan& plan) noexcept;

int probeLastCap() noexcept;  // reads /proc/sys/kernel/cap_last_cap (parent side)

}  // namespace vessel::runtime::caps
