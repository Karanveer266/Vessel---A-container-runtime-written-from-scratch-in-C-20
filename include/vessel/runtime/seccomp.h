// SPDX-License-Identifier: Apache-2.0
// Seccomp-BPF filter compiler.
//
// Produces a classic-BPF program implementing a DENY-LIST with these
// properties:
//   1. Architecture pinned: seccomp_data.arch must equal the build arch,
//      otherwise SECCOMP_RET_KILL_PROCESS (blocks i386/x32 ABI smuggling —
//      the same syscall *number* means different syscalls per ABI).
//   2. On x86_64, any nr with the __X32_SYSCALL_BIT set is killed.
//   3. Denied syscalls return EPERM (not KILL) so well-behaved programs that
//      probe for features (e.g. glibc trying clone3) degrade gracefully.
//   4. The deny set is compiled into a BALANCED BINARY SEARCH over sorted
//      syscall numbers: O(log n) BPF instructions executed per syscall instead
//      of the O(n) linear chain libseccomp historically emitted. Seccomp runs
//      on *every* syscall of the workload, so this is a hot path.
//
// The program is pure data (std::vector<sock_filter>) so it is fully unit
// testable with the cBPF interpreter in tests/seccomp_test.cpp, without
// installing it into the test process.
#pragma once

#include <linux/filter.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "vessel/common/result.h"
#include "vessel/runtime/spec.h"

namespace vessel::runtime::seccomp {

struct Program {
  std::vector<sock_filter> insns;
  std::vector<int> denied;   // sorted syscall numbers that return EPERM
  uint32_t arch = 0;         // AUDIT_ARCH_* this program is pinned to
};

uint32_t nativeAuditArch() noexcept;

// Syscall numbers (for the native arch) denied by each profile.
std::vector<int> deniedSyscalls(SeccompProfile p);

// Compile an arbitrary sorted+unique deny list. Exposed for tests.
Result<Program> compile(std::span<const int> denied_sorted, uint32_t arch, int errno_value = 1 /*EPERM*/);
Result<Program> build(SeccompProfile p);

// Install into the *calling* thread (async-signal-safe: no allocation).
// Requires no_new_privs or CAP_SYS_ADMIN.
int install(const Program& prog) noexcept;

}  // namespace vessel::runtime::seccomp
