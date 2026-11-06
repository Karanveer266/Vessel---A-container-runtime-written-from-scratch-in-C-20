// SPDX-License-Identifier: Apache-2.0
#include "vessel/runtime/seccomp.h"

#include <linux/audit.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>

#include "vessel/runtime/syscalls.h"

namespace vessel::runtime::seccomp {
namespace {

constexpr uint32_t kOffNr = offsetof(seccomp_data, nr);
constexpr uint32_t kOffArch = offsetof(seccomp_data, arch);

#if defined(__x86_64__)
constexpr uint32_t kX32Bit = 0x40000000u;  // __X32_SYSCALL_BIT
#endif

// Syscalls a container must never make. Rationale per group in details.md.
constexpr const char* kDefaultDeny[] = {
    // Kernel/module/boot manipulation.
    "init_module", "finit_module", "delete_module", "kexec_load", "kexec_file_load", "reboot",
    // Global clock / swap / accounting / quota.
    "settimeofday", "clock_settime", "clock_adjtime", "adjtimex", "stime", "swapon", "swapoff", "acct",
    "quotactl", "quotactl_fd",
    // Filesystem/namespace (re)configuration — would let the workload undo
    // our pivot_root / mount hardening.
    "mount", "umount2", "pivot_root", "move_mount", "open_tree", "fsopen", "fsconfig", "fsmount", "fspick",
    "mount_setattr", "chroot",
    // Kernel keyring is not namespaced.
    "add_key", "request_key", "keyctl",
    // Obsolete / historically dangerous.
    "uselib", "ustat", "sysfs", "_sysctl", "nfsservctl", "vm86", "vm86old", "lookup_dcookie",
    "iopl", "ioperm", "create_module", "get_kernel_syms", "query_module",
    // Handle-based open bypasses path-based confinement (CVE-2014-? "shocker").
    "open_by_handle_at", "name_to_handle_at",
    // NUMA policy of other processes / page migration.
    "move_pages", "migrate_pages", "set_mempolicy_home_node",
};

constexpr const char* kStrictExtra[] = {
    // Introspection & kernel attack surface often used for exploitation.
    "ptrace", "process_vm_readv", "process_vm_writev", "kcmp", "bpf", "perf_event_open", "userfaultfd",
    "io_uring_setup", "io_uring_enter", "io_uring_register", "personality", "unshare", "setns",
    "clone3",  // forces glibc to fall back to clone(2), whose flags a future
               // argument filter can inspect (clone3 passes a struct pointer)
    "fanotify_init", "lsm_set_self_attr", "landlock_create_ruleset",
};

void addNames(std::vector<int>& out, const char* const* b, const char* const* e) {
  for (auto it = b; it != e; ++it)
    if (auto nr = syscalls::number(*it)) out.push_back(*nr);
}

sock_filter stmt(uint16_t code, uint32_t k) { return BPF_STMT(code, k); }
sock_filter jump(uint16_t code, uint32_t k, uint8_t jt, uint8_t jf) { return BPF_JUMP(code, k, jt, jf); }

// Emits a binary search over denied[lo, hi) into `out`.
//
// cBPF jumps are forward-only with 8-bit conditional offsets, so a leaf can
// NOT jump to a single shared RET at the end of a large program (the
// distance exceeds 255 — found by the randomized test). Instead every leaf
// carries its own two return instructions:
//
//     jeq #a, deny     ; up to 3 compares
//     jeq #b, deny
//     ret ALLOW
//   deny:
//     ret ERRNO
//
// All jumps are therefore local (< 8 instructions), and the only long jump
// is the JGE at each inner node, whose true-branch skips the *left subtree*
// — checked for overflow below. Cost: two extra instructions per leaf.
void emitSearch(std::vector<sock_filter>& out, const std::vector<int>& d, size_t lo, size_t hi, uint32_t ret_deny,
                bool& overflow) {
  if (hi - lo <= 3) {
    const size_t n = hi - lo;
    for (size_t i = 0; i < n; ++i) {
      // distance from this compare to the deny RET: remaining compares + ALLOW
      auto to_deny = static_cast<uint8_t>(n - i);
      out.push_back(jump(BPF_JMP | BPF_JEQ | BPF_K, static_cast<uint32_t>(d[lo + i]), to_deny, 0));
    }
    out.push_back(stmt(BPF_RET | BPF_K, SECCOMP_RET_ALLOW));
    out.push_back(stmt(BPF_RET | BPF_K, ret_deny));
    return;
  }
  size_t mid = lo + (hi - lo) / 2;
  // if (nr >= d[mid]) goto right-subtree else fall through to left-subtree
  size_t branch = out.size();
  out.push_back(jump(BPF_JMP | BPF_JGE | BPF_K, static_cast<uint32_t>(d[mid]), 0, 0));
  emitSearch(out, d, lo, mid, ret_deny, overflow);
  size_t off = out.size() - branch - 1;
  if (off > 255) overflow = true;
  out[branch].jt = static_cast<uint8_t>(off);
  emitSearch(out, d, mid, hi, ret_deny, overflow);
}

}  // namespace

uint32_t nativeAuditArch() noexcept {
#if defined(__x86_64__)
  return AUDIT_ARCH_X86_64;
#elif defined(__aarch64__)
  return AUDIT_ARCH_AARCH64;
#else
#error unsupported arch
#endif
}

std::vector<int> deniedSyscalls(SeccompProfile p) {
  std::vector<int> out;
  if (p == SeccompProfile::kUnconfined) return out;
  addNames(out, std::begin(kDefaultDeny), std::end(kDefaultDeny));
  if (p == SeccompProfile::kStrict) addNames(out, std::begin(kStrictExtra), std::end(kStrictExtra));
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

Result<Program> compile(std::span<const int> denied_sorted, uint32_t arch, int errno_value) {
  if (!std::is_sorted(denied_sorted.begin(), denied_sorted.end()))
    return Error::logic("seccomp: deny list must be sorted");
  // The only long jump is each inner JGE skipping its left subtree (~2.3
  // insns per entry); 200 entries keeps that far below the 8-bit limit, and
  // emitSearch() still detects overflow rather than assuming.
  if (denied_sorted.size() > 200) return Error::logic("seccomp: deny list too large (max 200)");

  Program p;
  p.arch = arch;
  p.denied.assign(denied_sorted.begin(), denied_sorted.end());
  auto& o = p.insns;

  // [0] A = arch; if (A != arch) KILL_PROCESS
  o.push_back(stmt(BPF_LD | BPF_W | BPF_ABS, kOffArch));
  o.push_back(jump(BPF_JMP | BPF_JEQ | BPF_K, arch, 1, 0));
  o.push_back(stmt(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS));
  // A = nr
  o.push_back(stmt(BPF_LD | BPF_W | BPF_ABS, kOffNr));
#if defined(__x86_64__)
  if (arch == AUDIT_ARCH_X86_64) {
    // x32 ABI syscalls share AUDIT_ARCH_X86_64 but have bit 30 set.
    o.push_back(jump(BPF_JMP | BPF_JGE | BPF_K, kX32Bit, 0, 1));
    o.push_back(stmt(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS));
  }
#endif

  const uint32_t ret_deny = SECCOMP_RET_ERRNO | (static_cast<uint32_t>(errno_value) & SECCOMP_RET_DATA);
  bool overflow = false;
  if (p.denied.empty()) o.push_back(stmt(BPF_RET | BPF_K, SECCOMP_RET_ALLOW));
  else emitSearch(o, p.denied, 0, p.denied.size(), ret_deny, overflow);
  if (overflow) return Error::logic("seccomp: branch offset overflow");
  if (o.size() > BPF_MAXINSNS) return Error::logic("seccomp: program too long");
  return p;
}

Result<Program> build(SeccompProfile p) {
  auto d = deniedSyscalls(p);
  return compile(d, nativeAuditArch());
}

int install(const Program& prog) noexcept {
  sock_fprog fprog{};
  fprog.len = static_cast<unsigned short>(prog.insns.size());
  fprog.filter = const_cast<sock_filter*>(prog.insns.data());
  // Prefer seccomp(2) with TSYNC so every thread gets the filter; fall back
  // to prctl on kernels/sandboxes without the syscall.
  long r = ::syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, 0, &fprog);
  if (r == 0) return 0;
  if (::prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &fprog, 0, 0) == 0) return 0;
  return -errno;
}

}  // namespace vessel::runtime::seccomp
