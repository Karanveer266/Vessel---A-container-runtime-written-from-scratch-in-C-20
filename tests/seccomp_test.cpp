// SPDX-License-Identifier: Apache-2.0
// The seccomp compiler is verified by *executing* the generated program in a
// small classic-BPF interpreter for every syscall number 0..1023 plus
// adversarial inputs (wrong arch, x32 bit). This proves the binary-search
// layout is equivalent to a naive membership test — without installing a
// filter into the test process.
#include <gtest/gtest.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>

#include <algorithm>
#include <cstddef>
#include <random>

#include "vessel/runtime/seccomp.h"
#include "vessel/runtime/syscalls.h"

using namespace vessel::runtime;

namespace {

// Minimal interpreter for the subset the compiler emits.
uint32_t runFilter(const std::vector<sock_filter>& prog, const seccomp_data& d, size_t* executed = nullptr) {
  uint32_t A = 0;
  size_t pc = 0, n = 0;
  const auto* bytes = reinterpret_cast<const uint8_t*>(&d);
  while (pc < prog.size()) {
    ++n;
    const auto& i = prog[pc];
    switch (i.code) {
      case BPF_LD | BPF_W | BPF_ABS:
        std::memcpy(&A, bytes + i.k, 4);
        ++pc;
        break;
      case BPF_JMP | BPF_JEQ | BPF_K: pc += 1 + (A == i.k ? i.jt : i.jf); break;
      case BPF_JMP | BPF_JGE | BPF_K: pc += 1 + (A >= i.k ? i.jt : i.jf); break;
      case BPF_JMP | BPF_JA: pc += 1 + i.k; break;
      case BPF_RET | BPF_K:
        if (executed) *executed = n;
        return i.k;
      default: ADD_FAILURE() << "unexpected opcode " << i.code; return 0;
    }
  }
  ADD_FAILURE() << "fell off the end of the program";
  return 0;
}

seccomp_data data(int nr, uint32_t arch = seccomp::nativeAuditArch()) {
  seccomp_data d{};
  d.nr = nr;
  d.arch = arch;
  return d;
}

}  // namespace

TEST(Seccomp, DefaultProfileDeniesExactlyTheList) {
  auto p = seccomp::build(SeccompProfile::kDefault);
  ASSERT_TRUE(p.ok()) << p.error().str();
  ASSERT_FALSE(p->denied.empty());
  for (int nr = 0; nr < 1024; ++nr) {
    bool want_deny = std::binary_search(p->denied.begin(), p->denied.end(), nr);
    uint32_t r = runFilter(p->insns, data(nr));
    if (want_deny) EXPECT_EQ(r, SECCOMP_RET_ERRNO | EPERM) << vessel::syscalls::name(nr);
    else EXPECT_EQ(r, SECCOMP_RET_ALLOW) << nr << " " << vessel::syscalls::name(nr);
  }
}

TEST(Seccomp, CriticalSyscallsDenied) {
  auto p = seccomp::build(SeccompProfile::kDefault);
  ASSERT_TRUE(p.ok());
  for (const char* n : {"mount", "umount2", "pivot_root", "init_module", "kexec_load", "reboot", "keyctl",
                        "open_by_handle_at", "swapon", "settimeofday"}) {
    auto nr = vessel::syscalls::number(n);
    ASSERT_TRUE(nr) << n;
    EXPECT_EQ(runFilter(p->insns, data(*nr)), SECCOMP_RET_ERRNO | EPERM) << n;
  }
  for (const char* n : {"read", "write", "openat", "mmap", "clone", "execve", "futex", "exit_group"}) {
    EXPECT_EQ(runFilter(p->insns, data(*vessel::syscalls::number(n))), SECCOMP_RET_ALLOW) << n;
  }
}

TEST(Seccomp, StrictIsSupersetOfDefault) {
  auto d = seccomp::deniedSyscalls(SeccompProfile::kDefault);
  auto s = seccomp::deniedSyscalls(SeccompProfile::kStrict);
  EXPECT_TRUE(std::includes(s.begin(), s.end(), d.begin(), d.end()));
  EXPECT_GT(s.size(), d.size());
  auto p = seccomp::build(SeccompProfile::kStrict);
  ASSERT_TRUE(p.ok());
  EXPECT_EQ(runFilter(p->insns, data(*vessel::syscalls::number("ptrace"))), SECCOMP_RET_ERRNO | EPERM);
  EXPECT_EQ(runFilter(p->insns, data(*vessel::syscalls::number("bpf"))), SECCOMP_RET_ERRNO | EPERM);
  EXPECT_TRUE(seccomp::deniedSyscalls(SeccompProfile::kUnconfined).empty());
}

TEST(Seccomp, ForeignArchitectureIsKilled) {
  auto p = seccomp::build(SeccompProfile::kDefault);
  ASSERT_TRUE(p.ok());
  // e.g. int 0x80 from a 64-bit process reports AUDIT_ARCH_I386.
  EXPECT_EQ(runFilter(p->insns, data(0, AUDIT_ARCH_I386)), SECCOMP_RET_KILL_PROCESS);
  EXPECT_EQ(runFilter(p->insns, data(1, 0xdeadbeef)), SECCOMP_RET_KILL_PROCESS);
}

#if defined(__x86_64__)
TEST(Seccomp, X32AbiIsKilled) {
  auto p = seccomp::build(SeccompProfile::kDefault);
  ASSERT_TRUE(p.ok());
  EXPECT_EQ(runFilter(p->insns, data(0x40000000 | 1)), SECCOMP_RET_KILL_PROCESS);
}
#endif

TEST(Seccomp, BinarySearchIsLogarithmic) {
  auto p = seccomp::build(SeccompProfile::kStrict);
  ASSERT_TRUE(p.ok());
  size_t worst = 0;
  for (int nr = 0; nr < 512; ++nr) {
    size_t n = 0;
    runFilter(p->insns, data(nr), &n);
    worst = std::max(worst, n);
  }
  // Linear chain would execute ~|denied|+5 instructions in the worst case.
  EXPECT_LT(worst, p->denied.size() / 3) << "worst=" << worst << " denied=" << p->denied.size();
}

TEST(Seccomp, RandomDenyListsMatchNaiveMembership) {
  std::mt19937 rng(42);
  for (int trial = 0; trial < 200; ++trial) {
    std::vector<int> deny;
    int n = static_cast<int>(rng() % 150);
    for (int i = 0; i < n; ++i) deny.push_back(static_cast<int>(rng() % 600));
    std::sort(deny.begin(), deny.end());
    deny.erase(std::unique(deny.begin(), deny.end()), deny.end());
    auto p = seccomp::compile(deny, seccomp::nativeAuditArch(), 13);
    ASSERT_TRUE(p.ok()) << p.error().str();
    for (int nr = 0; nr < 640; ++nr) {
      bool in = std::binary_search(deny.begin(), deny.end(), nr);
      ASSERT_EQ(runFilter(p->insns, data(nr)), in ? (SECCOMP_RET_ERRNO | 13u) : SECCOMP_RET_ALLOW)
          << "trial " << trial << " nr " << nr;
    }
  }
}

TEST(Seccomp, RejectsUnsortedInput) {
  std::vector<int> bad = {5, 3};
  EXPECT_FALSE(seccomp::compile(bad, seccomp::nativeAuditArch()).ok());
}
