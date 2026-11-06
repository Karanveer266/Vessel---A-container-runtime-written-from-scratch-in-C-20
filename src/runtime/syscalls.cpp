// SPDX-License-Identifier: Apache-2.0
#include "vessel/runtime/syscalls.h"

#include <algorithm>
#include <array>
#include <string_view>

namespace vessel::syscalls {
namespace {

struct SyscallEntry {
  int nr;
  const char* name;
};
#include "syscall_table.inc"

constexpr int kMax = [] {
  int m = 0;
  for (const auto& e : kSyscalls) m = std::max(m, e.nr);
  return m;
}();

// Dense nr -> name index built at compile time: name() is called for every
// trace event, so it must be O(1) and allocation-free.
constexpr auto kByNr = [] {
  std::array<const char*, kMax + 1> a{};
  for (const auto& e : kSyscalls) a[e.nr] = e.name;
  return a;
}();

}  // namespace

std::string_view name(int nr) noexcept {
  if (nr < 0 || nr > kMax || kByNr[nr] == nullptr) return {};
  return kByNr[nr];
}

std::optional<int> number(std::string_view n) noexcept {
  for (const auto& e : kSyscalls)
    if (n == e.name) return e.nr;
  return std::nullopt;
}

int maxNr() noexcept { return kMax; }

Category categorize(int nr) noexcept {
  std::string_view n = name(nr);
  if (n.empty()) return Category::kOther;
  auto in = [&](std::initializer_list<std::string_view> l) { return std::find(l.begin(), l.end(), n) != l.end(); };
  if (in({"mmap", "munmap", "mremap", "brk", "mprotect", "madvise", "mlock", "munlock", "mlockall",
          "munlockall", "msync", "mincore", "process_madvise", "membarrier", "memfd_create"}))
    return Category::kMemory;
  if (in({"socket", "connect", "accept", "accept4", "bind", "listen", "sendto", "recvfrom", "sendmsg",
          "recvmsg", "sendmmsg", "recvmmsg", "shutdown", "getsockopt", "setsockopt", "getsockname",
          "getpeername", "socketpair"}))
    return Category::kNet;
  if (in({"clone", "clone3", "fork", "vfork", "execve", "execveat", "exit", "exit_group", "wait4",
          "waitid", "kill", "tgkill", "tkill", "setns", "unshare", "prctl", "ptrace"}))
    return Category::kProcess;
  if (in({"futex", "futex_waitv", "pipe", "pipe2", "eventfd", "eventfd2", "msgget", "msgsnd", "msgrcv",
          "semget", "semop", "shmget", "shmat", "shmdt", "mq_open"}))
    return Category::kIpc;
  if (in({"nanosleep", "clock_nanosleep", "clock_gettime", "gettimeofday", "timerfd_create",
          "timerfd_settime", "setitimer", "alarm"}))
    return Category::kTime;
  if (in({"read", "write", "pread64", "pwrite64", "readv", "writev", "open", "openat", "openat2", "close",
          "stat", "fstat", "lstat", "newfstatat", "statx", "lseek", "getdents64", "fsync", "fdatasync",
          "ioctl", "fcntl", "dup", "dup2", "dup3", "access", "faccessat", "faccessat2", "readlink",
          "readlinkat", "unlink", "unlinkat", "mkdir", "mkdirat", "rename", "renameat", "renameat2",
          "close_range", "sendfile", "splice", "copy_file_range", "io_uring_enter"}))
    return Category::kFile;
  return Category::kOther;
}

}  // namespace vessel::syscalls
