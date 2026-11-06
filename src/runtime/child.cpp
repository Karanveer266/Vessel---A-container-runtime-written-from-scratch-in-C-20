// SPDX-License-Identifier: Apache-2.0
// Code that runs in the freshly cloned container process.
//
// RULES FOR THIS FILE (see launcher.h for the why):
//   * Only async-signal-safe functions: raw syscalls / thin libc wrappers.
//   * No heap allocation, no C++ exceptions, no std::string, no logging.
//   * All data comes from the ChildPlan prepared by the parent.
#include "child.h"

#include <fcntl.h>
#include <linux/sched.h>
#include <net/if.h>
#include <sched.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/mount.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include "vessel/runtime/seccomp.h"

#ifndef CLOSE_RANGE_CLOEXEC
#define CLOSE_RANGE_CLOEXEC (1U << 2)
#endif

namespace vessel::runtime::detail {

const char* stageName(Stage s) noexcept {
  switch (s) {
    case Stage::kNone: return "none";
    case Stage::kSyncWait: return "sync-wait";
    case Stage::kSetsid: return "setsid";
    case Stage::kSetId: return "setresuid/setresgid";
    case Stage::kHostname: return "sethostname";
    case Stage::kMount: return "mount";
    case Stage::kMkdir: return "mkdir";
    case Stage::kTouch: return "create mount target";
    case Stage::kSymlink: return "symlink";
    case Stage::kPivotRoot: return "pivot_root";
    case Stage::kUmountOld: return "umount old root";
    case Stage::kChdir: return "chdir";
    case Stage::kCgroupNs: return "unshare(CLONE_NEWCGROUP)";
    case Stage::kLoopback: return "bring up loopback";
    case Stage::kStdio: return "stdio setup";
    case Stage::kCloseFds: return "close inherited fds";
    case Stage::kCaps: return "drop capabilities";
    case Stage::kNoNewPrivs: return "PR_SET_NO_NEW_PRIVS";
    case Stage::kSeccomp: return "install seccomp filter";
    case Stage::kInitFork: return "fork workload from init";
    case Stage::kExec: return "execve";
  }
  return "?";
}

void childFail(const ChildPlan& plan, Stage stage, int err, int op_index) noexcept {
  ChildError e{stage, err, op_index};
  if (plan.err_wr >= 0) {
    ssize_t r;
    do {
      r = ::write(plan.err_wr, &e, sizeof e);
    } while (r < 0 && errno == EINTR);
  }
  ::_exit(127);
}

namespace {

void resetSignals() noexcept {
  // The daemon ignores SIGPIPE (gRPC requires it) and blocks signals in
  // worker threads; ignored dispositions and the mask survive execve, which
  // would silently change workload semantics (e.g. `yes | head` never ending).
  for (int sig = 1; sig < NSIG; ++sig) {
    if (sig == SIGKILL || sig == SIGSTOP) continue;
    ::signal(sig, SIG_DFL);
  }
  sigset_t none;
  sigemptyset(&none);
  ::sigprocmask(SIG_SETMASK, &none, nullptr);
}

void runOps(const ChildPlan& p) noexcept {
  for (int i = 0; i < p.n_ops; ++i) {
    const Op& op = p.ops[i];
    int r = 0;
    Stage st = Stage::kMount;
    switch (op.kind) {
      case OpKind::kMount:
        r = ::mount(op.source, op.target, op.fstype, op.flags, op.data);
        st = Stage::kMount;
        break;
      case OpKind::kMkdir:
        r = ::mkdir(op.target, op.mode);
        if (r != 0 && errno == EEXIST) r = 0;
        st = Stage::kMkdir;
        break;
      case OpKind::kTouch: {
        int fd = ::open(op.target, O_WRONLY | O_CREAT | O_CLOEXEC | O_NOFOLLOW, op.mode);
        if (fd >= 0) ::close(fd);
        r = fd >= 0 ? 0 : -1;
        st = Stage::kTouch;
        break;
      }
      case OpKind::kSymlink:
        r = ::symlink(op.source, op.target);
        if (r != 0 && errno == EEXIST) r = 0;
        st = Stage::kSymlink;
        break;
      case OpKind::kChdir:
        r = ::chdir(op.target);
        st = Stage::kChdir;
        break;
      case OpKind::kPivot:
        // pivot_root(".", ".") stacks the old root on top of the new one;
        // the subsequent lazy umount of "." removes it. This avoids needing a
        // writable directory inside the rootfs for the old root (works with
        // read-only images) — the trick runc uses.
        r = static_cast<int>(::syscall(SYS_pivot_root, ".", "."));
        if (r != 0) childFail(p, Stage::kPivotRoot, errno, i);
        r = ::umount2(".", MNT_DETACH);
        if (r != 0) childFail(p, Stage::kUmountOld, errno, i);
        r = ::chdir("/");
        st = Stage::kChdir;
        break;
      case OpKind::kUnshareCgroup:
        r = ::unshare(CLONE_NEWCGROUP);
        st = Stage::kCgroupNs;
        break;
    }
    if (r != 0 && !op.optional) childFail(p, st, errno, i);
  }
}

void loopbackUp(const ChildPlan& p) noexcept {
  int s = ::socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
  if (s < 0) childFail(p, Stage::kLoopback, errno);
  ifreq ifr{};
  std::memcpy(ifr.ifr_name, "lo", 3);
  // Only issue SIOCSIFFLAGS if lo is actually down: some environments
  // (gVisor's netstack, certain vendor kernels) pre-configure it and reject
  // the ioctl with ENOTTY.
  if (::ioctl(s, SIOCGIFFLAGS, &ifr) == 0 && !(ifr.ifr_flags & IFF_UP)) {
    ifr.ifr_flags = static_cast<short>(ifr.ifr_flags | IFF_UP | IFF_RUNNING);
    if (::ioctl(s, SIOCSIFFLAGS, &ifr) != 0) {
      int e = errno;
      ::close(s);
      childFail(p, Stage::kLoopback, e);
    }
  }
  ::close(s);
}

void markFdsCloexec(const ChildPlan& p) noexcept {
  // Defense in depth: every daemon fd is opened O_CLOEXEC, but a library we
  // do not control (gRPC, libbpf, CUDA) might not be. close_range() is 5.9+;
  // fall back to a bounded fcntl loop.
  if (::syscall(SYS_close_range, 3U, ~0U, CLOSE_RANGE_CLOEXEC) == 0) return;
  for (int fd = 3; fd < 4096; ++fd) {
    int fl = ::fcntl(fd, F_GETFD);
    if (fl >= 0 && !(fl & FD_CLOEXEC)) ::fcntl(fd, F_SETFD, fl | FD_CLOEXEC);
  }
  (void)p;
}

[[noreturn]] void execWorkload(const ChildPlan& p) noexcept {
  int last = ENOENT;
  for (int i = 0; i < p.n_exec_candidates; ++i) {
    ::execve(p.exec_candidates[i], p.argv, p.envp);
    // Keep the most informative error: EACCES beats ENOENT (like execvp).
    if (errno != ENOENT || last == ENOENT) last = errno;
  }
  childFail(p, Stage::kExec, last);
}

// Minimal PID-1. A process running as PID 1 in a pid namespace gets two
// special behaviours that break naive workloads:
//   1. It inherits every orphan; if it never wait()s, zombies accumulate
//      until pids.max is hit.
//   2. The kernel drops signals for which PID 1 has no handler (SIGTERM
//      from `docker stop`-style shutdown would be silently ignored).
// So the container's PID 1 is this loop (like tini/dumb-init): it blocks all
// signals, forwards them to the workload, reaps everything, and exits with
// the workload's status. It never calls exec, which also means the ptrace
// backend never reports its syscalls (only post-exec tasks are "armed").
[[noreturn]] void initLoop(const ChildPlan& p, pid_t child) noexcept {
  sigset_t all;
  sigfillset(&all);
  for (;;) {
    siginfo_t si{};
    int sig = ::sigwaitinfo(&all, &si);
    if (sig < 0) continue;
    if (sig == SIGCHLD) {
      for (;;) {
        int status = 0;
        pid_t w = ::waitpid(-1, &status, WNOHANG);
        if (w <= 0) break;
        if (w == child) {
          // Workload is gone: kill stragglers (daemonized grandchildren) so
          // the cgroup drains, then mirror its exit status.
          ::kill(-1, SIGKILL);
          while (::waitpid(-1, nullptr, 0) > 0) {
          }
          if (WIFEXITED(status)) ::_exit(WEXITSTATUS(status));
          ::_exit(128 + WTERMSIG(status));
        }
      }
      continue;
    }
    // Forward everything else (SIGTERM, SIGINT, SIGHUP, SIGUSR1, ...).
    ::kill(child, sig);
  }
  (void)p;
}

}  // namespace

void finalizeAndExec(const ChildPlan& p) noexcept {
  if (::chdir(p.workdir) != 0) childFail(p, Stage::kChdir, errno);
  markFdsCloexec(p);
  if (p.drop_caps) {
    int r = caps::apply(p.caps);
    if (r != 0) childFail(p, Stage::kCaps, -r);
  }
  if (p.no_new_privs && ::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) childFail(p, Stage::kNoNewPrivs, errno);
  if (p.seccomp != nullptr) {
    sock_fprog fprog{p.seccomp_len, const_cast<sock_filter*>(p.seccomp)};
    if (::syscall(SYS_seccomp, 1 /*SECCOMP_SET_MODE_FILTER*/, 0, &fprog) != 0 &&
        ::prctl(PR_SET_SECCOMP, 2 /*SECCOMP_MODE_FILTER*/, &fprog, 0, 0) != 0)
      childFail(p, Stage::kSeccomp, errno);
  }
  // From here on, mount/pivot_root/etc. are denied by seccomp.
  if (p.use_init) {
    sigset_t all, old;
    sigfillset(&all);
    ::sigprocmask(SIG_BLOCK, &all, &old);  // block before fork: no lost SIGCHLD
    pid_t c = ::fork();
    if (c < 0) childFail(p, Stage::kInitFork, errno);
    if (c == 0) {
      ::sigprocmask(SIG_SETMASK, &old, nullptr);
      execWorkload(p);
    }
    // Init must not keep the error pipe open, or the parent would never see
    // EOF on a successful exec.
    if (p.err_wr >= 0) ::close(p.err_wr);
    int devnull = ::open("/dev/null", O_RDWR | O_CLOEXEC);
    if (devnull >= 0) {
      ::dup2(devnull, 1);  // drop our references to the log pipes so log EOF
      ::dup2(devnull, 2);  // tracks the workload, not init
      ::close(devnull);
    }
    initLoop(p, c);
  }
  execWorkload(p);
}

void childMain(const ChildPlan& p) noexcept {
  for (int i = 0; i < p.n_close_fds; ++i) ::close(p.close_fds[i]);
  resetSignals();
  // Die with the daemon: if vesseld crashes, orphaned containers must not
  // linger unsupervised (they would escape resource accounting UX).
  ::prctl(PR_SET_PDEATHSIG, SIGKILL, 0, 0, 0);

  // Wait until the parent has written uid/gid maps and moved us into the
  // cgroup. Until then we have no credentials in our own namespace.
  char go = 0;
  ssize_t n;
  do {
    n = ::read(p.sync_rd, &go, 1);
  } while (n < 0 && errno == EINTR);
  if (n != 1) childFail(p, Stage::kSyncWait, n < 0 ? errno : EPIPE);
  ::close(p.sync_rd);

  if (::setsid() < 0) childFail(p, Stage::kSetsid, errno);
  if (p.in_userns) {
    if (::setresgid(0, 0, 0) != 0) childFail(p, Stage::kSetId, errno);
    if (::setresuid(0, 0, 0) != 0) childFail(p, Stage::kSetId, errno);
  }

  // stdio: stdin=/dev/null of the host (opened before pivot is fine: it is a
  // char device, not a path into the host fs), stdout/stderr = log pipes.
  int devnull = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
  if (devnull < 0 || ::dup2(devnull, 0) < 0) childFail(p, Stage::kStdio, errno);
  ::close(devnull);
  if (p.stdout_wr >= 0 && ::dup2(p.stdout_wr, 1) < 0) childFail(p, Stage::kStdio, errno);
  if (p.stderr_wr >= 0 && ::dup2(p.stderr_wr, 2) < 0) childFail(p, Stage::kStdio, errno);
  if (p.stdout_wr > 2) ::close(p.stdout_wr);
  if (p.stderr_wr > 2) ::close(p.stderr_wr);

  if (p.hostname != nullptr && ::sethostname(p.hostname, std::strlen(p.hostname)) != 0)
    childFail(p, Stage::kHostname, errno);

  runOps(p);
  if (p.loopback_up) loopbackUp(p);
  finalizeAndExec(p);
}

}  // namespace vessel::runtime::detail
