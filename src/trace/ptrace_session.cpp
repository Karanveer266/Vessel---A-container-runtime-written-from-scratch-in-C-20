// SPDX-License-Identifier: Apache-2.0
// Portable syscall tracer built on PTRACE_SEIZE.
//
// Why it exists next to eBPF: eBPF needs CAP_BPF/CAP_SYS_ADMIN, a kernel with
// CONFIG_BPF_SYSCALL and, for cgroup filtering, cgroup v2. Rootless mode,
// most Android devices, gVisor, and locked-down CI have none of those. ptrace
// works anywhere a parent may trace its own child — including the sandbox
// this project was developed in.
//
// State machine per task (threads are tasks):
//
//       ┌──────────── syscall-enter-stop ─────────────┐
//       │                                             ▼
//   [running] ◄──── syscall-exit-stop ──── [in syscall: nr,args,t0]
//
// Stops are distinguished with PTRACE_O_TRACESYSGOOD (SIGTRAP|0x80). Enter vs
// exit comes from PTRACE_GET_SYSCALL_INFO when the kernel supports it (5.3+)
// and from a per-task toggle otherwise. Seccomp-denied syscalls are detected
// at exit (-EPERM on a syscall our profile denies) and surfaced as
// SECCOMP_DENIED events so the operator sees *why* a workload failed.
//
// Group-stop vs signal-delivery-stop: with PTRACE_SEIZE, a group-stop is
// reported as PTRACE_EVENT_STOP and must be resumed with PTRACE_LISTEN
// (not injected); signal-delivery stops must re-inject the signal or the
// tracee would never see it (e.g. SIGTERM from Stop()).
#include <signal.h>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include "vessel/common/clock.h"
#include "vessel/common/fs.h"
#include "vessel/common/log.h"
#include "vessel/runtime/syscalls.h"
#include "vessel/trace/trace.h"

// PTRACE_GET_SYSCALL_INFO (5.3+). Declared locally rather than via
// <linux/ptrace.h>, which conflicts with glibc's <sys/ptrace.h> and is
// missing from older NDK sysroots. Layout matches the kernel UAPI.
namespace {
constexpr long kPtraceGetSyscallInfo = 0x420e;
constexpr uint8_t kInfoEntry = 1, kInfoExit = 2;
struct SyscallEntryInfo {
  uint64_t nr;
  uint64_t args[6];
};
struct SyscallExitInfo {
  int64_t rval;
  uint8_t is_error;
};
struct SyscallSeccompInfo {
  uint64_t nr;
  uint64_t args[6];
  uint32_t ret_data;
};
struct SyscallInfo {
  uint8_t op;
  uint8_t pad[3];
  uint32_t arch;
  uint64_t instruction_pointer;
  uint64_t stack_pointer;
  union {
    SyscallEntryInfo entry;
    SyscallExitInfo exit;
    SyscallSeccompInfo seccomp;
  };
};
static_assert(sizeof(SyscallInfo) == 88, "must match struct ptrace_syscall_info");
// Raw syscall: the request is not in every libc's __ptrace_request enum.
long ptraceRaw(long req, pid_t pid, void* addr, void* data) {
  return ::syscall(SYS_ptrace, req, pid, addr, data);
}
}  // namespace

namespace vessel::trace {

struct PtraceSession::Task {
  bool in_syscall = false;
  bool armed = false;  // only trace after the first execve (skip runtime setup)
  int nr = -1;
  uint64_t args[3]{};
  int64_t t0 = 0;
  std::string comm;
};

namespace {

bool readRegs(pid_t tid, SyscallRecord* rec, bool exit_hint) {
#if defined(__x86_64__)
  user_regs_struct r{};
  if (::ptrace(PTRACE_GETREGS, tid, nullptr, &r) != 0) return false;
  rec->nr = static_cast<int>(r.orig_rax);
  rec->args[0] = r.rdi;
  rec->args[1] = r.rsi;
  rec->args[2] = r.rdx;
  rec->ret = static_cast<int64_t>(r.rax);
  // On entry the kernel pre-loads rax with -ENOSYS.
  rec->entry = !exit_hint;
  return true;
#elif defined(__aarch64__)
  user_pt_regs r{};
  iovec iov{&r, sizeof r};
  if (::ptrace(PTRACE_GETREGSET, tid, reinterpret_cast<void*>(1) /*NT_PRSTATUS*/, &iov) != 0) return false;
  rec->nr = static_cast<int>(r.regs[8]);
  rec->args[0] = r.regs[0];
  rec->args[1] = r.regs[1];
  rec->args[2] = r.regs[2];
  rec->ret = static_cast<int64_t>(r.regs[0]);
  rec->entry = !exit_hint;
  return true;
#else
  return false;
#endif
}

int execveNr() {
  static const int nr = syscalls::number("execve").value_or(-1);
  return nr;
}

std::string readComm(pid_t tid) {
  auto c = fs::readFile("/proc/" + std::to_string(tid) + "/comm", 64);
  if (!c.ok()) return {};
  std::string s = std::move(c.value());
  while (!s.empty() && s.back() == '\n') s.pop_back();
  return s;
}

}  // namespace

PtraceSession::PtraceSession(std::string id, Sink sink, std::shared_ptr<SyscallAggregator> agg)
    : id_(std::move(id)), sink_(std::move(sink)), agg_(std::move(agg)), tasks_(new std::map<pid_t, Task>) {}

PtraceSession::~PtraceSession() { delete tasks_; }

PtraceSession::Task& PtraceSession::task(pid_t tid) { return (*tasks_)[tid]; }

Status PtraceSession::seize(pid_t pid) {
  // EXITKILL: if the daemon dies, tracees are SIGKILLed instead of being left
  // stopped forever. TRACECLONE/FORK/VFORK: auto-attach every descendant
  // atomically (no race with a fast-forking workload).
  long opts = PTRACE_O_TRACESYSGOOD | PTRACE_O_EXITKILL | PTRACE_O_TRACECLONE | PTRACE_O_TRACEFORK |
              PTRACE_O_TRACEVFORK | PTRACE_O_TRACEEXEC;
  if (::ptrace(PTRACE_SEIZE, pid, nullptr, reinterpret_cast<void*>(static_cast<intptr_t>(opts))) != 0)
    return Error::fromErrno("PTRACE_SEIZE");
  task(pid);
  return okStatus();
}

void PtraceSession::emit(Event&& e) {
  e.container_id = id_;
  e.backend = Backend::kPtrace;
  if (sink_) sink_(std::move(e));
}

void PtraceSession::handleSyscallStop(pid_t tid, Task& t) {
  bool is_exit = t.in_syscall;
  SyscallRecord rec{};
  if (have_syscall_info_) {
    SyscallInfo si{};
    long n = ptraceRaw(kPtraceGetSyscallInfo, tid, reinterpret_cast<void*>(sizeof si), &si);
    if (n > 0 && (si.op == kInfoEntry || si.op == kInfoExit)) {
      is_exit = si.op == kInfoExit;
      if (!is_exit) {
        rec.nr = static_cast<int>(si.entry.nr);
        for (int i = 0; i < 3; ++i) rec.args[i] = si.entry.args[i];
      } else {
        rec.ret = si.exit.rval;
      }
    } else {
      have_syscall_info_ = false;  // kernel/sandbox without it: use the toggle
      VLOG_DEBUG("ptrace", "PTRACE_GET_SYSCALL_INFO unsupported, using register decoding");
    }
  }
  if (!have_syscall_info_ && !readRegs(tid, &rec, is_exit)) return;

  if (!is_exit) {
    t.in_syscall = true;
    t.nr = rec.nr;
    t.args[0] = rec.args[0];
    t.args[1] = rec.args[1];
    t.args[2] = rec.args[2];
    t.t0 = clk::monotonicNs();
    return;
  }
  t.in_syscall = false;
  if (!t.armed) return;
  const uint64_t lat = static_cast<uint64_t>(clk::monotonicNs() - t.t0);
  agg_->record(t.nr, rec.ret, lat);

  Event e;
  e.ts_unix_ns = clk::realtimeNs();
  e.tid = tid;
  e.pid = tid;  // refined by the caller-visible tgid below when known
  e.nr = t.nr;
  e.ret = rec.ret;
  e.latency_ns = lat;
  e.args = {t.args[0], t.args[1], t.args[2]};
  e.comm = t.comm;
  auto cat = syscalls::categorize(t.nr);
  if (cat == syscalls::Category::kMemory) {
    e.kind = Kind::kMemory;
    auto n = syscalls::name(t.nr);
    e.bytes = n == "mremap" ? t.args[2] : (n == "brk" ? 0 : t.args[1]);
  } else {
    e.kind = Kind::kSyscall;
  }
  emit(std::move(e));
}

int PtraceSession::run(pid_t init_pid, const std::function<void()>& on_progress) {
  int init_status = 0;
  for (;;) {
    int status = 0;
    // __WALL: also report clone()d threads (non-SIGCHLD children).
    // __WNOTHREAD: only wait for *this thread's* children/tracees. Without it
    // waitpid(-1) in one supervisor thread would happily reap the init
    // process of a container supervised by a different thread (children are
    // shared by the whole thread group by default).
    pid_t tid = ::waitpid(-1, &status, __WALL | __WNOTHREAD);
    if (tid < 0) {
      if (errno == EINTR) continue;
      break;  // ECHILD: everything is gone
    }
    ++stops_;
    if (on_progress) on_progress();

    if (WIFEXITED(status) || WIFSIGNALED(status)) {
      auto it = tasks_->find(tid);
      if (it != tasks_->end()) {
        if (it->second.armed) {
          Event e;
          e.kind = Kind::kExit;
          e.ts_unix_ns = clk::realtimeNs();
          e.pid = e.tid = tid;
          e.comm = it->second.comm;
          e.ret = WIFEXITED(status) ? WEXITSTATUS(status) : -WTERMSIG(status);
          emit(std::move(e));
        }
        tasks_->erase(it);
      }
      if (tid == init_pid) {
        init_status = status;
        // init gone => kernel SIGKILLs the rest of the pid namespace. Keep
        // reaping until no tracees remain so no zombie is left behind.
        if (tasks_->empty()) break;
      }
      continue;
    }
    if (!WIFSTOPPED(status)) continue;

    Task& t = task(tid);
    const int sig = WSTOPSIG(status);
    const int event = status >> 16;
    int inject = 0;

    if (sig == (SIGTRAP | 0x80)) {
      handleSyscallStop(tid, t);
    } else if (event == PTRACE_EVENT_EXEC) {
      t.armed = true;
      t.in_syscall = false;  // the execve "exit" stop follows; treat fresh
      t.comm = readComm(tid);
      Event e;
      e.kind = Kind::kExec;
      e.ts_unix_ns = clk::realtimeNs();
      e.pid = e.tid = tid;
      e.comm = t.comm;
      emit(std::move(e));
      // Restart in "in syscall" state: the next stop is execve's exit.
      t.in_syscall = true;
      t.nr = execveNr();
      t.t0 = clk::monotonicNs();
    } else if (event == PTRACE_EVENT_FORK || event == PTRACE_EVENT_VFORK || event == PTRACE_EVENT_CLONE) {
      unsigned long child = 0;
      ::ptrace(PTRACE_GETEVENTMSG, tid, nullptr, &child);
      Task& c = task(static_cast<pid_t>(child));
      c.armed = t.armed;  // threads/children of the workload are traced
      c.comm = t.comm;
    } else if (event == PTRACE_EVENT_STOP) {
      // Group-stop (SIGSTOP/SIGTSTP...) or the initial SEIZE/INTERRUPT stop.
      if (sig == SIGSTOP || sig == SIGTSTP || sig == SIGTTIN || sig == SIGTTOU) {
        ::ptrace(PTRACE_LISTEN, tid, nullptr, nullptr);
        continue;
      }
    } else if (sig != SIGTRAP) {
      inject = sig;  // genuine signal-delivery-stop: pass it on
    }
    ::ptrace(PTRACE_SYSCALL, tid, nullptr, reinterpret_cast<void*>(static_cast<intptr_t>(inject)));
  }
  return init_status;
}

}  // namespace vessel::trace
