// SPDX-License-Identifier: Apache-2.0
// Low-level container process launcher.
//
// launch() turns a RuntimeSpec into a running, fully isolated process:
//
//   parent (supervisor thread)                child (new namespaces)
//   ────────────────────────────              ─────────────────────────────
//   prepare(): all allocation happens here
//   clone3(CLONE_INTO_CGROUP|NEW*) ──────────► blocks on sync pipe
//     (fallback: clone(2) + cgroup.procs)
//   write setgroups/uid_map/gid_map
//   [optional] PTRACE_SEIZE
//   write 1 byte to sync pipe ───────────────► setsid, become root in userns
//                                              sethostname
//                                              unshare(CLONE_NEWCGROUP)
//                                              mount plan + pivot_root
//                                              lo up, mini-init fork
//                                              caps, no_new_privs, seccomp
//   read(error pipe):                          execve()  (error pipe is
//     EOF  -> exec succeeded                     O_CLOEXEC → EOF on success)
//     data -> {stage, errno} of the failure
//
// After clone the child is a copy of a MULTI-THREADED process. Only the
// cloning thread exists in the child, and any mutex another thread held
// (malloc arenas, iostream, the logger) is held forever. Therefore the child
// path performs raw syscalls only on memory prepared by the parent: no malloc,
// no std::string, no logging, no exceptions. This is the single most
// important invariant in the runtime; it is enforced by code structure
// (child code lives in child.cpp and only touches the ChildPlan POD).
#pragma once

#include <sys/types.h>

#include <functional>
#include <string>
#include <vector>

#include "vessel/cgroup/cgroup.h"
#include "vessel/common/result.h"
#include "vessel/common/unique_fd.h"
#include "vessel/runtime/spec.h"

namespace vessel::runtime {

struct Process {
  pid_t pid = -1;            // container PID 1, host pid namespace
  UniqueFd pidfd;            // pidfd_open() handle, -1 if unsupported
  UniqueFd stdout_fd;        // read ends of the workload's stdout/stderr
  UniqueFd stderr_fd;
  // Read end of the O_CLOEXEC error pipe. EOF = the workload exec'd; a
  // ChildError record = setup failed. Consumed by awaitExec(). Left to the
  // caller (instead of being consumed inside launch()) because a ptrace
  // tracer must keep servicing tracee stops while waiting for it — blocking
  // here would deadlock against a tracee stopped at PTRACE_EVENT_FORK.
  UniqueFd exec_status_fd;
  bool used_clone3 = false;  // CLONE_INTO_CGROUP path taken
};

// Blocks until the workload exec'd (ok) or setup failed (error naming the
// failed stage). With `nonblocking`, returns EAGAIN if undecided yet.
Status awaitExec(Process& p, bool nonblocking = false);

struct LaunchOptions {
  cgroup::Cgroup* cgroup = nullptr;   // may be null (no resource limits)
  // Called after namespaces exist but before the child runs user code. Used
  // by the ptrace backend to PTRACE_SEIZE the child so that the very first
  // syscall of the workload (its execve) is observed.
  std::function<Status(pid_t)> before_release;
  bool rootless = false;              // daemon is not real root
  // Per-container scratch directory on the host (used when rootfs == "/":
  // the host root is rbind-mounted here and pivoted into, read-only).
  std::string state_dir;
  // When true, nothing is mounted over the rootfs' /sys/fs/cgroup.
  bool skip_cgroupfs = false;
};

Result<Process> launch(const RuntimeSpec& spec, const LaunchOptions& opts);

// Run `args` inside an existing container (namespaces of `init_pid`) and
// collect its output. Implements `vesselctl exec` / the Exec RPC.
struct ExecResult {
  int exit_code = -1;
  std::string out, err;
  bool timed_out = false;
};
Result<ExecResult> execIn(pid_t init_pid, const RuntimeSpec& spec, cgroup::Cgroup* cg,
                          const std::vector<std::string>& args, const std::vector<std::string>& env,
                          int timeout_ms);

// Default id mappings for the current daemon credentials.
std::vector<IdMap> defaultUidMap(bool rootless);
std::vector<IdMap> defaultGidMap(bool rootless);

// Decode a wait(2) status into the conventional shell exit code
// (exit status, or 128+signal).
int exitCodeFromWaitStatus(int status) noexcept;

}  // namespace vessel::runtime
