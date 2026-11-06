// SPDX-License-Identifier: Apache-2.0
// Internal: the plan executed by the cloned child.
//
// ChildPlan is a POD-ish bundle of raw pointers into storage owned by the
// parent's PlanStorage. After clone() the child sees a copy-on-write snapshot
// of that memory, so the pointers stay valid without the child allocating.
#pragma once

#include <linux/filter.h>
#include <sys/types.h>

#include <cstdint>

#include "vessel/runtime/capabilities.h"

namespace vessel::runtime::detail {

enum class Stage : int32_t {
  kNone = 0,
  kSyncWait,
  kSetsid,
  kSetId,
  kHostname,
  kMount,
  kMkdir,
  kTouch,
  kSymlink,
  kPivotRoot,
  kUmountOld,
  kChdir,
  kCgroupNs,
  kLoopback,
  kStdio,
  kCloseFds,
  kCaps,
  kNoNewPrivs,
  kSeccomp,
  kInitFork,
  kExec,
};
const char* stageName(Stage s) noexcept;

struct ChildError {
  Stage stage;
  int32_t err;
  int32_t op_index;  // index into the op list for mount/mkdir failures
};

enum class OpKind : uint8_t { kMount, kMkdir, kTouch, kSymlink, kPivot, kChdir, kUnshareCgroup };

struct Op {
  OpKind kind;
  bool optional;              // failure is ignored (feature not available)
  const char* source;         // mount source / symlink target
  const char* target;         // path the op acts on
  const char* fstype;
  const char* data;
  unsigned long flags;
  unsigned mode;
};

struct ChildPlan {
  // Synchronisation.
  int sync_rd = -1;       // child blocks reading 1 byte until parent is done
  int err_wr = -1;        // O_CLOEXEC: EOF on successful exec
  int stdout_wr = -1, stderr_wr = -1;
  int close_fds[8];       // parent-side fds to close in the child
  int n_close_fds = 0;

  bool in_userns = false;
  const char* hostname = nullptr;
  bool loopback_up = false;

  const Op* ops = nullptr;  // filesystem plan, executed in order
  int n_ops = 0;

  const char* workdir = "/";
  char* const* argv = nullptr;
  char* const* envp = nullptr;
  const char* const* exec_candidates = nullptr;  // absolute paths to try for argv[0]
  int n_exec_candidates = 0;

  bool use_init = true;
  bool no_new_privs = true;
  bool drop_caps = true;
  caps::Plan caps{};
  const sock_filter* seccomp = nullptr;  // null = unconfined
  unsigned short seccomp_len = 0;
};

// Entry point in the child; never returns.
[[noreturn]] void childMain(const ChildPlan& plan) noexcept;

// Shared with execIn(): drop privileges and exec. Never returns.
[[noreturn]] void finalizeAndExec(const ChildPlan& plan) noexcept;

// Report {stage, errno} through the error pipe and _exit(127).
[[noreturn]] void childFail(const ChildPlan& plan, Stage stage, int err, int op_index = -1) noexcept;

}  // namespace vessel::runtime::detail
