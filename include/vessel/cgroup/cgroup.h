// SPDX-License-Identifier: Apache-2.0
// cgroup management: one abstract interface, two drivers.
//
//   CgroupV2Driver  — unified hierarchy (/sys/fs/cgroup, "0::/..." in
//                     /proc/self/cgroup). Primary target. Uses cpu.max,
//                     memory.max/high/swap.max, pids.max, cpuset.cpus,
//                     cgroup.freeze, cgroup.kill, memory.events, PSI.
//   CgroupV1Driver  — legacy per-controller hierarchies. Still what many
//                     Android vendor kernels (and some CI sandboxes) expose,
//                     so it is a first-class citizen, not an afterthought.
//
// A Cgroup handle owns exactly one container's cgroup (one directory per
// hierarchy). Its destructor never touches the filesystem: removal is an
// explicit, fallible operation (destroy()) because rmdir on a cgroup fails
// with EBUSY while any task is still alive, and silently leaking a cgroup is
// better than blocking in a destructor.
#pragma once

#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vessel/common/result.h"

namespace vessel::cgroup {

enum class Version { kNone = 0, kV1 = 1, kV2 = 2, kHybrid = 3 };

struct Limits {
  uint32_t cpu_millicores = 0;   // 0 = unlimited
  uint32_t cpu_weight = 0;       // 1..10000, 0 = default
  uint64_t memory_max = 0;       // bytes, 0 = unlimited
  uint64_t memory_high = 0;      // bytes, 0 = unset
  uint32_t pids_max = 0;         // 0 = unlimited
  std::string cpuset_cpus;       // "" = inherit
  bool disable_swap = false;
  // CFS period used to express millicores. 100ms is the kernel default; a
  // shorter period reduces worst-case throttle latency at the cost of more
  // scheduler overhead (see details.md §cgroups).
  uint32_t cpu_period_us = 100'000;
};

struct PsiLine {
  float avg10 = 0, avg60 = 0, avg300 = 0;
  uint64_t total_usec = 0;
};
struct Psi {
  PsiLine some, full;
  bool valid = false;
};

struct Stats {
  uint64_t cpu_usage_usec = 0, cpu_user_usec = 0, cpu_system_usec = 0;
  uint64_t cpu_nr_periods = 0, cpu_nr_throttled = 0, cpu_throttled_usec = 0;
  uint64_t memory_current = 0, memory_max = 0, memory_peak = 0;
  uint64_t events_low = 0, events_high = 0, events_max = 0, events_oom = 0, events_oom_kill = 0;
  uint32_t pids_current = 0, pids_max = 0;
  Psi cpu_psi, memory_psi, io_psi;
};

// Pure functions on file contents. Exposed so they can be unit-tested against
// captured kernel output without needing a cgroup filesystem.
namespace parse {
// "max 100000" / "50000 100000"
bool cpuMax(const std::string& s, uint64_t* quota_us /*UINT64_MAX=max*/, uint64_t* period_us);
std::string cpuMaxValue(uint32_t millicores, uint32_t period_us);
// Flat keyed files: cpu.stat, memory.events, memory.stat ("key value\n")
std::optional<uint64_t> keyed(const std::string& s, const std::string& key);
// "max" or integer
std::optional<uint64_t> maxOrInt(const std::string& s);  // "max" -> UINT64_MAX
// PSI: "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull ..."
Psi psi(const std::string& s);
// v1 cpu.shares (2..262144) from v2-style weight (1..10000), mirrors the
// formula systemd/runc use so that both drivers agree on relative priority.
uint64_t weightToShares(uint32_t weight);
// /proc/self/cgroup line for the unified hierarchy ("0::/path").
std::optional<std::string> unifiedPathFromProcSelf(const std::string& proc_self_cgroup);
// v1: controller -> path ("4:memory:/foo" ...). Handles "cpu,cpuacct" co-mounts.
std::optional<std::string> v1PathFor(const std::string& proc_self_cgroup, const std::string& controller);
struct MountEntry {
  std::string root;         // field 4: root of the mount within the fs
  std::string mount_point;  // field 5
  std::string fstype;       // after the " - " separator
  std::string super_opts;   // e.g. "rw,memory" for v1 hierarchies
};
// /proc/self/mountinfo parser (handles optional fields and \040 escapes).
std::vector<MountEntry> mountinfo(const std::string& content);
}  // namespace parse

class Cgroup {
 public:
  virtual ~Cgroup() = default;
  virtual Status apply(const Limits& l) = 0;
  virtual Status addProcess(pid_t pid) = 0;
  virtual Result<Stats> stats() = 0;
  virtual Status freeze(bool frozen) = 0;
  // SIGKILL every task in the cgroup atomically (cgroup.kill on >=5.14,
  // otherwise an iterate-until-empty loop over cgroup.procs).
  virtual Status killAll() = 0;
  virtual Result<std::vector<pid_t>> procs() = 0;
  // rmdir all directories. Retries briefly on EBUSY (tasks still exiting).
  virtual Status destroy() = 0;
  // Kernel cgroup id (inode number of the v2 dir) — what bpf_get_current_
  // cgroup_id() returns. 0 when not on v2.
  virtual uint64_t id() const = 0;
  // fd of the v2 directory, usable with clone3(CLONE_INTO_CGROUP). -1 on v1.
  virtual int dirFd() const = 0;
  virtual std::string path() const = 0;  // primary path, for display
  virtual bool oomKilledSinceStart() = 0;
};

class Manager {
 public:
  virtual ~Manager() = default;
  // Auto-detect the host configuration and the subtree we are allowed to
  // manage. `root_override` forces a base path (tests / unusual mounts).
  static Result<std::unique_ptr<Manager>> detect(const std::string& root_override = {});

  virtual Version version() const = 0;
  virtual std::vector<std::string> controllers() const = 0;
  virtual bool psiAvailable() const = 0;
  virtual std::string base() const = 0;
  virtual Result<std::unique_ptr<Cgroup>> create(const std::string& name) = 0;
  // Re-attach to an existing cgroup created by a previous daemon instance.
  virtual Result<std::unique_ptr<Cgroup>> open(const std::string& name) = 0;
  // Remove empty container cgroups left behind by a crashed daemon instance
  // (names matching the 12-hex-char id format only; anything with live tasks
  // is left alone). Returns the number removed.
  virtual size_t reapStale() = 0;
};

const char* versionName(Version v) noexcept;

}  // namespace vessel::cgroup
