// SPDX-License-Identifier: Apache-2.0
// cgroup v2 (unified hierarchy) driver.
//
// Layout created under the delegated base B (the cgroup the daemon starts in):
//
//   B/                      cgroup.subtree_control = +cpu +memory +pids ...
//   ├── vessel.supervisor/  the daemon's own processes (leaf)
//   └── vessel/             cgroup.subtree_control = +cpu +memory +pids ...
//       ├── 3f9a0c12d4e1/   one leaf per container
//       └── ...
//
// Why the supervisor leaf: cgroup v2's "no internal processes" rule forbids
// enabling controllers in B/cgroup.subtree_control while B itself contains
// tasks (EBUSY) unless B is the root. When running under systemd with
// Delegate=yes, or inside another container, B is not the root, so the daemon
// first evacuates itself into a sibling leaf. This is the same trick used by
// systemd-nspawn and Docker-in-Docker.
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <limits>
#include <thread>

#include "drivers.h"
#include "vessel/common/fs.h"
#include "vessel/common/log.h"
#include "vessel/common/strings.h"

namespace vessel::cgroup {
namespace {

uint64_t cgroupIdOf(int dirfd) {
  // The kernel cgroup id is the kernfs node id. name_to_handle_at() on a
  // kernfs dir returns it as an 8-byte handle (exactly what bpftool and
  // systemd use); st_ino equals it on 64-bit kernels >= 5.5 and is the fallback.
  alignas(file_handle) unsigned char buf[sizeof(file_handle) + sizeof(uint64_t)] = {};
  auto* fh = reinterpret_cast<file_handle*>(buf);
  fh->handle_bytes = sizeof(uint64_t);
  int mount_id = 0;
  if (::name_to_handle_at(dirfd, "", fh, &mount_id, AT_EMPTY_PATH) == 0 && fh->handle_bytes == sizeof(uint64_t)) {
    uint64_t id = 0;
    std::memcpy(&id, buf + sizeof(file_handle), sizeof id);
    return id;
  }
  struct stat st {};
  if (::fstat(dirfd, &st) == 0) return st.st_ino;
  return 0;
}

class V2Cgroup final : public Cgroup {
 public:
  V2Cgroup(std::string path, UniqueFd dir, bool psi)
      : path_(std::move(path)), dir_(std::move(dir)), psi_(psi), id_(cgroupIdOf(dir_.get())) {
    oom_baseline_ = readOomKills();
  }

  Status apply(const Limits& l) override {
    // Order matters: memory.high before memory.max so that a transient state
    // never has high > max (the kernel accepts it, but it is meaningless).
    VESSEL_TRY(write("cpu.max", parse::cpuMaxValue(l.cpu_millicores, l.cpu_period_us)));
    if (l.cpu_weight) VESSEL_TRY(write("cpu.weight", std::to_string(std::min(l.cpu_weight, 10000u))));
    VESSEL_TRY(write("memory.high", l.memory_high ? std::to_string(l.memory_high) : "max"));
    VESSEL_TRY(write("memory.max", l.memory_max ? std::to_string(l.memory_max) : "max"));
    if (l.disable_swap) {
      // memory.swap.max is absent when the kernel has no swap accounting.
      auto s = write("memory.swap.max", "0");
      if (!s.ok() && s.error().code != ENOENT) return s;
    }
    // Make the OOM killer take out the whole container, not one random task
    // inside it: a half-dead workload is worse than a cleanly dead one.
    (void)write("memory.oom.group", "1");
    VESSEL_TRY(write("pids.max", l.pids_max ? std::to_string(l.pids_max) : "max"));
    if (!l.cpuset_cpus.empty()) VESSEL_TRY(write("cpuset.cpus", l.cpuset_cpus));
    return okStatus();
  }

  Status addProcess(pid_t pid) override { return write("cgroup.procs", std::to_string(pid)); }

  Result<Stats> stats() override {
    Stats s;
    if (auto c = read("cpu.stat"); c.ok()) {
      const auto& t = c.value();
      s.cpu_usage_usec = parse::keyed(t, "usage_usec").value_or(0);
      s.cpu_user_usec = parse::keyed(t, "user_usec").value_or(0);
      s.cpu_system_usec = parse::keyed(t, "system_usec").value_or(0);
      s.cpu_nr_periods = parse::keyed(t, "nr_periods").value_or(0);
      s.cpu_nr_throttled = parse::keyed(t, "nr_throttled").value_or(0);
      s.cpu_throttled_usec = parse::keyed(t, "throttled_usec").value_or(0);
    } else {
      return c.error();  // cpu.stat always exists in v2; its absence = cgroup gone
    }
    if (auto c = read("memory.current"); c.ok()) s.memory_current = str::parseInt<uint64_t>(c.value()).value_or(0);
    if (auto c = read("memory.max"); c.ok()) {
      auto v = parse::maxOrInt(c.value()).value_or(0);
      s.memory_max = v == std::numeric_limits<uint64_t>::max() ? 0 : v;
    }
    if (auto c = read("memory.peak"); c.ok()) s.memory_peak = str::parseInt<uint64_t>(c.value()).value_or(0);
    if (auto c = read("memory.events"); c.ok()) {
      const auto& t = c.value();
      s.events_low = parse::keyed(t, "low").value_or(0);
      s.events_high = parse::keyed(t, "high").value_or(0);
      s.events_max = parse::keyed(t, "max").value_or(0);
      s.events_oom = parse::keyed(t, "oom").value_or(0);
      s.events_oom_kill = parse::keyed(t, "oom_kill").value_or(0);
    }
    if (auto c = read("pids.current"); c.ok()) s.pids_current = str::parseInt<uint32_t>(c.value()).value_or(0);
    if (auto c = read("pids.max"); c.ok()) {
      auto v = parse::maxOrInt(c.value()).value_or(0);
      s.pids_max = v > std::numeric_limits<uint32_t>::max() ? 0 : static_cast<uint32_t>(v);
    }
    if (psi_) {
      if (auto c = read("cpu.pressure"); c.ok()) s.cpu_psi = parse::psi(c.value());
      if (auto c = read("memory.pressure"); c.ok()) s.memory_psi = parse::psi(c.value());
      if (auto c = read("io.pressure"); c.ok()) s.io_psi = parse::psi(c.value());
    }
    return s;
  }

  Status freeze(bool frozen) override {
    VESSEL_TRY(write("cgroup.freeze", frozen ? "1" : "0"));
    // cgroup.freeze is asynchronous; wait (bounded) for cgroup.events to
    // report the new state so that Pause() returning means "really paused".
    for (int i = 0; i < 200; ++i) {
      auto ev = read("cgroup.events");
      if (ev.ok() && parse::keyed(ev.value(), "frozen") == (frozen ? 1u : 0u)) return okStatus();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return Error{ETIMEDOUT, "cgroup.freeze did not settle"};
  }

  Status killAll() override {
    // cgroup.kill (5.14+) is atomic w.r.t. concurrent forks. Fall back to a
    // kill loop on older kernels.
    auto s = write("cgroup.kill", "1");
    if (s.ok()) return s;
    return detail::killLoop(fs::join(path_, "cgroup.procs"));
  }

  Result<std::vector<pid_t>> procs() override { return detail::readProcs(fs::join(path_, "cgroup.procs")); }

  Status destroy() override {
    dir_.reset();
    return detail::rmdirRetry(path_);
  }

  uint64_t id() const override { return id_; }
  int dirFd() const override { return dir_.get(); }
  std::string path() const override { return path_; }

  bool oomKilledSinceStart() override { return readOomKills() > oom_baseline_; }

 private:
  uint64_t readOomKills() {
    auto c = read("memory.events");
    return c.ok() ? parse::keyed(c.value(), "oom_kill").value_or(0) : 0;
  }
  Status write(const char* file, const std::string& v) {
    if (!dir_) return Error{EBADF, std::string("cgroup write ") + file};
    auto s = fs::writeFileAt(dir_.get(), file, v);
    if (!s.ok()) return Error{s.error().code, path_ + "/" + file + " <- '" + v + "'"};
    return s;
  }
  Result<std::string> read(const char* file) {
    if (!dir_) return Error{EBADF, std::string("cgroup read ") + file};
    return fs::readFileAt(dir_.get(), file);
  }

  std::string path_;
  UniqueFd dir_;
  bool psi_;
  uint64_t id_;
  uint64_t oom_baseline_ = 0;
};

class V2Manager final : public Manager {
 public:
  V2Manager(std::string base, bool hybrid) : base_(std::move(base)), hybrid_(hybrid) {}

  Status init() {
    auto ctrl = fs::readFile(fs::join(base_, "cgroup.controllers"));
    if (!ctrl.ok()) return ctrl.error();
    for (auto c : str::split(str::trim(ctrl.value()), ' ')) controllers_.emplace_back(c);
    psi_ = fs::exists(fs::join(base_, "cpu.pressure")) ||
           fs::exists("/proc/pressure/cpu");

    std::string enable;
    for (const char* c : {"cpu", "memory", "pids", "cpuset", "io"})
      if (std::find(controllers_.begin(), controllers_.end(), c) != controllers_.end())
        enable += std::string(enable.empty() ? "+" : " +") + c;

    auto s = fs::writeFile(fs::join(base_, "cgroup.subtree_control"), enable);
    if (!s.ok() && s.error().code == EBUSY) {
      VLOG_INFO("cgroup", "base {} has member processes; evacuating to vessel.supervisor", base_);
      std::string sup = fs::join(base_, "vessel.supervisor");
      VESSEL_TRY(fs::mkdirs(sup));
      // Move *every* process (not just us) — subtree_control needs B empty.
      VESSEL_ASSIGN_OR_RETURN(auto pids, detail::readProcs(fs::join(base_, "cgroup.procs")));
      for (pid_t p : pids) (void)fs::writeFile(fs::join(sup, "cgroup.procs"), std::to_string(p));
      s = fs::writeFile(fs::join(base_, "cgroup.subtree_control"), enable);
    }
    if (!s.ok()) return Error{s.error().code, "enable controllers in " + base_ + ": " + s.error().message};

    root_ = fs::join(base_, "vessel");
    VESSEL_TRY(fs::mkdirs(root_));
    VESSEL_TRY(fs::writeFile(fs::join(root_, "cgroup.subtree_control"), enable));
    return okStatus();
  }

  Version version() const override { return hybrid_ ? Version::kHybrid : Version::kV2; }
  std::vector<std::string> controllers() const override { return controllers_; }
  bool psiAvailable() const override { return psi_; }
  std::string base() const override { return root_; }

  Result<std::unique_ptr<Cgroup>> create(const std::string& name) override {
    std::string p = fs::join(root_, name);
    if (::mkdir(p.c_str(), 0755) != 0) return Error::fromErrno("mkdir " + p);
    return open(name);
  }

  size_t reapStale() override { return detail::reapEmptyChildren(root_); }

  Result<std::unique_ptr<Cgroup>> open(const std::string& name) override {
    std::string p = fs::join(root_, name);
    VESSEL_ASSIGN_OR_RETURN(UniqueFd fd, fs::openDir(p));
    return std::unique_ptr<Cgroup>(std::make_unique<V2Cgroup>(p, std::move(fd), psi_));
  }

 private:
  std::string base_, root_;
  bool hybrid_;
  bool psi_ = false;
  std::vector<std::string> controllers_;
};

}  // namespace

Result<std::unique_ptr<Manager>> makeV2Manager(std::string base, bool hybrid) {
  auto m = std::make_unique<V2Manager>(std::move(base), hybrid);
  VESSEL_TRY(m->init());
  return std::unique_ptr<Manager>(std::move(m));
}

}  // namespace vessel::cgroup
