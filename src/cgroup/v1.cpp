// SPDX-License-Identifier: Apache-2.0
// cgroup v1 (legacy) driver.
//
// One directory per controller hierarchy: <ctrl-root>/vessel/<id>. Every
// controller is optional except memory/cpu/pids for the corresponding limit;
// asking for a limit whose controller is not mounted is a hard error (never
// silently run a container "unlimited" when the user asked for a limit).
//
// v1 pitfalls handled here:
//   * cpuset: a new child has EMPTY cpuset.cpus/mems and refuses tasks
//     (ENOSPC) until both are populated — copied from the parent.
//   * cpu/cpuacct are often co-mounted ("cpu,cpuacct") -> same directory.
//   * freezer may be absent (Android, gVisor) -> pause degrades to SIGSTOP
//     of every task, reported as such.
//   * no atomic cgroup.kill -> kill loop over cgroup.procs.
//   * cpuacct.usage is in ns, cpuacct.stat in USER_HZ ticks.
//   * oom: memory.oom_control "oom_kill N" (4.13+) or memory.failcnt.
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <csignal>
#include <limits>
#include <map>
#include <set>
#include <thread>

#include "drivers.h"
#include "vessel/common/fs.h"
#include "vessel/common/log.h"
#include "vessel/common/strings.h"

namespace vessel::cgroup {
namespace {

// v1 reports "unlimited" as a huge page-aligned number (LONG_MAX rounded).
constexpr uint64_t kV1Unlimited = 0x7FFFFFFFFFFFF000ULL;

class V1Cgroup final : public Cgroup {
 public:
  explicit V1Cgroup(std::map<std::string, std::string> dirs) : dirs_(std::move(dirs)) {
    oom_baseline_ = readOomKills();
  }

  Status apply(const Limits& l) override {
    if (l.cpu_millicores || l.cpu_weight) {
      if (!has("cpu")) return Error{ENOTSUP, "cpu limit requested but cpu controller not mounted"};
      if (l.cpu_millicores) {
        uint64_t quota = 0, period = 0;
        auto v = parse::cpuMaxValue(l.cpu_millicores, l.cpu_period_us);
        parse::cpuMax(v, &quota, &period);
        VESSEL_TRY(write("cpu", "cpu.cfs_period_us", std::to_string(period)));
        VESSEL_TRY(write("cpu", "cpu.cfs_quota_us", std::to_string(quota)));
      }
      if (l.cpu_weight) VESSEL_TRY(write("cpu", "cpu.shares", std::to_string(parse::weightToShares(l.cpu_weight))));
    }
    if (l.memory_max || l.memory_high) {
      if (!has("memory")) return Error{ENOTSUP, "memory limit requested but memory controller not mounted"};
      if (l.memory_max) VESSEL_TRY(write("memory", "memory.limit_in_bytes", std::to_string(l.memory_max)));
      // Closest v1 analogue of memory.high is the soft limit (reclaim target
      // under global pressure). It does not throttle like v2 does — documented.
      if (l.memory_high) (void)write("memory", "memory.soft_limit_in_bytes", std::to_string(l.memory_high));
      if (l.disable_swap && l.memory_max)
        (void)write("memory", "memory.memsw.limit_in_bytes", std::to_string(l.memory_max));
    }
    if (l.pids_max) {
      if (!has("pids")) return Error{ENOTSUP, "pids limit requested but pids controller not mounted"};
      VESSEL_TRY(write("pids", "pids.max", std::to_string(l.pids_max)));
    }
    if (!l.cpuset_cpus.empty()) {
      if (!has("cpuset")) return Error{ENOTSUP, "cpuset requested but cpuset controller not mounted"};
      VESSEL_TRY(write("cpuset", "cpuset.cpus", l.cpuset_cpus));
    }
    return okStatus();
  }

  Status addProcess(pid_t pid) override {
    std::string p = std::to_string(pid);
    for (const auto& [ctrl, dir] : dirs_) {
      if (ctrl == "cpuacct" && dirs_.contains("cpu") && dirs_.at("cpu") == dir) continue;
      auto s = fs::writeFile(fs::join(dir, "cgroup.procs"), p);
      if (!s.ok()) {
        // "tasks" is the pre-3.x per-thread interface; some kernels only
        // accept writes there.
        s = fs::writeFile(fs::join(dir, "tasks"), p);
        if (!s.ok()) return Error{s.error().code, "attach pid " + p + " to " + dir};
      }
    }
    return okStatus();
  }

  Result<Stats> stats() override {
    Stats s;
    if (has("cpuacct")) {
      if (auto c = read("cpuacct", "cpuacct.usage"); c.ok())
        s.cpu_usage_usec = str::parseInt<uint64_t>(c.value()).value_or(0) / 1000;
      if (auto c = read("cpuacct", "cpuacct.usage_user"); c.ok())
        s.cpu_user_usec = str::parseInt<uint64_t>(c.value()).value_or(0) / 1000;
      if (auto c = read("cpuacct", "cpuacct.usage_sys"); c.ok())
        s.cpu_system_usec = str::parseInt<uint64_t>(c.value()).value_or(0) / 1000;
      if (s.cpu_user_usec == 0 && s.cpu_system_usec == 0) {
        if (auto c = read("cpuacct", "cpuacct.stat"); c.ok()) {
          static const long hz = ::sysconf(_SC_CLK_TCK);
          const uint64_t mult = 1'000'000ULL / static_cast<uint64_t>(hz > 0 ? hz : 100);
          s.cpu_user_usec = parse::keyed(c.value(), "user").value_or(0) * mult;
          s.cpu_system_usec = parse::keyed(c.value(), "system").value_or(0) * mult;
        }
      }
    }
    if (has("cpu")) {
      if (auto c = read("cpu", "cpu.stat"); c.ok()) {
        s.cpu_nr_periods = parse::keyed(c.value(), "nr_periods").value_or(0);
        s.cpu_nr_throttled = parse::keyed(c.value(), "nr_throttled").value_or(0);
        s.cpu_throttled_usec = parse::keyed(c.value(), "throttled_time").value_or(0) / 1000;
      }
    }
    if (has("memory")) {
      if (auto c = read("memory", "memory.usage_in_bytes"); c.ok())
        s.memory_current = str::parseInt<uint64_t>(c.value()).value_or(0);
      if (auto c = read("memory", "memory.limit_in_bytes"); c.ok()) {
        auto v = str::parseInt<uint64_t>(c.value()).value_or(0);
        s.memory_max = v >= kV1Unlimited ? 0 : v;
      }
      if (auto c = read("memory", "memory.max_usage_in_bytes"); c.ok())
        s.memory_peak = str::parseInt<uint64_t>(c.value()).value_or(0);
      if (auto c = read("memory", "memory.failcnt"); c.ok())
        s.events_max = str::parseInt<uint64_t>(c.value()).value_or(0);
      s.events_oom_kill = readOomKills();
    }
    if (has("pids")) {
      if (auto c = read("pids", "pids.current"); c.ok())
        s.pids_current = str::parseInt<uint32_t>(c.value()).value_or(0);
      if (auto c = read("pids", "pids.max"); c.ok()) {
        auto v = parse::maxOrInt(c.value()).value_or(0);
        s.pids_max = v > std::numeric_limits<uint32_t>::max() ? 0 : static_cast<uint32_t>(v);
      }
    }
    return s;
  }

  Status freeze(bool frozen) override {
    if (has("freezer")) {
      VESSEL_TRY(write("freezer", "freezer.state", frozen ? "FROZEN" : "THAWED"));
      for (int i = 0; i < 200; ++i) {
        auto st = read("freezer", "freezer.state");
        if (st.ok() && str::trim(st.value()) == (frozen ? "FROZEN" : "THAWED")) return okStatus();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
      return Error{ETIMEDOUT, "freezer.state did not settle"};
    }
    // Degraded mode: job-control signals. Observable by the workload (it can
    // catch SIGCONT) — the API reports this via a log line.
    VESSEL_ASSIGN_OR_RETURN(auto pids, procs());
    for (pid_t p : pids) ::kill(p, frozen ? SIGSTOP : SIGCONT);
    return okStatus();
  }

  Status killAll() override {
    auto it = dirs_.find("pids");
    if (it == dirs_.end()) it = dirs_.find("memory");
    if (it == dirs_.end()) it = dirs_.begin();
    if (it == dirs_.end()) return okStatus();
    return detail::killLoop(fs::join(it->second, "cgroup.procs"));
  }

  Result<std::vector<pid_t>> procs() override {
    if (dirs_.empty()) return std::vector<pid_t>{};
    auto it = dirs_.find("pids");
    if (it == dirs_.end()) it = dirs_.begin();
    return detail::readProcs(fs::join(it->second, "cgroup.procs"));
  }

  Status destroy() override {
    Status first;
    std::set<std::string> done;
    for (const auto& [ctrl, dir] : dirs_) {
      if (!done.insert(dir).second) continue;
      auto s = detail::rmdirRetry(dir);
      if (!s.ok() && first.ok()) first = s;
    }
    return first;
  }

  uint64_t id() const override { return 0; }
  int dirFd() const override { return -1; }
  std::string path() const override {
    if (auto it = dirs_.find("memory"); it != dirs_.end()) return it->second;
    return dirs_.empty() ? std::string() : dirs_.begin()->second;
  }
  bool oomKilledSinceStart() override { return readOomKills() > oom_baseline_; }

 private:
  bool has(const std::string& c) const { return dirs_.contains(c); }
  uint64_t readOomKills() {
    if (!has("memory")) return 0;
    auto c = read("memory", "memory.oom_control");
    return c.ok() ? parse::keyed(c.value(), "oom_kill").value_or(0) : 0;
  }
  Status write(const std::string& ctrl, const char* file, const std::string& v) {
    auto s = fs::writeFile(fs::join(dirs_.at(ctrl), file), v);
    if (!s.ok()) return Error{s.error().code, dirs_.at(ctrl) + "/" + file + " <- '" + v + "'"};
    return s;
  }
  Result<std::string> read(const std::string& ctrl, const char* file) {
    return fs::readFile(fs::join(dirs_.at(ctrl), file));
  }

  std::map<std::string, std::string> dirs_;
  uint64_t oom_baseline_ = 0;
};

class V1Manager final : public Manager {
 public:
  V1Manager(std::map<std::string, std::string> dirs, bool hybrid) : ctrl_base_(std::move(dirs)), hybrid_(hybrid) {}

  Status init() {
    for (auto& [ctrl, base] : ctrl_base_) {
      std::string root = fs::join(base, "vessel");
      auto s = fs::mkdirs(root);
      if (!s.ok()) {
        VLOG_WARN("cgroup", "v1 controller {} unusable ({}), skipping", ctrl, s.error().str());
        continue;
      }
      if (ctrl == "cpuset") copyCpuset(base, root);
      roots_[ctrl] = root;
    }
    if (roots_.empty()) return Error{EACCES, "no writable cgroup v1 controller"};
    return okStatus();
  }

  Version version() const override { return hybrid_ ? Version::kHybrid : Version::kV1; }
  std::vector<std::string> controllers() const override {
    std::vector<std::string> out;
    for (const auto& [c, _] : roots_) out.push_back(c);
    return out;
  }
  bool psiAvailable() const override { return false; }
  std::string base() const override {
    if (auto it = roots_.find("memory"); it != roots_.end()) return it->second;
    return roots_.begin()->second;
  }

  Result<std::unique_ptr<Cgroup>> create(const std::string& name) override {
    std::map<std::string, std::string> dirs;
    for (const auto& [ctrl, root] : roots_) {
      std::string d = fs::join(root, name);
      if (::mkdir(d.c_str(), 0755) != 0 && errno != EEXIST) {
        auto err = Error::fromErrno("mkdir " + d);
        for (const auto& [_, made] : dirs) ::rmdir(made.c_str());
        return err;
      }
      if (ctrl == "cpuset") copyCpuset(root, d);
      dirs[ctrl] = d;
    }
    return std::unique_ptr<Cgroup>(std::make_unique<V1Cgroup>(std::move(dirs)));
  }

  size_t reapStale() override {
    size_t n = 0;
    for (const auto& [ctrl, root] : roots_) n = std::max(n, detail::reapEmptyChildren(root));
    return n;
  }

  Result<std::unique_ptr<Cgroup>> open(const std::string& name) override {
    std::map<std::string, std::string> dirs;
    for (const auto& [ctrl, root] : roots_) {
      std::string d = fs::join(root, name);
      if (fs::isDir(d)) dirs[ctrl] = d;
    }
    if (dirs.empty()) return Error{ENOENT, "cgroup " + name + " not found"};
    return std::unique_ptr<Cgroup>(std::make_unique<V1Cgroup>(std::move(dirs)));
  }

 private:
  static void copyCpuset(const std::string& parent, const std::string& child) {
    for (const char* f : {"cpuset.cpus", "cpuset.mems"}) {
      auto cur = fs::readFile(fs::join(child, f));
      if (cur.ok() && !str::trim(cur.value()).empty()) continue;
      auto v = fs::readFile(fs::join(parent, f));
      if (v.ok()) (void)fs::writeFile(fs::join(child, f), std::string(str::trim(v.value())));
    }
  }

  std::map<std::string, std::string> ctrl_base_;
  std::map<std::string, std::string> roots_;
  bool hybrid_;
};

}  // namespace

Result<std::unique_ptr<Manager>> makeV1Manager(std::map<std::string, std::string> controller_dirs, bool hybrid) {
  auto m = std::make_unique<V1Manager>(std::move(controller_dirs), hybrid);
  VESSEL_TRY(m->init());
  return std::unique_ptr<Manager>(std::move(m));
}

namespace detail {

Result<std::vector<pid_t>> readProcs(const std::string& procs_file) {
  VESSEL_ASSIGN_OR_RETURN(std::string c, fs::readFile(procs_file));
  std::vector<pid_t> out;
  for (auto l : str::split(c, '\n'))
    if (auto p = str::parseInt<pid_t>(l)) out.push_back(*p);
  return out;
}

Status killLoop(const std::string& procs_file) {
  // Tasks can fork between our read and our kill, so iterate until the
  // cgroup is observed empty. pids.max bounds how fast it can refill.
  for (int round = 0; round < 100; ++round) {
    auto pids = readProcs(procs_file);
    if (!pids.ok()) return pids.error().code == ENOENT ? okStatus() : Status(pids.error());
    if (pids.value().empty()) return okStatus();
    for (pid_t p : pids.value()) ::kill(p, SIGKILL);
    std::this_thread::sleep_for(std::chrono::milliseconds(round < 10 ? 1 : 10));
  }
  return Error{EBUSY, "kill loop: cgroup did not drain: " + procs_file};
}

size_t reapEmptyChildren(const std::string& root) {
  auto ls = fs::listDir(root);
  if (!ls.ok()) return 0;
  size_t n = 0;
  for (const auto& name : ls.value()) {
    if (name.size() != 12 || name.find_first_not_of("0123456789abcdef") != std::string::npos) continue;
    // rmdir on a cgroup fails with EBUSY if it still has tasks or children,
    // so this can never destroy a live container.
    if (::rmdir(fs::join(root, name).c_str()) == 0) ++n;
  }
  return n;
}

Status rmdirRetry(const std::string& dir) {
  for (int i = 0; i < 50; ++i) {
    if (::rmdir(dir.c_str()) == 0 || errno == ENOENT) return okStatus();
    if (errno != EBUSY) return Error::fromErrno("rmdir " + dir);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return Error{EBUSY, "rmdir " + dir + " (tasks still present)"};
}

}  // namespace detail
}  // namespace vessel::cgroup
