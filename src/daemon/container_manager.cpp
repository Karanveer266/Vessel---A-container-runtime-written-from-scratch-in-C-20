// SPDX-License-Identifier: Apache-2.0
#include "container_manager.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>

#include "vessel/common/clock.h"
#include "vessel/common/fs.h"
#include "vessel/common/id.h"
#include "vessel/common/log.h"
#include "vessel/common/strings.h"
#include "vessel/runtime/seccomp.h"
#include "vessel/runtime/syscalls.h"

#ifndef VESSEL_VERSION
#define VESSEL_VERSION "0.0.0-dev"
#endif

namespace vessel::daemon {

using v1::ContainerState;

struct ContainerManager::Record {
  std::string id;
  v1::ContainerSpec spec;
  trace::Backend backend = trace::Backend::kNone;

  mutable std::mutex m;
  std::condition_variable cv;
  ContainerState state = v1::CONTAINER_STATE_CREATED;
  pid_t pid = -1;
  int exit_code = 0;
  bool oom_killed = false;
  bool stop_requested = false;
  int64_t created_ns = 0, started_ns = 0, finished_ns = 0;
  std::string error;

  std::unique_ptr<cgroup::Cgroup> cg;
  std::string cgroup_path;
  uint64_t cgroup_id = 0;
  int ebpf_slot = -1;
  std::shared_ptr<trace::SyscallAggregator> agg = std::make_shared<trace::SyscallAggregator>();
  std::vector<int> denied;  // seccomp deny list, for SECCOMP_DENIED detection
  std::shared_ptr<LogBuffer> logs = std::make_shared<LogBuffer>(1 << 20);

  // Telemetry state.
  cgroup::Stats last_stats;
  uint64_t prev_cpu_usec = 0;
  int64_t prev_cpu_ts = 0;
  uint32_t millicores = 0;
  std::vector<trace::SyscallStat> final_summary;
  trace::NetCounters final_net;
  std::atomic<bool> monitor_stop{false};

  std::thread supervisor, pump, monitor;
};

namespace {

trace::Backend toBackend(v1::TraceBackend b, trace::Backend dflt) {
  switch (b) {
    case v1::TRACE_BACKEND_NONE: return trace::Backend::kNone;
    case v1::TRACE_BACKEND_EBPF: return trace::Backend::kEbpf;
    case v1::TRACE_BACKEND_PTRACE: return trace::Backend::kPtrace;
    case v1::TRACE_BACKEND_AUTO: return trace::Backend::kAuto;
    default: return dflt;
  }
}

v1::TraceBackend toProto(trace::Backend b) {
  switch (b) {
    case trace::Backend::kEbpf: return v1::TRACE_BACKEND_EBPF;
    case trace::Backend::kPtrace: return v1::TRACE_BACKEND_PTRACE;
    case trace::Backend::kAuto: return v1::TRACE_BACKEND_AUTO;
    default: return v1::TRACE_BACKEND_NONE;
  }
}

cgroup::Limits toLimits(const v1::ResourceLimits& l) {
  cgroup::Limits c;
  c.cpu_millicores = l.cpu_millicores();
  c.cpu_weight = l.cpu_weight();
  c.memory_max = l.memory_max_bytes();
  c.memory_high = l.memory_high_bytes();
  c.pids_max = l.pids_max();
  c.cpuset_cpus = l.cpuset_cpus();
  c.disable_swap = l.disable_swap();
  return c;
}

void fillPsi(v1::Psi* out, const cgroup::Psi& p) {
  out->set_some_avg10(p.some.avg10);
  out->set_some_avg60(p.some.avg60);
  out->set_full_avg10(p.full.avg10);
  out->set_some_total_usec(p.some.total_usec);
}

std::string selfDir() {
  char buf[4096];
  ssize_t n = ::readlink("/proc/self/exe", buf, sizeof buf - 1);
  if (n <= 0) return ".";
  std::string p(buf, static_cast<size_t>(n));
  return p.substr(0, p.rfind('/'));
}

bool isRunningish(ContainerState s) {
  return s == v1::CONTAINER_STATE_RUNNING || s == v1::CONTAINER_STATE_PAUSED;
}

}  // namespace

ContainerManager::ContainerManager(DaemonConfig cfg) : cfg_(std::move(cfg)) {
  if (cfg_.gpu_worker.empty()) cfg_.gpu_worker = fs::join(selfDir(), "vessel-gpu-worker");
  janitor_ = std::thread([this] { janitorLoop(); });
}

void ContainerManager::janitorLoop() {
  ::pthread_setname_np(::pthread_self(), "janitor");
  std::unique_lock lk(janitor_mu_);
  for (;;) {
    janitor_cv_.wait(lk, [&] { return janitor_stop_ || !janitor_queue_.empty(); });
    if (janitor_stop_) return;
    auto ids = std::move(janitor_queue_);
    janitor_queue_.clear();
    lk.unlock();
    for (const auto& id : ids)
      if (auto s = remove(id, false); !s.ok() && s.error().code != ENOENT)
        VLOG_WARN("manager", "auto-remove {}: {}", id, s.error().str());
    lk.lock();
  }
}

ContainerManager::~ContainerManager() {
  {
    std::lock_guard lk(janitor_mu_);
    janitor_stop_ = true;
  }
  janitor_cv_.notify_all();
  if (janitor_.joinable()) janitor_.join();
  std::vector<std::shared_ptr<Record>> all;
  {
    std::lock_guard lk(mu_);
    for (auto& [_, r] : containers_) all.push_back(r);
  }
  for (auto& r : all) (void)remove(r->id, /*force=*/true);
  ebpf_.reset();
}

Status ContainerManager::init() {
  VESSEL_TRY(fs::mkdirs(cfg_.state_dir, 0700));
  auto cg = cgroup::Manager::detect(cfg_.cgroup_root);
  if (cg.ok()) {
    cgroups_ = std::move(cg.value());
    if (size_t n = cgroups_->reapStale(); n > 0)
      VLOG_INFO("manager", "removed {} stale container cgroup(s) from a previous run", n);
  } else {
    // Running without cgroups is allowed (e.g. unprivileged dev box) but
    // every limit request will then be rejected loudly.
    VLOG_WARN("manager", "cgroups unavailable: {} — resource limits disabled", cg.error().str());
  }
  if (cfg_.enable_ebpf && trace::EbpfTracer::compiledIn()) {
    auto t = trace::EbpfTracer::open([this](trace::Event&& e) { bus_.publish(std::move(e)); });
    if (t.ok()) {
      ebpf_ = std::move(t.value());
      VLOG_INFO("manager", "eBPF tracer loaded");
    } else {
      VLOG_WARN("manager", "eBPF unavailable ({}); ptrace backend will be used", t.error().str());
    }
  }
  bus_.setWatchHook([this](const std::string& id, size_t n) { onWatchersChanged(id, n); });
  return okStatus();
}

void ContainerManager::onWatchersChanged(const std::string& id, size_t) {
  if (!ebpf_) return;
  std::vector<std::shared_ptr<Record>> targets;
  {
    std::lock_guard lk(mu_);
    for (auto& [cid, r] : containers_)
      if ((id.empty() || cid == id) && r->ebpf_slot >= 0) targets.push_back(r);
  }
  for (auto& r : targets) {
    trace::FilterConfig f;
    f.mute = bus_.watchers(r->id) == 0;
    (void)ebpf_->configure(r->ebpf_slot, f);
  }
}

std::string ContainerManager::resolveId(const std::string& ref) {
  std::lock_guard lk(mu_);
  if (containers_.contains(ref)) return ref;
  std::string match;
  for (const auto& [id, r] : containers_) {
    if (r->spec.name() == ref) return id;
    if (ref.size() >= 3 && id.compare(0, ref.size(), ref) == 0) {
      if (!match.empty()) return {};  // ambiguous prefix
      match = id;
    }
  }
  return match;
}

std::shared_ptr<ContainerManager::Record> ContainerManager::find(const std::string& ref) {
  std::string id = resolveId(ref);
  std::lock_guard lk(mu_);
  auto it = containers_.find(id);
  return it == containers_.end() ? nullptr : it->second;
}

v1::Container ContainerManager::snapshot(const Record& r) const {
  v1::Container c;
  c.set_id(r.id);
  *c.mutable_spec() = r.spec;
  std::lock_guard lk(r.m);
  c.set_state(r.state);
  c.set_host_pid(r.pid);
  c.set_exit_code(r.exit_code);
  c.set_oom_killed(r.oom_killed);
  c.set_created_at_unix_ns(r.created_ns);
  c.set_started_at_unix_ns(r.started_ns);
  c.set_finished_at_unix_ns(r.finished_ns);
  c.set_cgroup_path(r.cgroup_path);
  c.set_cgroup_id(r.cgroup_id);
  c.set_active_trace_backend(toProto(r.backend));
  c.set_error(r.error);
  return c;
}

void ContainerManager::publishLifecycle(const Record& r, const std::string& what) {
  trace::Event e;
  e.container_id = r.id;
  e.kind = trace::Kind::kLifecycle;
  e.ts_unix_ns = clk::realtimeNs();
  e.detail = what;
  e.backend = r.backend;
  bus_.publish(std::move(e));
}

Result<v1::Container> ContainerManager::create(const v1::ContainerSpec& in) {
  if (in.args().empty()) return Error{EINVAL, "spec.args must not be empty"};
  if (!in.name().empty() && !isValidName(in.name())) return Error{EINVAL, "invalid container name '" + in.name() + "'"};
  if (!in.name().empty() && !resolveId(in.name()).empty())
    return Error{EEXIST, "a container named '" + in.name() + "' already exists"};

  auto r = std::make_shared<Record>();
  r->spec = in;
  if (r->spec.rootfs().empty()) r->spec.set_rootfs(cfg_.default_rootfs);
  {
    std::lock_guard lk(mu_);
    do r->id = newContainerId();
    while (containers_.contains(r->id));
  }
  r->created_ns = clk::realtimeNs();

  // Resolve tracing backend now so the API reports what will actually run.
  trace::Backend want = toBackend(in.trace(), cfg_.default_trace);
  if (want == trace::Backend::kAuto) want = ebpf_ ? trace::Backend::kEbpf : trace::Backend::kPtrace;
  if (want == trace::Backend::kEbpf && !ebpf_) return Error{ENOTSUP, "eBPF tracing requested but not available on this host"};
  r->backend = want;

  auto profile = in.seccomp() == v1::SECCOMP_PROFILE_STRICT       ? runtime::SeccompProfile::kStrict
                 : in.seccomp() == v1::SECCOMP_PROFILE_UNCONFINED ? runtime::SeccompProfile::kUnconfined
                                                                  : runtime::SeccompProfile::kDefault;
  r->denied = runtime::seccomp::deniedSyscalls(profile);

  // cgroup: created and limited *before* the process exists.
  const auto& lim = in.limits();
  const bool wants_limits = lim.cpu_millicores() || lim.memory_max_bytes() || lim.pids_max() ||
                            lim.memory_high_bytes() || !lim.cpuset_cpus().empty() || lim.cpu_weight();
  if (cgroups_) {
    VESSEL_ASSIGN_OR_RETURN(r->cg, cgroups_->create(r->id));
    if (auto s = r->cg->apply(toLimits(lim)); !s.ok()) {
      (void)r->cg->destroy();
      return s.error();
    }
    r->cgroup_path = r->cg->path();
    r->cgroup_id = r->cg->id();
  } else if (wants_limits) {
    return Error{ENOTSUP, "resource limits requested but no usable cgroup hierarchy"};
  }
  if (auto s = fs::mkdirs(fs::join(cfg_.state_dir, r->id), 0700); !s.ok()) {
    if (r->cg) (void)r->cg->destroy();
    return s.error();
  }
  {
    std::lock_guard lk(mu_);
    containers_[r->id] = r;
  }
  publishLifecycle(*r, "created");
  VLOG_INFO("manager", "created {} ({}) trace={} cgroup={}", r->id, in.name(), trace::backendName(r->backend),
            r->cgroup_path);
  return snapshot(*r);
}

runtime::RuntimeSpec ContainerManager::toRuntimeSpec(const Record& r) const {
  runtime::RuntimeSpec s;
  const auto& p = r.spec;
  s.id = r.id;
  s.rootfs = p.rootfs();
  s.args.assign(p.args().begin(), p.args().end());
  s.env.assign(p.env().begin(), p.env().end());
  s.hostname = p.hostname().empty() ? (p.name().empty() ? r.id : p.name()) : p.hostname();
  s.working_dir = p.working_dir().empty() ? "/" : p.working_dir();
  s.network = p.network() == v1::NETWORK_MODE_NONE   ? runtime::NetworkMode::kNone
              : p.network() == v1::NETWORK_MODE_HOST ? runtime::NetworkMode::kHost
                                                     : runtime::NetworkMode::kLoopback;
  s.seccomp = p.seccomp() == v1::SECCOMP_PROFILE_STRICT       ? runtime::SeccompProfile::kStrict
              : p.seccomp() == v1::SECCOMP_PROFILE_UNCONFINED ? runtime::SeccompProfile::kUnconfined
                                                              : runtime::SeccompProfile::kDefault;
  s.read_only_rootfs = p.read_only_rootfs();
  for (const auto& m : p.mounts()) s.mounts.push_back({m.source(), m.target(), m.read_only()});
  for (const auto& m : p.uid_mappings()) s.uid_map.push_back({m.container_id(), m.host_id(), m.size()});
  for (const auto& m : p.gid_mappings()) s.gid_map.push_back({m.container_id(), m.host_id(), m.size()});
  if (!p.gpu_devices().empty()) {
    // NVIDIA userspace needs the control nodes plus one node per GPU.
    for (const char* n : {"/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset"})
      if (fs::exists(n)) s.devices.emplace_back(n);
    for (int g : p.gpu_devices()) {
      std::string n = "/dev/nvidia" + std::to_string(g);
      if (fs::exists(n)) s.devices.push_back(n);
    }
  }
  return s;
}

Result<v1::Container> ContainerManager::start(const std::string& ref) {
  auto r = find(ref);
  if (!r) return Error{ENOENT, "no such container: " + ref};
  {
    std::lock_guard lk(r->m);
    if (r->state != v1::CONTAINER_STATE_CREATED) return Error{EALREADY, "container " + r->id + " was already started"};
    r->state = v1::CONTAINER_STATE_RUNNING;  // reserve; corrected by supervisor
  }
  if (r->backend == trace::Backend::kEbpf && ebpf_) {
    auto slot = ebpf_->attach(r->id, r->cgroup_id, r->cg ? r->cg->dirFd() : -1);
    if (slot.ok()) {
      r->ebpf_slot = slot.value();
      trace::FilterConfig f;
      f.mute = bus_.watchers(r->id) == 0;
      (void)ebpf_->configure(r->ebpf_slot, f);
    } else {
      VLOG_WARN("manager", "{}: eBPF attach failed ({}), using ptrace", r->id, slot.error().str());
      r->backend = trace::Backend::kPtrace;
    }
  }
  std::promise<Status> started;
  auto fut = started.get_future();
  r->supervisor = std::thread([this, r, p = std::move(started)]() mutable { supervise(r, std::move(p)); });
  Status s = fut.get();
  if (!s.ok()) return s.error();
  return snapshot(*r);
}

void ContainerManager::supervise(std::shared_ptr<Record> r, std::promise<Status> started) {
  ::pthread_setname_np(::pthread_self(), ("sv-" + r->id.substr(0, 8)).c_str());
  runtime::RuntimeSpec spec = toRuntimeSpec(*r);
  runtime::LaunchOptions opts;
  opts.cgroup = r->cg.get();
  opts.rootless = cfg_.rootless;
  opts.state_dir = fs::join(cfg_.state_dir, r->id);
  opts.skip_cgroupfs = !cgroups_ || cgroups_->version() == cgroup::Version::kV1;

  // Sink shared by both tracer backends: tags seccomp denials.
  const std::vector<int> denied = r->denied;
  auto sink = [this, denied](trace::Event&& e) {
    if ((e.kind == trace::Kind::kSyscall) && e.ret == -EPERM &&
        std::binary_search(denied.begin(), denied.end(), e.nr)) {
      e.kind = trace::Kind::kSeccompDenied;
      e.detail = "denied by seccomp profile";
    }
    bus_.publish(std::move(e));
  };

  std::unique_ptr<trace::PtraceSession> pt;
  if (r->backend == trace::Backend::kPtrace) pt = std::make_unique<trace::PtraceSession>(r->id, sink, r->agg);
  opts.before_release = [&](pid_t pid) -> Status {
    if (pt) VESSEL_TRY(pt->seize(pid));
    // cgroup v1 hosts: seed the eBPF task tree with init; fork tracepoint
    // propagates membership to every descendant from here on.
    if (ebpf_ && r->ebpf_slot >= 0 && r->cgroup_id == 0) VESSEL_TRY(ebpf_->trackTask(r->ebpf_slot, pid));
    return okStatus();
  };

  auto proc = runtime::launch(spec, opts);
  if (!proc.ok()) {
    {
      std::lock_guard lk(r->m);
      r->state = v1::CONTAINER_STATE_FAILED;
      r->error = proc.error().str();
      r->finished_ns = clk::realtimeNs();
    }
    r->logs->close();
    if (ebpf_ && r->ebpf_slot >= 0) ebpf_->detach(r->ebpf_slot);
    r->ebpf_slot = -1;
    started.set_value(proc.error());
    return;
  }
  {
    std::lock_guard lk(r->m);
    r->pid = proc->pid;
  }
  r->pump = std::thread([this, r, out = std::move(proc->stdout_fd), err = std::move(proc->stderr_fd)]() mutable {
    pumpLogs(r, std::move(out), std::move(err));
  });

  bool decided = false;
  auto settle = [&](const Status& s) {
    decided = true;
    if (s.ok()) {
      {
        std::lock_guard lk(r->m);
        r->started_ns = clk::realtimeNs();
      }
      r->monitor = std::thread([this, r] { monitor(r); });
      publishLifecycle(*r, "started");
      VLOG_INFO("manager", "started {} pid={} clone3={}", r->id, proc->pid, proc->used_clone3);
    } else {
      std::lock_guard lk(r->m);
      r->error = s.error().str();
      ::kill(proc->pid, SIGKILL);
    }
    started.set_value(s);
  };

  int status = 0;
  if (pt) {
    status = pt->run(proc->pid, [&] {
      if (decided) return;
      auto s = runtime::awaitExec(proc.value(), /*nonblocking=*/true);
      if (s.ok() || s.error().code != EAGAIN) settle(s);
    });
    if (!decided) settle(runtime::awaitExec(proc.value()));
  } else {
    settle(runtime::awaitExec(proc.value()));
    while (::waitpid(proc->pid, &status, __WALL) < 0 && errno == EINTR) {
    }
  }
  bool failed_setup;
  {
    std::lock_guard lk(r->m);
    failed_setup = !r->error.empty() && r->started_ns == 0;
  }
  if (failed_setup) {
    {
      std::lock_guard lk(r->m);
      r->state = v1::CONTAINER_STATE_FAILED;
    }
    if (r->pump.joinable()) r->pump.join();
    r->logs->close();
    if (ebpf_ && r->ebpf_slot >= 0) ebpf_->detach(r->ebpf_slot);
    if (r->cg) (void)r->cg->killAll();
    r->cv.notify_all();
    return;
  }
  finish(*r, status);
}

void ContainerManager::finish(Record& r, int status) {
  r.monitor_stop.store(true);
  if (r.monitor.joinable()) r.monitor.join();
  // Make sure nothing survives (daemonized grandchildren in host-pid mode,
  // or setup races): the cgroup is the source of truth for membership.
  if (r.cg) (void)r.cg->killAll();
  if (r.pump.joinable()) r.pump.join();
  r.logs->close();

  bool oom = false;
  cgroup::Stats final_stats;
  if (r.cg) {
    oom = r.cg->oomKilledSinceStart();
    if (auto s = r.cg->stats(); s.ok()) final_stats = s.value();
  }
  if (ebpf_ && r.ebpf_slot >= 0) {
    r.final_summary = ebpf_->summary(r.ebpf_slot);
    r.final_net = ebpf_->net(r.ebpf_slot);
    ebpf_->detach(r.ebpf_slot);
  } else {
    r.final_summary = r.agg->snapshot();
  }
  {
    std::lock_guard lk(r.m);
    r.ebpf_slot = -1;
    r.exit_code = runtime::exitCodeFromWaitStatus(status);
    // A SIGKILL with an oom_kill increment is the OOM killer, not the user.
    r.oom_killed = oom || (r.exit_code == 137 && final_stats.events_oom_kill > 0);
    r.state = v1::CONTAINER_STATE_STOPPED;
    r.finished_ns = clk::realtimeNs();
    if (final_stats.cpu_usage_usec || final_stats.memory_peak) r.last_stats = final_stats;
  }
  r.cv.notify_all();
  if (r.oom_killed) {
    trace::Event e;
    e.container_id = r.id;
    e.kind = trace::Kind::kOomKill;
    e.ts_unix_ns = clk::realtimeNs();
    e.detail = "container killed by the OOM killer (memory.max reached)";
    bus_.publish(std::move(e));
  }
  publishLifecycle(r, "stopped exit=" + std::to_string(r.exit_code) + (r.oom_killed ? " oom" : ""));
  VLOG_INFO("manager", "{} exited code={}{}", r.id, r.exit_code, r.oom_killed ? " (OOM killed)" : "");
  if (r.spec.auto_remove()) {
    {
      std::lock_guard lk(janitor_mu_);
      janitor_queue_.push_back(r.id);
    }
    janitor_cv_.notify_one();
  }
}

void ContainerManager::pumpLogs(std::shared_ptr<Record> r, UniqueFd out, UniqueFd err) {
  ::pthread_setname_np(::pthread_self(), ("log-" + r->id.substr(0, 7)).c_str());
  pollfd p[2] = {{out.get(), POLLIN, 0}, {err.get(), POLLIN, 0}};
  int open_fds = 2;
  char buf[16384];
  while (open_fds > 0) {
    int n = ::poll(p, 2, -1);
    if (n < 0) {
      if (errno == EINTR) continue;
      break;
    }
    for (int i = 0; i < 2; ++i) {
      if (p[i].fd < 0 || !(p[i].revents & (POLLIN | POLLHUP | POLLERR))) continue;
      ssize_t got = ::read(p[i].fd, buf, sizeof buf);
      if (got > 0) {
        r->logs->append(i == 0 ? LogBuffer::kStdout : LogBuffer::kStderr, buf, static_cast<size_t>(got));
      } else if (got == 0 || errno != EINTR) {
        p[i].fd = -1;
        --open_fds;
      }
    }
  }
}

void ContainerManager::monitor(std::shared_ptr<Record> r) {
  // Event-driven where the kernel allows it:
  //   * memory.events (v2) raises POLLPRI on every change (kernfs_notify).
  //   * PSI trigger "some 100ms per 1s" on memory.pressure raises POLLPRI
  //     when the container stalls >=10% on memory — earlier warning than OOM.
  // Otherwise (v1) fall back to a 500ms poll of the counters.
  ::pthread_setname_np(::pthread_self(), ("mon-" + r->id.substr(0, 7)).c_str());
  if (!r->cg) return;
  UniqueFd events_fd, psi_fd;
  const bool v2 = cgroups_ && cgroups_->version() != cgroup::Version::kV1 && r->cg->dirFd() >= 0;
  if (v2) {
    events_fd.reset(::openat(r->cg->dirFd(), "memory.events", O_RDONLY | O_CLOEXEC));
    if (cgroups_->psiAvailable()) {
      psi_fd.reset(::openat(r->cg->dirFd(), "memory.pressure", O_RDWR | O_NONBLOCK | O_CLOEXEC));
      constexpr std::string_view trig = "some 100000 1000000";
      if (psi_fd && ::write(psi_fd.get(), trig.data(), trig.size() + 1) < 0) psi_fd.reset();
    }
  }
  uint64_t last_oom = 0, last_max = 0, last_high = 0;
  if (auto s = r->cg->stats(); s.ok()) {
    last_oom = s->events_oom_kill;
    last_max = s->events_max;
    last_high = s->events_high;
  }
  int64_t last_psi_event = 0;
  while (!r->monitor_stop.load()) {
    pollfd p[2] = {{events_fd.get(), POLLPRI, 0}, {psi_fd.get(), POLLPRI, 0}};
    int n = ::poll(p, 2, 500);
    if (n < 0 && errno != EINTR) break;
    if (p[1].revents & POLLPRI) {
      int64_t now = clk::monotonicNs();
      if (now - last_psi_event > 1'000'000'000) {  // rate-limit to 1/s
        last_psi_event = now;
        trace::Event e;
        e.container_id = r->id;
        e.kind = trace::Kind::kMemoryPressure;
        e.ts_unix_ns = clk::realtimeNs();
        if (auto s = r->cg->stats(); s.ok())
          e.detail = "memory stall avg10=" + std::to_string(s->memory_psi.some.avg10) + "%";
        bus_.publish(std::move(e));
      }
    }
    auto s = r->cg->stats();
    if (!s.ok()) continue;
    if (s->events_oom_kill > last_oom) {
      trace::Event e;
      e.container_id = r->id;
      e.kind = trace::Kind::kOomKill;
      e.ts_unix_ns = clk::realtimeNs();
      e.bytes = s->memory_max;
      e.detail = std::to_string(s->events_oom_kill - last_oom) + " task(s) OOM-killed";
      bus_.publish(std::move(e));
    }
    if (s->events_max > last_max || s->events_high > last_high) {
      trace::Event e;
      e.container_id = r->id;
      e.kind = trace::Kind::kMemoryPressure;
      e.ts_unix_ns = clk::realtimeNs();
      e.detail = "memory limit hit (max+" + std::to_string(s->events_max - last_max) + ", high+" +
                 std::to_string(s->events_high - last_high) + ")";
      bus_.publish(std::move(e));
    }
    last_oom = s->events_oom_kill;
    last_max = s->events_max;
    last_high = s->events_high;
  }
}

Result<v1::Container> ContainerManager::stop(const std::string& ref, uint32_t timeout_ms) {
  auto r = find(ref);
  if (!r) return Error{ENOENT, "no such container: " + ref};
  // NB: snapshot() takes r->m itself, so it must never be called with r->m
  // held (std::mutex is non-recursive). Decide under the lock, act after.
  pid_t pid;
  bool running;
  {
    std::lock_guard lk(r->m);
    running = isRunningish(r->state);
    pid = r->pid;
    if (running) r->stop_requested = true;
  }
  if (!running) return snapshot(*r);
  // A frozen cgroup cannot handle SIGTERM; thaw first.
  if (r->cg) (void)r->cg->freeze(false);
  if (pid > 0) {
    ::kill(pid, SIGTERM);  // mini-init forwards it to the workload
    ::kill(pid, SIGCONT);  // in case of the SIGSTOP pause fallback
  }
  {
    std::unique_lock lk(r->m);
    running = !r->cv.wait_for(lk, std::chrono::milliseconds(timeout_ms ? timeout_ms : 10000),
                              [&] { return !isRunningish(r->state); });
  }
  if (!running) return snapshot(*r);
  VLOG_INFO("manager", "{}: grace period expired, killing cgroup", r->id);
  if (r->cg) (void)r->cg->killAll();
  if (pid > 0) ::kill(pid, SIGKILL);
  std::unique_lock lk(r->m);
  r->cv.wait_for(lk, std::chrono::seconds(10), [&] { return !isRunningish(r->state); });
  lk.unlock();
  return snapshot(*r);
}

Result<v1::Container> ContainerManager::pause(const std::string& ref, bool paused) {
  auto r = find(ref);
  if (!r) return Error{ENOENT, "no such container: " + ref};
  {
    std::lock_guard lk(r->m);
    auto want_from = paused ? v1::CONTAINER_STATE_RUNNING : v1::CONTAINER_STATE_PAUSED;
    if (r->state != want_from)
      return Error{EINVAL, std::string("container is not ") + (paused ? "running" : "paused")};
  }
  if (!r->cg) return Error{ENOTSUP, "pause requires cgroups"};
  VESSEL_TRY(r->cg->freeze(paused));
  {
    std::lock_guard lk(r->m);
    if (isRunningish(r->state)) r->state = paused ? v1::CONTAINER_STATE_PAUSED : v1::CONTAINER_STATE_RUNNING;
  }
  publishLifecycle(*r, paused ? "paused" : "resumed");
  return snapshot(*r);
}

Status ContainerManager::remove(const std::string& ref, bool force) {
  auto r = find(ref);
  if (!r) return Error{ENOENT, "no such container: " + ref};
  {
    std::lock_guard lk(r->m);
    if (isRunningish(r->state) && !force) return Error{EBUSY, "container is running (use force)"};
  }
  if (force) (void)stop(r->id, 2000);
  if (r->supervisor.joinable()) r->supervisor.join();
  if (r->pump.joinable()) r->pump.join();
  if (r->monitor.joinable()) r->monitor.join();
  if (r->cg) {
    if (auto s = r->cg->destroy(); !s.ok()) VLOG_WARN("manager", "{}: {}", r->id, s.error().str());
  }
  std::string dir = fs::join(cfg_.state_dir, r->id);
  ::rmdir(fs::join(dir, "rootfs").c_str());
  ::rmdir(dir.c_str());
  {
    std::lock_guard lk(mu_);
    containers_.erase(r->id);
  }
  publishLifecycle(*r, "removed");
  return okStatus();
}

Result<v1::Container> ContainerManager::get(const std::string& ref) {
  auto r = find(ref);
  if (!r) return Error{ENOENT, "no such container: " + ref};
  return snapshot(*r);
}

std::vector<v1::Container> ContainerManager::list(bool all) {
  std::vector<std::shared_ptr<Record>> rs;
  {
    std::lock_guard lk(mu_);
    for (auto& [_, r] : containers_) rs.push_back(r);
  }
  std::vector<v1::Container> out;
  for (auto& r : rs) {
    auto c = snapshot(*r);
    if (all || isRunningish(c.state()) || c.state() == v1::CONTAINER_STATE_CREATED) out.push_back(std::move(c));
  }
  std::sort(out.begin(), out.end(), [](auto& a, auto& b) { return a.created_at_unix_ns() < b.created_at_unix_ns(); });
  return out;
}

Result<v1::Container> ContainerManager::wait(const std::string& ref, int timeout_ms) {
  auto r = find(ref);
  if (!r) return Error{ENOENT, "no such container: " + ref};
  std::unique_lock lk(r->m);
  r->cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return !isRunningish(r->state); });
  lk.unlock();
  return snapshot(*r);
}

Result<runtime::ExecResult> ContainerManager::exec(const std::string& ref, const std::vector<std::string>& args,
                                                   const std::vector<std::string>& env, int timeout_ms) {
  auto r = find(ref);
  if (!r) return Error{ENOENT, "no such container: " + ref};
  pid_t pid;
  {
    std::lock_guard lk(r->m);
    if (r->state != v1::CONTAINER_STATE_RUNNING) return Error{EINVAL, "container is not running"};
    pid = r->pid;
  }
  return runtime::execIn(pid, toRuntimeSpec(*r), r->cg.get(), args, env, timeout_ms);
}

Result<v1::StatsSample> ContainerManager::sample(const std::string& ref) {
  auto r = find(ref);
  if (!r) return Error{ENOENT, "no such container: " + ref};
  v1::StatsSample out;
  out.set_container_id(r->id);
  out.set_timestamp_unix_ns(clk::realtimeNs());
  cgroup::Stats s;
  bool live;
  {
    std::lock_guard lk(r->m);
    live = isRunningish(r->state);
    s = r->last_stats;
  }
  if (live && r->cg) {
    auto st = r->cg->stats();
    if (st.ok()) s = st.value();
  }
  {
    // Millicores from the usage delta. Concurrent streams share the derived
    // value; a new delta is only computed if >=50ms elapsed, which keeps
    // N parallel subscribers from producing N noisy micro-deltas.
    std::lock_guard lk(r->m);
    if (live) r->last_stats = s;
    int64_t now = clk::monotonicNs();
    if (r->prev_cpu_ts == 0) {
      r->prev_cpu_ts = now;
      r->prev_cpu_usec = s.cpu_usage_usec;
    } else if (now - r->prev_cpu_ts >= 50'000'000) {
      double dt_us = static_cast<double>(now - r->prev_cpu_ts) / 1000.0;
      double du = static_cast<double>(s.cpu_usage_usec >= r->prev_cpu_usec ? s.cpu_usage_usec - r->prev_cpu_usec : 0);
      r->millicores = live ? static_cast<uint32_t>(du / dt_us * 1000.0 + 0.5) : 0;
      r->prev_cpu_ts = now;
      r->prev_cpu_usec = s.cpu_usage_usec;
    }
    out.set_cpu_millicores_used(r->millicores);
  }
  out.set_cpu_usage_usec(s.cpu_usage_usec);
  out.set_cpu_user_usec(s.cpu_user_usec);
  out.set_cpu_system_usec(s.cpu_system_usec);
  out.set_cpu_nr_throttled(s.cpu_nr_throttled);
  out.set_cpu_throttled_usec(s.cpu_throttled_usec);
  out.set_memory_current_bytes(s.memory_current);
  out.set_memory_max_bytes(s.memory_max);
  out.set_memory_peak_bytes(s.memory_peak);
  out.set_memory_events_high(s.events_high);
  out.set_memory_events_max(s.events_max);
  out.set_memory_events_oom(s.events_oom);
  out.set_memory_events_oom_kill(s.events_oom_kill);
  out.set_pids_current(s.pids_current);
  out.set_pids_max(s.pids_max);
  fillPsi(out.mutable_cpu_pressure(), s.cpu_psi);
  fillPsi(out.mutable_memory_pressure(), s.memory_psi);
  fillPsi(out.mutable_io_pressure(), s.io_psi);
  trace::NetCounters net = r->final_net;
  if (ebpf_ && r->ebpf_slot >= 0) net = ebpf_->net(r->ebpf_slot);
  out.set_net_rx_bytes(net.rx_bytes);
  out.set_net_tx_bytes(net.tx_bytes);
  return out;
}

Result<v1::SyscallSummary> ContainerManager::syscallSummary(const std::string& ref) {
  auto r = find(ref);
  if (!r) return Error{ENOENT, "no such container: " + ref};
  std::vector<trace::SyscallStat> stats;
  bool live;
  {
    std::lock_guard lk(r->m);
    live = isRunningish(r->state);
  }
  if (live && ebpf_ && r->ebpf_slot >= 0) stats = ebpf_->summary(r->ebpf_slot);
  else if (live) stats = r->agg->snapshot();
  else stats = r->final_summary;
  v1::SyscallSummary out;
  out.set_container_id(r->id);
  out.set_backend(toProto(r->backend));
  uint64_t total = 0;
  for (const auto& s : stats) {
    auto* c = out.add_syscalls();
    c->set_nr(s.nr);
    c->set_name(std::string(syscalls::name(s.nr)));
    c->set_count(s.count);
    c->set_errors(s.errors);
    c->set_total_latency_ns(s.total_ns);
    c->set_max_latency_ns(s.max_ns);
    total += s.count;
  }
  out.set_total_events(total);
  if (ebpf_) out.set_dropped_events(ebpf_->drops());
  return out;
}

Result<std::shared_ptr<LogBuffer>> ContainerManager::logs(const std::string& ref) {
  auto r = find(ref);
  if (!r) return Error{ENOENT, "no such container: " + ref};
  return r->logs;
}

v1::HostInfo ContainerManager::hostInfo() const {
  v1::HostInfo h;
  utsname u{};
  if (::uname(&u) == 0) {
    h.set_kernel_release(u.release);
    h.set_machine(u.machine);
  }
#ifdef __ANDROID__
  h.set_is_android(true);
#else
  h.set_is_android(fs::exists("/system/build.prop"));
#endif
  if (cgroups_) {
    switch (cgroups_->version()) {
      case cgroup::Version::kV1: h.set_cgroup_version(v1::CGROUP_VERSION_V1); break;
      case cgroup::Version::kV2: h.set_cgroup_version(v1::CGROUP_VERSION_V2); break;
      case cgroup::Version::kHybrid: h.set_cgroup_version(v1::CGROUP_VERSION_HYBRID); break;
      default: break;
    }
    for (const auto& c : cgroups_->controllers()) h.add_cgroup_controllers(c);
    h.set_psi_available(cgroups_->psiAvailable());
  }
  h.set_ebpf_available(ebpf_ != nullptr);
  h.set_rootless(cfg_.rootless);
  h.set_daemon_version(VESSEL_VERSION);
  h.set_online_cpus(static_cast<uint32_t>(::sysconf(_SC_NPROCESSORS_ONLN)));
  struct sysinfo si {};
  if (::sysinfo(&si) == 0) h.set_total_memory_bytes(static_cast<uint64_t>(si.totalram) * si.mem_unit);
  trace::Backend d = cfg_.default_trace;
  if (d == trace::Backend::kAuto) d = ebpf_ ? trace::Backend::kEbpf : trace::Backend::kPtrace;
  h.set_default_trace_backend(toProto(d));
  // GPUs as reported by the NVIDIA kernel driver (no CUDA dependency here).
  if (auto ls = fs::listDir("/proc/driver/nvidia/gpus"); ls.ok()) {
    int idx = 0;
    for (const auto& bus : ls.value()) {
      auto info = fs::readFile("/proc/driver/nvidia/gpus/" + bus + "/information");
      auto* g = h.add_gpus();
      g->set_index(idx++);
      if (info.ok())
        for (auto line : str::split(info.value(), '\n'))
          if (str::startsWith(line, "Model:")) g->set_name(std::string(str::trim(line.substr(6))));
    }
  }
  return h;
}

Result<v1::GpuJobResult> ContainerManager::runGpuJob(const v1::GpuJobRequest& req) {
  if (!fs::exists(cfg_.gpu_worker)) return Error{ENOENT, "gpu worker not found at " + cfg_.gpu_worker};
  static constexpr const char* kOps[] = {"gemm", "gemm", "splitk", "flash", "allreduce"};
  static constexpr const char* kBackends[] = {"auto", "cuda", "sim", "auto"};
  v1::ContainerSpec spec;
  spec.set_name("gpujob-" + newContainerId().substr(0, 6));
  spec.set_rootfs(req.rootfs().empty() ? cfg_.default_rootfs : req.rootfs());
  auto add = [&](const std::string& a) { spec.add_args(a); };
  // Bind the worker's directory read-only at a fixed path: the job must not
  // depend on where the binary lives on the host (e.g. under /tmp, which the
  // container gets as a fresh tmpfs) or on the rootfs containing it.
  const std::string worker_dir = cfg_.gpu_worker.substr(0, cfg_.gpu_worker.rfind('/'));
  const std::string worker_name = cfg_.gpu_worker.substr(cfg_.gpu_worker.rfind('/') + 1);
  auto* m = spec.add_mounts();
  m->set_source(worker_dir.empty() ? "/" : worker_dir);
  m->set_target("/run/vessel-gpu");
  m->set_read_only(true);
  add("/run/vessel-gpu/" + worker_name);
  add("--op"), add(kOps[std::min<int>(req.op(), 4)]);
  add("--backend"), add(kBackends[std::min<int>(req.backend(), 3)]);
  auto num = [&](const char* flag, uint32_t v) {
    if (v) add(flag), add(std::to_string(v));
  };
  num("--m", req.m()), num("--n", req.n()), num("--k", req.k()), num("--ranks", req.ranks());
  num("--bh", req.batch_heads()), num("--seq", req.seq_len()), num("--dim", req.head_dim());
  num("--iters", req.iterations());
  if (req.causal()) add("--causal");
  if (req.verify()) add("--verify");
  *spec.mutable_limits() = req.limits();
  spec.set_network(v1::NETWORK_MODE_NONE);
  spec.set_trace(v1::TRACE_BACKEND_NONE);
  if (req.backend() != v1::GPU_BACKEND_SIMULATED) {
    for (uint32_t i = 0; i < std::max<uint32_t>(req.ranks(), 1); ++i) spec.add_gpu_devices(static_cast<int>(i));
  }
  spec.add_env("CUDA_DEVICE_ORDER=PCI_BUS_ID");

  VESSEL_ASSIGN_OR_RETURN(auto c, create(spec));
  auto started = start(c.id());
  if (!started.ok()) {
    (void)remove(c.id(), true);
    return started.error();
  }
  auto done = wait(c.id(), 15 * 60 * 1000);
  auto lb = logs(c.id());
  std::string out, err;
  if (lb.ok()) {
    uint64_t cur = 0;
    std::vector<LogBuffer::Chunk> chunks;
    bool more = true;
    while (more) {
      chunks.clear();
      more = lb.value()->read(&cur, &chunks, 0);
      for (auto& ch : chunks) (ch.stream == LogBuffer::kStdout ? out : err) += ch.data;
      if (chunks.empty()) break;
    }
  }
  v1::GpuJobResult res;
  res.set_container_id(c.id());
  for (auto line : str::split(out, '\n')) {
    auto eq = line.find('=');
    if (eq == std::string_view::npos) continue;
    std::string k(line.substr(0, eq)), v(line.substr(eq + 1));
    (*res.mutable_metrics())[k] = v;
    auto d = [&] { return std::strtod(v.c_str(), nullptr); };
    if (k == "ok") res.set_ok(v == "1");
    else if (k == "error") res.set_error(v);
    else if (k == "backend") res.set_backend(v == "cuda" ? v1::GPU_BACKEND_CUDA : v1::GPU_BACKEND_SIMULATED);
    else if (k == "time_ms") res.set_time_ms(d());
    else if (k == "gflops") res.set_gflops(d());
    else if (k == "bus_bw_gbps") res.set_bus_bandwidth_gbps(d());
    else if (k == "max_abs_error") res.set_max_abs_error(d());
    else if (k == "devices") res.set_device_names(v);
  }
  if (done.ok() && done->exit_code() != 0 && res.error().empty()) {
    res.set_ok(false);
    res.set_error((done->oom_killed() ? "OOM killed; " : "") + std::string("worker exited with code ") +
                  std::to_string(done->exit_code()) + (err.empty() ? "" : ": " + err.substr(0, 512)));
  }
  (void)remove(c.id(), true);
  return res;
}

}  // namespace vessel::daemon
