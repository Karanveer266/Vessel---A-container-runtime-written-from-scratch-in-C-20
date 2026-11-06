// SPDX-License-Identifier: Apache-2.0
// ContainerManager: owns every container and its supervisor thread.
//
// Threading model (one container):
//
//   gRPC thread ── Start() ──► spawns ┌──────────────────────────────────┐
//        ▲                            │ supervisor thread                │
//        └── launch result (future) ──┤  launch() → clone child          │
//                                     │  [ptrace] tracer loop            │
//                                     │  waitpid(pid, __WNOTHREAD)       │
//                                     │  exit bookkeeping, cleanup       │
//                                     └──────────────────────────────────┘
//                                     ┌──────────────────────────────────┐
//                                     │ log pump thread: pipes → ring    │
//                                     └──────────────────────────────────┘
//                                     ┌──────────────────────────────────┐
//                                     │ monitor thread: poll(POLLPRI) on │
//                                     │ memory.events + PSI trigger      │
//                                     └──────────────────────────────────┘
//
// Why the supervisor launches the container itself (instead of the gRPC
// thread): ptrace(2) binds a tracee to the *thread* that attached, waitpid
// with __WNOTHREAD only sees the calling thread's children, and
// PR_SET_PDEATHSIG fires when the parent *thread* (not process) exits.
// Making the long-lived supervisor thread the parent satisfies all three.
#pragma once

#include <condition_variable>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "event_bus.h"
#include "log_buffer.h"
#include "vessel/cgroup/cgroup.h"
#include "vessel/runtime/launcher.h"
#include "vessel/trace/trace.h"
#include "vessel/v1/vessel.pb.h"

namespace vessel::daemon {

struct DaemonConfig {
  std::string state_dir = "/run/vessel";
  std::string default_rootfs = "/";      // host root, read-only
  std::string gpu_worker;                // path to vessel-gpu-worker
  bool rootless = false;
  trace::Backend default_trace = trace::Backend::kAuto;
  std::string cgroup_root;               // override (tests)
  bool enable_ebpf = true;
};

class ContainerManager {
 public:
  explicit ContainerManager(DaemonConfig cfg);
  ~ContainerManager();
  Status init();  // detect cgroups, try eBPF

  Result<v1::Container> create(const v1::ContainerSpec& spec);
  Result<v1::Container> start(const std::string& ref);
  Result<v1::Container> stop(const std::string& ref, uint32_t timeout_ms);
  Result<v1::Container> pause(const std::string& ref, bool paused);
  Status remove(const std::string& ref, bool force);
  Result<v1::Container> get(const std::string& ref);
  std::vector<v1::Container> list(bool all);
  Result<runtime::ExecResult> exec(const std::string& ref, const std::vector<std::string>& args,
                                   const std::vector<std::string>& env, int timeout_ms);
  // Blocks until the container leaves RUNNING/PAUSED or timeout. Returns state.
  Result<v1::Container> wait(const std::string& ref, int timeout_ms);

  Result<v1::StatsSample> sample(const std::string& ref);
  Result<v1::SyscallSummary> syscallSummary(const std::string& ref);
  Result<std::shared_ptr<LogBuffer>> logs(const std::string& ref);
  Result<v1::GpuJobResult> runGpuJob(const v1::GpuJobRequest& req);
  v1::HostInfo hostInfo() const;

  EventBus& bus() { return bus_; }
  std::string resolveId(const std::string& ref);  // "" if not found

 private:
  struct Record;
  std::shared_ptr<Record> find(const std::string& ref);
  void supervise(std::shared_ptr<Record> r, std::promise<Status> started);
  void pumpLogs(std::shared_ptr<Record> r, UniqueFd out, UniqueFd err);
  void monitor(std::shared_ptr<Record> r);
  void finish(Record& r, int wait_status);
  void publishLifecycle(const Record& r, const std::string& what);
  void onWatchersChanged(const std::string& id, size_t n);
  void janitorLoop();
  runtime::RuntimeSpec toRuntimeSpec(const Record& r) const;
  v1::Container snapshot(const Record& r) const;

  DaemonConfig cfg_;
  std::unique_ptr<cgroup::Manager> cgroups_;
  std::unique_ptr<trace::EbpfTracer> ebpf_;
  EventBus bus_;
  mutable std::mutex mu_;
  std::map<std::string, std::shared_ptr<Record>> containers_;

  // auto_remove: finish() runs on the supervisor thread, and remove() must
  // join that very thread, so removal is handed to a janitor thread.
  std::mutex janitor_mu_;
  std::condition_variable janitor_cv_;
  std::vector<std::string> janitor_queue_;
  bool janitor_stop_ = false;
  std::thread janitor_;
};

}  // namespace vessel::daemon
