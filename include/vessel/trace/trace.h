// SPDX-License-Identifier: Apache-2.0
// Tracing front-end shared by both backends.
//
//   EbpfTracer    one global instance; kernel programs filter by container
//                 (cgroup id on v2, task-tree map on v1); ring-buffer poller
//                 thread converts records to trace::Event.
//   PtraceSession one per container, runs on the container's supervisor
//                 thread (ptrace requests must come from the tracer thread).
//
// Both deliver the same Event type to a Sink and feed a SyscallAggregator, so
// everything above this layer (gRPC streams, CLI, Android UI) is backend
// agnostic. The active backend is reported so the operator knows the latency
// numbers' fidelity (ptrace latencies include two context switches per stop).
#pragma once

#include <sys/types.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "vessel/common/result.h"

namespace vessel::trace {

enum class Backend { kNone = 1, kEbpf = 2, kPtrace = 3, kAuto = 4 };
enum class Kind { kSyscall = 1, kMemory = 2, kNet = 3, kExec = 4, kExit = 5, kOomKill = 6,
                  kMemoryPressure = 7, kSeccompDenied = 8, kLifecycle = 9 };

const char* backendName(Backend b) noexcept;

struct Event {
  std::string container_id;
  Kind kind = Kind::kSyscall;
  Backend backend = Backend::kNone;
  int64_t ts_unix_ns = 0;
  int32_t pid = 0, tid = 0;
  std::string comm;
  int32_t nr = -1;
  int64_t ret = 0;
  uint64_t latency_ns = 0;
  std::array<uint64_t, 3> args{};
  uint64_t bytes = 0;
  bool egress = false;
  std::string detail;
  uint64_t dropped_before = 0;  // events lost (kernel ring / queue) before this one
};

using Sink = std::function<void(Event&&)>;

struct SyscallStat {
  int nr = -1;
  uint64_t count = 0, errors = 0, total_ns = 0, max_ns = 0;
};

// Thread-safe exact counters, used by the ptrace backend (eBPF aggregates in
// a kernel PERCPU_HASH instead). Fixed array indexed by nr: the hot path is a
// lock + three adds, no allocation, no hashing.
class SyscallAggregator {
 public:
  void record(int nr, int64_t ret, uint64_t latency_ns) noexcept;
  std::vector<SyscallStat> snapshot() const;
  uint64_t total() const noexcept { return total_.load(std::memory_order_relaxed); }

 private:
  static constexpr int kMaxNr = 512;
  mutable std::mutex mu_;
  std::array<SyscallStat, kMaxNr> by_nr_{};
  std::atomic<uint64_t> total_{0};
};

// Pure decoding logic for ptrace, separated for unit tests.
struct SyscallRecord {
  int nr;
  uint64_t args[3];
  int64_t ret;
  bool entry;
};

class PtraceSession {
 public:
  PtraceSession(std::string container_id, Sink sink, std::shared_ptr<SyscallAggregator> agg);
  ~PtraceSession();
  PtraceSession(const PtraceSession&) = delete;
  PtraceSession& operator=(const PtraceSession&) = delete;

  // Called from LaunchOptions::before_release, on the supervisor thread.
  Status seize(pid_t init_pid);

  // Runs the tracing loop until `init_pid` exits; returns its wait status.
  // `on_progress` is invoked after every stop so the caller can poll for
  // exec completion without a second thread.
  int run(pid_t init_pid, const std::function<void()>& on_progress);

  uint64_t stops() const noexcept { return stops_; }

 private:
  struct Task;
  void handleSyscallStop(pid_t tid, Task& t);
  void emit(Event&& e);
  Task& task(pid_t tid);

  std::string id_;
  Sink sink_;
  std::shared_ptr<SyscallAggregator> agg_;
  std::map<pid_t, Task>* tasks_;
  bool have_syscall_info_ = true;
  uint64_t stops_ = 0;
};

// eBPF backend --------------------------------------------------------------
struct NetCounters {
  uint64_t rx_bytes = 0, tx_bytes = 0, rx_packets = 0, tx_packets = 0;
};
struct FilterConfig {
  uint64_t min_latency_ns = 0;
  bool errors_only = false;
  bool emit_net = false;
  bool mute = false;  // nobody is listening: keep exact counters, emit no events
};

class EbpfTracer {
 public:
  virtual ~EbpfTracer() = default;
  // Loads and attaches the programs. Fails cleanly (ENOSYS/EPERM/ENOTSUP)
  // when the kernel or build lacks support; the caller then falls back to
  // ptrace. `ring_bytes` = 0 keeps the compiled default (4 MiB).
  static Result<std::unique_ptr<EbpfTracer>> open(Sink sink);
  static bool compiledIn() noexcept;

  virtual Result<int> attach(const std::string& container_id, uint64_t cgroup_id, int cgroup_fd) = 0;  // -> slot
  virtual Status trackTask(int slot, pid_t pid) = 0;  // cgroup-v1 hosts
  virtual Status configure(int slot, const FilterConfig& cfg) = 0;
  virtual void detach(int slot) = 0;
  virtual std::vector<SyscallStat> summary(int slot) = 0;
  virtual NetCounters net(int slot) = 0;
  virtual uint64_t drops() = 0;
};

}  // namespace vessel::trace
