// SPDX-License-Identifier: Apache-2.0
// Fan-out of trace events to any number of subscribers (gRPC streams).
//
// Producers (eBPF poller thread, ptrace supervisor threads, PSI monitor)
// call publish(); it never blocks: each subscriber owns a BoundedQueue with
// drop-oldest semantics. The bus also tracks how many subscribers watch each
// container so that tracers can be *muted* when nobody listens — in eBPF
// mode that keeps the ring buffer silent while exact in-kernel counters keep
// running (zero user-space cost for unobserved containers).
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "vessel/common/bounded_queue.h"
#include "vessel/trace/trace.h"

namespace vessel::daemon {

struct EventFilter {
  std::string container_id;              // empty = all
  std::set<trace::Kind> kinds;           // empty = all
  std::set<int> syscalls;                // empty = all
  uint64_t min_latency_ns = 0;
  bool errors_only = false;
  bool matches(const trace::Event& e) const;
};

class EventBus {
 public:
  using Queue = BoundedQueue<trace::Event>;
  struct Subscription {
    uint64_t id;
    std::shared_ptr<Queue> queue;
  };
  // Called with (container_id, subscriber_count) whenever the count changes.
  using WatchHook = std::function<void(const std::string&, size_t)>;

  explicit EventBus(size_t per_subscriber_capacity = 8192) : cap_(per_subscriber_capacity) {}

  void setWatchHook(WatchHook h);
  Subscription subscribe(EventFilter f);
  void unsubscribe(uint64_t id);
  void publish(trace::Event&& e);
  size_t watchers(const std::string& container_id) const;
  uint64_t published() const noexcept { return published_.load(std::memory_order_relaxed); }

 private:
  struct Sub {
    EventFilter filter;
    std::shared_ptr<Queue> queue;
  };
  const size_t cap_;
  mutable std::mutex mu_;
  std::map<uint64_t, Sub> subs_;
  std::map<std::string, size_t> watch_count_;  // "" = wildcard subscribers
  uint64_t next_id_ = 1;
  WatchHook hook_;
  std::atomic<uint64_t> published_{0};
};

}  // namespace vessel::daemon
