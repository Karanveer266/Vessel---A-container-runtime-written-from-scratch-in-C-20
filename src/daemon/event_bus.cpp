// SPDX-License-Identifier: Apache-2.0
#include "event_bus.h"

namespace vessel::daemon {

bool EventFilter::matches(const trace::Event& e) const {
  if (!container_id.empty() && e.container_id != container_id) return false;
  if (!kinds.empty() && !kinds.contains(e.kind)) return false;
  const bool is_sc = e.kind == trace::Kind::kSyscall || e.kind == trace::Kind::kMemory;
  if (is_sc) {
    if (!syscalls.empty() && !syscalls.contains(e.nr)) return false;
    if (e.latency_ns < min_latency_ns) return false;
    if (errors_only && e.ret >= 0) return false;
  }
  return true;
}

void EventBus::setWatchHook(WatchHook h) {
  std::lock_guard lk(mu_);
  hook_ = std::move(h);
}

EventBus::Subscription EventBus::subscribe(EventFilter f) {
  auto q = std::make_shared<Queue>(cap_);
  WatchHook hook;
  std::string cid = f.container_id;
  size_t count = 0;
  uint64_t id = 0;
  {
    std::lock_guard lk(mu_);
    id = next_id_++;
    subs_.emplace(id, Sub{std::move(f), q});
    count = ++watch_count_[cid];
    hook = hook_;
  }
  if (hook) hook(cid, count);
  return {id, q};
}

void EventBus::unsubscribe(uint64_t id) {
  WatchHook hook;
  std::string cid;
  size_t count = 0;
  {
    std::lock_guard lk(mu_);
    auto it = subs_.find(id);
    if (it == subs_.end()) return;
    it->second.queue->close();
    cid = it->second.filter.container_id;
    subs_.erase(it);
    count = --watch_count_[cid];
    if (count == 0) watch_count_.erase(cid);
    hook = hook_;
  }
  if (hook) hook(cid, count);
}

void EventBus::publish(trace::Event&& e) {
  published_.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard lk(mu_);
  // Find matches first so the common single-subscriber case moves the event
  // instead of copying it.
  Sub* last = nullptr;
  for (auto& [id, s] : subs_) {
    if (!s.filter.matches(e)) continue;
    if (last) last->queue->push(e);  // copy for all but the final match
    last = &s;
  }
  if (last) last->queue->push(std::move(e));
}

size_t EventBus::watchers(const std::string& cid) const {
  std::lock_guard lk(mu_);
  size_t n = 0;
  if (auto it = watch_count_.find(cid); it != watch_count_.end()) n += it->second;
  if (auto it = watch_count_.find(""); it != watch_count_.end()) n += it->second;
  return n;
}

}  // namespace vessel::daemon
