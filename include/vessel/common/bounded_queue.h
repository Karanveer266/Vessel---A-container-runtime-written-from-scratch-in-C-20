// SPDX-License-Identifier: Apache-2.0
// BoundedQueue<T>: multi-producer / single-consumer queue with a hard capacity
// and a DROP-OLDEST overflow policy.
//
// This is the back-pressure firewall between producers that must never block
// (the eBPF ring-buffer poller, the ptrace loop, the cgroup sampler) and
// consumers of unknown speed (a gRPC stream to a phone on flaky Wi-Fi).
//
//   * push() never blocks and never allocates once the deque reached capacity
//     in steady state (std::deque reuses its blocks).
//   * When full, the oldest element is discarded and `dropped_` incremented;
//     the consumer reads and resets the counter so the loss is reported
//     in-band (TraceEvent.dropped_events) rather than silently.
//   * Drop-oldest (not drop-newest) because for live telemetry the most recent
//     state is the most valuable.
//
// A lock-free ring was considered; at the event rates involved (<1M/s, already
// batched by the kernel ring buffer) a mutex+condvar is simpler to prove
// correct and the critical section is a pointer move.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>

namespace vessel {

template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(size_t capacity) : capacity_(capacity ? capacity : 1) {}

  // Returns false if an element had to be dropped to make room.
  bool push(T v) {
    bool dropped = false;
    {
      std::lock_guard lk(mu_);
      if (closed_) return false;
      if (q_.size() >= capacity_) {
        q_.pop_front();
        ++dropped_;
        dropped = true;
      }
      q_.push_back(std::move(v));
    }
    cv_.notify_one();
    return !dropped;
  }

  // Waits up to `timeout`. Returns nullopt on timeout or when closed+empty.
  template <typename Rep, typename Period>
  std::optional<T> popFor(std::chrono::duration<Rep, Period> timeout) {
    std::unique_lock lk(mu_);
    if (!cv_.wait_for(lk, timeout, [&] { return !q_.empty() || closed_; })) return std::nullopt;
    if (q_.empty()) return std::nullopt;
    T v = std::move(q_.front());
    q_.pop_front();
    return v;
  }

  std::optional<T> tryPop() {
    std::lock_guard lk(mu_);
    if (q_.empty()) return std::nullopt;
    T v = std::move(q_.front());
    q_.pop_front();
    return v;
  }

  // Returns and resets the number of elements dropped since the last call.
  uint64_t takeDropped() {
    std::lock_guard lk(mu_);
    uint64_t d = dropped_;
    dropped_ = 0;
    return d;
  }

  void close() {
    {
      std::lock_guard lk(mu_);
      closed_ = true;
    }
    cv_.notify_all();
  }
  bool closed() const {
    std::lock_guard lk(mu_);
    return closed_;
  }
  size_t size() const {
    std::lock_guard lk(mu_);
    return q_.size();
  }
  size_t capacity() const noexcept { return capacity_; }

 private:
  const size_t capacity_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<T> q_;
  uint64_t dropped_ = 0;
  bool closed_ = false;
};

}  // namespace vessel
