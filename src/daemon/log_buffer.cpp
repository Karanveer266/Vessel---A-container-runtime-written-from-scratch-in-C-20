// SPDX-License-Identifier: Apache-2.0
#include "log_buffer.h"

#include <chrono>

#include "vessel/common/clock.h"

namespace vessel::daemon {

void LogBuffer::append(Stream s, const char* data, size_t n) {
  if (n == 0) return;
  {
    std::lock_guard lk(mu_);
    if (closed_) return;
    if (n > cap_) {  // keep only the tail of a gigantic write
      data += n - cap_;
      n = cap_;
    }
    entries_.push_back(Entry{next_offset_, Chunk{s, clk::realtimeNs(), std::string(data, n)}});
    next_offset_ += n;
    live_bytes_ += n;
    evictLocked();
  }
  cv_.notify_all();
}

void LogBuffer::evictLocked() {
  while (live_bytes_ > cap_ && head_ < entries_.size()) {
    live_bytes_ -= entries_[head_].chunk.data.size();
    entries_[head_].chunk.data = std::string();
    ++head_;
  }
  // Compact once the dead prefix dominates: O(1) amortised per append.
  if (head_ > 64 && head_ * 2 > entries_.size()) {
    entries_.erase(entries_.begin(), entries_.begin() + static_cast<long>(head_));
    head_ = 0;
  }
}

void LogBuffer::close() {
  {
    std::lock_guard lk(mu_);
    closed_ = true;
  }
  cv_.notify_all();
}

bool LogBuffer::read(uint64_t* cursor, std::vector<Chunk>* out, int wait_ms) {
  std::unique_lock lk(mu_);
  auto has_data = [&] { return next_offset_ > *cursor || closed_; };
  if (!has_data()) cv_.wait_for(lk, std::chrono::milliseconds(wait_ms), has_data);
  for (size_t i = head_; i < entries_.size(); ++i) {
    const auto& e = entries_[i];
    uint64_t end = e.offset + e.chunk.data.size();
    if (end <= *cursor) continue;
    Chunk c = e.chunk;
    if (e.offset < *cursor) c.data.erase(0, *cursor - e.offset);  // partial
    out->push_back(std::move(c));
    *cursor = end;
  }
  // "More may arrive": false once closed and this reader is at the end.
  // Data appended to *out by this call must still be consumed by the caller.
  return !(closed_ && *cursor >= next_offset_);
}

uint64_t LogBuffer::tailCursor(size_t tail_bytes) const {
  std::lock_guard lk(mu_);
  uint64_t oldest = head_ < entries_.size() ? entries_[head_].offset : next_offset_;
  if (tail_bytes == 0 || tail_bytes >= next_offset_ - oldest) return oldest;
  return next_offset_ - tail_bytes;
}

uint64_t LogBuffer::bytesWritten() const {
  std::lock_guard lk(mu_);
  return next_offset_;
}

}  // namespace vessel::daemon
