// SPDX-License-Identifier: Apache-2.0
// Fixed-capacity ring of container output with monotonically increasing byte
// offsets. Readers keep an offset cursor; if the writer laps them they skip
// ahead (and learn how many bytes they missed) instead of blocking the
// container. Memory per container is bounded regardless of how much it logs —
// a must on a phone or an embedded target.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace vessel::daemon {

class LogBuffer {
 public:
  enum Stream : uint8_t { kStdout = 1, kStderr = 2 };
  struct Chunk {
    Stream stream;
    int64_t ts_ns;
    std::string data;
  };

  explicit LogBuffer(size_t capacity_bytes = 1 << 20) : cap_(capacity_bytes) {}

  void append(Stream s, const char* data, size_t n);
  void close();  // no more writes (container exited); wakes readers

  // Reads chunks with offset >= *cursor. Blocks up to `wait_ms` if none.
  // Returns false when closed and fully drained — chunks appended by the same
  // call are still valid and must be processed before stopping.
  bool read(uint64_t* cursor, std::vector<Chunk>* out, int wait_ms);
  uint64_t tailCursor(size_t tail_bytes) const;
  uint64_t bytesWritten() const;

 private:
  struct Entry {
    uint64_t offset;
    Chunk chunk;
  };
  void evictLocked();

  const size_t cap_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::vector<Entry> entries_;  // ordered by offset; front evicted first
  size_t head_ = 0;             // index of first live entry (amortised erase)
  size_t live_bytes_ = 0;
  uint64_t next_offset_ = 0;
  bool closed_ = false;
};

}  // namespace vessel::daemon
