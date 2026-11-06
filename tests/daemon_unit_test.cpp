// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <thread>

#include "event_bus.h"
#include "log_buffer.h"

using namespace vessel;
using namespace vessel::daemon;

namespace {
trace::Event ev(const std::string& id, int nr, int64_t ret = 0, uint64_t lat = 0) {
  trace::Event e;
  e.container_id = id;
  e.nr = nr;
  e.ret = ret;
  e.latency_ns = lat;
  return e;
}
}  // namespace

TEST(EventBus, FiltersByContainerKindLatencyErrors) {
  EventBus bus(16);
  EventFilter f1;
  f1.container_id = "a";
  auto s1 = bus.subscribe(f1);
  EventFilter f2;
  f2.errors_only = true;
  f2.min_latency_ns = 100;
  auto s2 = bus.subscribe(f2);

  bus.publish(ev("a", 1, 0, 50));
  bus.publish(ev("b", 2, -2, 500));
  bus.publish(ev("a", 3, -1, 10));  // error but too fast for s2
  EXPECT_EQ(s1.queue->size(), 2u);
  EXPECT_EQ(s2.queue->size(), 1u);
  EXPECT_EQ(s2.queue->tryPop()->nr, 2);

  trace::Event life;
  life.container_id = "a";
  life.kind = trace::Kind::kLifecycle;  // non-syscall events bypass latency/error filters
  bus.publish(std::move(life));
  EXPECT_EQ(s2.queue->size(), 1u);
}

TEST(EventBus, WatchHookCountsSubscribers) {
  EventBus bus;
  std::vector<std::pair<std::string, size_t>> calls;
  bus.setWatchHook([&](const std::string& id, size_t n) { calls.emplace_back(id, n); });
  EventFilter f;
  f.container_id = "x";
  auto a = bus.subscribe(f);
  auto b = bus.subscribe(f);
  EXPECT_EQ(bus.watchers("x"), 2u);
  bus.unsubscribe(a.id);
  bus.unsubscribe(b.id);
  EXPECT_EQ(bus.watchers("x"), 0u);
  ASSERT_EQ(calls.size(), 4u);
  EXPECT_EQ(calls.back().second, 0u);
  EXPECT_TRUE(b.queue->closed());
}

TEST(EventBus, SlowSubscriberDoesNotBlockProducer) {
  EventBus bus(10);
  auto s = bus.subscribe({});
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 100000; ++i) bus.publish(ev("a", i));
  EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::seconds(2));
  EXPECT_EQ(s.queue->size(), 10u);
  EXPECT_EQ(s.queue->takeDropped(), 100000u - 10u);
  EXPECT_EQ(s.queue->tryPop()->nr, 100000 - 10);  // newest retained
}

TEST(LogBuffer, ReadFollowsAndPartialChunks) {
  LogBuffer b(64);
  b.append(LogBuffer::kStdout, "hello ", 6);
  b.append(LogBuffer::kStderr, "world", 5);
  uint64_t cur = 0;
  std::vector<LogBuffer::Chunk> out;
  ASSERT_TRUE(b.read(&cur, &out, 0));
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0].data, "hello ");
  EXPECT_EQ(out[1].stream, LogBuffer::kStderr);
  EXPECT_EQ(cur, 11u);
  out.clear();
  uint64_t tail = b.tailCursor(3);
  ASSERT_TRUE(b.read(&tail, &out, 0));
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].data, "rld");
}

TEST(LogBuffer, BoundedMemoryAndLappedReaders) {
  LogBuffer b(100);
  std::string chunk(30, 'x');
  for (int i = 0; i < 100; ++i) b.append(LogBuffer::kStdout, chunk.data(), chunk.size());
  EXPECT_EQ(b.bytesWritten(), 3000u);
  uint64_t cur = 0;  // reader that never kept up
  std::vector<LogBuffer::Chunk> out;
  b.read(&cur, &out, 0);
  size_t total = 0;
  for (auto& c : out) total += c.data.size();
  EXPECT_LE(total, 100u);
  EXPECT_EQ(cur, 3000u);
}

TEST(LogBuffer, CloseEndsFollowers) {
  LogBuffer b;
  std::thread t([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    b.append(LogBuffer::kStdout, "x", 1);
    b.close();
  });
  uint64_t cur = 0;
  std::vector<LogBuffer::Chunk> out;
  int rounds = 0;
  while (b.read(&cur, &out, 1000) && rounds < 10) ++rounds;
  t.join();
  ASSERT_EQ(out.size(), 1u);
  EXPECT_LT(rounds, 10);
}
