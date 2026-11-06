// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <thread>

#include "vessel/trace/trace.h"

using namespace vessel::trace;

TEST(Aggregator, CountsErrorsLatencyAndSorts) {
  SyscallAggregator a;
  for (int i = 0; i < 10; ++i) a.record(0, 100, 1000);
  a.record(1, -2 /*ENOENT*/, 5000);
  a.record(1, 3, 100);
  a.record(1, -4096, 10);  // not an errno (e.g. a large mmap address)
  a.record(-1, 0, 0);      // ignored
  a.record(100000, 0, 0);  // ignored
  auto s = a.snapshot();
  ASSERT_EQ(s.size(), 2u);
  EXPECT_EQ(s[0].nr, 0);
  EXPECT_EQ(s[0].count, 10u);
  EXPECT_EQ(s[0].total_ns, 10000u);
  EXPECT_EQ(s[1].count, 3u);
  EXPECT_EQ(s[1].errors, 1u);
  EXPECT_EQ(s[1].max_ns, 5000u);
  EXPECT_EQ(a.total(), 13u);
}

TEST(Aggregator, ThreadSafe) {
  SyscallAggregator a;
  std::vector<std::thread> ts;
  for (int t = 0; t < 8; ++t)
    ts.emplace_back([&] {
      for (int i = 0; i < 20000; ++i) a.record(i % 7, 0, 1);
    });
  for (auto& t : ts) t.join();
  uint64_t sum = 0;
  for (const auto& s : a.snapshot()) sum += s.count;
  EXPECT_EQ(sum, 160000u);
}
