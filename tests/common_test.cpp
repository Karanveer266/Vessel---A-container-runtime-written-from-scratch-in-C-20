// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <fcntl.h>

#include <set>
#include <thread>

#include "vessel/common/bounded_queue.h"
#include "vessel/common/fs.h"
#include "vessel/common/id.h"
#include "vessel/common/result.h"
#include "vessel/common/strings.h"
#include "test_util.h"

using namespace vessel;

TEST(Strings, ParseBytes) {
  EXPECT_EQ(str::parseBytes("4096"), 4096u);
  EXPECT_EQ(str::parseBytes("64M"), 64ull << 20);
  EXPECT_EQ(str::parseBytes("64MiB").has_value(), false);  // only K/M/G/T(+B)
  EXPECT_EQ(str::parseBytes("64MB"), 64ull << 20);
  EXPECT_EQ(str::parseBytes("1.5G"), 3ull << 29);
  EXPECT_EQ(str::parseBytes("512k"), 512u << 10);
  EXPECT_FALSE(str::parseBytes(""));
  EXPECT_FALSE(str::parseBytes("-1M"));
  EXPECT_FALSE(str::parseBytes("12Q"));
}

TEST(Strings, ParseCpu) {
  EXPECT_EQ(str::parseCpu("1"), 1000u);
  EXPECT_EQ(str::parseCpu("0.25"), 250u);
  EXPECT_EQ(str::parseCpu("750m"), 750u);
  EXPECT_FALSE(str::parseCpu("abc"));
}

TEST(Strings, SplitTrim) {
  auto v = str::split("a,,b,c", ',');
  ASSERT_EQ(v.size(), 3u);
  EXPECT_EQ(v[2], "c");
  EXPECT_EQ(str::split("a,,b", ',', false).size(), 3u);
  EXPECT_EQ(str::trim("  x y \n"), "x y");
  EXPECT_EQ(str::parseInt<int>(" 42\n"), 42);
  EXPECT_FALSE(str::parseInt<int>("42x"));
}

TEST(Strings, HumanBytes) {
  EXPECT_EQ(str::humanBytes(512), "512 B");
  EXPECT_EQ(str::humanBytes(1536), "1.5 KiB");
  EXPECT_EQ(str::humanBytes(64ull << 20), "64.0 MiB");
}

TEST(Result, PropagatesErrors) {
  auto f = [](bool ok) -> Result<int> {
    if (!ok) return Error{EINVAL, "stage"};
    return 7;
  };
  auto g = [&](bool ok) -> Result<int> {
    VESSEL_ASSIGN_OR_RETURN(int v, f(ok));
    return v * 2;
  };
  EXPECT_EQ(g(true).value(), 14);
  ASSERT_FALSE(g(false).ok());
  EXPECT_EQ(g(false).error().code, EINVAL);
  EXPECT_NE(g(false).error().str().find("stage: Invalid argument"), std::string::npos);
}

TEST(Ids, FormatAndUniqueness) {
  std::set<std::string> seen;
  for (int i = 0; i < 1000; ++i) {
    auto id = newContainerId();
    ASSERT_EQ(id.size(), 12u);
    ASSERT_EQ(id.find_first_not_of("0123456789abcdef"), std::string::npos);
    seen.insert(id);
  }
  EXPECT_EQ(seen.size(), 1000u);
  EXPECT_TRUE(isValidName("web-1.prod_a"));
  EXPECT_FALSE(isValidName("-bad"));
  EXPECT_FALSE(isValidName("has space"));
  EXPECT_FALSE(isValidName(std::string(64, 'a')));
}

TEST(BoundedQueue, DropsOldestAndCounts) {
  BoundedQueue<int> q(3);
  for (int i = 0; i < 5; ++i) q.push(i);
  EXPECT_EQ(q.takeDropped(), 2u);
  EXPECT_EQ(q.takeDropped(), 0u);  // reset on read
  EXPECT_EQ(q.tryPop(), 2);        // 0 and 1 were evicted
  EXPECT_EQ(q.tryPop(), 3);
  EXPECT_EQ(q.tryPop(), 4);
  EXPECT_FALSE(q.tryPop());
}

TEST(BoundedQueue, CloseWakesConsumer) {
  BoundedQueue<int> q(4);
  std::thread t([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    q.close();
  });
  auto start = std::chrono::steady_clock::now();
  EXPECT_FALSE(q.popFor(std::chrono::seconds(5)));
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(2));
  t.join();
  EXPECT_FALSE(q.push(1));
}

TEST(BoundedQueue, ConcurrentProducersNeverExceedCapacity) {
  BoundedQueue<int> q(100);
  std::vector<std::thread> ts;
  for (int p = 0; p < 4; ++p)
    ts.emplace_back([&] {
      for (int i = 0; i < 10000; ++i) q.push(i);
    });
  for (auto& t : ts) t.join();
  EXPECT_EQ(q.size(), 100u);
  EXPECT_EQ(q.takeDropped(), 40000u - 100u);
}

TEST(Fs, WriteReadMkdirs) {
  auto d = testutil::tempDir("fs");
  ASSERT_FALSE(d.empty());
  ASSERT_TRUE(fs::mkdirs(d + "/a/b/c").ok());
  ASSERT_TRUE(fs::mkdirs(d + "/a/b/c").ok());  // idempotent
  ASSERT_TRUE(fs::touch(d + "/a/f").ok());
  ASSERT_TRUE(fs::writeFile(d + "/a/f", "hello", O_TRUNC).ok());
  EXPECT_EQ(fs::readFile(d + "/a/f").value(), "hello");
  auto big = std::string(100000, 'x');
  ASSERT_TRUE(fs::writeFile(d + "/a/f", big, O_TRUNC).ok());
  EXPECT_EQ(fs::readFile(d + "/a/f").value().size(), big.size());
  EXPECT_EQ(fs::readFile(d + "/a/f", 10).value().size(), 10u);  // bounded
  EXPECT_FALSE(fs::readFile(d + "/nope").ok());
  EXPECT_EQ(fs::readFile(d + "/nope").error().code, ENOENT);
  EXPECT_EQ(fs::join("/a/", "/b"), "/a/b");
  testutil::rmrf(d);
}
