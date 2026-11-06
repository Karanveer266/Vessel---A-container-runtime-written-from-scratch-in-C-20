// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "vessel/gpu/common.hpp"
#include "vessel/gpu/result.hpp"

using namespace vessel::gpu;

TEST(SplitRange, CoversExactlyOnceWithAlignment) {
  for (std::size_t n : {0UL, 1UL, 3UL, 4UL, 7UL, 99UL, 1000UL, 1023UL}) {
    for (std::size_t parts : {1UL, 2UL, 3UL, 5UL, 8UL}) {
      for (std::size_t align : {1UL, 4UL}) {
        std::size_t expect = 0;
        for (std::size_t i = 0; i < parts; ++i) {
          const Range r = split_range(n, parts, i, align);
          EXPECT_EQ(r.begin, expect) << n << "/" << parts << "/" << align;
          EXPECT_LE(r.begin, r.end);
          if (i + 1 < parts) {
            EXPECT_EQ(r.end % align, 0U);
          }
          expect = r.end;
        }
        EXPECT_EQ(expect, n);
      }
    }
  }
}

TEST(SplitRange, BalancedWithinOneUnit) {
  std::size_t mn = ~0UL, mx = 0;
  for (std::size_t i = 0; i < 7; ++i) {
    const auto s = split_range(100, 7, i).size();
    mn = std::min(mn, s);
    mx = std::max(mx, s);
  }
  EXPECT_LE(mx - mn, 1U);
}

TEST(FillUniform, DeterministicAndInRange) {
  std::vector<float> a(10000), b(10000);
  fill_uniform(a, 7);
  fill_uniform(b, 7);
  EXPECT_EQ(a, b);
  double mean = 0;
  for (float x : a) {
    ASSERT_GE(x, -1.0F);
    ASSERT_LT(x, 1.0F);
    mean += x;
  }
  EXPECT_NEAR(mean / 10000.0, 0.0, 0.05);
  fill_uniform(b, 8);
  EXPECT_NE(a, b);
}

TEST(MaxAbsDiff, PropagatesNaNAndSizeMismatch) {
  std::vector<float> a{1, 2, 3}, b{1, 2.5F, 3};
  EXPECT_DOUBLE_EQ(max_abs_diff(a, b), 0.5);
  b[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(std::isnan(max_abs_diff(a, b)));
  EXPECT_TRUE(std::isinf(max_abs_diff(a, std::vector<float>{1})));
}

TEST(ParallelFor, VisitsEveryIndexOnceAndPropagatesExceptions) {
  std::vector<std::atomic<int>> hits(1001);
  parallel_for(hits.size(), 4, [&](std::size_t b, std::size_t e, unsigned) {
    for (auto i = b; i < e; ++i) hits[i]++;
  });
  for (auto& h : hits) EXPECT_EQ(h.load(), 1);

  std::vector<std::atomic<int>> dyn(333);
  parallel_for_dynamic(dyn.size(), 3, [&](std::size_t i, unsigned) { dyn[i]++; });
  for (auto& h : dyn) EXPECT_EQ(h.load(), 1);

  EXPECT_THROW(parallel_for(10, 3,
                            [](std::size_t b, std::size_t, unsigned) {
                              if (b == 0) throw std::runtime_error("boom");
                            }),
               std::runtime_error);
}

TEST(Bandwidth, MatchesNcclTestsDefinition) {
  // 1 GB in 10 ms on 8 ranks: algbw = 100 GB/s, busbw = 100 * 14/8 = 175 GB/s.
  const auto bw = allreduce_bandwidth(1'000'000'000, 8, 10.0);
  EXPECT_DOUBLE_EQ(bw.algbw_gbps, 100.0);
  EXPECT_DOUBLE_EQ(bw.busbw_gbps, 175.0);
  EXPECT_DOUBLE_EQ(allreduce_bandwidth(1000, 1, 1.0).busbw_gbps, 0.0);  // 1 rank: no bus traffic
  EXPECT_DOUBLE_EQ(allreduce_bandwidth(1000, 2, 0.0).algbw_gbps, 0.0);
}
