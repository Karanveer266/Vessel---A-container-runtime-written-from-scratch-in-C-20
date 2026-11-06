// SPDX-License-Identifier: MIT
// Ring all-reduce: correctness, byte accounting and failure behaviour.
#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

#include "vessel/gpu/common.hpp"
#include "vessel/gpu/sim/communicator.hpp"

using namespace vessel::gpu;
using namespace vessel::gpu::sim;

namespace {

// Integer-valued inputs make the fp32 sum exact regardless of the ring's
// reduction order, so the comparison can be bit-exact (EXPECT_EQ).
float input(std::size_t rank, std::size_t i) { return static_cast<float>(static_cast<int>((rank * 131 + i * 7) % 61) - 30); }

struct Case {
  int ranks;
  std::size_t count;
  std::size_t chunk;
  std::size_t slots;
};

void run_case(const Case& c, bool in_place) {
  World world(c.ranks, CommConfig{c.chunk, c.slots});
  const auto R = static_cast<std::size_t>(c.ranks);
  std::vector<std::vector<float>> send(R), recv(R);
  for (std::size_t r = 0; r < R; ++r) {
    send[r].resize(c.count);
    recv[r].assign(c.count, -12345.0F);
    for (std::size_t i = 0; i < c.count; ++i) send[r][i] = input(r, i);
  }
  std::vector<CommStats> stats(R);
  world.run([&](Communicator& comm) {
    const auto r = static_cast<std::size_t>(comm.rank());
    if (in_place) {
      recv[r] = send[r];
      comm.allReduce(recv[r].data(), recv[r].data(), c.count);
    } else {
      comm.allReduce(send[r].data(), recv[r].data(), c.count);
    }
    stats[r] = comm.stats();
  });

  for (std::size_t i = 0; i < c.count; ++i) {
    float expect = 0;
    for (std::size_t r = 0; r < R; ++r) expect += input(r, i);
    for (std::size_t r = 0; r < R; ++r) ASSERT_EQ(recv[r][i], expect) << "rank " << r << " idx " << i;
  }

  // Byte accounting: measured == exact formula, and the ring total equals
  // 2(R-1)*N*4 for any N (each element crosses 2(R-1) links in total).
  std::uint64_t total = 0;
  for (std::size_t r = 0; r < R; ++r) {
    EXPECT_EQ(stats[r].bytes_sent, ring_allreduce_bytes_per_rank(c.count, R, r, c.chunk)) << "rank " << r;
    EXPECT_EQ(stats[r].bytes_received, stats[(r + R - 1) % R].bytes_sent);
    total += stats[r].bytes_sent;
  }
  EXPECT_EQ(total, 2 * (R - 1) * c.count * sizeof(float));
}

}  // namespace

std::string ring_case_name(const ::testing::TestParamInfo<std::tuple<int, std::size_t>>& p) {
  return "R" + std::to_string(std::get<0>(p.param)) + "_N" + std::to_string(std::get<1>(p.param));
}

class RingAllReduce : public ::testing::TestWithParam<std::tuple<int, std::size_t>> {};

TEST_P(RingAllReduce, OutOfPlaceSumIsExact) {
  const auto [R, count] = GetParam();
  // Tiny chunks force many loops and many slices -> exercises pipelining,
  // back-pressure and the multi-loop partition.
  run_case({R, count, 5, 4}, false);
  run_case({R, count, 1 << 15, 8}, false);
}

TEST_P(RingAllReduce, InPlaceSumIsExact) {
  const auto [R, count] = GetParam();
  run_case({R, count, 7, 4}, true);
}

INSTANTIATE_TEST_SUITE_P(Ranks, RingAllReduce,
                         ::testing::Combine(::testing::Values(1, 2, 3, 4, 7, 8),
                                            // incl. count < R, primes, not divisible by R
                                            ::testing::Values<std::size_t>(1, 2, 5, 6, 13, 97, 1000, 4099)),
                         ring_case_name);

TEST(RingAllReduceBytes, EqualsTheoryWhenDivisible) {
  // N divisible by R and a single loop: per-rank bytes == 2(R-1)/R * N * 4 exactly.
  for (int R : {2, 3, 4, 7, 8}) {
    const std::size_t count = static_cast<std::size_t>(R) * 1000;
    World world(R);
    std::vector<float> dummy;
    std::vector<std::uint64_t> sent(static_cast<std::size_t>(R));
    std::vector<std::vector<float>> bufs(static_cast<std::size_t>(R), std::vector<float>(count, 1.0F));
    world.run([&](Communicator& c) {
      auto& b = bufs[static_cast<std::size_t>(c.rank())];
      c.allReduce(b.data(), b.data(), count);
      sent[static_cast<std::size_t>(c.rank())] = c.stats().bytes_sent;
      for (float x : b) ASSERT_EQ(x, static_cast<float>(R));
    });
    for (auto s : sent) EXPECT_DOUBLE_EQ(static_cast<double>(s), ring_allreduce_theoretical_bytes(count, R));
  }
}

TEST(RingAllReduceBytes, LargeOddCountWithinOneElementPerSegment) {
  // Not divisible: every rank is within 2 elements per loop of the continuous bound.
  const std::size_t count = 1'000'003, chunk = 1000;
  for (std::size_t R : {3UL, 7UL}) {
    const std::size_t loops = (count + R * chunk - 1) / (R * chunk);
    for (std::size_t r = 0; r < R; ++r) {
      const double got = static_cast<double>(ring_allreduce_bytes_per_rank(count, R, r, chunk));
      EXPECT_NEAR(got, ring_allreduce_theoretical_bytes(count, R), 2.0 * 4.0 * static_cast<double>(loops));
    }
  }
}

TEST(RingAllReduce, OtherOps) {
  constexpr int R = 4;
  constexpr std::size_t N = 1001;
  for (RedOp op : {RedOp::Max, RedOp::Min, RedOp::Prod, RedOp::Avg}) {
    World world(R, {16, 4});
    std::vector<std::vector<float>> b(R, std::vector<float>(N));
    for (std::size_t r = 0; r < R; ++r)
      for (std::size_t i = 0; i < N; ++i) b[r][i] = static_cast<float>((r + 1) * (i % 3 + 1));
    world.run([&](Communicator& c) {
      auto& v = b[static_cast<std::size_t>(c.rank())];
      c.allReduce(v.data(), v.data(), N, op);
    });
    for (std::size_t i = 0; i < N; ++i) {
      const float base = static_cast<float>(i % 3 + 1);
      float e = 0;
      switch (op) {
        case RedOp::Max: e = 4 * base; break;
        case RedOp::Min: e = base; break;
        case RedOp::Prod: e = 24 * base * base * base * base; break;
        case RedOp::Avg: e = 2.5F * base; break;
        case RedOp::Sum: break;
      }
      for (std::size_t r = 0; r < R; ++r) ASSERT_FLOAT_EQ(b[r][i], e) << static_cast<int>(op);
    }
  }
}

TEST(RingAllReduce, RepeatedCollectivesReuseLinks) {
  World world(3, {64, 4});
  std::vector<std::vector<float>> b(3, std::vector<float>(777, 1.0F));
  world.run([&](Communicator& c) {
    auto& v = b[static_cast<std::size_t>(c.rank())];
    for (int it = 0; it < 5; ++it) {
      c.allReduce(v.data(), v.data(), v.size());
      c.barrier();
    }
  });
  for (auto& v : b)
    for (float x : v) ASSERT_EQ(x, 243.0F);  // 3^5
}

TEST(RingAllReduce, BackPressureIsObservedWithSlowConsumer) {
  // Rank 1 sleeps before participating: rank 0's outgoing FIFO fills up and
  // it must stall (bounded links), yet the result is still correct.
  World world(2, {64, 4});
  std::vector<std::vector<float>> b(2, std::vector<float>(64 * 64, 1.0F));
  std::vector<CommStats> st(2);
  world.run([&](Communicator& c) {
    if (c.rank() == 1) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    auto& v = b[static_cast<std::size_t>(c.rank())];
    c.allReduce(v.data(), v.data(), v.size());
    st[static_cast<std::size_t>(c.rank())] = c.stats();
  });
  EXPECT_GT(st[0].send_stalls + st[0].recv_stalls, 0U);
  for (auto& v : b)
    for (float x : v) ASSERT_EQ(x, 2.0F);
}

TEST(RingAllReduce, FailingRankAbortsWorldInsteadOfDeadlocking) {
  World world(4, {8, 4});
  std::vector<std::vector<float>> b(4, std::vector<float>(1000, 1.0F));
  EXPECT_THROW(world.run([&](Communicator& c) {
                 if (c.rank() == 2) throw std::runtime_error("rank 2 died");
                 auto& v = b[static_cast<std::size_t>(c.rank())];
                 c.allReduce(v.data(), v.data(), v.size());
                 c.barrier();
               }),
               std::runtime_error);
  EXPECT_TRUE(world.aborted());
}

TEST(RingAllReduce, RejectsPartialOverlap) {
  World world(1);
  std::vector<float> v(10, 1.0F);
  EXPECT_THROW(world.comm(0).allReduce(v.data(), v.data() + 1, 5), std::invalid_argument);
  EXPECT_THROW(World(0), std::invalid_argument);
}
