// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <cmath>
#include <tuple>
#include <vector>

#include "vessel/gpu/common.hpp"
#include "vessel/gpu/cpu_kernels.hpp"
#include "vessel/gpu/result.hpp"

using namespace vessel::gpu;

namespace {
struct Inputs {
  std::vector<float> Q, K, V;
  explicit Inputs(std::size_t n, std::uint64_t seed, float amp = 1.0F) : Q(n), K(n), V(n) {
    fill_uniform(Q, seed, -amp, amp);
    fill_uniform(K, seed + 1, -amp, amp);
    fill_uniform(V, seed + 2);
  }
};
}  // namespace

class FlashVsNaive : public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t, bool>> {};

TEST_P(FlashVsNaive, Matches) {
  const auto [seq, dim, causal] = GetParam();
  const cpu::AttentionShape s{3, seq, dim, causal};
  const std::size_t n = s.bh * seq * dim;
  Inputs in(n, 100 + seq);
  std::vector<float> O(n, std::nanf("")), ref(n);
  const float scale = 1.0F / std::sqrt(static_cast<float>(dim));
  cpu::attention_naive(in.Q.data(), in.K.data(), in.V.data(), ref.data(), s, scale);
  cpu::flash_attention(in.Q.data(), in.K.data(), in.V.data(), O.data(), s, scale, 3);
  EXPECT_LE(max_abs_diff(O, ref), kAttentionTolerance) << "seq=" << seq << " dim=" << dim << " causal=" << causal;
}

// Odd lengths: < Br, == Br, between Br and Bc, not multiples of either,
// several K/V blocks with a partial last one.
INSTANTIATE_TEST_SUITE_P(Shapes, FlashVsNaive,
                         ::testing::Combine(::testing::Values<std::size_t>(1, 2, 31, 32, 33, 63, 65, 100, 257),
                                            ::testing::Values<std::size_t>(16, 32, 64, 128),
                                            ::testing::Bool()));

TEST(Flash, CausalFirstRowCopiesFirstValue) {
  // Row 0 under a causal mask attends only key 0 -> output == V[0] exactly-ish.
  const cpu::AttentionShape s{1, 50, 32, true};
  Inputs in(s.seq * s.dim, 9);
  std::vector<float> O(in.Q.size());
  cpu::flash_attention(in.Q.data(), in.K.data(), in.V.data(), O.data(), s, 0.2F, 2);
  for (std::size_t t = 0; t < s.dim; ++t) EXPECT_NEAR(O[t], in.V[t], 1e-6);
}

TEST(Flash, NumericallyStableForLargeScores) {
  // |scores| up to ~ 30*30*64*0.125 = 7200: exp() would overflow without
  // max subtraction. Online softmax must stay finite and match the oracle.
  const cpu::AttentionShape s{2, 97, 64, false};
  Inputs in(s.bh * s.seq * s.dim, 21, 30.0F);
  std::vector<float> O(in.Q.size()), ref(in.Q.size());
  cpu::attention_naive(in.Q.data(), in.K.data(), in.V.data(), ref.data(), s, 0.125F);
  cpu::flash_attention(in.Q.data(), in.K.data(), in.V.data(), O.data(), s, 0.125F, 2);
  for (float x : O) ASSERT_TRUE(std::isfinite(x));
  EXPECT_LE(max_abs_diff(O, ref), kAttentionTolerance);
}

TEST(Flash, FlopAccounting) {
  EXPECT_DOUBLE_EQ(cpu::attention_flops({2, 4, 8, false}), 4.0 * 8 * 2 * 16);
  EXPECT_DOUBLE_EQ(cpu::attention_flops({2, 4, 8, true}), 4.0 * 8 * 2 * 10);
}
