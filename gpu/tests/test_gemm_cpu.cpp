// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <cmath>
#include <tuple>
#include <vector>

#include "vessel/gpu/common.hpp"
#include "vessel/gpu/cpu_kernels.hpp"
#include "vessel/gpu/result.hpp"
#include "vessel/gpu/sim/communicator.hpp"

using namespace vessel::gpu;

class GemmShapes : public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t, std::size_t>> {};

TEST_P(GemmShapes, TiledMatchesNaive) {
  const auto [M, N, K] = GetParam();
  std::vector<float> A(M * K), B(K * N), C(M * N, std::nanf("")), ref(M * N);
  fill_uniform(A, 1);
  fill_uniform(B, 2);
  cpu::gemm_naive(A.data(), B.data(), ref.data(), M, N, K, K, N, N);
  for (unsigned threads : {1U, 3U}) {
    std::fill(C.begin(), C.end(), std::nanf(""));  // beta=0 must never read C
    cpu::gemm_tiled(A.data(), B.data(), C.data(), M, N, K, K, N, N, false, threads);
    EXPECT_LE(max_abs_diff(C, ref), gemm_tolerance(K)) << M << "x" << N << "x" << K << " t=" << threads;
  }
}

// Odd/prime shapes hit every edge: partial MR/NR micro-tiles, partial MC/KC
// blocks, K spanning multiple KC panels (beta=1 path), and N > NC.
INSTANTIATE_TEST_SUITE_P(Odd, GemmShapes,
                         ::testing::Values(std::tuple{1UL, 1UL, 1UL}, std::tuple{1UL, 17UL, 3UL},
                                           std::tuple{7UL, 1UL, 5UL}, std::tuple{5UL, 13UL, 600UL},
                                           std::tuple{67UL, 45UL, 33UL}, std::tuple{73UL, 129UL, 257UL},
                                           std::tuple{128UL, 128UL, 128UL}, std::tuple{150UL, 31UL, 513UL},
                                           std::tuple{13UL, 2085UL, 9UL}, std::tuple{200UL, 200UL, 1UL}));

TEST(Gemm, AccumulateAddsToC) {
  constexpr std::size_t M = 37, N = 29, K = 300;
  std::vector<float> A(M * K), B(K * N), C(M * N), ref(M * N);
  fill_uniform(A, 3);
  fill_uniform(B, 4);
  fill_uniform(C, 5);
  ref = C;
  cpu::gemm_naive(A.data(), B.data(), ref.data(), M, N, K, K, N, N, true);
  cpu::gemm_tiled(A.data(), B.data(), C.data(), M, N, K, K, N, N, true, 2);
  EXPECT_LE(max_abs_diff(C, ref), gemm_tolerance(K));
}

TEST(Gemm, StridedSubmatrices) {
  // Operate on a sub-block of larger matrices via leading dimensions — the
  // exact access pattern split-K uses for A[:, k0:k1].
  constexpr std::size_t M = 19, N = 23, K = 41, lda = 64, ldb = 50, ldc = 31;
  std::vector<float> A(M * lda), B(K * ldb), C(M * ldc, 7.0F), ref(M * ldc, 7.0F);
  fill_uniform(A, 6);
  fill_uniform(B, 7);
  cpu::gemm_naive(A.data() + 3, B.data() + 5, ref.data(), M, N, K, lda, ldb, ldc);
  cpu::gemm_tiled(A.data() + 3, B.data() + 5, C.data(), M, N, K, lda, ldb, ldc, false, 2);
  EXPECT_LE(max_abs_diff(C, ref), gemm_tolerance(K));
  for (std::size_t i = 0; i < M; ++i)  // padding columns untouched
    for (std::size_t j = N; j < ldc; ++j) ASSERT_EQ(C[i * ldc + j], 7.0F);
}

TEST(Gemm, KZeroClearsC) {
  std::vector<float> C(12, 3.0F);
  cpu::gemm_tiled(nullptr, nullptr, C.data(), 3, 4, 0, 1, 4, 4);
  for (float x : C) EXPECT_EQ(x, 0.0F);
}

class SplitK : public ::testing::TestWithParam<std::tuple<int, std::size_t>> {};

TEST_P(SplitK, SumOfSlicesEqualsFullGemm) {
  const auto [R, K] = GetParam();
  constexpr std::size_t M = 45, N = 38;
  std::vector<float> A(M * K), B(K * N), full(M * N);
  fill_uniform(A, 10);
  fill_uniform(B, 11);
  cpu::gemm_naive(A.data(), B.data(), full.data(), M, N, K, K, N, N);

  const auto Rs = static_cast<std::size_t>(R);
  std::vector<std::vector<float>> C(Rs, std::vector<float>(M * N));
  sim::World world(R, {256, 4});
  world.run([&](sim::Communicator& comm) {
    const auto r = static_cast<std::size_t>(comm.rank());
    const Range ks = split_range(K, Rs, r, 4);
    cpu::gemm_tiled(A.data() + ks.begin, B.data() + ks.begin * N, C[r].data(), M, N, ks.size(), K, N, N, false, 1);
    comm.allReduce(C[r].data(), C[r].data(), M * N);
  });
  for (const auto& c : C) EXPECT_LE(max_abs_diff(c, full), gemm_tolerance(K));
  // All ranks must agree bit-for-bit: all-gather copies one reduced value.
  for (std::size_t r = 1; r < Rs; ++r) EXPECT_EQ(C[r], C[0]);
}

// Includes K < R*4 (some ranks get an empty K slice and contribute zeros).
INSTANTIATE_TEST_SUITE_P(Ranks, SplitK,
                         ::testing::Combine(::testing::Values(1, 2, 3, 4, 7),
                                            ::testing::Values<std::size_t>(3, 17, 99, 1000)));

TEST(Gemm, ConfigStringDescribesBlocking) {
  const auto s = cpu::gemm_tiled_config();
  EXPECT_NE(s.find("MR"), std::string::npos);
  EXPECT_NE(s.find("KC256"), std::string::npos);
}
