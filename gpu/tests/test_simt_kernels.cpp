// SPDX-License-Identifier: MIT
// Executes the *actual CUDA kernel source* (src/cuda/*_kernels.cuh) on the
// host SIMT emulator and compares against the CPU oracles. This is how the
// CUDA kernels' indexing / edge handling / barrier placement are tested on
// GPU-less CI; on a GPU box the worker's --verify covers the real hardware.
#include "simt_emu.hpp"  // must precede the kernel headers
//
#include <gtest/gtest.h>

#include <cmath>
#include <tuple>
#include <vector>

#include "cuda/flash_attention_kernels.cuh"
#include "cuda/sgemm_kernels.cuh"
#include "vessel/gpu/common.hpp"
#include "vessel/gpu/cpu_kernels.hpp"
#include "vessel/gpu/result.hpp"

using namespace vessel::gpu;
namespace kern = vessel::gpu::cuda::kern;

namespace {

void emu_sgemm(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C,
               int ldc, bool expect_vec) {
  using T = kern::SgemmTile;
  const dim3 grid(static_cast<unsigned>((N + T::BN - 1) / T::BN), static_cast<unsigned>((M + T::BM - 1) / T::BM));
  const bool vec = kern::sgemm_can_vectorize(A, B, C, N, K, lda, ldb, ldc);
  ASSERT_EQ(vec, expect_vec);
  if (vec)
    simt::launch(grid, dim3(T::kThreads), 0, kern::sgemm_128x128x8_kernel<true>, M, N, K, alpha, A, lda, B, ldb, beta,
                 C, ldc);
  else
    simt::launch(grid, dim3(T::kThreads), 0, kern::sgemm_128x128x8_kernel<false>, M, N, K, alpha, A, lda, B, ldb,
                 beta, C, ldc);
}

}  // namespace

class SimtSgemm : public ::testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(SimtSgemm, MatchesOracle) {
  const auto [M, N, K] = GetParam();
  const auto m = static_cast<std::size_t>(M), n = static_cast<std::size_t>(N), k = static_cast<std::size_t>(K);
  std::vector<float> A(m * k), B(k * n), C(m * n), ref(m * n);
  fill_uniform(A, 1);
  fill_uniform(B, 2);
  fill_uniform(C, 3);
  // ref = 0.5 * C + 1.5 * A B   (exercises alpha and the beta != 0 read path)
  std::vector<float> ab(m * n);
  cpu::gemm_naive(A.data(), B.data(), ab.data(), m, n, k, k, n, n);
  for (std::size_t i = 0; i < ref.size(); ++i) ref[i] = 0.5F * C[i] + 1.5F * ab[i];
  const bool vec = K % 4 == 0 && N % 4 == 0;
  emu_sgemm(M, N, K, 1.5F, A.data(), K, B.data(), N, 0.5F, C.data(), N, vec);
  EXPECT_LE(max_abs_diff(C, ref), 2 * gemm_tolerance(k));
}

// Vectorized path: multiples of 4 that are *not* multiples of the 128/8
// tile; scalar path: odd N and/or K. Each shape spans >1 block in M or N.
INSTANTIATE_TEST_SUITE_P(Shapes, SimtSgemm,
                         ::testing::Values(std::tuple{130, 132, 40}, std::tuple{1, 4, 4}, std::tuple{129, 8, 12},
                                           std::tuple{131, 67, 37}, std::tuple{3, 129, 5}, std::tuple{17, 1, 1}));

TEST(SimtSgemm, BetaZeroIgnoresNaNInC) {
  constexpr int M = 20, N = 24, K = 16;
  std::vector<float> A(M * K), B(K * N), C(M * N, std::nanf("")), ref(M * N);
  fill_uniform(A, 4);
  fill_uniform(B, 5);
  cpu::gemm_naive(A.data(), B.data(), ref.data(), M, N, K, K, N, N);
  emu_sgemm(M, N, K, 1.F, A.data(), K, B.data(), N, 0.F, C.data(), N, true);
  EXPECT_LE(max_abs_diff(C, ref), gemm_tolerance(K));
}

TEST(SimtSgemm, NaiveKernelMatches) {
  constexpr int M = 19, N = 37, K = 23;
  std::vector<float> A(M * K), B(K * N), C(M * N), ref(M * N);
  fill_uniform(A, 6);
  fill_uniform(B, 7);
  cpu::gemm_naive(A.data(), B.data(), ref.data(), M, N, K, K, N, N);
  simt::launch(dim3((N + 31) / 32, (M + 7) / 8), dim3(32, 8), 0, kern::sgemm_naive_kernel, M, N, K, 1.F, A.data(), K,
               B.data(), N, 0.F, C.data(), N);
  EXPECT_LE(max_abs_diff(C, ref), gemm_tolerance(K));
}

namespace {

template <int D>
void emu_flash(int bh, int seq, bool causal, std::uint64_t seed) {
  using Cfg = kern::FlashConfig<D>;
  const std::size_t n = static_cast<std::size_t>(bh) * static_cast<std::size_t>(seq) * D;
  std::vector<float> Q(n), K(n), V(n), O(n, std::nanf("")), ref(n);
  fill_uniform(Q, seed);
  fill_uniform(K, seed + 1);
  fill_uniform(V, seed + 2);
  const float scale = 1.0F / std::sqrt(static_cast<float>(D));
  cpu::attention_naive(Q.data(), K.data(), V.data(), ref.data(),
                       {static_cast<std::size_t>(bh), static_cast<std::size_t>(seq), D, causal}, scale);
  const dim3 grid(static_cast<unsigned>((seq + Cfg::Br - 1) / Cfg::Br), static_cast<unsigned>(bh));
  simt::launch(grid, dim3(Cfg::kThreads), Cfg::kSmemBytes, kern::flash_attention_fwd_kernel<D>,
               static_cast<const float*>(Q.data()), static_cast<const float*>(K.data()),
               static_cast<const float*>(V.data()), O.data(), seq, scale * 1.4426950408889634F, causal);
  EXPECT_LE(max_abs_diff(O, ref), kAttentionTolerance) << "D=" << D << " seq=" << seq << " causal=" << causal;
}

}  // namespace

class SimtFlash : public ::testing::TestWithParam<std::tuple<int, bool>> {};

TEST_P(SimtFlash, D32) { emu_flash<32>(2, std::get<0>(GetParam()), std::get<1>(GetParam()), 11); }
TEST_P(SimtFlash, D64) { emu_flash<64>(2, std::get<0>(GetParam()), std::get<1>(GetParam()), 22); }
TEST_P(SimtFlash, D128) { emu_flash<128>(1, std::get<0>(GetParam()), std::get<1>(GetParam()), 33); }

// 1: single key; 33/63: partial Q block + seq tail inside the first K tile;
// 130: several K/V tiles (both double-buffer stages) and a partial last tile.
INSTANTIATE_TEST_SUITE_P(Seq, SimtFlash,
                         ::testing::Combine(::testing::Values(1, 33, 63, 130), ::testing::Bool()));
