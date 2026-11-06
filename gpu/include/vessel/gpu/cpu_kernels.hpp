// SPDX-License-Identifier: MIT
// CPU reference + optimized kernels (GEMM, attention).
//
// Two tiers per operation:
//   *_naive : textbook loops, the "obviously correct" oracle used by tests and
//             by --verify. Accumulates in double so it is also *more* accurate
//             than the kernels it checks (the error we report is then
//             dominated by the kernel under test, not the oracle).
//   *_tiled : cache-blocked, register-tiled, multithreaded implementations
//             that the simulated backend actually benchmarks. They mirror the
//             blocking structure of the CUDA kernels (shared-memory tile <->
//             L1/L2 block, register micro-tile <-> SIMD micro-kernel), which is
//             the point: the sim backend exercises the same algorithms.
//
// All matrices are row-major with explicit leading dimensions so that a
// split-K slice A[:, k0:k1] can be passed as (A + k0, lda = K) without a copy.
#pragma once

#include <cstddef>
#include <string>

namespace vessel::gpu::cpu {

/// C[M,N] = A[M,K] * B[K,N]   (accumulate=false)
/// C[M,N] += A[M,K] * B[K,N]  (accumulate=true)
void gemm_naive(const float* A, const float* B, float* C, std::size_t M, std::size_t N,
                std::size_t K, std::size_t lda, std::size_t ldb, std::size_t ldc,
                bool accumulate = false) noexcept;

/// Blocked GEMM (BLIS/GotoBLAS loop structure, see gemm_cpu.cpp for details).
/// `threads` = 0 uses std::thread::hardware_concurrency().
void gemm_tiled(const float* A, const float* B, float* C, std::size_t M, std::size_t N,
                std::size_t K, std::size_t lda, std::size_t ldb, std::size_t ldc,
                bool accumulate = false, unsigned threads = 0);

/// Human-readable blocking configuration of gemm_tiled for this build, e.g.
/// "MR6xNR16,KC256,MC72,NC2048,avx2". MR/NR depend on the SIMD ISA the
/// translation unit was compiled for, hence a function and not constants.
[[nodiscard]] std::string gemm_tiled_config();

/// Attention problem shape. Q, K, V, O are dense [bh, seq, dim] row-major.
struct AttentionShape {
  std::size_t bh = 1;    // batch * heads (independent problems)
  std::size_t seq = 1;   // sequence length (queries == keys, self-attention)
  std::size_t dim = 64;  // head dimension
  bool causal = false;   // query i attends keys j <= i
};

/// O = softmax(Q K^T * scale [+ causal mask]) V, materializing each full
/// score row (O(seq) memory per row, O(seq^2) work). Oracle only.
void attention_naive(const float* Q, const float* K, const float* V, float* O,
                     const AttentionShape& s, float scale) noexcept;

/// FlashAttention-2 forward on the CPU: Q row-blocks x K/V column-blocks with
/// the online-softmax recurrence; never materializes more than a Br x Bc score
/// tile. Same algorithm as src/cuda/flash_attention.cu.
void flash_attention(const float* Q, const float* K, const float* V, float* O,
                     const AttentionShape& s, float scale, unsigned threads = 0);

struct FlashBlocking {
  static constexpr std::size_t Br = 32;  // query rows per task
  static constexpr std::size_t Bc = 64;  // keys per streamed K/V tile
};

/// Number of (query, key) pairs that contribute to attention; used for FLOP
/// accounting (causal masking roughly halves the work).
[[nodiscard]] constexpr double attention_pairs(const AttentionShape& s) noexcept {
  const double n = static_cast<double>(s.seq);
  return static_cast<double>(s.bh) * (s.causal ? n * (n + 1.0) / 2.0 : n * n);
}

/// FLOPs: QK^T (2*dim per pair) + PV (2*dim per pair). Softmax exp/max/sum is
/// O(pairs) and conventionally excluded (as in the FlashAttention papers).
[[nodiscard]] constexpr double attention_flops(const AttentionShape& s) noexcept {
  return 4.0 * static_cast<double>(s.dim) * attention_pairs(s);
}

}  // namespace vessel::gpu::cpu
