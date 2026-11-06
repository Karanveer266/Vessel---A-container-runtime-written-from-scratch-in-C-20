// SPDX-License-Identifier: MIT
// Host-callable launchers for the custom CUDA kernels (implemented in
// sgemm.cu / flash_attention.cu). All pointers are device pointers; all
// launches are asynchronous on `stream`.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace vessel::gpu::cuda {

/// C = alpha*A*B + beta*C, row-major. Picks the float4 or scalar-edge
/// instantiation of the 128x128x8 kernel based on shape/alignment.
/// Returns true if the vectorized path was used.
bool sgemm(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C,
           int ldc, cudaStream_t stream);

/// Reference: one thread per output element.
void sgemm_naive(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta,
                 float* C, int ldc, cudaStream_t stream);

/// Fused attention forward. Q,K,V,O: [bh, seq, dim] contiguous, dim in {32, 64, 128}.
/// Throws std::invalid_argument for other head dims.
void flash_attention_fwd(const float* Q, const float* K, const float* V, float* O, int bh, int seq, int dim,
                         float scale, bool causal, cudaStream_t stream);

/// Reference: materializes S = QK^T*scale in `scratch` ([bh, seq, seq] floats),
/// row softmax, then O = P V — the memory-hungry baseline for verification.
void attention_naive(const float* Q, const float* K, const float* V, float* O, float* scratch, int bh, int seq,
                     int dim, float scale, bool causal, cudaStream_t stream);

/// Human-readable tile descriptions (`tile=` output key).
const char* sgemm_tile_string() noexcept;
const char* flash_tile_string(int dim) noexcept;

}  // namespace vessel::gpu::cuda
