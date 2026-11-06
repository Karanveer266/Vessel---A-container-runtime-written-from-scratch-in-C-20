// SPDX-License-Identifier: MIT
// Launchers for the SGEMM kernels in sgemm_kernels.cuh.
#include "cuda/cuda_check.cuh"
#include "cuda/kernels.hpp"
#include "cuda/sgemm_kernels.cuh"

namespace vessel::gpu::cuda {

using kern::SgemmTile;

bool sgemm(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C,
           int ldc, cudaStream_t stream) {
  if (M <= 0 || N <= 0) return false;
  const dim3 block(SgemmTile::kThreads);
  // x over N, y over M: see the L2-reuse comment in the kernel. gridDim.y is
  // limited to 65535 -> M <= 8.3M rows, far beyond any realistic use.
  const dim3 grid((N + SgemmTile::BN - 1) / SgemmTile::BN, (M + SgemmTile::BM - 1) / SgemmTile::BM);
  const bool vec = kern::sgemm_can_vectorize(A, B, C, N, K, lda, ldb, ldc);
  if (vec) {
    kern::sgemm_128x128x8_kernel<true><<<grid, block, 0, stream>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  } else {
    kern::sgemm_128x128x8_kernel<false><<<grid, block, 0, stream>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  VESSEL_CUDA_CHECK_LAUNCH();
  return vec;
}

void sgemm_naive(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta,
                 float* C, int ldc, cudaStream_t stream) {
  if (M <= 0 || N <= 0) return;
  const dim3 block(32, 8);  // 32 threads along N: coalesced B loads / C stores
  const dim3 grid((N + 31) / 32, (M + 7) / 8);
  kern::sgemm_naive_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  VESSEL_CUDA_CHECK_LAUNCH();
}

const char* sgemm_tile_string() noexcept { return "BM128xBN128xBK8,TM8xTN8,256thr,smem_dbuf,float4"; }

}  // namespace vessel::gpu::cuda
