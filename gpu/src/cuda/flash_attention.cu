// SPDX-License-Identifier: MIT
// Launchers for the fused attention kernel + a naive (materializing)
// reference implementation used by --verify.
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <string>

#include "cuda/cuda_check.cuh"
#include "cuda/flash_attention_kernels.cuh"
#include "cuda/kernels.hpp"

namespace vessel::gpu::cuda {

namespace {

template <int D>
void launch_flash(const float* Q, const float* K, const float* V, float* O, int bh, int seq, float scale, bool causal,
                  cudaStream_t stream) {
  using Cfg = kern::FlashConfig<D>;
  auto* kernel = &kern::flash_attention_fwd_kernel<D>;
  // > 48 KiB dynamic SMEM requires an explicit opt-in per kernel. Cheap
  // (host-side attribute), but do it once per process per instantiation.
  // Function-local static init is thread-safe; the per-device aspect is
  // handled by the attribute being a property of the function, applied to
  // the current device context — re-set it if the device changes.
  static thread_local int configured_device = -1;
  int dev = 0;
  VESSEL_CUDA_CHECK(cudaGetDevice(&dev));
  if (configured_device != dev) {
    VESSEL_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           static_cast<int>(Cfg::kSmemBytes)));
    configured_device = dev;
  }
  const dim3 grid((seq + Cfg::Br - 1) / Cfg::Br, bh);
  // softmax(x) = 2^(x*log2 e) / sum: fold log2(e) into the Q pre-scale.
  const float scale_log2 = scale * 1.4426950408889634f;
  kernel<<<grid, Cfg::kThreads, Cfg::kSmemBytes, stream>>>(Q, K, V, O, seq, scale_log2, causal);
  VESSEL_CUDA_CHECK_LAUNCH();
}

// ---- naive reference: S = scale*QK^T (materialized), softmax, O = PV -------
__global__ void naive_scores_kernel(const float* __restrict__ Q, const float* __restrict__ K, float* __restrict__ S,
                                    int seq, int dim, float scale, bool causal) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int i = static_cast<int>(blockIdx.y);
  const std::size_t h = blockIdx.z;
  if (j >= seq) return;
  const float* q = Q + (h * seq + i) * dim;
  const float* k = K + (h * seq + j) * dim;
  float acc = 0.f;
  for (int t = 0; t < dim; ++t) acc = fmaf(q[t], k[t], acc);
  S[(h * seq + i) * seq + j] = (causal && j > i) ? -INFINITY : acc * scale;
}

// One warp per row: max, exp, sum via shuffles; normalizes in place.
__global__ void naive_softmax_kernel(float* __restrict__ S, int rows, int seq) {
  const int row = static_cast<int>(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
  const int lane = static_cast<int>(threadIdx.x % 32);
  if (row >= rows) return;
  float* s = S + static_cast<std::size_t>(row) * seq;
  float mx = -INFINITY;
  for (int j = lane; j < seq; j += 32) mx = fmaxf(mx, s[j]);
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  float sum = 0.f;
  for (int j = lane; j < seq; j += 32) {
    const float e = expf(s[j] - mx);
    s[j] = e;
    sum += e;
  }
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  const float inv = 1.f / sum;
  for (int j = lane; j < seq; j += 32) s[j] *= inv;
}

__global__ void naive_pv_kernel(const float* __restrict__ P, const float* __restrict__ V, float* __restrict__ O,
                                int seq, int dim) {
  const int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int i = static_cast<int>(blockIdx.y);
  const std::size_t h = blockIdx.z;
  if (t >= dim) return;
  const float* p = P + (h * seq + i) * seq;
  const float* v = V + h * seq * dim;
  float acc = 0.f;
  for (int j = 0; j < seq; ++j) acc = fmaf(p[j], v[static_cast<std::size_t>(j) * dim + t], acc);
  O[(h * seq + i) * dim + t] = acc;
}

}  // namespace

void flash_attention_fwd(const float* Q, const float* K, const float* V, float* O, int bh, int seq, int dim,
                         float scale, bool causal, cudaStream_t stream) {
  if (bh <= 0 || seq <= 0) return;
  if (bh > 65535) throw std::invalid_argument("flash_attention_fwd: bh must be <= 65535 (gridDim.y)");
  switch (dim) {
    case 32: launch_flash<32>(Q, K, V, O, bh, seq, scale, causal, stream); break;
    case 64: launch_flash<64>(Q, K, V, O, bh, seq, scale, causal, stream); break;
    case 128: launch_flash<128>(Q, K, V, O, bh, seq, scale, causal, stream); break;
    default:
      throw std::invalid_argument("flash_attention_fwd: head dim " + std::to_string(dim) +
                                  " not supported (compiled for 32, 64, 128)");
  }
}

void attention_naive(const float* Q, const float* K, const float* V, float* O, float* scratch, int bh, int seq,
                     int dim, float scale, bool causal, cudaStream_t stream) {
  if (bh <= 0 || seq <= 0) return;
  naive_scores_kernel<<<dim3((seq + 127) / 128, seq, bh), 128, 0, stream>>>(Q, K, scratch, seq, dim, scale, causal);
  VESSEL_CUDA_CHECK_LAUNCH();
  const int rows = bh * seq;
  naive_softmax_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(scratch, rows, seq);
  VESSEL_CUDA_CHECK_LAUNCH();
  naive_pv_kernel<<<dim3((dim + 127) / 128, seq, bh), 128, 0, stream>>>(scratch, V, O, seq, dim);
  VESSEL_CUDA_CHECK_LAUNCH();
}

const char* flash_tile_string(int dim) noexcept {
  switch (dim) {
    case 32: return "Br64xBc64,D32,256thr,cp.async_dbuf";
    case 64: return "Br64xBc64,D64,256thr,cp.async_dbuf";
    case 128: return "Br32xBc32,D128,128thr,cp.async_dbuf";
    default: return "unsupported";
  }
}

}  // namespace vessel::gpu::cuda
