// SPDX-License-Identifier: MIT
// Portability layer for the CUDA kernels in sgemm_kernels.cuh and
// flash_attention_kernels.cuh.
//
// The kernels are written against a handful of macros/helpers instead of raw
// CUDA keywords so that the *same source* compiles in two ways:
//
//   1. nvcc (__CUDACC__)          -> real device code (sgemm.cu, flash_attention.cu)
//   2. host C++ + SIMT emulator   -> tests/simt_emu.hpp runs every CUDA thread
//      (VESSEL_SIMT_EMULATION)       of a block as an OS thread, __syncthreads
//                                    as std::barrier, shuffles via a per-warp
//                                    exchange buffer.
//
// (2) is what lets a GPU-less CI box *execute* the kernels' indexing, edge
// guards, shared-memory layout and barrier placement against CPU references.
// It says nothing about performance, and it cannot catch bugs that need real
// warp-level lockstep or the GPU memory model — but those are exactly the
// constructs these kernels avoid (all cross-thread communication goes
// through __syncthreads or full-mask __shfl_xor_sync).
#pragma once

#include <cmath>
#include <cstddef>
#include <cstring>

#if defined(__CUDACC__)

#include <cuda_runtime.h>

#define VESSEL_DEVICE __device__ __forceinline__
#define VESSEL_GLOBAL __global__
#define VESSEL_LAUNCH_BOUNDS(threads, min_blocks) __launch_bounds__(threads, min_blocks)
#define VESSEL_SHARED __shared__
#define VESSEL_ALIGN16 __align__(16)
#define VESSEL_UNROLL _Pragma("unroll")
// Dynamic shared memory: one untyped, 16-byte aligned arena per block.
#define VESSEL_DYNAMIC_SMEM(T, name)                              \
  extern __shared__ __align__(16) unsigned char vessel_smem_raw_[]; \
  T* name = reinterpret_cast<T*>(vessel_smem_raw_)

#else  // host SIMT emulation

#if !defined(VESSEL_SIMT_EMULATION)
#error "device_compat.cuh: compile with nvcc, or include tests/simt_emu.hpp first"
#endif

#define VESSEL_DEVICE inline
#define VESSEL_GLOBAL inline
#define VESSEL_LAUNCH_BOUNDS(threads, min_blocks)
// One block runs at a time in the emulator, so a function-local static is
// exactly "one instance shared by all threads of the block".
#define VESSEL_SHARED static
#define VESSEL_ALIGN16 alignas(16)
#define VESSEL_UNROLL
#define VESSEL_DYNAMIC_SMEM(T, name) T* name = reinterpret_cast<T*>(::simt::dynamic_smem())

#endif

namespace vessel::gpu::cuda::dev {

/// 128-bit global load through the read-only (non-coherent) path. `p` must be
/// 16-byte aligned. __ldg is a hint on sm_70+ (the compiler often infers it
/// from const __restrict__), kept explicit for clarity.
VESSEL_DEVICE float4 ldg4(const float* p) {
#if defined(__CUDACC__)
  return __ldg(reinterpret_cast<const float4*>(p));
#else
  float4 v;
  std::memcpy(&v, p, sizeof v);
  return v;
#endif
}

VESSEL_DEVICE float ldg1(const float* p) {
#if defined(__CUDACC__)
  return __ldg(p);
#else
  return *p;
#endif
}

/// 128-/64-bit shared or global memory accesses (LDS.128 / STS.128 etc.).
VESSEL_DEVICE float4 ld4(const float* p) {
#if defined(__CUDACC__)
  return *reinterpret_cast<const float4*>(p);
#else
  float4 v;
  std::memcpy(&v, p, sizeof v);
  return v;
#endif
}
VESSEL_DEVICE void st4(float* p, float4 v) {
#if defined(__CUDACC__)
  *reinterpret_cast<float4*>(p) = v;
#else
  std::memcpy(p, &v, sizeof v);
#endif
}
VESSEL_DEVICE float2 ld2(const float* p) {
#if defined(__CUDACC__)
  return *reinterpret_cast<const float2*>(p);
#else
  float2 v;
  std::memcpy(&v, p, sizeof v);
  return v;
#endif
}
VESSEL_DEVICE void st2(float* p, float2 v) {
#if defined(__CUDACC__)
  *reinterpret_cast<float2*>(p) = v;
#else
  std::memcpy(p, &v, sizeof v);
#endif
}

/// Loads N (1, 2 or 4) consecutive floats into out[0..N) with one vector
/// instruction. N is a compile-time tile parameter, so this folds away.
template <int N>
VESSEL_DEVICE void ldN(const float* p, float* out) {
  if constexpr (N == 4) {
    const float4 v = ld4(p);
    out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
  } else if constexpr (N == 2) {
    const float2 v = ld2(p);
    out[0] = v.x; out[1] = v.y;
  } else {
    static_assert(N == 1, "ldN supports 1, 2, 4");
    out[0] = *p;
  }
}
template <int N>
VESSEL_DEVICE void stN(float* p, const float* in) {
  if constexpr (N == 4) {
    st4(p, make_float4(in[0], in[1], in[2], in[3]));
  } else if constexpr (N == 2) {
    st2(p, make_float2(in[0], in[1]));
  } else {
    static_assert(N == 1, "stN supports 1, 2, 4");
    *p = in[0];
  }
}

VESSEL_DEVICE float warp_xor(float v, int lane_mask) {
#if defined(__CUDACC__)
  return __shfl_xor_sync(0xffffffffu, v, lane_mask);
#else
  return ::simt::shfl_xor(v, lane_mask);
#endif
}

VESSEL_DEVICE float fast_exp2(float x) {
#if defined(__CUDACC__)
  return exp2f(x);  // -> ex2.approx.ftz.f32 with --use_fast_math, ~2 ulp otherwise
#else
  return std::exp2(x);
#endif
}

}  // namespace vessel::gpu::cuda::dev
