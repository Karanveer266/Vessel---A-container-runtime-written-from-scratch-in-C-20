// SPDX-License-Identifier: MIT
// Minimal host-side SIMT emulator for executing the CUDA kernels in
// src/cuda/*_kernels.cuh on a machine without a GPU.
//
// Model:
//  * blocks run one after another; inside a block every CUDA thread is a real
//    OS thread (std::jthread), so code between barriers genuinely runs
//    concurrently and a missing __syncthreads shows up as a data race / wrong
//    result (and under -fsanitize=thread as a report);
//  * __syncthreads  -> std::barrier over the whole block
//  * __syncwarp     -> std::barrier over the 32 threads of the warp
//  * __shfl_xor_sync-> per-warp exchange buffer + two warp barriers
//  * __shared__     -> function-local `static` (one block at a time => one
//                      instance per block), dynamic smem -> a 16 B aligned arena
//
// What it validates: index math, edge guards, shared-memory layout (any
// overlap/out-of-range would corrupt results), barrier placement,
// double-buffer stage selection, online-softmax numerics.
// What it cannot validate: performance, bank conflicts, cp.async semantics
// (emulated synchronously), warp-synchronous code without explicit sync
// (the kernels contain none), the GPU memory model.
#pragma once

#define VESSEL_SIMT_EMULATION 1

#include <barrier>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <thread>
#include <vector>

#define __restrict__ __restrict

struct float4 { float x, y, z, w; };
struct float2 { float x, y; };
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
inline float2 make_float2(float x, float y) { return {x, y}; }

struct uint3 { unsigned x = 0, y = 0, z = 0; };
struct dim3 {
  unsigned x = 1, y = 1, z = 1;
  constexpr dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};

namespace simt {

struct Warp {
  float buf[32] = {};
  std::barrier<> bar{32};
};

struct BlockState {
  std::barrier<> block_bar;
  std::vector<std::unique_ptr<Warp>> warps;
  explicit BlockState(unsigned threads) : block_bar(static_cast<std::ptrdiff_t>(threads)) {
    for (unsigned w = 0; w < (threads + 31) / 32; ++w) warps.push_back(std::make_unique<Warp>());
  }
};

inline thread_local BlockState* tl_block = nullptr;
inline thread_local unsigned tl_linear_tid = 0;

struct SmemArena {
  void* p = nullptr;
  std::size_t bytes = 0;
  ~SmemArena() { if (p) ::operator delete(p, std::align_val_t{16}); }
  void ensure(std::size_t n) {
    if (n <= bytes) return;
    if (p) ::operator delete(p, std::align_val_t{16});
    p = ::operator new(n, std::align_val_t{16});
    bytes = n;
  }
};
inline SmemArena g_smem;
inline unsigned char* dynamic_smem() { return static_cast<unsigned char*>(g_smem.p); }

inline float shfl_xor(float v, int lane_mask) {
  Warp& w = *tl_block->warps[tl_linear_tid / 32];
  const unsigned lane = tl_linear_tid % 32;
  w.buf[lane] = v;
  w.bar.arrive_and_wait();
  const float r = w.buf[lane ^ static_cast<unsigned>(lane_mask)];
  w.bar.arrive_and_wait();  // nobody overwrites buf before everyone has read
  return r;
}

}  // namespace simt

// CUDA built-in variables (per emulated thread).
inline thread_local uint3 threadIdx;
inline thread_local uint3 blockIdx;
inline thread_local dim3 blockDim;
inline thread_local dim3 gridDim;

inline void __syncthreads() { simt::tl_block->block_bar.arrive_and_wait(); }
inline void __syncwarp() { simt::tl_block->warps[simt::tl_linear_tid / 32]->bar.arrive_and_wait(); }

namespace simt {

/// Equivalent of kernel<<<grid, block, smem_bytes>>>(args...), synchronous.
/// Block sizes must be a multiple of 32 (true for every kernel we emulate).
template <class Kernel, class... Args>
void launch(dim3 grid, dim3 block, std::size_t smem_bytes, Kernel kernel, Args... args) {
  const unsigned threads = block.x * block.y * block.z;
  g_smem.ensure(smem_bytes ? smem_bytes : 16);
  for (unsigned bz = 0; bz < grid.z; ++bz)
    for (unsigned by = 0; by < grid.y; ++by)
      for (unsigned bx = 0; bx < grid.x; ++bx) {
        BlockState state(threads);
        std::vector<std::jthread> pool;
        pool.reserve(threads);
        for (unsigned t = 0; t < threads; ++t) {
          pool.emplace_back([&, t] {
            tl_block = &state;
            tl_linear_tid = t;
            threadIdx = {t % block.x, (t / block.x) % block.y, t / (block.x * block.y)};
            blockIdx = {bx, by, bz};
            blockDim = block;
            gridDim = grid;
            kernel(args...);
          });
        }
      }  // jthreads join -> block finished
}

}  // namespace simt
