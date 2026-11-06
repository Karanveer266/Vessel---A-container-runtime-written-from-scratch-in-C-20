// SPDX-License-Identifier: MIT
// FP32 SGEMM kernels:  C = alpha * A * B + beta * C   (row-major, explicit ld*)
//
// Included by sgemm.cu (nvcc) and by tests/test_simt_kernels.cpp (host SIMT
// emulator) — see device_compat.cuh.
//
// ===========================================================================
// Why this design (roofline reasoning)
// ===========================================================================
// A100 (sm_80): 19.5 TFLOP/s FP32 (non-tensor), ~1.55 TB/s HBM2e
//   -> ridge point ~ 12.6 FLOP/byte.  H100 SXM: 67 TFLOP/s, 3.35 TB/s -> ~20.
// A square GEMM has an *algorithmic* intensity of 2n^3 / (3 * 4n^2) = n/6
// FLOP/B (683 at n=4096): it is compute bound in principle. Whether a kernel
// realizes that depends on reuse at every level of the memory hierarchy:
//
//  DRAM/L2 -> SMEM  (block tile BM x BN, K step BK)
//    Each block streams a BM x BK panel of A and a BK x BN panel of B per
//    K step and performs 2*BM*BN*BK FLOPs on them:
//        intensity = 2*BM*BN*BK / (4*(BM+BN)*BK) = BM*BN / (2*(BM+BN))
//                  = 32 FLOP/B for 128x128
//    i.e. even with *zero* L2 hits the kernel sits above the ridge point.
//    (The naive kernel re-reads A and B from L1/L2 for every output: 0.25 FLOP/B.)
//
//  SMEM -> registers (thread micro-tile TM x TN = 8 x 8)
//    Per k each thread loads TM + TN = 16 floats (4 x LDS.128) and issues
//    TM*TN = 64 FFMA: 16 FFMA per shared-memory instruction. An SM can issue
//    ~4 warp-FFMA/clk but only ~1 warp-LDS.128 every 2+ clk (128 B/clk smem
//    bandwidth), so a 4x4 micro-tile (8 FFMA/LDS) would be smem-bound; 8x8
//    leaves headroom for address math and the epilogue.
//
// ===========================================================================
// Thread / data mapping (256 threads = 8 warps, 2 blocks/SM target)
// ===========================================================================
//  tx = tid % 16, ty = tid / 16.  Thread (ty, tx) owns the 8x8 outputs
//     rows {ty*4 + i, 64 + ty*4 + i | i<4}  x  cols {tx*4 + j, 64 + tx*4 + j | j<4}
//  Splitting the 8 rows/cols into two groups of 4 that are 64 apart (instead
//  of 8 contiguous) is what makes every shared-memory read a conflict-free
//  LDS.128 *and* makes the C stores of a half-warp one contiguous 256 B run.
//
//  Shared memory (double buffered, 16.25 KiB = 16640 B/block):
//     As[2][BK][BM + 4]   A tile stored TRANSPOSED (k-major)
//     Bs[2][BK][BN]       B tile as-is (already k-major)
//  Transposing A at store time turns the compute loop's A accesses into
//  contiguous float4 reads along M. The +4 padding makes the transposed
//  4 x STS.32 store conflict-free: even lanes (a_col = 0) write row k at
//  columns a_row = 0..15, odd lanes (a_col = 4) write row k+4 at the same
//  columns; the rows' bank offsets differ by 4 * 132 mod 32 = 16, so the two
//  lane sets cover banks [b, b+16) and [b+16, b+32): 32 distinct banks.
//  (Unpadded, 4 * 128 mod 32 = 0 -> every store would be a 2-way conflict.)
//  The padded stride (528 B) is still a multiple of 16 B, keeping LDS.128
//  aligned.  Reads: As[k][ty*4] is one address per 8-lane LDS.128 phase
//  (broadcast); Bs[k][tx*4] covers 8 consecutive 16 B chunks per phase
//  = all 32 banks once.  => zero bank conflicts on the hot path.
//
//  Global loads: A tile (128x8) and B tile (8x128) are 256 float4 each, i.e.
//  exactly one LDG.128 per thread per operand per K step. B loads are fully
//  coalesced (a warp reads 512 contiguous bytes); A loads read 2 x 16 B per
//  row, 16 rows per warp — each 32 B sector is fully used.
//
// ===========================================================================
// Software pipeline (double buffering)
// ===========================================================================
//   prologue: LDG tile 0 -> regs -> STS buf 0; __syncthreads
//   loop kt:  LDG tile kt+1 -> regs        (latency ~500 cycles, in flight...)
//             8 x (4 LDS.128 + 64 FFMA) on buf[kt&1]   (...hidden behind this)
//             STS regs -> buf[(kt+1)&1]
//             __syncthreads                       (single barrier per K step)
//  One barrier suffices: the buffer written in iteration kt was last read in
//  iteration kt-1, which ended with a barrier; the buffer read in kt was
//  written before that same barrier.
//  The BK=8 inner loop is fully unrolled so ptxas can hoist the LDS of step
//  k+1 above the FFMAs of step k (register-level pipelining for free).
//
// ===========================================================================
// Occupancy / registers
// ===========================================================================
//   Budget: 64 accumulators + 16 fragments + 8 staging + addresses ≈ 120.
//   __launch_bounds__(256, 2) caps ptxas at 128 regs/thread so 2 blocks
//   (16 warps, 25% occupancy) fit in the 64K-register file; smem is not the
//   limiter (2 x 16.25 KiB). Measured (ptxas 12.8 -v): 126/127 regs on sm_80,
//   127/128 on sm_90 (vec4/scalar), 0 bytes spilled. Low occupancy is
//   intentional: latency is hidden by ILP (64 independent FFMA chains per
//   thread), not by warp count — the classic Volkov "better performance at
//   lower occupancy" trade.
//
// ===========================================================================
// Edge handling
// ===========================================================================
//  kVec4 = true  : requires K % 4 == 0, N % 4 == 0, lda/ldb/ldc % 4 == 0 and
//                  16 B aligned base pointers (sgemm_can_vectorize). Any float4
//                  that starts in bounds is then entirely in bounds, so one
//                  predicate per LDG.128/STG.128 handles all M/N/K edges.
//  kVec4 = false : same tiling, per-element guarded scalar loads/stores; used
//                  for odd shapes/strides (e.g. a split-K slice of K = 1023).
//  Out-of-range tile elements are loaded as 0, so partial tiles need no
//  special case inside the FFMA loop.
#pragma once

#include <cstddef>
#include <cstdint>

#include "device_compat.cuh"

namespace vessel::gpu::cuda::kern {

struct SgemmTile {
  static constexpr int BM = 128;
  static constexpr int BN = 128;
  static constexpr int BK = 8;
  static constexpr int TM = 8;
  static constexpr int TN = 8;
  static constexpr int kThreads = (BM / TM) * (BN / TN);  // 256
  static constexpr int kALd = BM + 4;                     // padded transposed-A row
};
static_assert(SgemmTile::kThreads == 256, "mapping below assumes 256 threads");
static_assert(SgemmTile::BM * SgemmTile::BK / 4 == SgemmTile::kThreads, "one float4 of A per thread");
static_assert(SgemmTile::BN * SgemmTile::BK / 4 == SgemmTile::kThreads, "one float4 of B per thread");
static_assert((SgemmTile::kALd * 4) % 16 == 0, "padded A rows must keep 16 B alignment");

/// Host-side predicate selecting the float4 instantiation.
inline bool sgemm_can_vectorize(const float* A, const float* B, const float* C, int N, int K, int lda, int ldb,
                                int ldc) {
  auto aligned = [](const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; };
  return K % 4 == 0 && N % 4 == 0 && lda % 4 == 0 && ldb % 4 == 0 && ldc % 4 == 0 && aligned(A) && aligned(B) &&
         aligned(C);
}

template <bool kVec4>
VESSEL_GLOBAL void VESSEL_LAUNCH_BOUNDS(SgemmTile::kThreads, 2)
    sgemm_128x128x8_kernel(int M, int N, int K, float alpha, const float* __restrict__ A, int lda,
                           const float* __restrict__ B, int ldb, float beta, float* __restrict__ C, int ldc) {
  constexpr int BM = SgemmTile::BM, BN = SgemmTile::BN, BK = SgemmTile::BK, ALD = SgemmTile::kALd;
  VESSEL_ALIGN16 VESSEL_SHARED float As[2][BK][ALD];
  VESSEL_ALIGN16 VESSEL_SHARED float Bs[2][BK][BN];

  const int tid = static_cast<int>(threadIdx.x);
  const int tx = tid % 16;
  const int ty = tid / 16;
  // blockIdx.x walks N so that consecutively scheduled blocks share the same
  // A row-panel (good L2 reuse of the operand read BN/BM times more often).
  const int bm = static_cast<int>(blockIdx.y) * BM;
  const int bn = static_cast<int>(blockIdx.x) * BN;

  // Global -> shared mapping (one float4 of each operand per thread).
  const int a_row = tid / 2;         // 0..127
  const int a_col = (tid % 2) * 4;   // 0 or 4
  const int b_row = tid / 32;        // 0..7
  const int b_col = (tid % 32) * 4;  // 0..124

  float a_stage[4];
  float b_stage[4];

  auto load_a = [&](int k0) {
    const int gr = bm + a_row;
    const int gk = k0 + a_col;
    if constexpr (kVec4) {
      if (gr < M && gk < K) {
        const float4 v = dev::ldg4(A + static_cast<std::size_t>(gr) * lda + gk);
        a_stage[0] = v.x; a_stage[1] = v.y; a_stage[2] = v.z; a_stage[3] = v.w;
      } else {
        a_stage[0] = a_stage[1] = a_stage[2] = a_stage[3] = 0.0f;
      }
    } else {
      VESSEL_UNROLL
      for (int i = 0; i < 4; ++i)
        a_stage[i] = (gr < M && gk + i < K) ? dev::ldg1(A + static_cast<std::size_t>(gr) * lda + gk + i) : 0.0f;
    }
  };
  auto load_b = [&](int k0) {
    const int gk = k0 + b_row;
    const int gn = bn + b_col;
    if constexpr (kVec4) {
      if (gk < K && gn < N) {
        const float4 v = dev::ldg4(B + static_cast<std::size_t>(gk) * ldb + gn);
        b_stage[0] = v.x; b_stage[1] = v.y; b_stage[2] = v.z; b_stage[3] = v.w;
      } else {
        b_stage[0] = b_stage[1] = b_stage[2] = b_stage[3] = 0.0f;
      }
    } else {
      VESSEL_UNROLL
      for (int j = 0; j < 4; ++j)
        b_stage[j] = (gk < K && gn + j < N) ? dev::ldg1(B + static_cast<std::size_t>(gk) * ldb + gn + j) : 0.0f;
    }
  };
  auto store_stage = [&](int buf) {
    VESSEL_UNROLL
    for (int i = 0; i < 4; ++i) As[buf][a_col + i][a_row] = a_stage[i];  // transpose on the fly
    dev::stN<4>(&Bs[buf][b_row][b_col], b_stage);
  };

  float acc[8][8];
  VESSEL_UNROLL
  for (int i = 0; i < 8; ++i) {
    VESSEL_UNROLL
    for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;
  }

  const int k_tiles = (K + BK - 1) / BK;
  load_a(0);
  load_b(0);
  store_stage(0);
  __syncthreads();

  for (int kt = 0; kt < k_tiles; ++kt) {
    const int cur = kt & 1;
    const bool has_next = kt + 1 < k_tiles;
    if (has_next) {  // issue next tile's global loads first: they overlap the FFMAs below
      load_a((kt + 1) * BK);
      load_b((kt + 1) * BK);
    }

    VESSEL_UNROLL
    for (int k = 0; k < BK; ++k) {
      float fa[8];
      float fb[8];
      dev::ldN<4>(&As[cur][k][ty * 4], fa);
      dev::ldN<4>(&As[cur][k][64 + ty * 4], fa + 4);
      dev::ldN<4>(&Bs[cur][k][tx * 4], fb);
      dev::ldN<4>(&Bs[cur][k][64 + tx * 4], fb + 4);
      VESSEL_UNROLL
      for (int i = 0; i < 8; ++i) {
        VESSEL_UNROLL
        for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(fa[i], fb[j], acc[i][j]);
      }
    }

    if (has_next) store_stage(cur ^ 1);
    __syncthreads();
  }

  // Epilogue. beta == 0 must not read C (BLAS semantics: C may be garbage/NaN).
  const bool read_c = beta != 0.0f;
  VESSEL_UNROLL
  for (int i = 0; i < 8; ++i) {
    const int row = bm + (i < 4 ? ty * 4 + i : 64 + ty * 4 + (i - 4));
    if (row >= M) continue;
    float* c_row = C + static_cast<std::size_t>(row) * ldc;
    VESSEL_UNROLL
    for (int h = 0; h < 2; ++h) {
      const int col = bn + h * 64 + tx * 4;
      float v[4];
      VESSEL_UNROLL
      for (int j = 0; j < 4; ++j) v[j] = alpha * acc[i][h * 4 + j];
      if constexpr (kVec4) {
        if (col < N) {
          if (read_c) {
            float old[4];
            dev::ldN<4>(c_row + col, old);
            VESSEL_UNROLL
            for (int j = 0; j < 4; ++j) v[j] = fmaf(beta, old[j], v[j]);
          }
          dev::stN<4>(c_row + col, v);
        }
      } else {
        VESSEL_UNROLL
        for (int j = 0; j < 4; ++j) {
          if (col + j < N) c_row[col + j] = read_c ? fmaf(beta, c_row[col + j], v[j]) : v[j];
        }
      }
    }
  }
}

/// Reference kernel: one thread per output, no reuse beyond L1/L2.
/// 2 FLOP per 8 B loaded => ~0.25 FLOP/B, firmly memory/latency bound; it
/// exists to verify the tiled kernel and to quantify its speedup.
VESSEL_GLOBAL void sgemm_naive_kernel(int M, int N, int K, float alpha, const float* __restrict__ A, int lda,
                                      const float* __restrict__ B, int ldb, float beta, float* __restrict__ C,
                                      int ldc) {
  const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (row >= M || col >= N) return;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k)
    acc = fmaf(A[static_cast<std::size_t>(row) * lda + k], B[static_cast<std::size_t>(k) * ldb + col], acc);
  float* c = C + static_cast<std::size_t>(row) * ldc + col;
  *c = beta != 0.0f ? fmaf(beta, *c, alpha * acc) : alpha * acc;
}

}  // namespace vessel::gpu::cuda::kern
