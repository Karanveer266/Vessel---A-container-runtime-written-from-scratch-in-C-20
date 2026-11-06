// SPDX-License-Identifier: MIT
// CPU SGEMM: naive oracle + GotoBLAS/BLIS-structured blocked kernel.
//
// Loop nest (Goto & van de Geijn, "Anatomy of High-Performance Matrix
// Multiplication", TOMS 2008; BLIS terminology):
//
//   for jc in N step NC          B panel  KC x NC  -> lives in L3
//     for pc in K step KC          pack B~ (KC x NC, NR-wide column slivers)
//       for ic in M step MC  [||]   pack A~ (MC x KC, MR-tall row slivers) -> L2
//         for jr in NC step NR        B~ sliver (KC x NR)                 -> L1
//           for ir in MC step MR        micro-kernel: MR x NR of C in registers
//             for p in KC                 rank-1 update: MR broadcasts x NR vector
//
// This is exactly the hierarchy the CUDA SGEMM uses one level up:
//   CUDA thread-block tile (smem)  <->  MC x KC A~ block / KC x NC B~ panel
//   per-thread 8x8 register tile   <->  MR x NR micro-tile
//   transposed A tile in smem      <->  A~ packed column-major per sliver
// so the sim backend benchmarks the same data-reuse strategy.
//
// Arithmetic intensity of the micro-kernel per k step: loads MR + NR floats
// (one sliver column of A~, one row of B~), does 2*MR*NR FLOPs. With MR=6,
// NR=16 that is 192 FLOP / 22 loads: the FMA units (2 x 8-wide per cycle on
// AVX2) stay saturated while L1 delivers only ~1.4 loads/cycle.
//
// Register budget (AVX2, 16 ymm): 6*16/8 = 12 accumulators + 2 B vectors +
// 1 broadcast = 15 registers — the classic Haswell 6x16 shape. Without AVX
// (SSE, 16 xmm) we switch to 4x8 = 8 accumulators so nothing spills.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>

#include "vessel/gpu/common.hpp"
#include "vessel/gpu/cpu_kernels.hpp"

namespace vessel::gpu::cpu {

// ---------------------------------------------------------------------------
// Oracle
// ---------------------------------------------------------------------------
void gemm_naive(const float* A, const float* B, float* C, std::size_t M, std::size_t N, std::size_t K,
                std::size_t lda, std::size_t ldb, std::size_t ldc, bool accumulate) noexcept {
  // i-k-j order with a double accumulator row: streams B rows contiguously
  // (so the oracle is not unbearably slow for tests) while keeping a
  // higher-precision sum than any kernel under test.
  std::unique_ptr<double[]> acc(new (std::nothrow) double[N == 0 ? 1 : N]);
  if (!acc) return;  // OOM in an oracle: leave C untouched; the caller's comparison will fail.
  for (std::size_t i = 0; i < M; ++i) {
    std::fill_n(acc.get(), N, 0.0);
    for (std::size_t k = 0; k < K; ++k) {
      const double a = A[i * lda + k];
      const float* b = B + k * ldb;
      for (std::size_t j = 0; j < N; ++j) acc[j] += a * static_cast<double>(b[j]);
    }
    float* c = C + i * ldc;
    for (std::size_t j = 0; j < N; ++j)
      c[j] = static_cast<float>(accumulate ? static_cast<double>(c[j]) + acc[j] : acc[j]);
  }
}

// ---------------------------------------------------------------------------
// Blocked kernel
// ---------------------------------------------------------------------------
namespace {

#if defined(__AVX__)
constexpr std::size_t kVec = 8;  // floats per 256-bit register
constexpr std::size_t MR = 6;
constexpr const char* kIsa = "avx";
#else
constexpr std::size_t kVec = 4;  // floats per 128-bit register (SSE / NEON)
constexpr std::size_t MR = 4;
constexpr const char* kIsa = "sse";
#endif
constexpr std::size_t NR = 2 * kVec;
constexpr std::size_t KC = 256;
// MC multiple of MR; MC*KC*4 B = 72 KiB (AVX) fits comfortably in a 256 KiB+ L2
// next to the streaming B~ sliver.
constexpr std::size_t MC = MR * 12;
constexpr std::size_t NC = 2048;  // KC*NC*4 B = 2 MiB B~ panel -> L3
static_assert(MC % MR == 0 && NC % NR == 0);

// GCC/Clang generic vectors: lower to ymm on AVX, xmm on SSE/NEON, no
// intrinsics headers needed and portable across x86-64 and AArch64.
typedef float vfloat __attribute__((vector_size(kVec * sizeof(float))));

#if defined(__FMA__) || defined(__aarch64__)
constexpr const char* kFma = "+fma";
#else
constexpr const char* kFma = "";
#endif

inline vfloat vload(const float* p) noexcept {
  vfloat v;
  std::memcpy(&v, p, sizeof v);  // packed buffers are 64-B aligned; memcpy keeps it UB-free
  return v;
}
inline void vstore(float* p, vfloat v) noexcept { std::memcpy(p, &v, sizeof v); }

/// MR x NR micro-kernel: C_tile = beta*C_tile + A~sliver * B~sliver.
/// `a` is MR floats per k (column of the A sliver), `b` is NR floats per k.
/// beta is 0 or 1; when 0, C is never read (it may hold garbage/NaN).
/// Writes a full tile to `c` with stride ldc — edge tiles go through a
/// scratch buffer in the caller.
inline void micro_kernel(std::size_t kc, const float* __restrict a, const float* __restrict b,
                         float* __restrict c, std::size_t ldc, bool beta) noexcept {
  vfloat acc0[MR] = {};
  vfloat acc1[MR] = {};
  for (std::size_t p = 0; p < kc; ++p) {
    const vfloat b0 = vload(b);
    const vfloat b1 = vload(b + kVec);
#pragma GCC unroll 8
    for (std::size_t i = 0; i < MR; ++i) {
      const float ai = a[i];
      acc0[i] += ai * b0;  // contracts to vfmadd231ps with -mfma (-ffp-contract=fast is GCC's default)
      acc1[i] += ai * b1;
    }
    a += MR;
    b += NR;
  }
  for (std::size_t i = 0; i < MR; ++i) {
    float* row = c + i * ldc;
    if (beta) {
      vstore(row, vload(row) + acc0[i]);
      vstore(row + kVec, vload(row + kVec) + acc1[i]);
    } else {
      vstore(row, acc0[i]);
      vstore(row + kVec, acc1[i]);
    }
  }
}

/// Packs A[mc x kc] (row-major, stride lda) into MR-tall slivers stored
/// k-major: dst[s][p][i] = A[s*MR + i][p]. Rows beyond mc are zero-filled so
/// the micro-kernel never needs an M edge test in its inner loop.
void pack_a(const float* A, std::size_t lda, std::size_t mc, std::size_t kc, float* dst) noexcept {
  for (std::size_t s = 0; s < mc; s += MR) {
    const std::size_t rows = std::min(MR, mc - s);
    for (std::size_t p = 0; p < kc; ++p) {
      for (std::size_t i = 0; i < rows; ++i) dst[i] = A[(s + i) * lda + p];
      for (std::size_t i = rows; i < MR; ++i) dst[i] = 0.0F;
      dst += MR;
    }
  }
}

/// Packs NR-wide column slivers [j0, j1) of B[kc x nc] (stride ldb):
/// dst[s][p][j] = B[p][s*NR + j], zero-padded beyond nc.
void pack_b(const float* B, std::size_t ldb, std::size_t kc, std::size_t nc, std::size_t s0,
            std::size_t s1, float* dst) noexcept {
  for (std::size_t s = s0; s < s1; ++s) {
    const std::size_t j = s * NR;
    const std::size_t cols = std::min(NR, nc - j);
    float* d = dst + s * NR * kc;
    for (std::size_t p = 0; p < kc; ++p) {
      const float* src = B + p * ldb + j;
      if (cols == NR) {
        std::memcpy(d, src, NR * sizeof(float));
      } else {
        std::memcpy(d, src, cols * sizeof(float));
        std::fill(d + cols, d + NR, 0.0F);
      }
      d += NR;
    }
  }
}

struct AlignedDelete {
  void operator()(float* p) const noexcept { ::operator delete[](p, std::align_val_t{64}); }
};
using AlignedBuf = std::unique_ptr<float[], AlignedDelete>;
AlignedBuf alloc_aligned(std::size_t n) {
  return AlignedBuf(static_cast<float*>(::operator new[](n * sizeof(float), std::align_val_t{64})));
}

constexpr std::size_t ceil_div(std::size_t a, std::size_t b) noexcept { return (a + b - 1) / b; }

}  // namespace

std::string gemm_tiled_config() {
  char buf[96];
  std::snprintf(buf, sizeof buf, "MR%zuxNR%zu,KC%zu,MC%zu,NC%zu,%s%s", MR, NR, KC, MC, NC, kIsa, kFma);
  return buf;
}

void gemm_tiled(const float* A, const float* B, float* C, std::size_t M, std::size_t N, std::size_t K,
                std::size_t lda, std::size_t ldb, std::size_t ldc, bool accumulate, unsigned threads) {
  if (M == 0 || N == 0) return;
  if (K == 0) {  // C = 0 (or unchanged when accumulating), matching BLAS semantics.
    if (!accumulate)
      for (std::size_t i = 0; i < M; ++i) std::fill_n(C + i * ldc, N, 0.0F);
    return;
  }
  if (threads == 0) threads = default_thread_count();
  const std::size_t m_blocks = ceil_div(M, MC);
  // Never start more threads than there are A blocks to work on.
  const unsigned nthreads = static_cast<unsigned>(std::min<std::size_t>(threads, m_blocks));

  AlignedBuf bpack = alloc_aligned(KC * std::min(NC, ceil_div(N, NR) * NR));
  std::vector<AlignedBuf> apack;
  apack.reserve(nthreads);
  for (unsigned t = 0; t < nthreads; ++t) apack.push_back(alloc_aligned(MC * KC));

  for (std::size_t jc = 0; jc < N; jc += NC) {
    const std::size_t nc = std::min(NC, N - jc);
    const std::size_t n_slivers = ceil_div(nc, NR);
    for (std::size_t pc = 0; pc < K; pc += KC) {
      const std::size_t kc = std::min(KC, K - pc);
      const bool beta = accumulate || pc > 0;  // first K panel overwrites C
      const float* Bp = B + pc * ldb + jc;

      // B~ is shared by all threads: pack it cooperatively, then compute.
      // (The jthread join in parallel_for is the barrier between the phases.)
      parallel_for(n_slivers, nthreads, [&](std::size_t s0, std::size_t s1, unsigned) {
        pack_b(Bp, ldb, kc, nc, s0, s1, bpack.get());
      });

      // A blocks are independent: each thread packs its own A~ and owns a
      // disjoint set of C rows, so no synchronization is needed in the loop.
      parallel_for(m_blocks, nthreads, [&](std::size_t b0, std::size_t b1, unsigned tid) {
        float* ap = apack[tid].get();
        alignas(64) float scratch[MR * NR];
        for (std::size_t blk = b0; blk < b1; ++blk) {
          const std::size_t ic = blk * MC;
          const std::size_t mc = std::min(MC, M - ic);
          pack_a(A + ic * lda + pc, lda, mc, kc, ap);
          for (std::size_t s = 0; s < n_slivers; ++s) {
            const std::size_t jr = s * NR;
            const std::size_t nr = std::min(NR, nc - jr);
            const float* bs = bpack.get() + s * NR * kc;
            for (std::size_t ir = 0; ir < mc; ir += MR) {
              const std::size_t mr = std::min(MR, mc - ir);
              float* c = C + (ic + ir) * ldc + jc + jr;
              const float* as = ap + ir * kc;
              if (mr == MR && nr == NR) {
                micro_kernel(kc, as, bs, c, ldc, beta);
              } else {
                // Edge tile: compute into scratch, then merge the valid part.
                micro_kernel(kc, as, bs, scratch, NR, false);
                for (std::size_t i = 0; i < mr; ++i)
                  for (std::size_t j = 0; j < nr; ++j)
                    c[i * ldc + j] = beta ? c[i * ldc + j] + scratch[i * NR + j] : scratch[i * NR + j];
              }
            }
          }
        }
      });
    }
  }
}

}  // namespace vessel::gpu::cpu
