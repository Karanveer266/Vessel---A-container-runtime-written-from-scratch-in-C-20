// SPDX-License-Identifier: MIT
// FlashAttention-2-style fused attention forward, FP32 SIMT.
//
//     O = softmax(Q K^T * scale [+ causal mask]) V,   Q,K,V,O: [bh, seq, D]
//
// Included by flash_attention.cu (nvcc) and tests/test_simt_kernels.cpp
// (host SIMT emulator) — see device_compat.cuh.
//
// ===========================================================================
// Algorithm (Dao 2023, "FlashAttention-2", Alg. 1, forward only)
// ===========================================================================
//   grid  = (ceil(seq/Br), bh)        one block per (Q row-block, head)
//   block = 8 warps (D<=64) or 4 (D=128); warp w owns query rows
//           [w*RW, (w+1)*RW) of the block, RW = 8
//
//   Qs <- Q[q0:q0+Br] * (scale*log2 e)               (once, stays in SMEM)
//   m = -inf, l = 0, acc = 0                         (registers, per row)
//   for each K/V tile c (Bc keys), double-buffered via cp.async:
//       S   = Qs Kc^T                                (registers, RW x Bc per warp)
//       mask (causal / seq tail) only on boundary tiles
//       m'  = max(m, rowmax S);  P = exp2(S - m');  alpha = exp2(m - m')
//       l   = alpha*l + rowsum P;   acc = alpha*acc + P Vc
//   O = acc / l
//
// Folding scale*log2(e) into Q once means the inner loop uses exp2 (a single
// MUFU.EX2) instead of exp (= ex2 + an FMUL), and saves a multiply per score.
// The full seq x seq matrix never exists: per block we hold Br x Bc scores.
// HBM traffic is O(seq*D) per head for K/V *per Q block*, i.e.
// O(seq^2 * D / Br) total instead of O(seq^2) for materializing S + P.
//
// ===========================================================================
// Thread mapping (per warp, RW = 8 query rows, KPL = Bc/32 keys per lane)
// ===========================================================================
//   S phase : lane owns keys {lane + 32*c | c < KPL} for all RW rows.
//             Per 4-wide d step it issues KPL LDS.128 of K (its own keys)
//             and RW broadcast LDS.128 of Q, then 4*RW*KPL FFMA
//             (D=64: 10 LDS -> 64 FFMA).
//   softmax : row max / row sum = 5-step __shfl_xor_sync butterflies, so every
//             lane ends up with m, l for all RW rows (no SMEM round trip).
//   PV phase: lane owns output dims [lane*DPL, lane*DPL + DPL), DPL = D/32.
//             P goes through a per-warp SMEM scratch (written by key-owner
//             lanes, read back as broadcast LDS.128 by dim-owner lanes):
//             per 4 keys: RW LDS.128 (P) + 4 LDS.DPL (V) -> 4*RW*DPL FFMA.
//             Only __syncwarp is needed around it (warp-private buffer).
//
// ===========================================================================
// Shared memory layout & bank conflicts
// ===========================================================================
//   Qs [Br][D]          read as broadcast (all lanes same address)  -> free
//   Ks [2][Bc][D + 4]   lane j reads row j, 16 B at column d: LDS.128 is
//                       served in quarter-warp phases of 8 lanes; the row
//                       stride (D+4)*4 B = (D/4 + 1) 16-B chunks is odd, so
//                       8 consecutive rows hit 8 distinct chunk positions
//                       -> conflict-free. (Unpadded, D/4 is even and lanes
//                       would collide 2..8-way.) +4 keeps 16 B alignment
//                       for both LDS.128 and the cp.async destination.
//   Vs [2][Bc][D]       lane reads DPL consecutive floats at lane*DPL:
//                       consecutive addresses -> conflict-free.
//   Ps [4][RW][Bc]      broadcast reads; writes by lane j to column j
//                       -> consecutive -> conflict-free.
//   Sizes (fp32): D=32 (Br64,Bc64): 58 KiB   D=64 (Br64,Bc64): 98 KiB
//                 D=128 (Br32,Bc32): 85 KiB
//   All exceed the 48 KiB static limit -> cudaFuncAttributeMaxDynamicSharedMemorySize,
//   and all fit the 99 KiB per-block opt-in limit of sm_86/sm_89. Residency:
//   1 block/SM (8 or 4 warps) on sm_86/89; on sm_80 / sm_90 (164 / 228 KiB
//   per SM): D=32 -> 2 / 2 blocks (sm_90 is then register-limited: 3 x 32K
//   regs > 64K), D=64 -> 1 / 2, D=128 -> 1 / 2.
//   Low occupancy is the price of keeping Q resident and double-buffering
//   K/V; it is compensated by ILP (RW*KPL independent accumulators per lane).
//
// ===========================================================================
// Pipelining
// ===========================================================================
//   cp.async.cg (sm_80+, L2-only, bypasses registers and L1) copies tile c+1
//   into the other SMEM stage while tile c is being computed:
//     issue(c+1); commit; wait_group<1>  // tile c landed (c+1 may be in flight)
//     __syncthreads                       // ...and is visible to all threads
//     compute(c)
//     __syncthreads                       // stage c free for issue(c+2)
//   Out-of-range key rows use the zero-fill form (src-size 0): uninitialized
//   SMEM could hold NaNs and 0 * NaN = NaN would poison acc via P*V.
//   On < sm_80 the same code path falls back to synchronous ld/st.
//
// ===========================================================================
// Causal masking
// ===========================================================================
//   * Tiles entirely above the diagonal are never loaded: the tile loop stops
//     at min(seq, q0 + Br) — about half the work and K/V traffic.
//   * The mask (and the seq-tail mask) is only evaluated on tiles that
//     straddle the boundary; the check is warp-uniform, so no divergence on
//     interior tiles.
//   * Blocks are scheduled heaviest-first (reverse blockIdx) so the long
//     causal rows don't form a tail at the end of the grid.
//
// Registers (ptxas 12.8, sm_80, measured with -Xptxas -v): D=32: 128,
// D=64: 128, D=128: 140; 0 bytes spilled. D<=64 blocks use 256 x 128 = 32K
// registers (half the RF, consistent with the SMEM-imposed 1-2 blocks/SM).
#pragma once

#include <cstddef>
#include <cstdint>

#include "device_compat.cuh"

namespace vessel::gpu::cuda::kern {

template <int D>
struct FlashConfig {
  static_assert(D == 32 || D == 64 || D == 128, "head_dim must be 32, 64 or 128");
  // D <= 64: 8 warps (Br = 64) -> 8 resident warps/SM even at 1 block/SM.
  // D = 128: 4 warps (Br = 32) so Qs + double-buffered K/V stay < 99 KiB.
  static constexpr int kWarps = D == 128 ? 4 : 8;
  static constexpr int kThreads = kWarps * 32;
  static constexpr int RW = 8;                   // query rows per warp
  static constexpr int Br = kWarps * RW;         // query rows per block (64 or 32)
  static constexpr int Bc = D == 128 ? 32 : 64;  // keys per tile (D=128: keep SMEM < 99 KiB)
  static constexpr int KPL = Bc / 32;            // keys per lane (S phase)
  static constexpr int DPL = D / 32;             // output dims per lane (PV phase)
  static constexpr int KLD = D + 4;              // padded K row (see header comment)

  // SMEM carve-up, in floats; every region offset is a multiple of 4 floats (16 B).
  static constexpr int kQsOff = 0;
  static constexpr int kKsOff = kQsOff + Br * D;
  static constexpr int kVsOff = kKsOff + 2 * Bc * KLD;
  static constexpr int kPsOff = kVsOff + 2 * Bc * D;
  static constexpr int kSmemFloats = kPsOff + kWarps * RW * Bc;
  static constexpr std::size_t kSmemBytes = kSmemFloats * sizeof(float);
  // 99 KiB = largest opt-in dynamic SMEM per block on sm_86/sm_89 (the
  // tightest of our default architectures).
  static_assert(kSmemBytes <= 99 * 1024, "tile config exceeds the sm_86/89 shared-memory limit");
  static_assert(Bc % 32 == 0 && Bc % 4 == 0 && D % 4 == 0, "tile shape invariants");
};

// ---------------------------------------------------------------------------
// cp.async helpers (Ampere+). 16-byte global -> shared copy with zero-fill.
// ---------------------------------------------------------------------------
VESSEL_DEVICE void cp_async16(float* smem_dst, const float* gmem_src, bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const auto dst = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  const int src_bytes = valid ? 16 : 0;  // 0 => write 16 zero bytes, read nothing
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(gmem_src), "r"(src_bytes)
               : "memory");
#else
  dev::st4(smem_dst, valid ? dev::ldg4(gmem_src) : make_float4(0.f, 0.f, 0.f, 0.f));
#endif
}
VESSEL_DEVICE void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
template <int N>
VESSEL_DEVICE void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
#endif
}

template <int D>
VESSEL_GLOBAL void VESSEL_LAUNCH_BOUNDS(FlashConfig<D>::kThreads, 1)
    flash_attention_fwd_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                               const float* __restrict__ V, float* __restrict__ O, int seq, float scale_log2,
                               bool causal) {
  using Cfg = FlashConfig<D>;
  constexpr int Br = Cfg::Br, Bc = Cfg::Bc, RW = Cfg::RW, KPL = Cfg::KPL, DPL = Cfg::DPL, KLD = Cfg::KLD;
  constexpr int kThreads = Cfg::kThreads;
  constexpr int kVecPerRow = D / 4;

  VESSEL_DYNAMIC_SMEM(float, smem);
  float* Qs = smem + Cfg::kQsOff;
  float* Ks = smem + Cfg::kKsOff;
  float* Vs = smem + Cfg::kVsOff;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid % 32;
  float* Ps = smem + Cfg::kPsOff + warp * RW * Bc;

  const int n_qblocks = static_cast<int>(gridDim.x);
  // Heaviest-first scheduling for causal: the last Q block sees every key.
  const int qb = causal ? n_qblocks - 1 - static_cast<int>(blockIdx.x) : static_cast<int>(blockIdx.x);
  const int q0 = qb * Br;
  const std::size_t head_off = static_cast<std::size_t>(blockIdx.y) * static_cast<std::size_t>(seq) * D;
  Q += head_off;
  K += head_off;
  V += head_off;
  O += head_off;

  // ---- Q tile -> SMEM, pre-scaled into the log2 domain ----------------------
  for (int v = tid; v < Br * kVecPerRow; v += kThreads) {
    const int r = v / kVecPerRow;
    const int c = (v % kVecPerRow) * 4;
    float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
    if (q0 + r < seq) x = dev::ldg4(Q + static_cast<std::size_t>(q0 + r) * D + c);
    dev::st4(Qs + r * D + c, make_float4(x.x * scale_log2, x.y * scale_log2, x.z * scale_log2, x.w * scale_log2));
  }

  // ---- K/V tile loader (cp.async, zero-filled past seq) ---------------------
  auto issue_kv = [&](int tile, int stage) {
    const int c0 = tile * Bc;
    float* ks = Ks + stage * Bc * KLD;
    float* vs = Vs + stage * Bc * D;
    for (int v = tid; v < Bc * kVecPerRow; v += kThreads) {
      const int r = v / kVecPerRow;
      const int c = (v % kVecPerRow) * 4;
      const bool ok = c0 + r < seq;
      // Clamp the source address for zero-filled rows so it is never formed out of bounds.
      const std::size_t src = static_cast<std::size_t>(ok ? c0 + r : 0) * D + c;
      cp_async16(ks + r * KLD + c, K + src, ok);
      cp_async16(vs + r * D + c, V + src, ok);
    }
    cp_async_commit();
  };

  // Causal: keys beyond the block's last query are masked for every row -> skip.
  const int kv_end = causal ? (q0 + Br < seq ? q0 + Br : seq) : seq;
  const int n_tiles = (kv_end + Bc - 1) / Bc;

  float m[RW], l[RW], acc[RW][DPL];
  VESSEL_UNROLL
  for (int r = 0; r < RW; ++r) {
    m[r] = -INFINITY;
    l[r] = 0.f;
    VESSEL_UNROLL
    for (int t = 0; t < DPL; ++t) acc[r][t] = 0.f;
  }
  const int row0 = q0 + warp * RW;  // first query row of this warp
  const float* q_warp = Qs + warp * RW * D;

  if (n_tiles > 0) issue_kv(0, 0);

  for (int tile = 0; tile < n_tiles; ++tile) {
    const int stage = tile & 1;
    if (tile + 1 < n_tiles) {
      issue_kv(tile + 1, stage ^ 1);
      cp_async_wait<1>();  // all groups except the newest are complete -> tile landed
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();  // other threads' copies visible (+ Qs on the first iteration)

    const float* ks = Ks + stage * Bc * KLD;
    const float* vs = Vs + stage * Bc * D;
    const int c0 = tile * Bc;

    // ---- S = Q K^T for RW rows x KPL keys per lane ----
    float s[RW][KPL];
    VESSEL_UNROLL
    for (int r = 0; r < RW; ++r) {
      VESSEL_UNROLL
      for (int c = 0; c < KPL; ++c) s[r][c] = 0.f;
    }
    VESSEL_UNROLL
    for (int d = 0; d < D; d += 4) {
      float kf[KPL][4];
      VESSEL_UNROLL
      for (int c = 0; c < KPL; ++c) dev::ldN<4>(ks + (lane + 32 * c) * KLD + d, kf[c]);
      VESSEL_UNROLL
      for (int r = 0; r < RW; ++r) {
        float qf[4];
        dev::ldN<4>(q_warp + r * D + d, qf);  // broadcast
        VESSEL_UNROLL
        for (int c = 0; c < KPL; ++c) {
          VESSEL_UNROLL
          for (int e = 0; e < 4; ++e) s[r][c] = fmaf(qf[e], kf[c][e], s[r][c]);
        }
      }
    }

    // ---- masking: only on tiles crossing the seq tail or the diagonal ----
    // Warp-uniform condition (depends on warp's rows and tile, not on lane).
    const bool tail = c0 + Bc > seq;
    const bool diag = causal && c0 + Bc - 1 > row0;
    if (tail || diag) {
      VESSEL_UNROLL
      for (int c = 0; c < KPL; ++c) {
        const int key = c0 + lane + 32 * c;
        VESSEL_UNROLL
        for (int r = 0; r < RW; ++r) {
          if (key >= seq || (causal && key > row0 + r)) s[r][c] = -INFINITY;
        }
      }
    }

    // ---- online softmax update ----
    float alpha[RW];
    VESSEL_UNROLL
    for (int r = 0; r < RW; ++r) {
      float mx = s[r][0];
      VESSEL_UNROLL
      for (int c = 1; c < KPL; ++c) mx = fmaxf(mx, s[r][c]);
      VESSEL_UNROLL
      for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, dev::warp_xor(mx, o));
      const float m_new = fmaxf(m[r], mx);
      // A row whose keys are all masked so far keeps m = -inf; subtract 0
      // instead so exp2(-inf - (-inf)) = NaN can never occur.
      const float m_use = m_new == -INFINITY ? 0.f : m_new;
      alpha[r] = dev::fast_exp2(m[r] - m_use);
      float sum = 0.f;
      VESSEL_UNROLL
      for (int c = 0; c < KPL; ++c) {
        const float p = dev::fast_exp2(s[r][c] - m_use);
        sum += p;
        Ps[r * Bc + lane + 32 * c] = p;
      }
      VESSEL_UNROLL
      for (int o = 16; o > 0; o >>= 1) sum += dev::warp_xor(sum, o);
      l[r] = alpha[r] * l[r] + sum;
      m[r] = m_new;
    }
    __syncwarp();  // Ps written by key-owner lanes -> read by dim-owner lanes

    // ---- acc = alpha * acc + P V ----
    VESSEL_UNROLL
    for (int r = 0; r < RW; ++r) {
      VESSEL_UNROLL
      for (int t = 0; t < DPL; ++t) acc[r][t] *= alpha[r];
    }
#if defined(__CUDACC__)
#pragma unroll 4
#endif
    for (int j = 0; j < Bc; j += 4) {
      float vf[4][DPL];
      VESSEL_UNROLL
      for (int e = 0; e < 4; ++e) dev::ldN<DPL>(vs + (j + e) * D + lane * DPL, vf[e]);
      VESSEL_UNROLL
      for (int r = 0; r < RW; ++r) {
        float pf[4];
        dev::ldN<4>(Ps + r * Bc + j, pf);  // broadcast
        VESSEL_UNROLL
        for (int e = 0; e < 4; ++e) {
          VESSEL_UNROLL
          for (int t = 0; t < DPL; ++t) acc[r][t] = fmaf(pf[e], vf[e][t], acc[r][t]);
        }
      }
    }
    __syncthreads();  // everyone done with this stage (and Ps) before it is refilled
  }

  // ---- epilogue: normalize and store (vectorized, coalesced across lanes) ----
  VESSEL_UNROLL
  for (int r = 0; r < RW; ++r) {
    const int row = row0 + r;
    if (row >= seq) continue;
    const float inv = l[r] > 0.f ? 1.f / l[r] : 0.f;
    float out[DPL];
    VESSEL_UNROLL
    for (int t = 0; t < DPL; ++t) out[t] = acc[r][t] * inv;
    dev::stN<DPL>(O + static_cast<std::size_t>(row) * D + lane * DPL, out);
  }
}

}  // namespace vessel::gpu::cuda::kern
