// SPDX-License-Identifier: MIT
// CPU attention: naive oracle + FlashAttention-2-style tiled forward pass.
//
// Online softmax (Milakov & Gimelshein 2018; Dao et al. 2022/2023)
// ----------------------------------------------------------------
// For one query row q, keys are consumed in blocks B_1, B_2, ... while three
// running quantities are kept:
//     m  = max score seen so far
//     l  = sum_j exp(s_j - m)
//     o  = sum_j exp(s_j - m) * v_j        (un-normalized output, dim floats)
// When a new block with scores s (block max m_b) arrives:
//     m' = max(m, m_b)
//     alpha = exp(m - m')                   (rescales the old state)
//     p_j = exp(s_j - m')
//     l' = alpha * l + sum_j p_j
//     o' = alpha * o + sum_j p_j v_j
// and at the end O = o / l. This is exact (not an approximation): it is the
// same softmax, merely evaluated in a streaming, numerically safe way, so the
// N x N score matrix never exists — memory is O(N * d) instead of O(N^2).
//
// FlashAttention-2 work decomposition (used both here and in CUDA):
//  * the outer loop is over Q row-blocks (Br rows) — each is an independent
//    task (here: a thread-pool work item; on the GPU: a thread block);
//  * the inner loop streams K/V column-blocks (Bc keys) through fast memory
//    (here: L1/L2; on the GPU: shared memory);
//  * normalization by 1/l happens once at the end, not per block.
//  * causal: key block c is skipped entirely when its first key is past the
//    last query of the row block, and only the diagonal block is masked
//    element-wise.
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "vessel/gpu/common.hpp"
#include "vessel/gpu/cpu_kernels.hpp"

namespace vessel::gpu::cpu {

void attention_naive(const float* Q, const float* K, const float* V, float* O, const AttentionShape& s,
                     float scale) noexcept {
  const std::size_t n = s.seq;
  const std::size_t d = s.dim;
  std::vector<double> p(n);
  std::vector<double> acc(d);
  for (std::size_t h = 0; h < s.bh; ++h) {
    const float* q = Q + h * n * d;
    const float* k = K + h * n * d;
    const float* v = V + h * n * d;
    float* o = O + h * n * d;
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t kend = s.causal ? i + 1 : n;
      // 1) full score row, 2) max, 3) exp + sum, 4) weighted sum of V.
      double mx = -std::numeric_limits<double>::infinity();
      for (std::size_t j = 0; j < kend; ++j) {
        double dot = 0.0;
        for (std::size_t t = 0; t < d; ++t) dot += static_cast<double>(q[i * d + t]) * k[j * d + t];
        p[j] = dot * scale;
        mx = std::max(mx, p[j]);
      }
      double sum = 0.0;
      for (std::size_t j = 0; j < kend; ++j) sum += (p[j] = std::exp(p[j] - mx));
      std::fill(acc.begin(), acc.end(), 0.0);
      for (std::size_t j = 0; j < kend; ++j)
        for (std::size_t t = 0; t < d; ++t) acc[t] += p[j] * v[j * d + t];
      for (std::size_t t = 0; t < d; ++t) o[i * d + t] = static_cast<float>(acc[t] / sum);
    }
  }
}

namespace {

constexpr std::size_t Br = FlashBlocking::Br;
constexpr std::size_t Bc = FlashBlocking::Bc;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

/// Per-thread scratch reused across tasks (no allocation in the hot loop).
struct FlashScratch {
  std::vector<float> S;     // Br x Bc score / probability tile
  std::vector<float> Kt;    // dim x Bc transposed K tile (unit-stride inner loop)
  std::vector<float> acc;   // Br x dim un-normalized output
  std::vector<float> m, l;  // Br running max / sum
  explicit FlashScratch(std::size_t d) : S(Br * Bc), Kt(d * Bc), acc(Br * d), m(Br), l(Br) {}
};

/// One (head, query-block) task.
void flash_block(const float* q, const float* k, const float* v, float* o, std::size_t n, std::size_t d,
                 std::size_t r0, bool causal, float scale, FlashScratch& w) {
  const std::size_t br = std::min(Br, n - r0);
  std::fill_n(w.acc.data(), br * d, 0.0F);
  std::fill_n(w.m.data(), br, kNegInf);
  std::fill_n(w.l.data(), br, 0.0F);

  // Causal early exit: keys > r0+br-1 are masked for every row in the block.
  const std::size_t kv_end = causal ? std::min(n, r0 + br) : n;
  for (std::size_t c0 = 0; c0 < kv_end; c0 += Bc) {
    const std::size_t bc = std::min(Bc, kv_end - c0);
    // Only blocks that straddle the diagonal need an element-wise mask.
    const bool diag = causal && c0 + bc > r0 + 1;

    // Transpose the K tile once per (task, block) so S = Q K^T becomes a
    // row-broadcast x unit-stride-vector update that auto-vectorizes; the
    // transpose is O(Bc*d) vs O(Br*Bc*d) FMAs.
    for (std::size_t j = 0; j < bc; ++j)
      for (std::size_t t = 0; t < d; ++t) w.Kt[t * Bc + j] = k[(c0 + j) * d + t];

    for (std::size_t i = 0; i < br; ++i) {
      float* srow = w.S.data() + i * Bc;
      std::fill_n(srow, bc, 0.0F);
      const float* qi = q + (r0 + i) * d;
      for (std::size_t t = 0; t < d; ++t) {
        const float qv = qi[t] * scale;  // fold the scale into Q: one mul per element instead of per score
        const float* kt = w.Kt.data() + t * Bc;
        for (std::size_t j = 0; j < bc; ++j) srow[j] += qv * kt[j];
      }

      const std::size_t row = r0 + i;
      // Valid keys for this row inside the block (causal: j <= row).
      const std::size_t valid = diag ? (row >= c0 ? std::min(bc, row - c0 + 1) : 0) : bc;
      if (valid == 0) continue;  // whole block masked for this row: state unchanged

      float mb = kNegInf;
      for (std::size_t j = 0; j < valid; ++j) mb = std::max(mb, srow[j]);
      const float m_new = std::max(w.m[i], mb);  // finite: valid >= 1
      const float alpha = std::exp(w.m[i] - m_new);  // exp(-inf) = 0 on the first block
      float psum = 0.0F;
      for (std::size_t j = 0; j < valid; ++j) psum += (srow[j] = std::exp(srow[j] - m_new));
      w.l[i] = alpha * w.l[i] + psum;
      w.m[i] = m_new;

      float* ai = w.acc.data() + i * d;
      if (alpha != 1.0F)
        for (std::size_t t = 0; t < d; ++t) ai[t] *= alpha;
      // O += P V : axpy over contiguous V rows (unit stride, vectorized).
      for (std::size_t j = 0; j < valid; ++j) {
        const float pj = srow[j];
        const float* vj = v + (c0 + j) * d;
        for (std::size_t t = 0; t < d; ++t) ai[t] += pj * vj[t];
      }
    }
  }
  for (std::size_t i = 0; i < br; ++i) {
    const float inv = 1.0F / w.l[i];  // l >= 1: the max element contributes exp(0)
    const float* ai = w.acc.data() + i * d;
    float* oi = o + (r0 + i) * d;
    for (std::size_t t = 0; t < d; ++t) oi[t] = ai[t] * inv;
  }
}

}  // namespace

void flash_attention(const float* Q, const float* K, const float* V, float* O, const AttentionShape& s,
                     float scale, unsigned threads) {
  if (s.bh == 0 || s.seq == 0 || s.dim == 0) return;
  const std::size_t n = s.seq;
  const std::size_t d = s.dim;
  const std::size_t qblocks = (n + Br - 1) / Br;
  const std::size_t tasks = s.bh * qblocks;
  if (threads == 0) threads = default_thread_count();
  const auto nthreads = static_cast<unsigned>(std::min<std::size_t>(threads, tasks));

  std::vector<std::unique_ptr<FlashScratch>> scratch(nthreads);
  // Causal tasks have very different costs (block i sees i+1 key blocks), so
  // use dynamic scheduling; schedule longest tasks first (LPT heuristic) to
  // shrink the tail.
  parallel_for_dynamic(tasks, nthreads, [&](std::size_t task, unsigned tid) {
    auto& w = scratch[tid];
    if (!w) w = std::make_unique<FlashScratch>(d);
    const std::size_t h = task % s.bh;
    std::size_t qb = task / s.bh;
    if (s.causal) qb = qblocks - 1 - qb;  // heaviest (last) query blocks first
    const std::size_t off = h * n * d;
    flash_block(Q + off, K + off, V + off, O + off, n, d, qb * Br, s.causal, scale, *w);
  });
}

}  // namespace vessel::gpu::cpu
