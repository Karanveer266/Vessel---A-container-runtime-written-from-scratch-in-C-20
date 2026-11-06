// SPDX-License-Identifier: MIT
// CUDA backend for vessel-gpu-worker: single-GPU GEMM / attention benchmarks
// and dispatch to the NCCL multi-GPU ops.
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "cuda/cuda_check.cuh"
#include "cuda/kernels.hpp"
#include "vessel/gpu/common.hpp"
#include "vessel/gpu/cpu_kernels.hpp"
#include "vessel/gpu/cuda/backend.hpp"
#ifdef VESSEL_HAVE_NCCL
#include "cuda/nccl_ops.hpp"
#endif

namespace vessel::gpu::cuda {

namespace {

int device_count_impl() {
  // cudaGetDeviceCount fails with cudaErrorNoDevice / cudaErrorInsufficientDriver
  // on GPU-less hosts: that is "0 devices", not an error, for --backend auto.
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) {
    (void)cudaGetLastError();  // clear the sticky-free error so later calls are clean
    return 0;
  }
  return n;
}

std::string device_name(int dev) {
  cudaDeviceProp p{};
  VESSEL_CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  return p.name;
}

int checked_int(std::size_t v, const char* what) {
  if (v > static_cast<std::size_t>(INT32_MAX)) throw std::invalid_argument(std::string(what) + " exceeds INT32_MAX");
  return static_cast<int>(v);
}

/// Times `iters` launches of `body` on `stream` with one event pair (no host
/// sync inside the loop, so launch overhead is overlapped like in production).
template <class F>
double time_on_stream(const WorkerOptions& o, cudaStream_t stream, F&& body) {
  for (std::size_t i = 0; i < o.warmup; ++i) body();
  Event start, stop;
  VESSEL_CUDA_CHECK(cudaEventRecord(start.get(), stream));
  for (std::size_t i = 0; i < o.iters; ++i) body();
  VESSEL_CUDA_CHECK(cudaEventRecord(stop.get(), stream));
  return Event::elapsed_ms(start, stop) / static_cast<double>(o.iters);
}

OpResult run_gemm(const WorkerOptions& o) {
  const int M = checked_int(o.m, "--m"), N = checked_int(o.n, "--n"), K = checked_int(o.k, "--k");
  VESSEL_CUDA_CHECK(cudaSetDevice(0));
  OpResult r;
  r.devices = {device_name(0)};

  std::vector<float> hA(o.m * o.k), hB(o.k * o.n);
  fill_uniform(hA, o.seed);
  fill_uniform(hB, o.seed + 1);
  Stream stream;
  DeviceBuffer<float> A(hA.size()), B(hB.size()), C(o.m * o.n);
  VESSEL_CUDA_CHECK(cudaMemcpyAsync(A.get(), hA.data(), A.bytes(), cudaMemcpyHostToDevice, stream.get()));
  VESSEL_CUDA_CHECK(cudaMemcpyAsync(B.get(), hB.data(), B.bytes(), cudaMemcpyHostToDevice, stream.get()));

  bool vec = false;
  r.time_ms = time_on_stream(o, stream.get(), [&] {
    vec = sgemm(M, N, K, 1.f, A.get(), K, B.get(), N, 0.f, C.get(), N, stream.get());
  });
  const double flops = 2.0 * static_cast<double>(o.m) * static_cast<double>(o.n) * static_cast<double>(o.k);
  r.gflops = flops / (r.time_ms * 1e6);

  if (o.verify) {
    // Two independent checks: GPU naive kernel (same inputs, different
    // algorithm) and the CPU double-accumulating oracle for small problems.
    DeviceBuffer<float> Cref(o.m * o.n);
    const double naive_ms = time_on_stream(WorkerOptions{.iters = 1, .warmup = 0}, stream.get(), [&] {
      sgemm_naive(M, N, K, 1.f, A.get(), K, B.get(), N, 0.f, Cref.get(), N, stream.get());
    });
    std::vector<float> got(o.m * o.n), ref(o.m * o.n);
    VESSEL_CUDA_CHECK(cudaMemcpy(got.data(), C.get(), C.bytes(), cudaMemcpyDeviceToHost));
    VESSEL_CUDA_CHECK(cudaMemcpy(ref.data(), Cref.get(), Cref.bytes(), cudaMemcpyDeviceToHost));
    r.max_abs_error = max_abs_diff(got, ref);
    r.add("naive_time_ms", naive_ms);
    r.add("speedup_vs_naive", naive_ms / r.time_ms);
    r.add("tolerance", gemm_tolerance(o.k));
    if (!(r.max_abs_error <= gemm_tolerance(o.k))) r.error = "verification failed: max_abs_error above tolerance";
  }
  r.add("kernel", std::string(vec ? "sgemm_128x128x8_vec4" : "sgemm_128x128x8_scalar_edge"));
  r.add("tile", std::string(sgemm_tile_string()));
  return r;
}

OpResult run_flash(const WorkerOptions& o) {
  const int bh = checked_int(o.bh, "--bh"), seq = checked_int(o.seq, "--seq"), dim = checked_int(o.dim, "--dim");
  if (dim != 32 && dim != 64 && dim != 128)
    throw std::invalid_argument("cuda flash: --dim must be 32, 64 or 128 (got " + std::to_string(dim) + ")");
  VESSEL_CUDA_CHECK(cudaSetDevice(0));
  OpResult r;
  r.devices = {device_name(0)};

  const std::size_t elems = o.bh * o.seq * o.dim;
  std::vector<float> hQ(elems), hK(elems), hV(elems);
  fill_uniform(hQ, o.seed);
  fill_uniform(hK, o.seed + 1);
  fill_uniform(hV, o.seed + 2);
  Stream stream;
  DeviceBuffer<float> Q(elems), K(elems), V(elems), O(elems);
  VESSEL_CUDA_CHECK(cudaMemcpyAsync(Q.get(), hQ.data(), Q.bytes(), cudaMemcpyHostToDevice, stream.get()));
  VESSEL_CUDA_CHECK(cudaMemcpyAsync(K.get(), hK.data(), K.bytes(), cudaMemcpyHostToDevice, stream.get()));
  VESSEL_CUDA_CHECK(cudaMemcpyAsync(V.get(), hV.data(), V.bytes(), cudaMemcpyHostToDevice, stream.get()));
  const float scale = 1.f / std::sqrt(static_cast<float>(dim));

  r.time_ms = time_on_stream(o, stream.get(), [&] {
    flash_attention_fwd(Q.get(), K.get(), V.get(), O.get(), bh, seq, dim, scale, o.causal, stream.get());
  });
  const cpu::AttentionShape shape{o.bh, o.seq, o.dim, o.causal};
  r.gflops = cpu::attention_flops(shape) / (r.time_ms * 1e6);

  if (o.verify) {
    const std::size_t scores = o.bh * o.seq * o.seq;
    if (scores * sizeof(float) > (std::size_t{4} << 30))
      throw std::invalid_argument("--verify: naive reference would need > 4 GiB for the score matrix");
    DeviceBuffer<float> S(scores), Oref(elems);
    attention_naive(Q.get(), K.get(), V.get(), Oref.get(), S.get(), bh, seq, dim, scale, o.causal, stream.get());
    std::vector<float> got(elems), ref(elems);
    VESSEL_CUDA_CHECK(cudaMemcpyAsync(got.data(), O.get(), O.bytes(), cudaMemcpyDeviceToHost, stream.get()));
    VESSEL_CUDA_CHECK(cudaMemcpyAsync(ref.data(), Oref.get(), Oref.bytes(), cudaMemcpyDeviceToHost, stream.get()));
    VESSEL_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    r.max_abs_error = max_abs_diff(got, ref);
    r.add("tolerance", kAttentionTolerance);
    if (!(r.max_abs_error <= kAttentionTolerance)) r.error = "verification failed: max_abs_error above tolerance";
  }
  r.add("kernel", std::string("flash_attention_fwd_fp32"));
  r.add("tile", std::string(flash_tile_string(dim)));
  r.add("causal", static_cast<std::size_t>(o.causal ? 1 : 0));
  // HBM traffic of the fused kernel (Q, O once; K, V once per Q block) — vs.
  // the naive path that additionally writes + reads the bh*seq*seq matrix.
  r.add("naive_scores_bytes", o.bh * o.seq * o.seq * sizeof(float));
  return r;
}

OpResult run_impl(const WorkerOptions& o) {
  if (o.iters == 0) throw std::invalid_argument("--iters must be >= 1");
  if (device_count_impl() == 0) throw CudaError("no CUDA device available (driver missing or no GPU)");
  OpResult r;
  switch (o.op) {
    case Op::Gemm: r = run_gemm(o); break;
    case Op::Flash: r = run_flash(o); break;
    case Op::SplitK:
    case Op::AllReduce:
#ifdef VESSEL_HAVE_NCCL
      r = o.op == Op::SplitK ? run_splitk_nccl(o) : run_allreduce_nccl(o);
      break;
#else
      throw CudaError("op '" + std::string(to_string(o.op)) + "' needs NCCL; rebuild with NCCL (NCCL_ROOT=...)");
#endif
  }
  r.backend = "cuda";
  return r;
}

}  // namespace

const CudaBackend& backend() noexcept {
  static const CudaBackend b{&device_count_impl, &run_impl};
  return b;
}

bool have_nccl() noexcept {
#ifdef VESSEL_HAVE_NCCL
  return true;
#else
  return false;
#endif
}

}  // namespace vessel::gpu::cuda
