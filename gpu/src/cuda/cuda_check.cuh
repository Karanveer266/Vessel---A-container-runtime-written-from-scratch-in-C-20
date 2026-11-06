// SPDX-License-Identifier: MIT
// Error-checking macros for the CUDA runtime and NCCL.
//
// Policy: errors become exceptions carrying "<expr> failed at file:line: <msg>".
// run_worker() turns any exception into ok=0 + error=..., so a broken driver,
// OOM or NCCL failure surfaces to the daemon as a structured error instead of
// a crash. Kernel launches are checked with cudaGetLastError() right after
// the launch (configuration errors) — asynchronous faults surface at the
// next synchronizing call, which is also checked.
#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#ifdef VESSEL_HAVE_NCCL
#include <nccl.h>
#endif

namespace vessel::gpu::cuda {

class CudaError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[noreturn]] inline void throw_cuda(cudaError_t e, const char* expr, const char* file, int line) {
  throw CudaError(std::string(expr) + " failed at " + file + ":" + std::to_string(line) + ": " +
                  cudaGetErrorName(e) + " (" + cudaGetErrorString(e) + ")");
}

#ifdef VESSEL_HAVE_NCCL
[[noreturn]] inline void throw_nccl(ncclResult_t r, const char* expr, const char* file, int line) {
  throw CudaError(std::string(expr) + " failed at " + file + ":" + std::to_string(line) + ": NCCL error " +
                  std::to_string(static_cast<int>(r)) + " (" + ncclGetErrorString(r) + ")");
}
#endif

}  // namespace vessel::gpu::cuda

#define VESSEL_CUDA_CHECK(expr)                                                         \
  do {                                                                                  \
    const cudaError_t vessel_err_ = (expr);                                             \
    if (vessel_err_ != cudaSuccess)                                                     \
      ::vessel::gpu::cuda::throw_cuda(vessel_err_, #expr, __FILE__, __LINE__);          \
  } while (0)

// After a <<<>>> launch: catches invalid configurations (too much smem, bad
// grid) immediately instead of at the next sync.
#define VESSEL_CUDA_CHECK_LAUNCH() VESSEL_CUDA_CHECK(cudaGetLastError())

#ifdef VESSEL_HAVE_NCCL
#define VESSEL_NCCL_CHECK(expr)                                                         \
  do {                                                                                  \
    const ncclResult_t vessel_res_ = (expr);                                            \
    if (vessel_res_ != ncclSuccess)                                                     \
      ::vessel::gpu::cuda::throw_nccl(vessel_res_, #expr, __FILE__, __LINE__);          \
  } while (0)
#endif

namespace vessel::gpu::cuda {

/// RAII device buffer (cudaMalloc/cudaFree). Move-only.
template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t n) : n_(n) {
    if (n) VESSEL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p_), n * sizeof(T)));
  }
  DeviceBuffer(DeviceBuffer&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) { reset(); p_ = o.p_; n_ = o.n_; o.p_ = nullptr; o.n_ = 0; }
    return *this;
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { reset(); }

  void reset() noexcept {
    if (p_) cudaFree(p_);  // never throw from a destructor path
    p_ = nullptr;
    n_ = 0;
  }
  [[nodiscard]] T* get() const noexcept { return p_; }
  [[nodiscard]] std::size_t size() const noexcept { return n_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return n_ * sizeof(T); }

 private:
  T* p_ = nullptr;
  std::size_t n_ = 0;
};

/// RAII stream / event wrappers.
class Stream {
 public:
  Stream() { VESSEL_CUDA_CHECK(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking)); }
  ~Stream() { if (s_) cudaStreamDestroy(s_); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  Stream(Stream&& o) noexcept : s_(o.s_) { o.s_ = nullptr; }
  [[nodiscard]] cudaStream_t get() const noexcept { return s_; }

 private:
  cudaStream_t s_ = nullptr;
};

class Event {
 public:
  Event() { VESSEL_CUDA_CHECK(cudaEventCreate(&e_)); }
  ~Event() { if (e_) cudaEventDestroy(e_); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  Event(Event&& o) noexcept : e_(o.e_) { o.e_ = nullptr; }
  [[nodiscard]] cudaEvent_t get() const noexcept { return e_; }

  /// Milliseconds between two recorded events (synchronizes on `end`).
  static float elapsed_ms(const Event& start, const Event& end) {
    VESSEL_CUDA_CHECK(cudaEventSynchronize(end.e_));
    float ms = 0.f;
    VESSEL_CUDA_CHECK(cudaEventElapsedTime(&ms, start.e_, end.e_));
    return ms;
  }

 private:
  cudaEvent_t e_ = nullptr;
};

/// Restores the calling thread's current device on scope exit.
class DeviceGuard {
 public:
  explicit DeviceGuard(int dev) {
    VESSEL_CUDA_CHECK(cudaGetDevice(&prev_));
    VESSEL_CUDA_CHECK(cudaSetDevice(dev));
  }
  ~DeviceGuard() { cudaSetDevice(prev_); }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int prev_ = 0;
};

}  // namespace vessel::gpu::cuda
