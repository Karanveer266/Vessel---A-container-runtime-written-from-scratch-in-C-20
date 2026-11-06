// SPDX-License-Identifier: MIT
// CUDA backend entry point. Deliberately free of CUDA types so that the
// worker's main.cpp (compiled by the host C++ compiler) can include it.
// Only available when the build defines VESSEL_HAVE_CUDA.
#pragma once

#include "vessel/gpu/worker.hpp"

namespace vessel::gpu::cuda {

/// Function table for run_worker(). device_count() returns 0 when there is
/// no driver or no device (it never throws), so `--backend auto` degrades to
/// the simulator on CUDA-enabled builds running on GPU-less hosts.
[[nodiscard]] const CudaBackend& backend() noexcept;

/// True when the binary was built with NCCL (split-K / all-reduce on CUDA).
[[nodiscard]] bool have_nccl() noexcept;

}  // namespace vessel::gpu::cuda
