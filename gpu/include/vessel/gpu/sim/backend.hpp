// SPDX-License-Identifier: MIT
// Simulated (CPU) backend for vessel-gpu-worker.
#pragma once

#include "vessel/gpu/result.hpp"

namespace vessel::gpu::sim {

/// Runs opts.op on CPU threads:
///   gemm      : cpu::gemm_tiled
///   splitk    : R rank threads, each a K-slice GEMM, then ring allReduce(sum)
///   flash     : cpu::flash_attention
///   allreduce : ring allReduce bandwidth benchmark
/// Throws std::exception on invalid options (run_worker converts to ok=0).
[[nodiscard]] OpResult run(const WorkerOptions& opts);

}  // namespace vessel::gpu::sim
