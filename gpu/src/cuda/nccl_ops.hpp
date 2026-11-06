// SPDX-License-Identifier: MIT
// Multi-GPU operations built on NCCL (single process, one thread, R devices).
#pragma once

#include "vessel/gpu/result.hpp"

namespace vessel::gpu::cuda {

/// Split-K SGEMM across opts.ranks GPUs + in-place ncclAllReduce(sum).
[[nodiscard]] OpResult run_splitk_nccl(const WorkerOptions& opts);

/// ncclAllReduce bandwidth benchmark (nccl-tests all_reduce_perf methodology).
[[nodiscard]] OpResult run_allreduce_nccl(const WorkerOptions& opts);

}  // namespace vessel::gpu::cuda
