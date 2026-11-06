// SPDX-License-Identifier: MIT
// vessel-gpu-worker entry point: a thin shim over worker_main() so that the
// whole CLI contract is testable in-process.
#include <iostream>
#include <string_view>
#include <vector>

#include "vessel/gpu/worker.hpp"
#ifdef VESSEL_HAVE_CUDA
#include "vessel/gpu/cuda/backend.hpp"
#endif

int main(int argc, char** argv) {
  std::ios::sync_with_stdio(false);
  std::vector<std::string_view> args(argv + 1, argv + argc);
#ifdef VESSEL_HAVE_CUDA
  const vessel::gpu::CudaBackend* cuda = &vessel::gpu::cuda::backend();
#else
  const vessel::gpu::CudaBackend* cuda = nullptr;
#endif
  return vessel::gpu::worker_main(args, cuda, std::cout, std::cerr);
}
