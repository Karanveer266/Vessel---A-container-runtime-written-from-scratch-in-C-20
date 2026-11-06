// SPDX-License-Identifier: MIT
// vessel-gpu-worker front-end: argument parsing, backend selection and the
// key=value output contract consumed by the Vessel daemon.
//
// Everything except main() lives in the library so the contract is unit
// tested in-process (tests/test_worker.cpp) without exec'ing the binary.
#pragma once

#include <iosfwd>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vessel/gpu/result.hpp"

namespace vessel::gpu {

/// The CUDA backend is injected rather than linked into vessel_gpu_core, so
/// the core library (and its tests) stay CUDA-free and `--backend auto`
/// selection can be tested with a fake device.
struct CudaBackend {
  int (*device_count)() = nullptr;  // 0 on no driver / no device; must not throw
  OpResult (*run)(const WorkerOptions&) = nullptr;
};

/// Keys the daemon relies on, in output order.
inline constexpr std::string_view kRequiredKeys[] = {
    "ok", "error", "backend", "time_ms", "gflops", "bus_bw_gbps", "max_abs_error", "devices"};

struct ParseOutcome {
  WorkerOptions opts;
  bool help = false;
  std::string error;  // non-empty => invalid command line
};

/// Parses argv[1..]. Accepts both `--key value` and `--key=value`.
[[nodiscard]] ParseOutcome parse_args(std::span<const std::string_view> args);
[[nodiscard]] std::string usage();

/// Resolves --backend auto and runs the op. Never throws: every failure is
/// reported as ok=false with a one-line error.
[[nodiscard]] OpResult run_worker(const WorkerOptions& opts, const CudaBackend* cuda);

/// Contract keys first (always present), then OpResult::extra in order.
/// Values are sanitized to a single line; device names have ',' replaced.
[[nodiscard]] std::vector<std::pair<std::string, std::string>> to_key_values(const OpResult& r);
[[nodiscard]] std::string format_key_values(const OpResult& r);

/// Inverse of format_key_values (splits on the first '='; skips blank lines).
/// Provided for consumers and the contract tests.
[[nodiscard]] std::map<std::string, std::string> parse_key_values(std::string_view text);

/// Whole program: parse, run, print. Returns the process exit code
/// (0 iff ok=1; --help returns 0).
int worker_main(std::span<const std::string_view> args, const CudaBackend* cuda, std::ostream& out,
                std::ostream& err);

[[nodiscard]] std::string_view to_string(Op op) noexcept;
[[nodiscard]] std::string_view to_string(Backend b) noexcept;

}  // namespace vessel::gpu
