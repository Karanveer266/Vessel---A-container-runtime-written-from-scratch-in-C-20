# Vessel GPU module (`gpu/`)

This module runs the GPU workloads for Vessel and exposes them through one CLI, `vessel-gpu-worker`, which the daemon executes and whose stdout it parses.

| Backend | What runs | Needs |
|---|---|---|
| `cuda` | Custom FP32 SGEMM, fused FlashAttention-style forward pass, NCCL split-K GEMM and all-reduce benchmark | CUDA toolkit ≥ 12.0, sm_80+ for `cp.async` (older GPUs use a synchronous fallback), NCCL for multi-GPU ops |
| `sim` | The same algorithms on CPU threads: blocked GEMM, tiled online-softmax attention, and a **ring all-reduce over bounded SPSC links** that follows NCCL's ring step sequence | C++20 compiler only |

`--backend auto` uses CUDA when the binary was built with CUDA **and** `cudaGetDeviceCount() > 0`. Otherwise it uses the simulator. A CUDA build therefore still works on a GPU-less host.

## Layout

```
include/vessel/gpu/      public headers (no CUDA types anywhere)
  common.hpp             RNG, Stopwatch, split_range, parallel_for
  result.hpp             WorkerOptions / OpResult, busbw formula, tolerances
  cpu_kernels.hpp        CPU GEMM + attention (oracle and optimized)
  worker.hpp             CLI parsing + key=value contract
  sim/communicator.hpp   simulated NCCL (World, Communicator, Link)
  sim/backend.hpp        sim backend entry
  cuda/backend.hpp       CUDA backend entry (function table, CUDA-free header)
src/common/              common.cpp
src/cpu/                 gemm_cpu.cpp (BLIS-style), attention_cpu.cpp (flash + naive)
src/sim/                 communicator.cpp (ring all-reduce), backend.cpp
src/worker/              worker.cpp (contract), main.cpp (3-line shim)
src/cuda/                device_compat.cuh, sgemm_kernels.cuh, flash_attention_kernels.cuh,
                         sgemm.cu, flash_attention.cu, nccl_ops.cu, backend.cu,
                         cuda_check.cuh (CUDA/NCCL check macros + RAII), kernels.hpp
tests/                   GoogleTest suites + simt_emu.hpp (host SIMT emulator)
```

## Building

```sh
# CPU only (always works)
cmake -S gpu -B build -G Ninja && cmake --build build && ctest --test-dir build

# With CUDA (+ NCCL for splitk/allreduce)
cmake -S gpu -B build -G Ninja -DVESSEL_GPU_CUDA=ON \
      -DNCCL_ROOT=/path/to/nccl -DCMAKE_CUDA_ARCHITECTURES="80;90"
```

| Option | Default | Meaning |
|---|---|---|
| `VESSEL_GPU_CUDA` | `AUTO` | `AUTO`: build the CUDA backend if `check_language(CUDA)` finds nvcc. `ON`: fail if nvcc is missing. `OFF`: never build it. Defines `VESSEL_HAVE_CUDA`. |
| `VESSEL_GPU_NCCL` | `AUTO` | Same semantics for NCCL (`find_path`/`find_library`, `NCCL_ROOT` hint). Defines `VESSEL_HAVE_NCCL`. |
| `VESSEL_GPU_CUDA_ARCHITECTURES` | `80;86;89;90` | GPU targets. A user-set `CMAKE_CUDA_ARCHITECTURES` takes precedence. |
| `VESSEL_GPU_NATIVE` | `ON` | `-march=native` for the CPU kernels. Turn off for relocatable binaries. |
| `VESSEL_GPU_WERROR` | `ON` | `-Werror` (host) / `-Werror=all-warnings` (nvcc). |
| `VESSEL_GPU_TESTS` | `ON` | Build `vessel_gpu_tests` and the CTest smoke tests. |

> **Known toolchain issue:** CUDA 12.8/12.9 headers conflict with glibc ≥ 2.41
> (Debian 13, Fedora 42) on `sinpi/cospi/sinpif/cospif` (missing `noexcept`).
> Use CUDA 13+, an older-glibc distro / the `nvidia/cuda` container (as in CI),
> or add `noexcept(true)` to those four declarations in the toolkit's
> `crt/math_functions.h`. This is a toolkit bug, not a Vessel one.

From the top-level project, `add_subdirectory(gpu)` is enough. All flags are scoped to targets, so nothing leaks into the parent project.

## Worker contract

```
vessel-gpu-worker --op gemm|splitk|flash|allreduce --backend cuda|sim|auto
                  [--m N --n N --k N] [--ranks N] [--count N]
                  [--bh N --seq N --dim N] [--causal] [--iters N] [--warmup N]
                  [--threads N] [--seed N] [--verify] [--help]
```

- **Defaults:** m=n=k=1024, ranks=2, bh=8, seq=512, dim=64, iters=10, warmup=1, count=m·n.
- **Output:** stdout carries `key=value` lines only. The first 8 keys are always, in order: `ok error backend time_ms gflops bus_bw_gbps max_abs_error devices`. Extra metrics follow, e.g. `tile`, `ranks`, `bytes_per_rank`, `bytes_per_rank_theory`, `algbw_gbps`, `compute_ms`, `allreduce_ms`, `tolerance`.
- **Exit code:** 0 if and only if `ok=1`.
- **Errors:** every failure, including bad arguments, CUDA/NCCL errors and failed verification, still prints the full contract with `ok=0`. Values are sanitized to one line, so an error message cannot inject keys.
- **Diagnostics:** go to stderr only.

## Kernel design (summary — details live in the source headers)

**SGEMM** (`sgemm_kernels.cuh`)
- Block tile 128×128×8, 256 threads, 8×8 register micro-tile per thread, split as 2×(4+4) so every shared-memory read is a conflict-free LDS.128.
- The A tile is stored transposed and padded to 132 floats per row, which makes the transposing store conflict-free as well.
- Each thread issues one float4 LDG per operand per K step.
- Shared memory is double-buffered: the next tile is prefetched into registers while the current one is computed, with a single `__syncthreads` per K step.
- Arithmetic intensity: 32 FLOP/B at the shared-memory level and 16 FFMA per LDS at the register level.
- `__launch_bounds__(256,2)` allows 2 blocks per SM. ptxas reports 126–128 registers and **0 spills** on sm_80/sm_90.
- Shapes or strides that are not multiples of 4 go through a scalar-guarded instantiation of the same kernel.
- A naive one-thread-per-output kernel is included as the reference.

**FlashAttention forward** (`flash_attention_kernels.cuh`)
- Follows the FlashAttention-2 work split: one block per (Q block, head), with Q pre-scaled by `scale·log2e` and kept in shared memory.
- K/V tiles are double-buffered with `cp.async.cg` (zero-fill past `seq`).
- Each warp owns 8 query rows. Lanes own keys during S = QKᵀ and own head dims during PV.
- Online softmax uses `exp2` and warp-shuffle row reductions.
- Causal mode skips tiles above the diagonal, masks only boundary tiles, and schedules the heaviest blocks first.
- Tiles: D=32/64 use Br=64, Bc=64 (58/98 KiB shared memory); D=128 uses Br=32, Bc=32 (85 KiB). All fit the 99 KiB opt-in limit on sm_86/sm_89.
- ptxas reports 128/128/140 registers and 0 spills.

**NCCL** (`nccl_ops.cu`)
- One process drives R GPUs via `ncclCommInitAll`.
- Each rank computes a K-slice of the GEMM (slices aligned to 4 elements), then runs an in-place `ncclAllReduce(sum)` inside `ncclGroupStart/End`.
- Every device has its own stream; per-device CUDA events time the compute and all-reduce phases.
- The all-reduce benchmark follows the nccl-tests methodology: busbw = algbw·2(n−1)/n.

**Simulated ring all-reduce** (`sim/communicator.cpp`)
- R ranks run as `std::jthread`s. Neighbours are connected by bounded lock-free SPSC FIFOs with 8 slots and chunk/(slots/2) elements per slot, matching NCCL's NCCL_STEPS and chunkSteps.
- Blocked threads wait with spin-then-futex.
- The step sequence is the same as NCCL's: `send`, then `recvReduceSend`×(R−2), then `recvReduceCopySend`, then `recvCopySend`×(R−2), then `recv`, per loop of R·chunk elements.
- A proof sketch in the source explains why the bounded ring cannot deadlock.
- Per-rank byte counters are checked against the exact formula and against the 2(R−1)/R·N·4 bound.
- If one rank fails, the whole world aborts (the remaining ranks get `CommAborted`) instead of deadlocking.

## Verification status

| What | How | Status |
|---|---|---|
| Sim backend, CPU kernels, ring all-reduce, worker contract | GoogleTest + CTest smoke runs of the binary | **executed**: 271/271 pass (g++ 14 and clang 19); ASan+UBSan and TSan clean |
| CUDA kernel *source* (SGEMM vec4/scalar, naive SGEMM, flash D=32/64/128, causal/non-causal, odd lengths) | `tests/simt_emu.hpp` runs the real `.cuh` kernels with every CUDA thread as an OS thread, `__syncthreads` as `std::barrier`, and emulated shuffles | **executed** against the CPU oracles. Mutation-checked: removing a barrier, the online-softmax rescale, or the double-buffer swap each makes tests fail |
| CUDA/NCCL host code, `cp.async`, real performance | nvcc 12.8, `-Werror=all-warnings`, sm_80/86/89/90, full link against cudart + NCCL | **compiled and linked only**. It cannot run without a GPU. On GPU hardware, run the worker with `--verify` |

## Results

### GPU — *fill in on your hardware* (deliberately left empty; no numbers invented)

| GPU | op | shape | time_ms | GFLOP/s | busbw GB/s | max_abs_error |
|---|---|---|---|---|---|---|
| _fill in_ | gemm | 4096³ | | | – | |
| _fill in_ | flash | bh=32 seq=4096 d=64 causal | | | – | |
| _fill in_ ×N | splitk | 4096×4096×16384 | | | | |
| _fill in_ ×N | allreduce | 256 MiB | | – | | |

Commands: `vessel-gpu-worker --backend cuda --op gemm --m 4096 --n 4096 --k 4096 --verify`, and so on.

### Simulated backend — measured on the dev container (CPU sim, *not* GPU numbers)

The host is a gVisor sandbox: 8 vCPUs are visible but only about 2 cores' worth of throughput is available, and it has an AVX2+FMA CPU. Expect run-to-run noise of ±30%.

| op | args | time_ms | GFLOP/s | busbw GB/s | max_abs_error |
|---|---|---|---|---|---|
| gemm | 1024³, 1 thread | 31.0 | 69 | – | – |
| gemm | 1024³ | 20.8 | 103 | – | 2.4e-5 |
| gemm | 2048³ | 98.3 | 175 | – | – |
| splitk | 1024³, 2 ranks | 14.9 | 144 | 2.29 | 2.3e-5 |
| splitk | 1024³, 4 ranks | 17.0 | 126 | 1.29 | 2.3e-5 |
| flash | bh8 seq512 d64 | 11.9 | 45 | – | 1.0e-7 |
| flash | bh8 seq512 d64 causal | 7.0 | 39 | – | 1.3e-7 |
| allreduce | 4 MiB, 2 ranks | 1.71 | – | 2.45 | 0 |
| allreduce | 4 MiB, 4 ranks | 3.86 | – | 1.63 | 2.4e-7 |
| allreduce | 16 MiB, 8 ranks | 40.1 | – | 0.73 | 0 |

The measured `bytes_per_rank` matches the theoretical 2(R−1)/R·N·4 exactly in every run above: 4 194 304 B for R=2 and 6 291 456 B for R=4 at N=1 M floats.
