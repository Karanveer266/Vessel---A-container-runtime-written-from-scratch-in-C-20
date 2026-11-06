# Vessel

**A container runtime written from scratch in C++20, with kernel-level observability, a gRPC control plane, an Android operator console, and sandboxed CUDA/NCCL GPU jobs.**

Vessel builds containers directly from Linux primitives: `clone3`, namespaces,
`pivot_root`, cgroups v1/v2, capabilities and seccomp-BPF. It traces what runs
inside them with an eBPF program, falling back to a portable `ptrace` engine
where eBPF is unavailable. A phone can drive the whole thing.

```
 ┌──────────────────────┐       gRPC / HTTP2 (TLS + bearer token)       ┌───────────────────────────────────────┐
 │  Vessel Console      │◄─────────────────────────────────────────────►│ vesseld                               │
 │  Android · Compose   │   unary: lifecycle, exec, GPU jobs            │  ContainerManager ─ supervisor thread │
 │  live charts, trace  │   server-streams: stats, trace, logs          │  EventBus (drop-oldest, per sub)      │
 └──────────────────────┘                                               │  LogBuffer (bounded ring per ctr)     │
 ┌──────────────────────┐                                               └──┬─────────────┬───────────────┬──────┘
 │  vesselctl (CLI)     │◄──────────── unix socket ───────────────────────►│             │               │
 └──────────────────────┘                                    libvessel_runtime   libvessel_trace   vessel-gpu-worker
                                                             clone3 · userns     eBPF (libbpf)     CUDA SGEMM
                                                             pivot_root · seccomp ptrace fallback  FlashAttention
                                                             caps · mini-init                      NCCL AllReduce
                                                             libvessel_cgroup                      (or CPU ring sim)
                                                             v2 + v1 drivers, PSI
```

## Highlights

| Area | What's there |
|---|---|
| **Isolation** | user, pid, mount, uts, ipc, net and cgroup namespaces; `pivot_root(".", ".")`; private `/proc`, minimal `/dev`, read-only `/sys`; masked procfs paths; read-only rootfs option; bind mounts; GPU device passthrough |
| **Hardening** | Docker-default capability set (all 5 sets + bounding), `PR_SET_NO_NEW_PRIVS`, seccomp-BPF with arch pinning and x32 rejection, fd leak defence via `close_range(CLOSE_RANGE_CLOEXEC)` |
| **Seccomp compiler** | Own cBPF code generator that emits a **balanced binary search**: at most 12 instructions per syscall, against 68 for a linear chain. Verified by running every syscall number, plus 200 random deny lists, through a cBPF interpreter |
| **Resource control** | One `Cgroup` interface with a **cgroup v2** driver (cpu.max, memory.max/high, swap.max, pids.max, cpuset, freeze, cgroup.kill, PSI) and a **cgroup v1** driver (most Android kernels). `clone3(CLONE_INTO_CGROUP)` means the child starts life already limited |
| **Process model** | Built-in mini-init as PID 1 that forwards signals and reaps zombies. Staged error reporting over an `O_CLOEXEC` pipe, so a failure reads e.g. *"failed at stage 'pivot_root': EINVAL"* |
| **eBPF tracing** | Raw-syscall tracepoints with exact per-syscall counters kept in-kernel (`PERCPU_HASH`); filtered events go to a ring buffer; `cgroup_skb` network counters; OOM, exec and exit events; muted when nobody is subscribed |
| **ptrace tracing** | `PTRACE_SEIZE` engine with automatic fork/clone following, group-stop vs signal-stop handling, `PTRACE_GET_SYSCALL_INFO` with a fallback to reading registers. Only traces after `execve`, so runtime setup noise is never reported. Flags syscalls that seccomp denied |
| **Control plane** | gRPC API (15 RPCs, 3 of them server-streaming); bounded drop-oldest queues so a slow phone can never stall the tracer; constant-time bearer auth; TLS; keepalive tuned for mobile clients |
| **GPU** | Hand-tuned CUDA SGEMM (128×128 tiles, 8×8 register micro-tiles, double buffering), fused FlashAttention forward pass with online softmax, NCCL split-K GEMM + `ncclAllReduce`. A CPU backend runs the **same ring all-reduce algorithm** on threads wired together by bounded SPSC "links" |
| **Android** | Kotlin/Compose console: live Canvas charts, a 2,000-event trace view that never grows past its cap, reconnect with backoff, lifecycle-aware streams. The daemon itself can run on a rooted device (init.rc + SELinux sketch included) |

## Quick start (Linux)

```bash
# Debian/Ubuntu dependencies
sudo apt install g++ cmake ninja-build pkg-config libgrpc++-dev protobuf-compiler-grpc \
                 libprotobuf-dev protobuf-compiler libssl-dev libgtest-dev \
                 clang libbpf-dev bpftool busybox-static        # last line optional (eBPF + demo rootfs)

cmake -S . -B build -G Ninja && cmake --build build
ctest --test-dir build --output-on-failure        # integration tests need root for cgroups

scripts/make-rootfs.sh busybox /tmp/rootfs
sudo ./build/vesseld --rootfs /tmp/rootfs &
sudo ./build/vesselctl info
```

### Examples

```console
$ vesselctl run --name hello --hostname box --memory 64M --pids 32 --cpus 0.5 -- \
      sh -c 'echo "I am $(id -u) on $(hostname) pid=$$"; ls /'
I am 0 on box pid=2
bin
dev
proc
sys

$ vesselctl run -d --name worker --memory 128M --trace ptrace -- sh -c 'while :; do cat /nope; sleep .2; done'
$ vesselctl trace worker --errors
08:50:30.722133 90d19099 SYSCALL   15689 cat   openat(0xffffff9c, 0x7ff7ff6ddf6e, 0x0) = -ENOENT  <49.8us>
08:50:30.716643 90d19099 SYSCALL   14692 sh    wait4(0xffffffffffffffff, 0x7f2e7f9692fc, 0x1) = -ECHILD  <44.2us>

$ vesselctl stats worker
ID             CPU(m)            MEM / LIMIT       PEAK      PIDS  THROTTLE  PSI-MEM OOM
90d1909941ce       60    2.6 MiB / 128.0 MiB        0 B      3/64         0    0.00% 0

$ vesselctl top-syscalls worker
backend=ptrace total=21690 dropped(stream)=0
SYSCALL                   COUNT   ERRORS          AVG          MAX
wait4                      3040     1840       21.6ms     1004.3ms
brk                        3010        0       32.0us      198.4us

$ vesselctl exec worker -- hostname
worker
$ vesselctl pause worker && vesselctl resume worker && vesselctl stop worker

$ vesselctl gpu --op splitk --backend auto --m 2048 --n 2048 --k 4096 --ranks 4 --verify
```

The output above is real, captured from the development machine.

### Serving the Android console

```bash
sudo vesseld --listen 0.0.0.0:50051 --token "$(openssl rand -hex 24)" \
             --tls-cert server.pem --tls-key server.key
# or over USB:  vesseld --listen 127.0.0.1:50051  &&  adb reverse tcp:50051 tcp:50051
```

See [`android/README.md`](android/README.md) to build the app and for every
connection scenario: USB, emulator, LAN, or the daemon on the same rooted
device.

## Repository layout

```
proto/vessel/v1/vessel.proto   the one API contract (daemon, CLI, Android)
include/vessel/                public headers: common, cgroup, runtime, trace
src/common/                    Result<T>, UniqueFd, fs helpers, logger, ids
src/cgroup/                    mountinfo-driven detection, v1 + v2 drivers, parsers
src/runtime/                   launcher (parent), child.cpp (async-signal-safe), exec (setns),
                               seccomp compiler, capabilities, syscall tables (generated)
src/trace/                     ptrace engine, eBPF loader/poller, aggregator
bpf/                           vessel_trace.bpf.c (GPL-2.0) + shared event ABI header
src/daemon/                    ContainerManager, EventBus, LogBuffer, gRPC service, main
src/cli/                       vesselctl
gpu/                           CUDA kernels, NCCL, simulated ring backend, worker, tests
android/                       Vessel Console (Kotlin, Compose, grpc-kotlin)
tests/                         unit + integration + gRPC end-to-end tests (GoogleTest)
bench/                         micro-benchmarks behind the numbers below
platform/                      systemd unit (Delegate=yes), Android init.rc, SELinux sketch
scripts/                       rootfs builder, Android deploy, syscall table generator
.github/workflows/ci.yml       gcc/clang/ASan matrix, CUDA build, NDK cross-build, Android app
```

## Build options

| Option | Default | Meaning |
|---|---|---|
| `VESSEL_BUILD_DAEMON` | `ON` | vesseld + vesselctl (needs gRPC/protobuf) |
| `VESSEL_EBPF` | `AUTO` | `AUTO` builds the eBPF tracer if clang, bpftool and libbpf are found |
| `VESSEL_BUILD_GPU` | `ON` | GPU worker; the CUDA path builds only when nvcc is found (`VESSEL_GPU_CUDA`) |
| `VESSEL_SANITIZE` | `OFF` | ASan + UBSan |
| `VESSEL_WERROR` | `OFF` | `-Werror` (on in CI) |

The runtime libraries (`vessel_common`, `vessel_cgroup`, `vessel_runtime`,
`vessel_trace`) have **no gRPC dependency**. You can embed them on their own,
or cross-compile them with just the NDK.

## Measured numbers

All numbers come from `bench/bench_runtime.cpp` on the development sandbox
(8 vCPU, gVisor kernel 4.19, cgroup v1). gVisor intercepts every syscall in
user space, so on a normal kernel expect startup and ptrace numbers several
times lower.

| Metric | Result |
|---|---|
| Container start → exit (`/bin/true`, full namespace set, pivot_root, seccomp) | p50 **9.5–13.6 ms** (across runs), p99 ≈ 19–21 ms |
| Seccomp: instructions executed per syscall (strict profile, 62 rules) | worst case **12**, vs 68 for a linear chain |
| ptrace backend overhead | ~60 µs per traced syscall under gVisor (it is two context switches per stop) |
| EventBus/BoundedQueue publish with a stalled consumer | **18 M events/s**; the producer never blocks |
| CUDA SGEMM 128×128×8 (ptxas, sm_80) | 126 registers, **0 spills**, 16,640 B shared memory |
| Simulated split-K GEMM 1024³, 2 ranks (CPU) | 77 GFLOP/s; bytes per rank = 2(R−1)/R·N·4 **exactly** |

No GPU timings are claimed: the development machine has no GPU. The CUDA
code compiles and links (nvcc 12.8, sm_80/86/89/90, NCCL), and its kernel
source runs on a host SIMT emulator in the tests. `gpu/README.md` has a table
to fill in on real hardware.

## Tests

```
342 CTest tests from one top-level build, all passing:
   71  runtime / cgroup / seccomp / ptrace / daemon (incl. gRPC end-to-end and a GPU job in a container)
  271  GPU module (ring all-reduce, CPU kernels, CUDA kernel source on a SIMT host emulator, worker contract)
128 JVM tests in the Android app (incl. an in-process gRPC server)
gcc 14 and clang 19 warning-free; ASan+UBSan clean; GPU module also TSan-clean
```

The integration tests start real containers. Each one first checks for the
kernel feature it needs and **skips with the reason** if that feature is
missing, e.g. *"kernel accounts but does not enforce pids.max"*. That way a
missing feature never shows up as a silent pass.

## Platform support

| | x86_64 Linux ≥ 5.8 | arm64 Linux / Android | Older kernels / sandboxes |
|---|---|---|---|
| Namespaces + pivot_root | ✓ | ✓ | ✓ (needs userns) |
| cgroup v2 / v1 | ✓ / ✓ | ✓ / ✓ (common on Android) | v1 |
| `clone3(CLONE_INTO_CGROUP)` | ✓ (5.7+) | ✓ | falls back to `clone` + `cgroup.procs` |
| eBPF tracer | ✓ (5.8+, ring buffer) | ✓ | falls back to ptrace |
| PSI events | ✓ (4.20+, `CONFIG_PSI`) | often off on Android | polled counters |

## Security note

`vesseld` runs as root and starts processes on request, so treat its socket
like the Docker socket. Over the network, always set `--token` and use TLS.
Running as real root, the default user-namespace mapping is the identity
mapping (Docker's default model). In that mode, capabilities, seccomp and
namespaces form the security boundary. See `details.md` for the threat model
and the trade-offs.

## License

Apache-2.0. The eBPF program in `bpf/` is GPL-2.0-only, as the kernel requires
for the helpers it uses (see `bpf/LICENSE.md`).
