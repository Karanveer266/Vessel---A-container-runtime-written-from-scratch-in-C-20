// SPDX-License-Identifier: Apache-2.0
// Micro-benchmarks for the claims made in README/details.md:
//   1. container start latency (clone -> exec -> exit), p50/p99
//   2. seccomp filter cost: instructions executed per syscall (binary search
//      vs. linear chain) using the same cBPF interpreter as the tests
//   3. ptrace tracing overhead on a syscall-heavy workload
//   4. EventBus publish throughput with a slow subscriber
// Usage: bench_runtime <rootfs-with-busybox>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

#include "vessel/common/bounded_queue.h"
#include "vessel/runtime/launcher.h"
#include "vessel/runtime/seccomp.h"
#include "vessel/trace/trace.h"

using namespace vessel;
using clk = std::chrono::steady_clock;

namespace {

double ms(clk::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

void drainAndWait(runtime::Process& p) {
  (void)runtime::awaitExec(p);
  char b[4096];
  while (::read(p.stdout_fd.get(), b, sizeof b) > 0) {
  }
  ::waitpid(p.pid, nullptr, 0);
}

void benchStart(const char* rootfs) {
  runtime::RuntimeSpec s;
  s.id = "bench";
  s.rootfs = rootfs;
  s.args = {"/bin/true"};
  runtime::LaunchOptions o;
  o.state_dir = "/tmp/vessel-bench";
  o.skip_cgroupfs = true;
  std::vector<double> t;
  for (int i = 0; i < 200; ++i) {
    auto t0 = clk::now();
    auto p = runtime::launch(s, o);
    if (!p.ok()) {
      std::printf("launch failed: %s\n", p.error().str().c_str());
      return;
    }
    drainAndWait(p.value());
    t.push_back(ms(clk::now() - t0));
  }
  std::sort(t.begin(), t.end());
  std::printf("container start->exit (/bin/true, 200 runs): p50=%.2f ms  p90=%.2f ms  p99=%.2f ms\n", t[100], t[180],
              t[198]);
}

uint32_t interp(const std::vector<sock_filter>& prog, int nr, size_t* n) {
  seccomp_data d{};
  d.nr = nr;
  d.arch = runtime::seccomp::nativeAuditArch();
  uint32_t A = 0;
  size_t pc = 0;
  *n = 0;
  const auto* bytes = reinterpret_cast<const uint8_t*>(&d);
  while (pc < prog.size()) {
    ++*n;
    const auto& i = prog[pc];
    switch (i.code) {
      case BPF_LD | BPF_W | BPF_ABS: std::memcpy(&A, bytes + i.k, 4); ++pc; break;
      case BPF_JMP | BPF_JEQ | BPF_K: pc += 1 + (A == i.k ? i.jt : i.jf); break;
      case BPF_JMP | BPF_JGE | BPF_K: pc += 1 + (A >= i.k ? i.jt : i.jf); break;
      case BPF_JMP | BPF_JA: pc += 1 + i.k; break;
      case BPF_RET | BPF_K: return i.k;
      default: return 0;
    }
  }
  return 0;
}

void benchSeccomp() {
  for (auto prof : {runtime::SeccompProfile::kDefault, runtime::SeccompProfile::kStrict}) {
    auto p = runtime::seccomp::build(prof);
    size_t worst = 0, total = 0, n = 0;
    for (int nr = 0; nr <= 450; ++nr) {
      interp(p->insns, nr, &n);
      worst = std::max(worst, n);
      total += n;
    }
    // Linear chain: 4 header insns + one JEQ per denied entry until a match.
    size_t linear_worst = 5 + p->denied.size() + 1;
    std::printf("seccomp %-7s: %zu denied, program %zu insns; executed per syscall avg=%.1f worst=%zu (linear chain worst=%zu)\n",
                prof == runtime::SeccompProfile::kDefault ? "default" : "strict", p->denied.size(), p->insns.size(),
                static_cast<double>(total) / 451.0, worst, linear_worst);
  }
}

void benchPtrace() {
  // Workload: 200k getppid() calls. Compare untraced vs traced wall time.
  auto run = [](bool traced) {
    auto agg = std::make_shared<trace::SyscallAggregator>();
    uint64_t events = 0;
    trace::PtraceSession s("b", [&](trace::Event&&) { ++events; }, agg);
    int sync[2];
    if (::pipe(sync) != 0) return 0.0;
    pid_t pid = ::fork();
    if (pid == 0) {
      char c;
      if (::read(sync[0], &c, 1) != 1) ::_exit(1);
      ::execl("/proc/self/exe", "bench", "--child-loop", nullptr);
      ::_exit(127);
    }
    if (traced && !s.seize(pid).ok()) return 0.0;
    auto t0 = clk::now();
    if (::write(sync[1], "g", 1) != 1) return 0.0;
    if (traced) s.run(pid, nullptr);
    else ::waitpid(pid, nullptr, 0);
    double t = ms(clk::now() - t0);
    if (traced) std::printf("  traced: %lu events, %lu exact syscalls counted\n", static_cast<unsigned long>(events),
                            static_cast<unsigned long>(agg->total()));
    ::close(sync[0]);
    ::close(sync[1]);
    return t;
  };
  double a = run(false), b = run(true);
  std::printf("ptrace overhead, 200k getppid(): untraced %.1f ms, traced %.1f ms => %.2f us/syscall added\n", a, b,
              (b - a) * 1000.0 / 200000.0);
}

void benchQueue() {
  BoundedQueue<trace::Event> q(8192);
  trace::Event e;
  e.container_id = "abcdef123456";
  e.comm = "worker";
  const int N = 2'000'000;
  auto t0 = clk::now();
  for (int i = 0; i < N; ++i) {
    e.nr = i;
    q.push(e);
  }
  double t = ms(clk::now() - t0);
  std::printf("BoundedQueue push with stalled consumer: %.1f M events/s (dropped %lu, retained %zu)\n",
              N / t / 1000.0, static_cast<unsigned long>(q.takeDropped()), q.size());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "--child-loop") == 0) {
    for (int i = 0; i < 200000; ++i) ::syscall(SYS_getppid);
    return 0;
  }
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <rootfs-with-/bin/true>\n", argv[0]);
    return 2;
  }
  benchStart(argv[1]);
  benchSeccomp();
  benchPtrace();
  benchQueue();
}
