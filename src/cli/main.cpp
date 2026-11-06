// SPDX-License-Identifier: Apache-2.0
// vesselctl — command line client for vesseld.
#include <grpcpp/grpcpp.h>
#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "vessel/common/strings.h"
#include "vessel/v1/vessel.grpc.pb.h"

namespace v1 = vessel::v1;
namespace str = vessel::str;

namespace {

const char* kUsage = R"(usage: vesselctl [--addr ADDR] [--token T] <command> [args]

Commands:
  info                                 host capabilities
  run [opts] -- CMD [ARGS...]          create + start (+ wait unless -d)
      --name N   --rootfs DIR   --hostname H   --workdir DIR   -e KEY=VAL
      --cpus 0.5|500m   --memory 64M   --memory-high 48M   --pids 64
      --cpuset 0-1   --no-swap   --net none|loopback|host
      --seccomp default|strict|unconfined   --trace auto|ebpf|ptrace|none
      -v SRC:DST[:ro]   --read-only   -d (detach)   --rm (auto-remove)
  ps [-a]                              list containers
  inspect ID                           full container record
  stop ID [--timeout MS]               SIGTERM, then cgroup-wide SIGKILL
  pause ID | resume ID                 freeze / thaw the cgroup
  rm [-f] ID                           remove
  exec ID -- CMD [ARGS...]             run a command inside a container
  logs [-f] ID                         container stdout/stderr
  stats [ID] [--interval MS]           live cgroup statistics
  trace [ID] [--errors] [--min-lat-us N] [--syscall NAME]...
                                       live syscall / memory / OOM event stream
  top-syscalls ID                      exact per-syscall counters & latency
  gpu --op gemm|splitk|flash|allreduce [--backend cuda|sim|auto]
      [--m N --n N --k N --ranks R --bh N --seq N --dim N --causal --iters N --verify]

ADDR defaults to $VESSEL_ADDR or unix:///run/vessel/vessel.sock
)";

struct Ctx {
  std::unique_ptr<v1::Vessel::Stub> stub;
  std::string token;
  std::unique_ptr<grpc::ClientContext> make(int deadline_s = 0) const {
    auto c = std::make_unique<grpc::ClientContext>();
    if (!token.empty()) c->AddMetadata("authorization", "Bearer " + token);
    if (deadline_s > 0) c->set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(deadline_s));
    return c;
  }
};

[[noreturn]] void die(const std::string& m, int code = 1) {
  std::cerr << "vesselctl: " << m << "\n";
  std::exit(code);
}
void check(const grpc::Status& s) {
  if (!s.ok()) die(s.error_message().empty() ? "rpc failed (" + std::to_string(s.error_code()) + ")" : s.error_message());
}

const char* stateName(v1::ContainerState s) {
  switch (s) {
    case v1::CONTAINER_STATE_CREATED: return "created";
    case v1::CONTAINER_STATE_RUNNING: return "running";
    case v1::CONTAINER_STATE_PAUSED: return "paused";
    case v1::CONTAINER_STATE_STOPPED: return "stopped";
    case v1::CONTAINER_STATE_FAILED: return "failed";
    default: return "?";
  }
}
const char* backendName(v1::TraceBackend b) {
  switch (b) {
    case v1::TRACE_BACKEND_EBPF: return "ebpf";
    case v1::TRACE_BACKEND_PTRACE: return "ptrace";
    case v1::TRACE_BACKEND_NONE: return "none";
    default: return "auto";
  }
}
std::string ago(int64_t ns) {
  if (ns == 0) return "-";
  int64_t s = (static_cast<int64_t>(std::time(nullptr)) * 1'000'000'000LL - ns) / 1'000'000'000LL;
  if (s < 60) return std::to_string(s) + "s";
  if (s < 3600) return std::to_string(s / 60) + "m";
  return std::to_string(s / 3600) + "h";
}
std::string fmtNs(uint64_t ns) {
  char b[32];
  if (ns < 10'000) std::snprintf(b, sizeof b, "%luns", static_cast<unsigned long>(ns));
  else if (ns < 10'000'000) std::snprintf(b, sizeof b, "%.1fus", static_cast<double>(ns) / 1e3);
  else std::snprintf(b, sizeof b, "%.1fms", static_cast<double>(ns) / 1e6);
  return b;
}

struct Args {
  std::vector<std::string> v;
  size_t i = 0;
  bool more() const { return i < v.size(); }
  const std::string& peek() const { return v[i]; }
  std::string next(const char* what) {
    if (!more()) die(std::string("missing ") + what);
    return v[i++];
  }
};

void printContainer(const v1::Container& c) {
  std::printf("%-12s  %-16s  %-8s  pid=%-7d exit=%-4d%s trace=%s\n", c.id().c_str(), c.spec().name().c_str(),
              stateName(c.state()), c.host_pid(), c.exit_code(), c.oom_killed() ? " OOM" : "",
              backendName(c.active_trace_backend()));
  if (!c.error().empty()) std::printf("  error: %s\n", c.error().c_str());
}

int cmdInfo(Ctx& ctx) {
  v1::HostInfo h;
  check(ctx.stub->GetHostInfo(ctx.make(10).get(), {}, &h));
  static const char* cg[] = {"?", "v1", "v2", "hybrid"};
  std::printf("daemon        %s\nkernel        %s (%s)%s\ncgroups       %s [", h.daemon_version().c_str(),
              h.kernel_release().c_str(), h.machine().c_str(), h.is_android() ? " android" : "", cg[h.cgroup_version()]);
  for (int i = 0; i < h.cgroup_controllers_size(); ++i) std::printf("%s%s", i ? " " : "", h.cgroup_controllers(i).c_str());
  std::printf("]\npsi           %s\nebpf          %s\nrootless      %s\ntrace default %s\ncpus          %u\nmemory        %s\n",
              h.psi_available() ? "yes" : "no", h.ebpf_available() ? "yes" : "no", h.rootless() ? "yes" : "no",
              backendName(h.default_trace_backend()), h.online_cpus(), str::humanBytes(h.total_memory_bytes()).c_str());
  for (const auto& g : h.gpus()) std::printf("gpu%-2d         %s\n", g.index(), g.name().c_str());
  return 0;
}

int streamLogs(Ctx& ctx, const std::string& id, bool follow) {
  v1::StreamLogsRequest req;
  req.set_id(id);
  req.set_follow(follow);
  auto c = ctx.make();
  auto rd = ctx.stub->StreamLogs(c.get(), req);
  v1::LogChunk ch;
  while (rd->Read(&ch)) {
    FILE* f = ch.stream() == v1::LogChunk::STREAM_STDERR ? stderr : stdout;
    std::fwrite(ch.data().data(), 1, ch.data().size(), f);
    std::fflush(f);
  }
  check(rd->Finish());
  return 0;
}

int cmdRun(Ctx& ctx, Args& a) {
  v1::CreateContainerRequest req;
  auto* s = req.mutable_spec();
  auto* l = s->mutable_limits();
  bool detach = false, rm = false;
  while (a.more() && a.peek() != "--") {
    std::string o = a.next("option");
    if (o == "--name") s->set_name(a.next("name"));
    else if (o == "--rootfs") s->set_rootfs(a.next("rootfs"));
    else if (o == "--hostname") s->set_hostname(a.next("hostname"));
    else if (o == "--workdir") s->set_working_dir(a.next("dir"));
    else if (o == "-e") s->add_env(a.next("KEY=VAL"));
    else if (o == "--cpus") {
      auto v = str::parseCpu(a.next("cpus"));
      if (!v) die("bad --cpus");
      l->set_cpu_millicores(*v);
    } else if (o == "--memory" || o == "--memory-high") {
      auto v = str::parseBytes(a.next("size"));
      if (!v) die("bad " + o);
      o == "--memory" ? l->set_memory_max_bytes(*v) : l->set_memory_high_bytes(*v);
    } else if (o == "--pids") {
      auto v = str::parseInt<uint32_t>(a.next("pids"));
      if (!v) die("bad --pids");
      l->set_pids_max(*v);
    } else if (o == "--cpuset") l->set_cpuset_cpus(a.next("cpus"));
    else if (o == "--no-swap") l->set_disable_swap(true);
    else if (o == "--net") {
      std::string n = a.next("mode");
      s->set_network(n == "none" ? v1::NETWORK_MODE_NONE : n == "host" ? v1::NETWORK_MODE_HOST : v1::NETWORK_MODE_LOOPBACK);
    } else if (o == "--seccomp") {
      std::string n = a.next("profile");
      s->set_seccomp(n == "strict" ? v1::SECCOMP_PROFILE_STRICT : n == "unconfined" ? v1::SECCOMP_PROFILE_UNCONFINED
                                                                                     : v1::SECCOMP_PROFILE_DEFAULT);
    } else if (o == "--trace") {
      std::string n = a.next("backend");
      s->set_trace(n == "ebpf" ? v1::TRACE_BACKEND_EBPF : n == "ptrace" ? v1::TRACE_BACKEND_PTRACE
                   : n == "none" ? v1::TRACE_BACKEND_NONE : v1::TRACE_BACKEND_AUTO);
    } else if (o == "-v") {
      auto parts = str::split(a.next("SRC:DST"), ':');
      if (parts.size() < 2) die("bad -v (want SRC:DST[:ro])");
      auto* m = s->add_mounts();
      m->set_source(std::string(parts[0]));
      m->set_target(std::string(parts[1]));
      m->set_read_only(parts.size() > 2 && parts[2] == "ro");
    } else if (o == "--read-only") s->set_read_only_rootfs(true);
    else if (o == "-d") detach = true;
    else if (o == "--rm") {
      rm = true;
      s->set_auto_remove(true);  // daemon-side, so it also works with -d
    }
    else die("unknown run option " + o);
  }
  if (a.more()) a.next("--");
  while (a.more()) s->add_args(a.next("arg"));
  if (s->args().empty()) die("run: missing command (use -- CMD ARGS)");
  req.set_start(true);

  v1::Container c;
  check(ctx.stub->CreateContainer(ctx.make(60).get(), req, &c));
  if (detach) {
    std::printf("%s\n", c.id().c_str());
    return 0;
  }
  streamLogs(ctx, c.id(), true);
  v1::ContainerRef ref;
  ref.set_id(c.id());
  for (int i = 0; i < 100; ++i) {  // logs EOF slightly precedes the state flip
    auto st = ctx.stub->GetContainer(ctx.make(10).get(), ref, &c);
    if (st.error_code() == grpc::StatusCode::NOT_FOUND) break;  // auto-removed; c holds last state
    check(st);
    if (c.state() == v1::CONTAINER_STATE_STOPPED || c.state() == v1::CONTAINER_STATE_FAILED) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (c.oom_killed()) std::fprintf(stderr, "vesselctl: container %s was OOM-killed\n", c.id().c_str());
  (void)rm;  // removal is done by the daemon (auto_remove)
  return c.exit_code();
}

int cmdStats(Ctx& ctx, Args& a) {
  v1::StreamStatsRequest req;
  while (a.more()) {
    std::string o = a.next("arg");
    if (o == "--interval") req.set_interval_ms(str::parseInt<uint32_t>(a.next("ms")).value_or(1000));
    else req.set_id(o);
  }
  auto c = ctx.make();
  auto rd = ctx.stub->StreamStats(c.get(), req);
  v1::StatsSample s;
  std::printf("%-12s %8s %22s %10s %9s %9s %8s %s\n", "ID", "CPU(m)", "MEM / LIMIT", "PEAK", "PIDS", "THROTTLE", "PSI-MEM",
              "OOM");
  while (rd->Read(&s)) {
    std::string mem = str::humanBytes(s.memory_current_bytes()) + " / " +
                      (s.memory_max_bytes() ? str::humanBytes(s.memory_max_bytes()) : std::string("-"));
    std::string pids = std::to_string(s.pids_current()) + "/" + (s.pids_max() ? std::to_string(s.pids_max()) : "-");
    std::printf("%-12s %8u %22s %10s %9s %9lu %7.2f%% %lu\n", s.container_id().c_str(), s.cpu_millicores_used(),
                mem.c_str(), str::humanBytes(s.memory_peak_bytes()).c_str(), pids.c_str(),
                static_cast<unsigned long>(s.cpu_nr_throttled()), static_cast<double>(s.memory_pressure().some_avg10()),
                static_cast<unsigned long>(s.memory_events_oom_kill()));
    std::fflush(stdout);
  }
  check(rd->Finish());
  return 0;
}

int cmdTrace(Ctx& ctx, Args& a) {
  v1::StreamTraceRequest req;
  while (a.more()) {
    std::string o = a.next("arg");
    if (o == "--errors") req.set_errors_only(true);
    else if (o == "--min-lat-us") req.set_min_latency_ns(str::parseInt<uint64_t>(a.next("us")).value_or(0) * 1000);
    else if (o == "--syscall") req.add_syscalls(a.next("name"));
    else req.set_id(o);
  }
  auto c = ctx.make();
  auto rd = ctx.stub->StreamTrace(c.get(), req);
  v1::TraceEvent e;
  static const char* kinds[] = {"?", "SYSCALL", "MEMORY", "NET", "EXEC", "EXIT", "OOM_KILL", "MEM_PRESSURE",
                                "SECCOMP_DENIED", "LIFECYCLE"};
  while (rd->Read(&e)) {
    if (e.dropped_events()) std::printf("... %lu events dropped ...\n", static_cast<unsigned long>(e.dropped_events()));
    time_t t = static_cast<time_t>(e.timestamp_unix_ns() / 1'000'000'000);
    tm tmv{};
    localtime_r(&t, &tmv);
    char ts[32];
    std::snprintf(ts, sizeof ts, "%02d:%02d:%02d.%06ld", tmv.tm_hour, tmv.tm_min, tmv.tm_sec,
                  static_cast<long>((e.timestamp_unix_ns() / 1000) % 1'000'000));
    const char* k = kinds[std::min<int>(e.kind(), 9)];
    if (e.kind() == v1::TRACE_EVENT_KIND_SYSCALL || e.kind() == v1::TRACE_EVENT_KIND_MEMORY ||
        e.kind() == v1::TRACE_EVENT_KIND_SECCOMP_DENIED) {
      std::string ret = std::to_string(e.ret());
      if (e.ret() < 0 && e.ret() > -4096) {
        // strerrorname_np() returns NULL for kernel-internal codes such as
        // ERESTARTSYS (512) that a ptrace syscall-exit stop can observe.
        const char* name = strerrorname_np(static_cast<int>(-e.ret()));
        ret = name ? std::string("-") + name : "-" + std::to_string(-e.ret());
      }
      std::string extra;
      if (e.bytes()) extra += " bytes=" + std::to_string(e.bytes());
      if (!e.detail().empty()) extra += "  # " + e.detail();
      std::printf("%s %.8s %-14s %6d %-15s %s(0x%lx, 0x%lx, 0x%lx) = %s  <%s>%s\n", ts, e.container_id().c_str(), k,
                  e.host_tid(), e.comm().c_str(), e.syscall_name().c_str(), static_cast<unsigned long>(e.arg0()),
                  static_cast<unsigned long>(e.arg1()), static_cast<unsigned long>(e.arg2()), ret.c_str(),
                  fmtNs(e.latency_ns()).c_str(), extra.c_str());
    } else {
      std::printf("%s %.8s %-14s %6d %-15s %s\n", ts, e.container_id().c_str(), k, e.host_pid(), e.comm().c_str(),
                  e.detail().c_str());
    }
    std::fflush(stdout);
  }
  check(rd->Finish());
  return 0;
}

int cmdTop(Ctx& ctx, const std::string& id) {
  v1::ContainerRef ref;
  ref.set_id(id);
  v1::SyscallSummary s;
  check(ctx.stub->GetSyscallSummary(ctx.make(10).get(), ref, &s));
  std::printf("backend=%s total=%lu dropped(stream)=%lu\n%-20s %10s %8s %12s %12s\n", backendName(s.backend()),
              static_cast<unsigned long>(s.total_events()), static_cast<unsigned long>(s.dropped_events()), "SYSCALL",
              "COUNT", "ERRORS", "AVG", "MAX");
  for (const auto& c : s.syscalls())
    std::printf("%-20s %10lu %8lu %12s %12s\n", c.name().c_str(), static_cast<unsigned long>(c.count()),
                static_cast<unsigned long>(c.errors()), fmtNs(c.count() ? c.total_latency_ns() / c.count() : 0).c_str(),
                fmtNs(c.max_latency_ns()).c_str());
  return 0;
}

int cmdGpu(Ctx& ctx, Args& a) {
  v1::GpuJobRequest req;
  req.set_op(v1::GPU_OP_GEMM);
  req.set_backend(v1::GPU_BACKEND_AUTO);
  req.set_verify(false);
  while (a.more()) {
    std::string o = a.next("arg");
    auto num = [&] { return str::parseInt<uint32_t>(a.next("number")).value_or(0); };
    if (o == "--op") {
      std::string v = a.next("op");
      req.set_op(v == "splitk" ? v1::GPU_OP_GEMM_SPLITK_ALLREDUCE : v == "flash" ? v1::GPU_OP_FLASH_ATTENTION
                 : v == "allreduce" ? v1::GPU_OP_ALLREDUCE_BENCH : v1::GPU_OP_GEMM);
    } else if (o == "--backend") {
      std::string v = a.next("backend");
      req.set_backend(v == "cuda" ? v1::GPU_BACKEND_CUDA : v == "sim" ? v1::GPU_BACKEND_SIMULATED : v1::GPU_BACKEND_AUTO);
    } else if (o == "--m") req.set_m(num());
    else if (o == "--n") req.set_n(num());
    else if (o == "--k") req.set_k(num());
    else if (o == "--ranks") req.set_ranks(num());
    else if (o == "--bh") req.set_batch_heads(num());
    else if (o == "--seq") req.set_seq_len(num());
    else if (o == "--dim") req.set_head_dim(num());
    else if (o == "--iters") req.set_iterations(num());
    else if (o == "--causal") req.set_causal(true);
    else if (o == "--verify") req.set_verify(true);
    else if (o == "--memory") req.mutable_limits()->set_memory_max_bytes(str::parseBytes(a.next("size")).value_or(0));
    else if (o == "--cpus") req.mutable_limits()->set_cpu_millicores(str::parseCpu(a.next("cpus")).value_or(0));
    else die("unknown gpu option " + o);
  }
  v1::GpuJobResult r;
  check(ctx.stub->RunGpuJob(ctx.make(1800).get(), req, &r));
  std::printf("ok            %s\n", r.ok() ? "yes" : "NO");
  if (!r.error().empty()) std::printf("error         %s\n", r.error().c_str());
  std::printf("backend       %s\ndevices       %s\ntime/iter     %.3f ms\nthroughput    %.2f GFLOP/s\nbus bandwidth %.2f GB/s\nmax |err|     %.3g\ncontainer     %s\n",
              r.backend() == v1::GPU_BACKEND_CUDA ? "cuda" : "sim", r.device_names().c_str(), r.time_ms(), r.gflops(),
              r.bus_bandwidth_gbps(), r.max_abs_error(), r.container_id().c_str());
  std::map<std::string, std::string> extra(r.metrics().begin(), r.metrics().end());
  for (const auto& [k, v] : extra) std::printf("  %-20s %s\n", k.c_str(), v.c_str());
  return r.ok() ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  std::string addr = std::getenv("VESSEL_ADDR") ? std::getenv("VESSEL_ADDR") : "unix:///run/vessel/vessel.sock";
  std::string token = std::getenv("VESSEL_TOKEN") ? std::getenv("VESSEL_TOKEN") : "";
  Args a;
  for (int i = 1; i < argc; ++i) a.v.emplace_back(argv[i]);
  while (a.more() && a.peek().rfind("--", 0) == 0 && a.peek() != "--") {
    std::string o = a.next("option");
    if (o == "--addr") addr = a.next("address");
    else if (o == "--token") token = a.next("token");
    else if (o == "--help") { std::cout << kUsage; return 0; }
    else die("unknown global option " + o + "\n" + kUsage, 2);
  }
  if (!a.more()) {
    std::cerr << kUsage;
    return 2;
  }
  Ctx ctx;
  ctx.token = token;
  ctx.stub = v1::Vessel::NewStub(grpc::CreateChannel(addr, grpc::InsecureChannelCredentials()));
  std::string cmd = a.next("command");

  auto simple = [&](auto rpc, auto req) {
    v1::Container c;
    check((ctx.stub.get()->*rpc)(ctx.make(60).get(), req, &c));
    printContainer(c);
    return 0;
  };
  if (cmd == "info") return cmdInfo(ctx);
  if (cmd == "run") return cmdRun(ctx, a);
  if (cmd == "ps") {
    v1::ListContainersRequest req;
    req.set_all(a.more() && a.peek() == "-a");
    v1::ListContainersResponse resp;
    check(ctx.stub->ListContainers(ctx.make(10).get(), req, &resp));
    std::printf("%-12s  %-16s  %-8s  %-7s  %-6s  %-6s  %s\n", "ID", "NAME", "STATE", "PID", "EXIT", "AGE", "COMMAND");
    for (const auto& c : resp.containers()) {
      std::string cmdline;
      for (const auto& x : c.spec().args()) cmdline += (cmdline.empty() ? "" : " ") + x;
      std::printf("%-12s  %-16s  %-8s  %-7d  %-6d  %-6s  %.40s\n", c.id().c_str(), c.spec().name().c_str(),
                  stateName(c.state()), c.host_pid(), c.exit_code(), ago(c.created_at_unix_ns()).c_str(), cmdline.c_str());
    }
    return 0;
  }
  if (cmd == "inspect") {
    v1::ContainerRef r;
    r.set_id(a.next("id"));
    v1::Container c;
    check(ctx.stub->GetContainer(ctx.make(10).get(), r, &c));
    std::cout << c.DebugString();
    return 0;
  }
  if (cmd == "stop") {
    v1::StopContainerRequest r;
    r.set_id(a.next("id"));
    if (a.more() && a.next("opt") == "--timeout") r.set_timeout_ms(str::parseInt<uint32_t>(a.next("ms")).value_or(10000));
    return simple(&v1::Vessel::Stub::StopContainer, r);
  }
  if (cmd == "pause" || cmd == "resume" || cmd == "start") {
    v1::ContainerRef r;
    r.set_id(a.next("id"));
    return simple(cmd == "pause" ? &v1::Vessel::Stub::PauseContainer
                  : cmd == "resume" ? &v1::Vessel::Stub::ResumeContainer : &v1::Vessel::Stub::StartContainer, r);
  }
  if (cmd == "rm") {
    v1::RemoveContainerRequest r;
    if (a.peek() == "-f") { r.set_force(true); a.next("-f"); }
    r.set_id(a.next("id"));
    v1::RemoveContainerResponse resp;
    check(ctx.stub->RemoveContainer(ctx.make(30).get(), r, &resp));
    return 0;
  }
  if (cmd == "exec") {
    v1::ExecRequest r;
    r.set_id(a.next("id"));
    if (a.more() && a.peek() == "--") a.next("--");
    while (a.more()) r.add_args(a.next("arg"));
    v1::ExecResponse resp;
    check(ctx.stub->Exec(ctx.make(120).get(), r, &resp));
    std::fwrite(resp.stdout().data(), 1, resp.stdout().size(), stdout);
    std::fwrite(resp.stderr().data(), 1, resp.stderr().size(), stderr);
    if (resp.timed_out()) std::fprintf(stderr, "vesselctl: exec timed out\n");
    return resp.exit_code();
  }
  if (cmd == "logs") {
    bool f = false;
    if (a.peek() == "-f") { f = true; a.next("-f"); }
    return streamLogs(ctx, a.next("id"), f);
  }
  if (cmd == "stats") return cmdStats(ctx, a);
  if (cmd == "trace") return cmdTrace(ctx, a);
  if (cmd == "top-syscalls") return cmdTop(ctx, a.next("id"));
  if (cmd == "gpu") return cmdGpu(ctx, a);
  die("unknown command '" + cmd + "'\n" + kUsage, 2);
}
