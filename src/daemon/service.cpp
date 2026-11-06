// SPDX-License-Identifier: Apache-2.0
#include "service.h"

#include <openssl/crypto.h>

#include <chrono>
#include <thread>

#include "vessel/common/log.h"
#include "vessel/runtime/syscalls.h"

namespace vessel::daemon {

// trace::Kind / trace::Backend are cast directly to their proto enums on the
// hot streaming path; pin the numeric equivalence at compile time.
static_assert(static_cast<int>(trace::Kind::kSyscall) == v1::TRACE_EVENT_KIND_SYSCALL);
static_assert(static_cast<int>(trace::Kind::kMemory) == v1::TRACE_EVENT_KIND_MEMORY);
static_assert(static_cast<int>(trace::Kind::kNet) == v1::TRACE_EVENT_KIND_NET);
static_assert(static_cast<int>(trace::Kind::kExec) == v1::TRACE_EVENT_KIND_PROCESS_EXEC);
static_assert(static_cast<int>(trace::Kind::kExit) == v1::TRACE_EVENT_KIND_PROCESS_EXIT);
static_assert(static_cast<int>(trace::Kind::kOomKill) == v1::TRACE_EVENT_KIND_OOM_KILL);
static_assert(static_cast<int>(trace::Kind::kMemoryPressure) == v1::TRACE_EVENT_KIND_MEMORY_PRESSURE);
static_assert(static_cast<int>(trace::Kind::kSeccompDenied) == v1::TRACE_EVENT_KIND_SECCOMP_DENIED);
static_assert(static_cast<int>(trace::Kind::kLifecycle) == v1::TRACE_EVENT_KIND_LIFECYCLE);
static_assert(static_cast<int>(trace::Backend::kNone) == v1::TRACE_BACKEND_NONE);
static_assert(static_cast<int>(trace::Backend::kEbpf) == v1::TRACE_BACKEND_EBPF);
static_assert(static_cast<int>(trace::Backend::kPtrace) == v1::TRACE_BACKEND_PTRACE);

grpc::Status toGrpc(const Error& e) {
  using grpc::StatusCode;
  StatusCode c = StatusCode::INTERNAL;
  switch (e.code) {
    case ENOENT: c = StatusCode::NOT_FOUND; break;
    case EEXIST: c = StatusCode::ALREADY_EXISTS; break;
    case EINVAL: c = StatusCode::INVALID_ARGUMENT; break;
    case EPERM:
    case EACCES: c = StatusCode::PERMISSION_DENIED; break;
    case EBUSY:
    case EALREADY: c = StatusCode::FAILED_PRECONDITION; break;
    case ENOTSUP:
    case ENOSYS: c = StatusCode::UNIMPLEMENTED; break;
    case ETIMEDOUT: c = StatusCode::DEADLINE_EXCEEDED; break;
    case ENOSPC:
    case ENOMEM: c = StatusCode::RESOURCE_EXHAUSTED; break;
    default: break;
  }
  return grpc::Status(c, e.str());
}

#define VESSEL_AUTH(ctx)                          \
  do {                                            \
    if (auto st_ = authorize(ctx); !st_.ok()) return st_; \
  } while (0)

grpc::Status VesselService::authorize(grpc::ServerContext* ctx) const {
  if (token_.empty()) return grpc::Status::OK;  // auth disabled (unix socket / localhost dev)
  const auto& md = ctx->client_metadata();
  auto it = md.find("authorization");
  if (it == md.end()) return {grpc::StatusCode::UNAUTHENTICATED, "missing authorization metadata"};
  std::string_view v(it->second.data(), it->second.size());
  constexpr std::string_view kBearer = "Bearer ";
  if (v.substr(0, kBearer.size()) != kBearer) return {grpc::StatusCode::UNAUTHENTICATED, "expected Bearer token"};
  v.remove_prefix(kBearer.size());
  // Constant-time compare: the token is the only secret protecting a root
  // daemon reachable over the network; don't leak it through timing.
  if (v.size() != token_.size() || CRYPTO_memcmp(v.data(), token_.data(), v.size()) != 0)
    return {grpc::StatusCode::UNAUTHENTICATED, "invalid token"};
  return grpc::Status::OK;
}

grpc::Status VesselService::GetHostInfo(grpc::ServerContext* ctx, const v1::GetHostInfoRequest*, v1::HostInfo* out) {
  VESSEL_AUTH(ctx);
  *out = mgr_.hostInfo();
  return grpc::Status::OK;
}

grpc::Status VesselService::CreateContainer(grpc::ServerContext* ctx, const v1::CreateContainerRequest* req,
                                            v1::Container* out) {
  VESSEL_AUTH(ctx);
  auto c = mgr_.create(req->spec());
  if (!c.ok()) return toGrpc(c.error());
  if (req->start()) {
    auto s = mgr_.start(c->id());
    if (!s.ok()) {
      auto g = mgr_.get(c->id());
      if (g.ok()) *out = g.value();
      return toGrpc(s.error());
    }
    *out = s.value();
  } else {
    *out = c.value();
  }
  return grpc::Status::OK;
}

#define VESSEL_UNARY(expr)                     \
  do {                                         \
    auto r_ = (expr);                          \
    if (!r_.ok()) return toGrpc(r_.error());   \
    *out = std::move(r_).value();              \
    return grpc::Status::OK;                   \
  } while (0)

grpc::Status VesselService::StartContainer(grpc::ServerContext* ctx, const v1::ContainerRef* req, v1::Container* out) {
  VESSEL_AUTH(ctx);
  VESSEL_UNARY(mgr_.start(req->id()));
}
grpc::Status VesselService::StopContainer(grpc::ServerContext* ctx, const v1::StopContainerRequest* req,
                                          v1::Container* out) {
  VESSEL_AUTH(ctx);
  VESSEL_UNARY(mgr_.stop(req->id(), req->timeout_ms()));
}
grpc::Status VesselService::PauseContainer(grpc::ServerContext* ctx, const v1::ContainerRef* req, v1::Container* out) {
  VESSEL_AUTH(ctx);
  VESSEL_UNARY(mgr_.pause(req->id(), true));
}
grpc::Status VesselService::ResumeContainer(grpc::ServerContext* ctx, const v1::ContainerRef* req, v1::Container* out) {
  VESSEL_AUTH(ctx);
  VESSEL_UNARY(mgr_.pause(req->id(), false));
}
grpc::Status VesselService::GetContainer(grpc::ServerContext* ctx, const v1::ContainerRef* req, v1::Container* out) {
  VESSEL_AUTH(ctx);
  VESSEL_UNARY(mgr_.get(req->id()));
}
grpc::Status VesselService::GetSyscallSummary(grpc::ServerContext* ctx, const v1::ContainerRef* req,
                                              v1::SyscallSummary* out) {
  VESSEL_AUTH(ctx);
  VESSEL_UNARY(mgr_.syscallSummary(req->id()));
}
grpc::Status VesselService::RunGpuJob(grpc::ServerContext* ctx, const v1::GpuJobRequest* req, v1::GpuJobResult* out) {
  VESSEL_AUTH(ctx);
  VESSEL_UNARY(mgr_.runGpuJob(*req));
}

grpc::Status VesselService::RemoveContainer(grpc::ServerContext* ctx, const v1::RemoveContainerRequest* req,
                                            v1::RemoveContainerResponse*) {
  VESSEL_AUTH(ctx);
  auto s = mgr_.remove(req->id(), req->force());
  return s.ok() ? grpc::Status::OK : toGrpc(s.error());
}

grpc::Status VesselService::ListContainers(grpc::ServerContext* ctx, const v1::ListContainersRequest* req,
                                           v1::ListContainersResponse* out) {
  VESSEL_AUTH(ctx);
  for (auto& c : mgr_.list(req->all())) *out->add_containers() = std::move(c);
  return grpc::Status::OK;
}

grpc::Status VesselService::Exec(grpc::ServerContext* ctx, const v1::ExecRequest* req, v1::ExecResponse* out) {
  VESSEL_AUTH(ctx);
  std::vector<std::string> args(req->args().begin(), req->args().end());
  std::vector<std::string> env(req->env().begin(), req->env().end());
  auto r = mgr_.exec(req->id(), args, env, static_cast<int>(req->timeout_ms()));
  if (!r.ok()) return toGrpc(r.error());
  out->set_exit_code(r->exit_code);
  out->set_stdout(r->out);
  out->set_stderr(r->err);
  out->set_timed_out(r->timed_out);
  return grpc::Status::OK;
}

grpc::Status VesselService::StreamStats(grpc::ServerContext* ctx, const v1::StreamStatsRequest* req,
                                        grpc::ServerWriter<v1::StatsSample>* w) {
  VESSEL_AUTH(ctx);
  const auto interval = std::chrono::milliseconds(std::clamp<uint32_t>(req->interval_ms() ? req->interval_ms() : 1000, 100, 60000));
  if (!req->id().empty() && mgr_.resolveId(req->id()).empty())
    return {grpc::StatusCode::NOT_FOUND, "no such container: " + req->id()};
  auto next = std::chrono::steady_clock::now();
  while (!ctx->IsCancelled() && !shutting_down_.load()) {
    std::vector<std::string> ids;
    if (req->id().empty()) {
      for (const auto& c : mgr_.list(false)) ids.push_back(c.id());
    } else {
      ids.push_back(req->id());
    }
    for (const auto& id : ids) {
      auto s = mgr_.sample(id);
      if (!s.ok()) {
        if (!req->id().empty()) return toGrpc(s.error());  // container removed
        continue;
      }
      // Write() blocks under flow control when the client is slow; we simply
      // sample less often (no queue builds up — samples are state, not events).
      if (!w->Write(s.value())) return grpc::Status::OK;
    }
    next += interval;
    auto now = std::chrono::steady_clock::now();
    if (next < now) next = now;  // don't burst to catch up after a stall
    std::this_thread::sleep_until(next);
  }
  return grpc::Status::OK;
}

grpc::Status VesselService::StreamTrace(grpc::ServerContext* ctx, const v1::StreamTraceRequest* req,
                                        grpc::ServerWriter<v1::TraceEvent>* w) {
  VESSEL_AUTH(ctx);
  EventFilter f;
  if (!req->id().empty()) {
    f.container_id = mgr_.resolveId(req->id());
    if (f.container_id.empty()) return {grpc::StatusCode::NOT_FOUND, "no such container: " + req->id()};
  }
  for (int k : req->kinds()) f.kinds.insert(static_cast<trace::Kind>(k));
  for (const auto& n : req->syscalls()) {
    auto nr = syscalls::number(n);
    if (!nr) return {grpc::StatusCode::INVALID_ARGUMENT, "unknown syscall '" + n + "'"};
    f.syscalls.insert(*nr);
  }
  f.min_latency_ns = req->min_latency_ns();
  f.errors_only = req->errors_only();

  auto sub = mgr_.bus().subscribe(std::move(f));
  struct Unsub {
    EventBus& b;
    uint64_t id;
    ~Unsub() { b.unsubscribe(id); }
  } guard{mgr_.bus(), sub.id};

  v1::TraceEvent ev;
  while (!ctx->IsCancelled() && !shutting_down_.load()) {
    auto e = sub.queue->popFor(std::chrono::milliseconds(250));
    if (!e) continue;
    ev.Clear();
    ev.set_container_id(e->container_id);
    ev.set_kind(static_cast<v1::TraceEventKind>(e->kind));
    ev.set_timestamp_unix_ns(e->ts_unix_ns);
    ev.set_host_pid(e->pid);
    ev.set_host_tid(e->tid);
    ev.set_comm(e->comm);
    ev.set_syscall_nr(e->nr);
    if (e->nr >= 0) ev.set_syscall_name(std::string(syscalls::name(e->nr)));
    ev.set_ret(e->ret);
    ev.set_latency_ns(e->latency_ns);
    ev.set_arg0(e->args[0]);
    ev.set_arg1(e->args[1]);
    ev.set_arg2(e->args[2]);
    ev.set_bytes(e->bytes);
    ev.set_egress(e->egress);
    ev.set_detail(e->detail);
    ev.set_backend(static_cast<v1::TraceBackend>(e->backend));
    ev.set_dropped_events(e->dropped_before + sub.queue->takeDropped());
    if (!w->Write(ev)) break;
  }
  return grpc::Status::OK;
}

grpc::Status VesselService::StreamLogs(grpc::ServerContext* ctx, const v1::StreamLogsRequest* req,
                                       grpc::ServerWriter<v1::LogChunk>* w) {
  VESSEL_AUTH(ctx);
  auto lb = mgr_.logs(req->id());
  if (!lb.ok()) return toGrpc(lb.error());
  auto& buf = *lb.value();
  uint64_t cursor = buf.tailCursor(req->tail_bytes());
  std::vector<LogBuffer::Chunk> chunks;
  v1::LogChunk out;
  while (!ctx->IsCancelled() && !shutting_down_.load()) {
    chunks.clear();
    bool more = buf.read(&cursor, &chunks, req->follow() ? 250 : 0);
    for (auto& c : chunks) {
      out.set_stream(c.stream == LogBuffer::kStdout ? v1::LogChunk::STREAM_STDOUT : v1::LogChunk::STREAM_STDERR);
      out.set_data(std::move(c.data));
      out.set_timestamp_unix_ns(c.ts_ns);
      if (!w->Write(out)) return grpc::Status::OK;
    }
    if (!more || (!req->follow() && chunks.empty())) break;
  }
  return grpc::Status::OK;
}

}  // namespace vessel::daemon
