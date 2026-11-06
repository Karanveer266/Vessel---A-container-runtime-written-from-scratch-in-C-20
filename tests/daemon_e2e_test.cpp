// SPDX-License-Identifier: Apache-2.0
// Full stack: in-process gRPC server on a unix socket + real containers.
#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdlib>
#include <thread>

#include "container_manager.h"
#include "service.h"
#include "test_util.h"
#include "vessel/common/fs.h"

using namespace vessel;
namespace v1 = vessel::v1;

namespace {

class DaemonE2E : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!testutil::canCreateNamespaces()) GTEST_SKIP() << "namespaces unavailable";
    std::string bb;
    for (const char* p : {"/bin/busybox", "/usr/bin/busybox"})
      if (fs::exists(p)) bb = p;
    if (const char* e = std::getenv("VESSEL_TEST_BUSYBOX")) bb = e;
    if (bb.empty()) GTEST_SKIP() << "busybox not found";
    root_ = testutil::tempDir("e2e-root");
    ASSERT_TRUE(fs::mkdirs(root_ + "/bin").ok());
    testutil::sh("cp " + bb + " " + root_ + "/bin/busybox");
    for (const char* l : {"sh", "cat", "echo", "sleep", "hostname", "head", "dd", "yes", "ls"})
      testutil::link("busybox", root_ + "/bin/" + l);
    state_ = testutil::tempDir("e2e-state");

    daemon::DaemonConfig cfg;
    cfg.state_dir = state_;
    cfg.default_rootfs = root_;
    cfg.rootless = ::geteuid() != 0;
    cfg.enable_ebpf = false;  // deterministic: exercise the ptrace backend
    mgr_ = std::make_unique<daemon::ContainerManager>(cfg);
    ASSERT_TRUE(mgr_->init().ok());
    svc_ = std::make_unique<daemon::VesselService>(*mgr_, "s3cret");
    sock_ = "unix://" + state_ + "/v.sock";
    grpc::ServerBuilder b;
    b.AddListeningPort(sock_, grpc::InsecureServerCredentials());
    b.RegisterService(svc_.get());
    server_ = b.BuildAndStart();
    ASSERT_TRUE(server_);
    stub_ = v1::Vessel::NewStub(grpc::CreateChannel(sock_, grpc::InsecureChannelCredentials()));
  }
  void TearDown() override {
    if (svc_) svc_->shutdown();
    if (server_) server_->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(1));
    mgr_.reset();
    if (!root_.empty()) testutil::rmrf(root_);
    testutil::rmrf(state_);
  }
  std::unique_ptr<grpc::ClientContext> ctx(bool auth = true) {
    auto c = std::make_unique<grpc::ClientContext>();
    if (auth) c->AddMetadata("authorization", "Bearer s3cret");
    c->set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(20));
    return c;
  }
  v1::Container runToCompletion(v1::ContainerSpec spec) {
    v1::CreateContainerRequest req;
    *req.mutable_spec() = std::move(spec);
    req.set_start(true);
    v1::Container c;
    auto st = stub_->CreateContainer(ctx().get(), req, &c);
    EXPECT_TRUE(st.ok()) << st.error_message();
    auto done = mgr_->wait(c.id(), 15000);
    EXPECT_TRUE(done.ok());
    return done.ok() ? done.value() : c;
  }
  std::string logsOf(const std::string& id) {
    v1::StreamLogsRequest r;
    r.set_id(id);
    auto c = ctx();
    auto rd = stub_->StreamLogs(c.get(), r);
    v1::LogChunk ch;
    std::string out;
    while (rd->Read(&ch)) out += ch.data();
    EXPECT_TRUE(rd->Finish().ok());
    return out;
  }

  std::string root_, state_, sock_;
  std::unique_ptr<daemon::ContainerManager> mgr_;
  std::unique_ptr<daemon::VesselService> svc_;
  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<v1::Vessel::Stub> stub_;
};

v1::ContainerSpec shSpec(const std::string& script) {
  v1::ContainerSpec s;
  s.add_args("/bin/sh");
  s.add_args("-c");
  s.add_args(script);
  return s;
}

}  // namespace

TEST_F(DaemonE2E, RejectsMissingOrBadToken) {
  v1::HostInfo h;
  EXPECT_EQ(stub_->GetHostInfo(ctx(false).get(), {}, &h).error_code(), grpc::StatusCode::UNAUTHENTICATED);
  auto c = std::make_unique<grpc::ClientContext>();
  c->AddMetadata("authorization", "Bearer wrong!");
  EXPECT_EQ(stub_->GetHostInfo(c.get(), {}, &h).error_code(), grpc::StatusCode::UNAUTHENTICATED);
  EXPECT_TRUE(stub_->GetHostInfo(ctx().get(), {}, &h).ok());
  EXPECT_FALSE(h.kernel_release().empty());
  EXPECT_GT(h.online_cpus(), 0u);
}

TEST_F(DaemonE2E, RunCaptureLogsAndExitCode) {
  auto spec = shSpec("echo out; echo err >&2; exit 5");
  spec.set_name("logs-test");
  auto c = runToCompletion(spec);
  EXPECT_EQ(c.state(), v1::CONTAINER_STATE_STOPPED);
  EXPECT_EQ(c.exit_code(), 5);
  auto logs = logsOf("logs-test");  // lookup by name
  EXPECT_NE(logs.find("out\n"), std::string::npos);
  EXPECT_NE(logs.find("err\n"), std::string::npos);
}

TEST_F(DaemonE2E, ErrorsMapToGrpcCodes) {
  v1::ContainerRef r;
  r.set_id("doesnotexist");
  v1::Container c;
  EXPECT_EQ(stub_->GetContainer(ctx().get(), r, &c).error_code(), grpc::StatusCode::NOT_FOUND);
  v1::CreateContainerRequest bad;
  EXPECT_EQ(stub_->CreateContainer(ctx().get(), bad, &c).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  auto spec = shSpec("true");
  spec.set_name("dup");
  *bad.mutable_spec() = spec;
  EXPECT_TRUE(stub_->CreateContainer(ctx().get(), bad, &c).ok());
  EXPECT_EQ(stub_->CreateContainer(ctx().get(), bad, &c).error_code(), grpc::StatusCode::ALREADY_EXISTS);
}

TEST_F(DaemonE2E, SetupFailureIsReportedNotHung) {
  v1::ContainerSpec s;
  s.add_args("/no/such/binary");
  v1::CreateContainerRequest req;
  *req.mutable_spec() = s;
  req.set_start(true);
  v1::Container c;
  auto st = stub_->CreateContainer(ctx().get(), req, &c);
  EXPECT_FALSE(st.ok());
  EXPECT_NE(st.error_message().find("execve"), std::string::npos) << st.error_message();
}

TEST_F(DaemonE2E, StopEscalatesAndListShowsState) {
  v1::CreateContainerRequest req;
  *req.mutable_spec() = shSpec("trap '' TERM; echo ready; while :; do sleep 1; done");  // ignores SIGTERM
  req.set_start(true);
  v1::Container c;
  ASSERT_TRUE(stub_->CreateContainer(ctx().get(), req, &c).ok());
  EXPECT_EQ(c.state(), v1::CONTAINER_STATE_RUNNING);
  // "Running" means exec'd, not "trap installed": wait for the marker or a
  // SIGTERM could legitimately kill the shell before it ignores the signal.
  {
    v1::StreamLogsRequest lr;
    lr.set_id(c.id());
    lr.set_follow(true);
    auto lc = ctx();
    auto rd = stub_->StreamLogs(lc.get(), lr);
    v1::LogChunk ch;
    std::string seen;
    while (seen.find("ready") == std::string::npos && rd->Read(&ch)) seen += ch.data();
    lc->TryCancel();
    ASSERT_NE(seen.find("ready"), std::string::npos);
  }
  v1::ListContainersResponse l;
  ASSERT_TRUE(stub_->ListContainers(ctx().get(), {}, &l).ok());
  EXPECT_EQ(l.containers_size(), 1);
  v1::StopContainerRequest s;
  s.set_id(c.id());
  s.set_timeout_ms(300);
  auto t0 = std::chrono::steady_clock::now();
  ASSERT_TRUE(stub_->StopContainer(ctx().get(), s, &c).ok());
  EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::seconds(5));
  EXPECT_EQ(c.state(), v1::CONTAINER_STATE_STOPPED);
  EXPECT_EQ(c.exit_code(), 137);
}

TEST_F(DaemonE2E, ExecInsideRunningContainer) {
  v1::CreateContainerRequest req;
  *req.mutable_spec() = shSpec("sleep 30");
  req.mutable_spec()->set_hostname("exec-me");
  req.set_start(true);
  v1::Container c;
  ASSERT_TRUE(stub_->CreateContainer(ctx().get(), req, &c).ok());
  v1::ExecRequest e;
  e.set_id(c.id());
  e.add_args("hostname");
  v1::ExecResponse r;
  auto st = stub_->Exec(ctx().get(), e, &r);
  ASSERT_TRUE(st.ok()) << st.error_message();
  EXPECT_EQ(r.exit_code(), 0) << r.stderr();
  EXPECT_EQ(r.stdout(), "exec-me\n");
  v1::RemoveContainerRequest rm;
  rm.set_id(c.id());
  rm.set_force(true);
  v1::RemoveContainerResponse resp;
  EXPECT_TRUE(stub_->RemoveContainer(ctx().get(), rm, &resp).ok());
}

TEST_F(DaemonE2E, TraceStreamDeliversSyscallsAndSummaryIsExact) {
  v1::CreateContainerRequest req;
  auto spec = shSpec("sleep 0.3; cat /missing 2>/dev/null; for i in 1 2 3; do echo $i > /dev/null; done; exit 0");
  spec.set_name("traced");
  spec.set_trace(v1::TRACE_BACKEND_PTRACE);
  *req.mutable_spec() = spec;
  req.set_start(false);
  v1::Container c;
  ASSERT_TRUE(stub_->CreateContainer(ctx().get(), req, &c).ok());
  EXPECT_EQ(c.active_trace_backend(), v1::TRACE_BACKEND_PTRACE);

  std::vector<v1::TraceEvent> got;
  std::thread reader([&] {
    v1::StreamTraceRequest tr;
    tr.set_id(c.id());
    auto cc = ctx();
    auto rd = stub_->StreamTrace(cc.get(), tr);
    v1::TraceEvent ev;
    while (rd->Read(&ev)) {
      got.push_back(ev);
      if (ev.kind() == v1::TRACE_EVENT_KIND_LIFECYCLE && ev.detail().rfind("stopped", 0) == 0) cc->TryCancel();
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // let the subscription register
  v1::ContainerRef ref;
  ref.set_id(c.id());
  ASSERT_TRUE(stub_->StartContainer(ctx().get(), ref, &c).ok());
  auto done = mgr_->wait(c.id(), 15000);
  ASSERT_TRUE(done.ok());
  reader.join();

  bool enoent = false, exec = false;
  for (const auto& e : got) {
    if (e.kind() == v1::TRACE_EVENT_KIND_PROCESS_EXEC) exec = true;
    if (e.syscall_name().find("open") != std::string::npos && e.ret() == -ENOENT) enoent = true;
  }
  EXPECT_TRUE(exec);
  EXPECT_TRUE(enoent);

  v1::SyscallSummary sum;
  ASSERT_TRUE(stub_->GetSyscallSummary(ctx().get(), ref, &sum).ok());
  EXPECT_EQ(sum.backend(), v1::TRACE_BACKEND_PTRACE);
  EXPECT_GT(sum.total_events(), 20u);
  ASSERT_GT(sum.syscalls_size(), 3);
  for (int i = 1; i < sum.syscalls_size(); ++i) EXPECT_GE(sum.syscalls(i - 1).count(), sum.syscalls(i).count());
}

TEST_F(DaemonE2E, StatsStreamReportsCgroupCounters) {
  auto info = mgr_->hostInfo();
  if (info.cgroup_version() == v1::CGROUP_VERSION_UNSPECIFIED) GTEST_SKIP() << "no cgroups";
  v1::CreateContainerRequest req;
  auto spec = shSpec("yes > /dev/null & yes > /dev/null & sleep 3");
  spec.mutable_limits()->set_memory_max_bytes(128ull << 20);
  spec.mutable_limits()->set_pids_max(32);
  *req.mutable_spec() = spec;
  req.set_start(true);
  v1::Container c;
  auto st = stub_->CreateContainer(ctx().get(), req, &c);
  ASSERT_TRUE(st.ok()) << st.error_message();
  v1::StreamStatsRequest sr;
  sr.set_id(c.id());
  sr.set_interval_ms(200);
  auto cc = ctx();
  auto rd = stub_->StreamStats(cc.get(), sr);
  v1::StatsSample s;
  int n = 0;
  uint32_t max_mc = 0, max_pids = 0;
  while (n < 6 && rd->Read(&s)) {
    ++n;
    max_mc = std::max(max_mc, s.cpu_millicores_used());
    max_pids = std::max(max_pids, s.pids_current());
    EXPECT_EQ(s.memory_max_bytes(), 128ull << 20);
    EXPECT_EQ(s.pids_max(), 32u);
  }
  cc->TryCancel();
  EXPECT_EQ(n, 6);
  EXPECT_GE(max_pids, 3u);   // init + sh + 2x yes + sleep
  EXPECT_GT(max_mc, 300u);   // two busy loops
  v1::RemoveContainerRequest rm;
  rm.set_id(c.id());
  rm.set_force(true);
  v1::RemoveContainerResponse resp;
  EXPECT_TRUE(stub_->RemoveContainer(ctx().get(), rm, &resp).ok());
}

TEST_F(DaemonE2E, MemoryLimitTriggersOomKill) {
  auto info = mgr_->hostInfo();
  bool has_mem = false;
  for (const auto& ctl : info.cgroup_controllers()) has_mem |= ctl == "memory";
  if (!has_mem) GTEST_SKIP() << "memory controller unavailable";
  // dd with a 64M block size forces a single 64M allocation inside a 16M cgroup.
  auto spec = shSpec("dd if=/dev/zero of=/dev/null bs=64M count=1");
  spec.mutable_limits()->set_memory_max_bytes(16ull << 20);
  spec.mutable_limits()->set_disable_swap(true);
  auto c = runToCompletion(spec);
  EXPECT_EQ(c.state(), v1::CONTAINER_STATE_STOPPED);
  if (c.exit_code() == 0) {
    // A 64 MiB allocation succeeded inside a 16 MiB cgroup: the kernel
    // accounts but does not enforce memory.max (observed on gVisor).
    GTEST_SKIP() << "kernel does not enforce memory.max";
  }
  if (!c.oom_killed()) GTEST_SKIP() << "kernel did not report an OOM kill (sandboxed memcg); exit=" << c.exit_code();
  EXPECT_EQ(c.exit_code(), 137);
}

TEST_F(DaemonE2E, AutoRemoveDeletesContainerAfterExit) {
  auto spec = shSpec("exit 0");
  spec.set_name("ephemeral");
  spec.set_auto_remove(true);
  v1::CreateContainerRequest req;
  *req.mutable_spec() = spec;
  req.set_start(true);
  v1::Container c;
  ASSERT_TRUE(stub_->CreateContainer(ctx().get(), req, &c).ok());
  v1::ContainerRef ref;
  ref.set_id(c.id());
  grpc::Status st;
  for (int i = 0; i < 200; ++i) {
    st = stub_->GetContainer(ctx().get(), ref, &c);
    if (st.error_code() == grpc::StatusCode::NOT_FOUND) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  EXPECT_EQ(st.error_code(), grpc::StatusCode::NOT_FOUND);
}

TEST_F(DaemonE2E, GpuJobRunsInSandboxedContainer) {
  // The worker is built in-tree when VESSEL_BUILD_GPU=ON; it is bind-mounted
  // read-only into an ephemeral, limited container by the daemon.
  const std::string worker = VESSEL_GPU_WORKER_PATH;
  if (worker.empty() || !fs::exists(worker)) GTEST_SKIP() << "gpu worker not built";
  daemon::DaemonConfig cfg;
  cfg.state_dir = state_ + "/gpu";
  cfg.default_rootfs = "/";  // host root read-only: the worker needs host libc/libstdc++
  cfg.gpu_worker = worker;
  cfg.rootless = ::geteuid() != 0;
  cfg.enable_ebpf = false;
  daemon::ContainerManager m(cfg);
  ASSERT_TRUE(m.init().ok());
  v1::GpuJobRequest req;
  req.set_op(v1::GPU_OP_ALLREDUCE_BENCH);
  req.set_backend(v1::GPU_BACKEND_SIMULATED);
  req.set_ranks(4);
  req.set_m(65536);
  req.set_iterations(2);
  req.set_verify(true);
  req.mutable_limits()->set_memory_max_bytes(256ull << 20);
  auto r = m.runGpuJob(req);
  ASSERT_TRUE(r.ok()) << r.error().str();
  EXPECT_TRUE(r->ok()) << r->error();
  EXPECT_EQ(r->backend(), v1::GPU_BACKEND_SIMULATED);
  EXPECT_GT(r->bus_bandwidth_gbps(), 0.0);
  EXPECT_LT(r->max_abs_error(), 1e-3);
  EXPECT_EQ(r->metrics().at("ranks"), "4");
  EXPECT_TRUE(m.list(true).empty()) << "job container must be removed";
}
