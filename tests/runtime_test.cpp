// SPDX-License-Identifier: Apache-2.0
// End-to-end tests of the launcher: real namespaces, real pivot_root, real
// seccomp. The rootfs is a single static busybox (tests/fixtures or
// $VESSEL_TEST_BUSYBOX), so tests are hermetic and fast (~5ms per container).
#include <gtest/gtest.h>
#include <poll.h>
#include <sys/wait.h>

#include <cstdlib>
#include <thread>

#include "test_util.h"
#include "vessel/common/fs.h"
#include "vessel/common/strings.h"
#include "vessel/runtime/launcher.h"

using namespace vessel;
using namespace vessel::runtime;

namespace {

std::string findBusybox() {
  if (const char* e = std::getenv("VESSEL_TEST_BUSYBOX")) return e;
  for (const char* p : {"/bin/busybox", "/usr/bin/busybox", "/usr/sbin/busybox"})
    if (fs::exists(p)) return p;
  return {};
}

class RuntimeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!testutil::canCreateNamespaces()) GTEST_SKIP() << "user/pid/mount namespaces unavailable";
    busybox_ = findBusybox();
    if (busybox_.empty()) GTEST_SKIP() << "static busybox not found (set VESSEL_TEST_BUSYBOX)";
    root_ = testutil::tempDir("rootfs");
    ASSERT_TRUE(fs::mkdirs(root_ + "/bin").ok());
    testutil::sh("cp " + busybox_ + " " + root_ + "/bin/busybox");
    for (const char* l : {"sh", "cat", "echo", "id", "hostname", "ls", "sleep", "ip", "mount", "grep", "ps", "head"})
      testutil::link("busybox", root_ + "/bin/" + l);
    state_ = testutil::tempDir("state");
  }
  void TearDown() override {
    if (!root_.empty()) testutil::rmrf(root_);
    testutil::rmrf(state_);
  }

  struct Out {
    int code = -1;
    std::string out, err;
  };
  Out run(RuntimeSpec spec) {
    spec.id = spec.id.empty() ? "testcontainer" : spec.id;
    if (spec.rootfs.empty()) spec.rootfs = root_;
    LaunchOptions o;
    o.rootless = ::geteuid() != 0;
    o.state_dir = state_;
    o.skip_cgroupfs = true;
    auto p = launch(spec, o);
    EXPECT_TRUE(p.ok()) << (p.ok() ? "" : p.error().str());
    if (!p.ok()) return {};
    Out r;
    auto st = awaitExec(p.value());
    if (!st.ok()) r.err = "SETUP: " + st.error().str();
    auto drain = [](int fd, std::string& s) {
      char b[4096];
      ssize_t n;
      while ((n = ::read(fd, b, sizeof b)) > 0) s.append(b, static_cast<size_t>(n));
    };
    std::thread t([&] { drain(p->stderr_fd.get(), r.err); });
    drain(p->stdout_fd.get(), r.out);
    t.join();
    int status = 0;
    ::waitpid(p->pid, &status, 0);
    r.code = exitCodeFromWaitStatus(status);
    return r;
  }
  RuntimeSpec sh(const std::string& script) {
    RuntimeSpec s;
    s.args = {"/bin/sh", "-c", script};
    return s;
  }

  std::string busybox_, root_, state_;
};

}  // namespace

TEST_F(RuntimeTest, RunsAsPid1AndRootInUserNamespace) {
  auto r = run(sh("echo pid=$$ uid=$(id -u)"));
  EXPECT_EQ(r.code, 0) << r.err;
  // With mini-init, init is PID 1 and the workload PID 2.
  EXPECT_EQ(r.out, "pid=2 uid=0\n") << r.err;
}

TEST_F(RuntimeTest, WithoutInitWorkloadIsPid1) {
  auto s = sh("echo $$");
  s.use_init = false;
  auto r = run(s);
  EXPECT_EQ(r.out, "1\n") << r.err;
}

TEST_F(RuntimeTest, ExitCodeAndSignalsPropagate) {
  EXPECT_EQ(run(sh("exit 42")).code, 42);
  EXPECT_EQ(run(sh("kill -9 $$")).code, 137);
}

TEST_F(RuntimeTest, HostnameIsIsolated) {
  auto s = sh("hostname");
  s.hostname = "vessel-test";
  EXPECT_EQ(run(s).out, "vessel-test\n");
}

TEST_F(RuntimeTest, FilesystemIsPivoted) {
  auto r = run(sh("ls /"));
  EXPECT_EQ(r.code, 0) << r.err;
  EXPECT_NE(r.out.find("bin"), std::string::npos);
  EXPECT_NE(r.out.find("proc"), std::string::npos);
  EXPECT_NE(r.out.find("dev"), std::string::npos);
  EXPECT_EQ(r.out.find("etc"), std::string::npos);  // host /etc is not visible
  EXPECT_EQ(r.out.find("home"), std::string::npos);
}

TEST_F(RuntimeTest, ProcShowsOnlyContainerProcesses) {
  auto r = run(sh("ls /proc | grep -E '^[0-9]+$' | head -5"));
  EXPECT_EQ(r.code, 0) << r.err;
  EXPECT_NE(r.out.find("1\n"), std::string::npos);
  // Every visible pid is tiny: we only see our own namespace.
  for (auto& line : vessel::str::split(r.out, '\n')) EXPECT_LT(std::stoi(std::string(line)), 50) << r.out;
}

TEST_F(RuntimeTest, DevHasMinimalNodes) {
  auto r = run(sh("echo hi > /dev/null && head -c 4 /dev/zero | wc -c && ls /dev"));
  EXPECT_EQ(r.code, 0) << r.err;
  EXPECT_NE(r.out.find("null"), std::string::npos);
  EXPECT_NE(r.out.find("urandom"), std::string::npos);
  EXPECT_EQ(r.out.find("sda"), std::string::npos);
}

TEST_F(RuntimeTest, PrivateNetworkHasOnlyLoopback) {
  auto r = run(sh("cat /proc/net/dev"));
  EXPECT_EQ(r.code, 0) << r.err;
  EXPECT_NE(r.out.find("lo:"), std::string::npos);
  EXPECT_EQ(r.out.find("eth0"), std::string::npos);
}

TEST_F(RuntimeTest, SeccompDeniesMountEvenAsContainerRoot) {
  auto r = run(sh("mkdir -p /tmp/x; mount -t tmpfs none /tmp/x; echo rc=$?"));
  EXPECT_NE(r.out.find("rc=1"), std::string::npos) << r.out << r.err;
  // EPERM, phrased by busybox as "permission denied" / by util-linux as
  // "Operation not permitted".
  EXPECT_TRUE(r.err.find("not permitted") != std::string::npos || r.err.find("permission denied") != std::string::npos)
      << r.err;
}

TEST_F(RuntimeTest, SeccompFilterIsActive) {
  auto r = run(sh("grep Seccomp: /proc/self/status"));
  EXPECT_EQ(r.code, 0) << r.err;
  EXPECT_NE(r.out.find("2"), std::string::npos) << r.out;  // SECCOMP_MODE_FILTER
}

TEST_F(RuntimeTest, CapabilitiesAreDropped) {
  auto r = run(sh("grep CapBnd /proc/self/status"));
  EXPECT_EQ(r.code, 0) << r.err;
  // Docker default set = 0x00000000a80425fb.
  EXPECT_NE(r.out.find("00000000a80425fb"), std::string::npos) << r.out;
}

TEST_F(RuntimeTest, NoNewPrivsIsSet) {
  auto r = run(sh("grep NoNewPrivs: /proc/self/status || echo absent"));
  if (r.out == "absent\n") GTEST_SKIP() << "procfs does not expose NoNewPrivs (e.g. gVisor)";
  EXPECT_NE(r.out.find("1"), std::string::npos) << r.out;
}

TEST_F(RuntimeTest, ReadOnlyRootfs) {
  auto s = sh("touch /bin/x 2>&1; echo rc=$?; touch /tmp/ok && echo tmp-ok");
  s.read_only_rootfs = true;
  auto r = run(s);
  EXPECT_NE(r.out.find("Read-only file system"), std::string::npos) << r.out << r.err;
  EXPECT_NE(r.out.find("tmp-ok"), std::string::npos) << r.out;
}

TEST_F(RuntimeTest, BindMountsWork) {
  auto host = testutil::tempDir("bind");
  ASSERT_TRUE(fs::writeFile(host + "/f", "from-host", O_CREAT).ok());
  auto s = sh("cat /data/f; echo; echo x > /data/g 2>&1 || echo ro-ok");
  s.mounts.push_back({host, "/data", true});
  auto r = run(s);
  EXPECT_NE(r.out.find("from-host"), std::string::npos) << r.out << r.err;
  EXPECT_NE(r.out.find("ro-ok"), std::string::npos) << r.out << r.err;
  testutil::rmrf(host);
}

TEST_F(RuntimeTest, EnvironmentAndWorkdir) {
  auto s = sh("echo $FOO:$container:$(pwd)");
  s.env = {"FOO=bar"};
  s.working_dir = "/bin";
  EXPECT_EQ(run(s).out, "bar:vessel:/bin\n");
}

TEST_F(RuntimeTest, MissingBinaryReportsExecStage) {
  RuntimeSpec s;
  s.args = {"/does/not/exist"};
  auto r = run(s);
  EXPECT_NE(r.err.find("execve"), std::string::npos) << r.err;
  EXPECT_NE(r.err.find("No such file"), std::string::npos) << r.err;
}

TEST_F(RuntimeTest, PathResolution) {
  RuntimeSpec s;
  s.args = {"echo", "resolved"};  // no slash: resolved through PATH by the parent
  EXPECT_EQ(run(s).out, "resolved\n");
}

TEST_F(RuntimeTest, InitReapsOrphansAndForwardsSignals) {
  // Background job orphaned to init; workload exits; init must reap and exit
  // with the workload's status.
  auto r = run(sh("(sleep 0.05 &) ; exit 3"));
  EXPECT_EQ(r.code, 3);
}

TEST_F(RuntimeTest, ExecInJoinsNamespaces) {
  auto spec = sh("sleep 5");
  spec.id = "exectest";
  spec.rootfs = root_;
  spec.hostname = "exec-host";
  LaunchOptions o;
  o.rootless = ::geteuid() != 0;
  o.state_dir = state_;
  o.skip_cgroupfs = true;
  auto p = launch(spec, o);
  ASSERT_TRUE(p.ok()) << p.error().str();
  ASSERT_TRUE(awaitExec(p.value()).ok());
  auto r = execIn(p->pid, spec, nullptr, {"/bin/sh", "-c", "hostname; echo $$; ls /proc | grep -c '^[0-9]'"}, {}, 5000);
  ASSERT_TRUE(r.ok()) << r.error().str();
  EXPECT_EQ(r->exit_code, 0) << r->err;
  EXPECT_NE(r->out.find("exec-host"), std::string::npos) << r->out;
  ::kill(p->pid, SIGKILL);
  ::waitpid(p->pid, nullptr, 0);
}
