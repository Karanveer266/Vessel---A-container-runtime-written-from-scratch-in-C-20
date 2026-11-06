// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <signal.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <mutex>
#include <thread>

#include "test_util.h"
#include "vessel/runtime/syscalls.h"
#include "vessel/trace/trace.h"

using namespace vessel;
using namespace vessel::trace;

TEST(Ptrace, TracesSyscallsAfterExecOnly) {
  std::mutex mu;
  std::vector<Event> events;
  auto agg = std::make_shared<SyscallAggregator>();
  PtraceSession s("c1", [&](Event&& e) {
    std::lock_guard lk(mu);
    events.push_back(std::move(e));
  }, agg);

  int sync[2];
  ASSERT_EQ(::pipe(sync), 0);
  pid_t pid = ::fork();
  if (pid == 0) {
    ::close(sync[1]);
    testutil::readByte(sync[0]);
    ::syscall(SYS_getcpu, nullptr, nullptr, nullptr);  // pre-exec, and never used by sh/cat: must NOT be reported
    ::execl("/bin/sh", "sh", "-c", "cat /nonexistent-file 2>/dev/null; exit 7", nullptr);
    ::_exit(127);
  }
  ::close(sync[0]);
  ASSERT_TRUE(s.seize(pid).ok());
  testutil::writeByte(sync[1]);
  ::close(sync[1]);
  int st = s.run(pid, nullptr);
  ASSERT_TRUE(WIFEXITED(st));
  EXPECT_EQ(WEXITSTATUS(st), 7);

  bool saw_exec = false, saw_enoent = false, saw_marker = false, saw_child = false;
  for (const auto& e : events) {
    EXPECT_EQ(e.container_id, "c1");
    EXPECT_EQ(e.backend, Backend::kPtrace);
    if (e.kind == Kind::kExec) saw_exec = true;
    if (e.nr == SYS_getcpu) saw_marker = true;
    if (e.tid != pid) saw_child = true;  // cat runs in a forked child
    auto n = syscalls::name(e.nr);
    if ((n == "openat" || n == "open") && e.ret == -ENOENT) saw_enoent = true;
  }
  EXPECT_TRUE(saw_exec);
  EXPECT_TRUE(saw_enoent) << "failed openat of /nonexistent-file not observed";
  EXPECT_TRUE(saw_child) << "fork children were not auto-attached";
  EXPECT_FALSE(saw_marker) << "pre-exec runtime syscalls leaked into the trace";
  EXPECT_GT(agg->total(), 10u);
  bool found = false;
  for (const auto& st2 : agg->snapshot())
    if (syscalls::name(st2.nr) == "execve") found = true;
  EXPECT_TRUE(found);
}

TEST(Ptrace, SignalsAreReinjected) {
  auto agg = std::make_shared<SyscallAggregator>();
  PtraceSession s("c2", nullptr, agg);
  int sync[2];
  ASSERT_EQ(::pipe(sync), 0);
  pid_t pid = ::fork();
  if (pid == 0) {
    testutil::readByte(sync[0]);
    ::execl("/bin/sh", "sh", "-c", "trap 'exit 9' TERM; while :; do sleep 0.01; done", nullptr);
    ::_exit(127);
  }
  ASSERT_TRUE(s.seize(pid).ok());
  testutil::writeByte(sync[1]);
  std::thread killer([pid] {
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    ::kill(pid, SIGTERM);
  });
  int st = s.run(pid, nullptr);
  killer.join();
  ASSERT_TRUE(WIFEXITED(st)) << st;
  EXPECT_EQ(WEXITSTATUS(st), 9) << "SIGTERM was swallowed by the tracer";
}
