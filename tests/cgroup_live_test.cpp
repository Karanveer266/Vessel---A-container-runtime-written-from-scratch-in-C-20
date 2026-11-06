// SPDX-License-Identifier: Apache-2.0
// Exercises the detected cgroup driver against the live kernel.
#include <gtest/gtest.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <thread>

#include "test_util.h"
#include "vessel/cgroup/cgroup.h"
#include "vessel/common/id.h"

using namespace vessel;
using namespace vessel::cgroup;

namespace {
std::unique_ptr<Manager> detectOrSkip() {
  auto m = Manager::detect();
  if (!m.ok()) return nullptr;
  return std::move(m.value());
}
}  // namespace

TEST(CgroupLive, CreateLimitStatsDestroy) {
  auto m = detectOrSkip();
  if (!m) GTEST_SKIP() << "no writable cgroup hierarchy";
  auto cg = m->create("t-" + newContainerId());
  ASSERT_TRUE(cg.ok()) << cg.error().str();
  Limits l;
  l.memory_max = 64ull << 20;
  l.pids_max = 16;
  l.cpu_millicores = 500;
  auto s = (*cg)->apply(l);
  ASSERT_TRUE(s.ok()) << s.error().str();
  auto st = (*cg)->stats();
  ASSERT_TRUE(st.ok()) << st.error().str();
  EXPECT_EQ(st->memory_max, 64ull << 20);
  EXPECT_EQ(st->pids_max, 16u);
  EXPECT_TRUE((*cg)->destroy().ok());
}

TEST(CgroupLive, ProcessAccountingAndKillAll) {
  auto m = detectOrSkip();
  if (!m) GTEST_SKIP() << "no writable cgroup hierarchy";
  auto cg = m->create("t-" + newContainerId());
  ASSERT_TRUE(cg.ok()) << cg.error().str();
  pid_t p = ::fork();
  if (p == 0) {
    for (;;) ::pause();
  }
  ASSERT_TRUE((*cg)->addProcess(p).ok());
  auto procs = (*cg)->procs();
  ASSERT_TRUE(procs.ok());
  ASSERT_EQ(procs->size(), 1u);
  EXPECT_EQ((*procs)[0], p);
  auto st = (*cg)->stats();
  ASSERT_TRUE(st.ok());
  EXPECT_EQ(st->pids_current, 1u);
  ASSERT_TRUE((*cg)->killAll().ok());
  int status = 0;
  ::waitpid(p, &status, 0);
  EXPECT_TRUE(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL);
  EXPECT_TRUE((*cg)->destroy().ok());
}

TEST(CgroupLive, PidsLimitStopsForkBomb) {
  auto m = detectOrSkip();
  if (!m) GTEST_SKIP() << "no writable cgroup hierarchy";
  auto cg = m->create("t-" + newContainerId());
  ASSERT_TRUE(cg.ok());
  Limits l;
  l.pids_max = 5;
  if (!(*cg)->apply(l).ok()) GTEST_SKIP() << "pids controller unavailable";
  int pipefd[2];
  ASSERT_EQ(::pipe(pipefd), 0);
  pid_t p = ::fork();
  if (p == 0) {
    testutil::readByte(pipefd[0]);  // wait until moved into the cgroup
    int ok = 0, failed = 0;
    for (int i = 0; i < 20; ++i) {
      pid_t c2 = ::fork();
      if (c2 == 0) {
        ::pause();
        ::_exit(0);
      }
      (c2 > 0 ? ok : failed)++;
    }
    ::_exit(ok * 10 + (failed > 0 ? 1 : 0));
  }
  ASSERT_TRUE((*cg)->addProcess(p).ok());
  testutil::writeByte(pipefd[1]);
  int status = 0;
  ::waitpid(p, &status, 0);
  ASSERT_TRUE(WIFEXITED(status));
  const int ok = WEXITSTATUS(status) / 10, any_failed = WEXITSTATUS(status) % 10;
  auto st = (*cg)->stats();
  if (st.ok() && st->pids_max == 5 && st->pids_current > st->pids_max) {
    // Accounting works but the limit is not enforced: a property of the
    // kernel (observed on gVisor), not of the driver. Don't report a pass.
    (void)(*cg)->killAll();
    (void)(*cg)->destroy();
    GTEST_SKIP() << "kernel accounts but does not enforce pids.max (pids.current=" << st->pids_current << ")";
  }
  EXPECT_EQ(any_failed, 1) << "fork was never refused";
  EXPECT_LE(ok, 4) << "more forks succeeded than pids.max allows";
  (void)(*cg)->killAll();
  EXPECT_TRUE((*cg)->destroy().ok());
  ::close(pipefd[0]);
  ::close(pipefd[1]);
}
