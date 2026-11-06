// SPDX-License-Identifier: Apache-2.0
// Parsers tested against verbatim kernel output captured from real hosts.
#include <gtest/gtest.h>

#include <limits>

#include "vessel/cgroup/cgroup.h"

using namespace vessel::cgroup;

TEST(CgroupParse, CpuMax) {
  uint64_t q = 0, p = 0;
  ASSERT_TRUE(parse::cpuMax("max 100000\n", &q, &p));
  EXPECT_EQ(q, std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(p, 100000u);
  ASSERT_TRUE(parse::cpuMax("50000 100000", &q, &p));
  EXPECT_EQ(q, 50000u);
  EXPECT_FALSE(parse::cpuMax("garbage", &q, &p));
  EXPECT_FALSE(parse::cpuMax("100 0", &q, &p));
}

TEST(CgroupParse, CpuMaxValue) {
  EXPECT_EQ(parse::cpuMaxValue(0, 100000), "max 100000");
  EXPECT_EQ(parse::cpuMaxValue(500, 100000), "50000 100000");
  EXPECT_EQ(parse::cpuMaxValue(2500, 100000), "250000 100000");
  // Below the kernel's 1ms minimum quota -> clamped rather than EINVAL.
  EXPECT_EQ(parse::cpuMaxValue(5, 100000), "1000 100000");
}

TEST(CgroupParse, KeyedFiles) {
  const std::string cpu_stat =
      "usage_usec 1234567\nuser_usec 1000000\nsystem_usec 234567\n"
      "nr_periods 120\nnr_throttled 37\nthrottled_usec 845123\n"
      "nr_bursts 0\nburst_usec 0\n";
  EXPECT_EQ(parse::keyed(cpu_stat, "usage_usec"), 1234567u);
  EXPECT_EQ(parse::keyed(cpu_stat, "nr_throttled"), 37u);
  EXPECT_EQ(parse::keyed(cpu_stat, "throttled_usec"), 845123u);
  EXPECT_FALSE(parse::keyed(cpu_stat, "usage"));  // no prefix matching
  const std::string events = "low 0\nhigh 12\nmax 3\noom 1\noom_kill 1\noom_group_kill 0\n";
  EXPECT_EQ(parse::keyed(events, "oom"), 1u);
  EXPECT_EQ(parse::keyed(events, "oom_kill"), 1u);
  EXPECT_EQ(parse::keyed(events, "max"), 3u);
}

TEST(CgroupParse, MaxOrInt) {
  EXPECT_EQ(parse::maxOrInt("max\n"), std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(parse::maxOrInt("67108864\n"), 67108864u);
  EXPECT_FALSE(parse::maxOrInt("nope"));
}

TEST(CgroupParse, Psi) {
  auto p = parse::psi(
      "some avg10=12.34 avg60=5.00 avg300=1.25 total=987654\n"
      "full avg10=1.50 avg60=0.10 avg300=0.00 total=12345\n");
  ASSERT_TRUE(p.valid);
  EXPECT_FLOAT_EQ(p.some.avg10, 12.34f);
  EXPECT_FLOAT_EQ(p.some.avg300, 1.25f);
  EXPECT_EQ(p.some.total_usec, 987654u);
  EXPECT_FLOAT_EQ(p.full.avg10, 1.5f);
  EXPECT_EQ(p.full.total_usec, 12345u);
  // cpu.pressure on kernels < 5.13 has only a "some" line.
  auto c = parse::psi("some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n");
  EXPECT_TRUE(c.valid);
  EXPECT_FALSE(parse::psi("").valid);
}

TEST(CgroupParse, WeightToSharesMatchesRuncFormula) {
  EXPECT_EQ(parse::weightToShares(0), 1024u);  // default
  EXPECT_EQ(parse::weightToShares(1), 2u);
  EXPECT_EQ(parse::weightToShares(10000), 262144u);
  // Round trip with runc's shares->weight: 1 + ((shares-2)*9999)/262142.
  for (uint32_t w : {1u, 39u, 100u, 5000u, 10000u}) {
    uint64_t s = parse::weightToShares(w);
    uint64_t back = 1 + ((s - 2) * 9999) / 262142;
    EXPECT_NEAR(static_cast<double>(back), w, 1.0) << w;
  }
}

TEST(CgroupParse, ProcSelfCgroup) {
  EXPECT_EQ(parse::unifiedPathFromProcSelf("0::/user.slice/user-1000.slice/session-2.scope\n"),
            "/user.slice/user-1000.slice/session-2.scope");
  const std::string v1 =
      "12:pids:/docker/abc\n11:cpu,cpuacct:/docker/abc\n4:memory:/docker/abc\n1:name=systemd:/docker/abc\n0::/\n";
  EXPECT_EQ(parse::v1PathFor(v1, "cpuacct"), "/docker/abc");
  EXPECT_EQ(parse::v1PathFor(v1, "memory"), "/docker/abc");
  EXPECT_FALSE(parse::v1PathFor(v1, "freezer"));
  EXPECT_EQ(parse::unifiedPathFromProcSelf(v1), "/");
}

TEST(CgroupParse, MountInfo) {
  // Real lines: systemd v2 host, an optional-fields line, an Android v1 memcg,
  // and a path with an escaped space.
  const std::string mi =
      "35 24 0:30 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:9 - cgroup2 cgroup2 rw,nsdelegate,memory_recursiveprot\n"
      "46 45 0:10 /dd64 /sys/fs/cgroup/cpu rw - cgroup none rw,cpu\n"
      "29 18 0:24 / /dev/memcg rw,nosuid,nodev,noexec,relatime shared:13 master:4 - cgroup none rw,memory\n"
      "60 36 0:50 / /mnt/my\\040disk rw - ext4 /dev/sda1 rw\n"
      "malformed line\n";
  auto m = parse::mountinfo(mi);
  ASSERT_EQ(m.size(), 4u);
  EXPECT_EQ(m[0].fstype, "cgroup2");
  EXPECT_EQ(m[0].mount_point, "/sys/fs/cgroup");
  EXPECT_EQ(m[1].root, "/dd64");
  EXPECT_EQ(m[1].super_opts, "rw,cpu");
  EXPECT_EQ(m[2].mount_point, "/dev/memcg");
  EXPECT_EQ(m[2].super_opts, "rw,memory");
  EXPECT_EQ(m[3].mount_point, "/mnt/my disk");
}
