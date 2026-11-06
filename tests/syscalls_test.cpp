// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <sys/syscall.h>

#include "vessel/runtime/syscalls.h"

using namespace vessel;

TEST(Syscalls, TableMatchesLibcHeaders) {
  EXPECT_EQ(syscalls::number("read"), SYS_read);
  EXPECT_EQ(syscalls::number("openat"), SYS_openat);
  EXPECT_EQ(syscalls::number("execve"), SYS_execve);
  EXPECT_EQ(syscalls::number("clone3"), SYS_clone3);
  EXPECT_EQ(syscalls::number("mmap"), SYS_mmap);
  EXPECT_EQ(syscalls::name(SYS_write), "write");
  EXPECT_EQ(syscalls::name(SYS_pivot_root), "pivot_root");
  EXPECT_EQ(syscalls::name(-1), "");
  EXPECT_EQ(syscalls::name(100000), "");
  EXPECT_FALSE(syscalls::number("not_a_syscall"));
  EXPECT_GE(syscalls::maxNr(), SYS_clone3);
}

TEST(Syscalls, Categories) {
  EXPECT_EQ(syscalls::categorize(SYS_mmap), syscalls::Category::kMemory);
  EXPECT_EQ(syscalls::categorize(SYS_brk), syscalls::Category::kMemory);
  EXPECT_EQ(syscalls::categorize(SYS_connect), syscalls::Category::kNet);
  EXPECT_EQ(syscalls::categorize(SYS_openat), syscalls::Category::kFile);
  EXPECT_EQ(syscalls::categorize(SYS_clone), syscalls::Category::kProcess);
  EXPECT_EQ(syscalls::categorize(SYS_futex), syscalls::Category::kIpc);
}
