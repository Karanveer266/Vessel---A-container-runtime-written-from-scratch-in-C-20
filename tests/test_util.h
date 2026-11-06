// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <fcntl.h>
#include <sched.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <string>

#include "vessel/common/fs.h"

namespace vessel::testutil {

// Can this process create the namespaces a container needs? Probed in a
// forked child so the test process itself is never modified.
inline bool canCreateNamespaces() {
  pid_t p = ::fork();
  if (p == 0) {
    int flags = CLONE_NEWUSER | CLONE_NEWNS | CLONE_NEWPID | CLONE_NEWUTS | CLONE_NEWIPC | CLONE_NEWNET;
    ::_exit(::unshare(flags) == 0 ? 0 : 1);
  }
  int st = 0;
  ::waitpid(p, &st, 0);
  return WIFEXITED(st) && WEXITSTATUS(st) == 0;
}

// Test-side wrappers that make ignored results explicit (and fail loudly).
inline void sh(const std::string& cmd) {
  if (std::system(cmd.c_str()) != 0) std::abort();
}
inline void rmrf(const std::string& path) {
  if (!path.empty() && path.rfind("/tmp/vessel-test-", 0) == 0) sh("rm -rf '" + path + "'");
}
inline void link(const char* target, const std::string& path) {
  if (::symlink(target, path.c_str()) != 0 && errno != EEXIST) std::abort();
}
inline void writeByte(int fd) {
  if (::write(fd, "g", 1) != 1) std::abort();
}
inline void readByte(int fd) {
  char c;
  if (::read(fd, &c, 1) != 1) ::_exit(99);
}

inline std::string tempDir(const std::string& tag) {
  std::string t = "/tmp/vessel-test-" + tag + "-XXXXXX";
  char* d = ::mkdtemp(t.data());
  return d ? std::string(d) : std::string();
}

}  // namespace vessel::testutil
