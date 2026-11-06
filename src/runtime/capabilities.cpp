// SPDX-License-Identifier: Apache-2.0
#include "vessel/runtime/capabilities.h"

#include <fcntl.h>
#include <linux/capability.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>

namespace vessel::runtime::caps {

std::vector<int> defaultSet() {
  // Same set as Docker / containerd's default OCI spec.
  return {CAP_CHOWN,  CAP_DAC_OVERRIDE, CAP_FSETID,         CAP_FOWNER, CAP_MKNOD,
          CAP_NET_RAW, CAP_SETGID,      CAP_SETUID,         CAP_SETFCAP, CAP_SETPCAP,
          CAP_NET_BIND_SERVICE, CAP_SYS_CHROOT, CAP_KILL,   CAP_AUDIT_WRITE};
}

Plan makePlan(const std::vector<int>& keep) {
  Plan p;
  for (int c : keep)
    if (c >= 0 && c < 64) p.keep_mask |= (1ULL << c);
  p.last_cap = probeLastCap();
  return p;
}

int probeLastCap() noexcept {
  int fd = ::open("/proc/sys/kernel/cap_last_cap", O_RDONLY | O_CLOEXEC);
  if (fd < 0) return CAP_LAST_CAP;
  char buf[16] = {};
  ssize_t n = ::read(fd, buf, sizeof buf - 1);
  ::close(fd);
  if (n <= 0) return CAP_LAST_CAP;
  int v = std::atoi(buf);
  return v > 0 && v < 64 ? v : CAP_LAST_CAP;
}

int apply(const Plan& plan) noexcept {
  // 1. Bounding set: once dropped, no execve (even of a setuid/file-cap
  //    binary) can regain the capability. Requires CAP_SETPCAP, which we still
  //    hold at this point (we are root in the user namespace).
  for (int c = 0; c <= plan.last_cap; ++c) {
    if (plan.keep_mask & (1ULL << c)) continue;
    if (::prctl(PR_CAPBSET_DROP, c, 0, 0, 0) != 0 && errno != EINVAL) return -errno;
  }
  // 2. Ambient set: clear entirely (ambient caps survive execve of
  //    non-privileged binaries — we do not want that implicit inheritance).
  ::prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL, 0, 0, 0);

  // 3. effective / permitted / inheritable via capset(2) (v3 = 64-bit).
  __user_cap_header_struct hdr{_LINUX_CAPABILITY_VERSION_3, 0};
  __user_cap_data_struct data[2]{};
  const auto lo = static_cast<uint32_t>(plan.keep_mask & 0xFFFFFFFFu);
  const auto hi = static_cast<uint32_t>(plan.keep_mask >> 32);
  data[0].effective = data[0].permitted = data[0].inheritable = lo;
  data[1].effective = data[1].permitted = data[1].inheritable = hi;
  if (::syscall(SYS_capset, &hdr, data) != 0) return -errno;
  return 0;
}

}  // namespace vessel::runtime::caps
