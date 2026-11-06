// SPDX-License-Identifier: Apache-2.0
// RAII owner for a POSIX file descriptor.
//
// Every fd in Vessel is owned by exactly one UniqueFd; raw ints only appear at
// syscall boundaries. Combined with O_CLOEXEC on every open this guarantees no
// daemon descriptor (gRPC sockets, cgroup dirs, BPF maps...) leaks into a
// container, which would be a container-escape primitive.
#pragma once

#include <unistd.h>

#include <utility>

namespace vessel {

class UniqueFd {
 public:
  UniqueFd() noexcept = default;
  explicit UniqueFd(int fd) noexcept : fd_(fd) {}
  ~UniqueFd() { reset(); }

  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& o) noexcept : fd_(std::exchange(o.fd_, -1)) {}
  UniqueFd& operator=(UniqueFd&& o) noexcept {
    if (this != &o) reset(std::exchange(o.fd_, -1));
    return *this;
  }

  [[nodiscard]] int get() const noexcept { return fd_; }
  [[nodiscard]] bool valid() const noexcept { return fd_ >= 0; }
  explicit operator bool() const noexcept { return valid(); }

  int release() noexcept { return std::exchange(fd_, -1); }
  void reset(int fd = -1) noexcept {
    if (fd_ >= 0) ::close(fd_);
    fd_ = fd;
  }

 private:
  int fd_ = -1;
};

}  // namespace vessel
