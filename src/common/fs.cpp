// SPDX-License-Identifier: Apache-2.0
#include "vessel/common/fs.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>

namespace vessel::fs {
namespace {

Result<std::string> readAll(int fd, const std::string& what, size_t max_bytes) {
  std::string out;
  out.resize(std::min<size_t>(max_bytes, 4096));
  size_t used = 0;
  for (;;) {
    if (used == out.size()) {
      if (out.size() >= max_bytes) break;
      out.resize(std::min(max_bytes, out.size() * 2));
    }
    ssize_t n = ::read(fd, out.data() + used, out.size() - used);
    if (n < 0) {
      if (errno == EINTR) continue;
      return Error::fromErrno("read " + what);
    }
    if (n == 0) break;
    used += static_cast<size_t>(n);
  }
  out.resize(used);
  return out;
}

Status writeOnce(int fd, std::string_view data, const std::string& what) {
  for (;;) {
    ssize_t n = ::write(fd, data.data(), data.size());
    if (n < 0 && errno == EINTR) continue;
    if (n < 0) return Error::fromErrno("write " + what);
    if (static_cast<size_t>(n) != data.size()) return Error{EIO, "short write " + what};
    return okStatus();
  }
}

}  // namespace

Result<std::string> readFile(const std::string& path, size_t max_bytes) {
  UniqueFd fd(::open(path.c_str(), O_RDONLY | O_CLOEXEC));
  if (!fd) return Error::fromErrno("open " + path);
  return readAll(fd.get(), path, max_bytes);
}

Result<std::string> readFileAt(int dirfd, const char* name, size_t max_bytes) {
  UniqueFd fd(::openat(dirfd, name, O_RDONLY | O_CLOEXEC));
  if (!fd) return Error::fromErrno(std::string("open ") + name);
  return readAll(fd.get(), name, max_bytes);
}

Status writeFile(const std::string& path, std::string_view data, int flags) {
  UniqueFd fd(::open(path.c_str(), O_WRONLY | O_CLOEXEC | flags, 0644));
  if (!fd) return Error::fromErrno("open " + path);
  return writeOnce(fd.get(), data, path);
}

Status writeFileAt(int dirfd, const char* name, std::string_view data) {
  UniqueFd fd(::openat(dirfd, name, O_WRONLY | O_CLOEXEC));
  if (!fd) return Error::fromErrno(std::string("open ") + name);
  return writeOnce(fd.get(), data, name);
}

bool exists(const std::string& path) noexcept {
  struct stat st {};
  return ::stat(path.c_str(), &st) == 0;
}

bool isDir(const std::string& path) noexcept {
  struct stat st {};
  return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

Status mkdirs(const std::string& path, unsigned mode) {
  if (path.empty()) return Error::logic("mkdirs: empty path");
  std::string cur;
  cur.reserve(path.size());
  size_t i = 0;
  while (i <= path.size()) {
    size_t j = path.find('/', i);
    if (j == std::string::npos) j = path.size();
    cur.assign(path, 0, j);
    if (!cur.empty() && ::mkdir(cur.c_str(), mode) != 0 && errno != EEXIST)
      return Error::fromErrno("mkdir " + cur);
    i = j + 1;
  }
  if (!isDir(path)) return Error{ENOTDIR, "mkdirs " + path};
  return okStatus();
}

Status touch(const std::string& path, unsigned mode) {
  UniqueFd fd(::open(path.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, mode));
  if (!fd) return Error::fromErrno("touch " + path);
  return okStatus();
}

Result<std::vector<std::string>> listDir(const std::string& path) {
  DIR* d = ::opendir(path.c_str());
  if (d == nullptr) return Error::fromErrno("opendir " + path);
  std::vector<std::string> out;
  while (dirent* e = ::readdir(d)) {
    std::string_view n = e->d_name;
    if (n == "." || n == "..") continue;
    out.emplace_back(n);
  }
  ::closedir(d);
  return out;
}

Result<UniqueFd> openDir(const std::string& path) {
  UniqueFd fd(::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (!fd) return Error::fromErrno("open dir " + path);
  return fd;
}

std::string join(std::string_view a, std::string_view b) {
  std::string out(a);
  if (!out.empty() && out.back() == '/') out.pop_back();
  if (!b.empty() && b.front() == '/') b.remove_prefix(1);
  out.push_back('/');
  out.append(b);
  return out;
}

}  // namespace vessel::fs
