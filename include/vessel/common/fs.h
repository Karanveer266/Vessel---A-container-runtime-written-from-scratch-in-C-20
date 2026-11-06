// SPDX-License-Identifier: Apache-2.0
// Small, allocation-conscious filesystem helpers used for cgroupfs/procfs I/O.
//
// cgroupfs and procfs files are "seq files": reads must use a single read(2)
// at offset 0 to get a consistent snapshot, and writes must be a single
// write(2) because the kernel parses each write() call as one command (a
// short write of "+memory" split over two calls would be two bogus commands).
// That is why these helpers exist instead of using <fstream>.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "vessel/common/result.h"
#include "vessel/common/unique_fd.h"

namespace vessel::fs {

Result<std::string> readFile(const std::string& path, size_t max_bytes = 1 << 20);
// Writes `data` with exactly one write(2). Fails on short write.
Status writeFile(const std::string& path, std::string_view data, int flags = 0);
// Like writeFile but relative to an open directory fd (openat-based, race free).
Status writeFileAt(int dirfd, const char* name, std::string_view data);
Result<std::string> readFileAt(int dirfd, const char* name, size_t max_bytes = 1 << 16);

bool exists(const std::string& path) noexcept;
bool isDir(const std::string& path) noexcept;
// mkdir -p. Existing directories are not an error.
Status mkdirs(const std::string& path, unsigned mode = 0755);
// Ensures a regular file exists at path (used as bind-mount target).
Status touch(const std::string& path, unsigned mode = 0644);
Result<std::vector<std::string>> listDir(const std::string& path);
Result<UniqueFd> openDir(const std::string& path);
std::string join(std::string_view a, std::string_view b);

}  // namespace vessel::fs
