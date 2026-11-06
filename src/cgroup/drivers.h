// SPDX-License-Identifier: Apache-2.0
// Internal: driver factories. Not part of the public include tree.
#pragma once

#include <map>
#include <memory>
#include <string>

#include "vessel/cgroup/cgroup.h"

namespace vessel::cgroup {

Result<std::unique_ptr<Manager>> makeV2Manager(std::string base, bool hybrid);
Result<std::unique_ptr<Manager>> makeV1Manager(std::map<std::string, std::string> controller_dirs,
                                               bool hybrid);

namespace detail {
std::string relativeToMount(const std::string& self_path, const std::string& mount_root);
// Common helper: SIGKILL everything listed in a cgroup.procs file until it is
// empty (handles tasks forking while we iterate). Bounded number of rounds.
Status killLoop(const std::string& procs_file);
Result<std::vector<pid_t>> readProcs(const std::string& procs_file);
// rmdir with bounded retries on EBUSY.
Status rmdirRetry(const std::string& dir);
// rmdir every empty child of `root` whose name looks like a container id.
size_t reapEmptyChildren(const std::string& root);
}  // namespace detail

}  // namespace vessel::cgroup
