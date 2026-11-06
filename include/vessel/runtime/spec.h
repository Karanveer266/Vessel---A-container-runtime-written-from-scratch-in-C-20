// SPDX-License-Identifier: Apache-2.0
// Runtime-level container description.
//
// Deliberately independent of protobuf: the runtime library (libvessel_runtime)
// can be embedded without gRPC (unit tests, a future OCI shim, an Android
// system service). The daemon maps vessel.v1.ContainerSpec -> RuntimeSpec.
#pragma once

#include <sys/types.h>

#include <cstdint>
#include <string>
#include <vector>

#include "vessel/cgroup/cgroup.h"

namespace vessel::runtime {

enum class NetworkMode { kNone, kLoopback, kHost };
enum class SeccompProfile { kDefault, kStrict, kUnconfined };

struct BindMount {
  std::string source;   // host path
  std::string target;   // absolute path inside the container
  bool read_only = false;
};

struct IdMap {
  uint32_t inside = 0, outside = 0, count = 1;
};

struct RuntimeSpec {
  std::string id;                       // container id (used for hostname default)
  std::string rootfs;                   // host path; "/" = read-only view of host root
  std::vector<std::string> args;
  std::vector<std::string> env;
  std::string hostname;
  std::string working_dir = "/";
  NetworkMode network = NetworkMode::kLoopback;
  SeccompProfile seccomp = SeccompProfile::kDefault;
  std::vector<BindMount> mounts;
  std::vector<std::string> devices;     // extra host device nodes to bind (e.g. /dev/nvidia0)
  bool read_only_rootfs = false;
  std::vector<IdMap> uid_map, gid_map;  // empty = daemon default
  bool use_init = true;                 // run vessel's mini-init as PID 1
  bool new_cgroup_ns = true;
  bool no_new_privs = true;
  // Capabilities retained inside the container (CAP_* numbers). Empty = the
  // Docker-compatible default set (see capabilities.cpp).
  std::vector<int> keep_caps;
  bool keep_default_caps = true;
};

}  // namespace vessel::runtime
