// SPDX-License-Identifier: Apache-2.0
// Host detection + driver selection.
//
// Detection is driven by /proc/self/mountinfo rather than by hard-coding
// /sys/fs/cgroup, because (a) Android mounts v1 controllers in odd places
// (/dev/memcg, /acct, /dev/cpuctl), (b) inside containers/CI the visible
// mount root is a sub-directory of the real hierarchy, and (c) on hybrid
// systems the cgroup2 mount at /sys/fs/cgroup/unified has no controllers.
#include <unistd.h>

#include <algorithm>
#include <set>

#include "drivers.h"
#include "vessel/common/fs.h"
#include "vessel/common/log.h"
#include "vessel/common/strings.h"

namespace vessel::cgroup {

const char* versionName(Version v) noexcept {
  switch (v) {
    case Version::kV1: return "v1";
    case Version::kV2: return "v2";
    case Version::kHybrid: return "hybrid";
    default: return "none";
  }
}

namespace detail {

std::string relativeToMount(const std::string& self_path, const std::string& mount_root) {
  // A cgroup namespace or a bind-mounted sub-hierarchy makes the mount root
  // non-"/"; our own path in /proc/self/cgroup is then reported relative to
  // the hierarchy root, so strip the common prefix.
  if (mount_root == "/" || mount_root.empty()) return self_path;
  if (str::startsWith(self_path, mount_root)) {
    std::string rel = self_path.substr(mount_root.size());
    return rel.empty() ? "/" : rel;
  }
  return "/";
}

}  // namespace detail

Result<std::unique_ptr<Manager>> Manager::detect(const std::string& root_override) {
  VESSEL_ASSIGN_OR_RETURN(std::string mi, fs::readFile("/proc/self/mountinfo"));
  VESSEL_ASSIGN_OR_RETURN(std::string self, fs::readFile("/proc/self/cgroup"));
  auto mounts = parse::mountinfo(mi);

  // --- cgroup v2 -----------------------------------------------------------
  const parse::MountEntry* v2 = nullptr;
  for (const auto& m : mounts)
    if (m.fstype == "cgroup2") { v2 = &m; break; }

  bool v2_has_controllers = false;
  if (v2 != nullptr) {
    auto c = fs::readFile(fs::join(v2->mount_point, "cgroup.controllers"));
    v2_has_controllers = c.ok() && !str::trim(c.value()).empty();
  }

  std::map<std::string, std::string> v1_paths;  // controller -> abs dir
  for (const auto& m : mounts) {
    if (m.fstype != "cgroup") continue;
    for (auto opt : str::split(m.super_opts, ',')) {
      static const std::set<std::string_view> kWanted = {"cpu", "cpuacct", "memory", "pids",
                                                         "cpuset", "freezer"};
      if (!kWanted.contains(opt)) continue;
      auto self_path = parse::v1PathFor(self, std::string(opt));
      std::string rel = detail::relativeToMount(self_path.value_or("/"), m.root);
      v1_paths.emplace(std::string(opt), fs::join(m.mount_point, rel));
    }
  }

  if (v2 != nullptr && v2_has_controllers) {
    std::string base = root_override;
    if (base.empty()) {
      auto self_path = parse::unifiedPathFromProcSelf(self).value_or("/");
      base = fs::join(v2->mount_point, detail::relativeToMount(self_path, v2->root));
    }
    VLOG_INFO("cgroup", "using cgroup v2 at {}", base);
    return makeV2Manager(base, !v1_paths.empty());
  }
  if (!v1_paths.empty()) {
    if (!root_override.empty())
      for (auto& [c, p] : v1_paths) p = fs::join(root_override, c);
    VLOG_INFO("cgroup", "using cgroup v1 ({} controllers{})", v1_paths.size(),
              v2 != nullptr ? ", hybrid host" : "");
    return makeV1Manager(std::move(v1_paths), v2 != nullptr);
  }
  return Error{ENOTSUP, "cgroup detect: no usable cgroup v1 or v2 hierarchy mounted"};
}

}  // namespace vessel::cgroup
