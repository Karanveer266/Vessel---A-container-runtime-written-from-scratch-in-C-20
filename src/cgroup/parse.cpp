// SPDX-License-Identifier: Apache-2.0
// Pure parsers for cgroupfs / procfs text formats. No I/O here: everything is
// unit-tested against captured kernel output in tests/cgroup_parse_test.cpp.
#include <cmath>
#include <cstdint>
#include <limits>
#include <string_view>

#include "vessel/cgroup/cgroup.h"
#include "vessel/common/strings.h"

namespace vessel::cgroup::parse {

bool cpuMax(const std::string& s, uint64_t* quota_us, uint64_t* period_us) {
  auto parts = str::split(str::trim(s), ' ');
  if (parts.size() != 2) return false;
  if (parts[0] == "max") {
    *quota_us = std::numeric_limits<uint64_t>::max();
  } else {
    auto q = str::parseInt<uint64_t>(parts[0]);
    if (!q) return false;
    *quota_us = *q;
  }
  auto p = str::parseInt<uint64_t>(parts[1]);
  if (!p || *p == 0) return false;
  *period_us = *p;
  return true;
}

std::string cpuMaxValue(uint32_t millicores, uint32_t period_us) {
  if (millicores == 0) return "max " + std::to_string(period_us);
  // quota = period * cores. The kernel rejects quotas < 1000us, so clamp:
  // asking for 5 millicores yields the minimum enforceable bandwidth instead
  // of an EINVAL that would abort container creation.
  uint64_t quota = static_cast<uint64_t>(period_us) * millicores / 1000;
  if (quota < 1000) quota = 1000;
  return std::to_string(quota) + " " + std::to_string(period_us);
}

std::optional<uint64_t> keyed(const std::string& s, const std::string& key) {
  std::string_view sv = s;
  size_t pos = 0;
  while (pos < sv.size()) {
    size_t eol = sv.find('\n', pos);
    if (eol == std::string_view::npos) eol = sv.size();
    std::string_view line = sv.substr(pos, eol - pos);
    if (line.size() > key.size() && line.substr(0, key.size()) == key && line[key.size()] == ' ')
      return str::parseInt<uint64_t>(line.substr(key.size() + 1));
    pos = eol + 1;
  }
  return std::nullopt;
}

std::optional<uint64_t> maxOrInt(const std::string& s) {
  auto t = str::trim(s);
  if (t == "max") return std::numeric_limits<uint64_t>::max();
  return str::parseInt<uint64_t>(t);
}

namespace {
bool psiLine(std::string_view line, PsiLine* out) {
  // "some avg10=0.12 avg60=0.05 avg300=0.01 total=123456"
  auto fields = str::split(line, ' ');
  if (fields.size() < 5) return false;
  for (size_t i = 1; i < fields.size(); ++i) {
    auto kv = fields[i];
    auto eq = kv.find('=');
    if (eq == std::string_view::npos) return false;
    auto k = kv.substr(0, eq);
    auto v = kv.substr(eq + 1);
    if (k == "total") {
      auto t = str::parseInt<uint64_t>(v);
      if (!t) return false;
      out->total_usec = *t;
    } else {
      float f = 0;
      auto [p, ec] = std::from_chars(v.data(), v.data() + v.size(), f);
      if (ec != std::errc{}) return false;
      if (k == "avg10") out->avg10 = f;
      else if (k == "avg60") out->avg60 = f;
      else if (k == "avg300") out->avg300 = f;
    }
  }
  return true;
}
}  // namespace

Psi psi(const std::string& s) {
  Psi p;
  bool some = false;
  for (auto line : str::split(s, '\n')) {
    if (str::startsWith(line, "some ")) some = psiLine(line, &p.some);
    else if (str::startsWith(line, "full ")) psiLine(line, &p.full);
  }
  p.valid = some;
  return p;
}

uint64_t weightToShares(uint32_t weight) {
  // Inverse of the systemd/runc mapping weight = 1 + ((shares-2)*9999)/262142.
  if (weight == 0) return 1024;
  if (weight > 10000) weight = 10000;
  return 2 + (static_cast<uint64_t>(weight - 1) * 262142) / 9999;
}

std::optional<std::string> unifiedPathFromProcSelf(const std::string& content) {
  for (auto line : str::split(content, '\n')) {
    if (str::startsWith(line, "0::")) return std::string(line.substr(3));
  }
  return std::nullopt;
}

std::optional<std::string> v1PathFor(const std::string& content, const std::string& controller) {
  // "hierarchy-ID:controller-list:path"; controller-list may be "cpu,cpuacct".
  for (auto line : str::split(content, '\n')) {
    auto c1 = line.find(':');
    if (c1 == std::string_view::npos) continue;
    auto c2 = line.find(':', c1 + 1);
    if (c2 == std::string_view::npos) continue;
    auto ctrls = line.substr(c1 + 1, c2 - c1 - 1);
    for (auto c : str::split(ctrls, ',')) {
      if (c == controller) return std::string(line.substr(c2 + 1));
    }
  }
  return std::nullopt;
}

namespace {
std::string unescapeOctal(std::string_view s) {
  // mountinfo escapes space, tab, newline and backslash as \ooo.
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '\\' && i + 3 < s.size()) {
      auto o = str::parseInt<unsigned>(s.substr(i + 1, 3), 8);
      if (o && *o < 256) {
        out.push_back(static_cast<char>(*o));
        i += 3;
        continue;
      }
    }
    out.push_back(s[i]);
  }
  return out;
}
}  // namespace

std::vector<MountEntry> mountinfo(const std::string& content) {
  // 36 35 98:0 /mnt1 /mnt2 rw,noatime master:1 - ext3 /dev/root rw,errors=continue
  // (1)(2)(3)   (4)   (5)      (6)      (7...)  sep (8)   (9)        (10)
  std::vector<MountEntry> out;
  for (auto line : str::split(content, '\n')) {
    auto f = str::split(line, ' ');
    size_t sep = 0;
    for (size_t i = 6; i < f.size(); ++i)
      if (f[i] == "-") { sep = i; break; }
    if (sep == 0 || sep + 3 > f.size() || f.size() < 7) continue;
    MountEntry m;
    m.root = unescapeOctal(f[3]);
    m.mount_point = unescapeOctal(f[4]);
    m.fstype = std::string(f[sep + 1]);
    m.super_opts = sep + 3 < f.size() ? std::string(f[sep + 3]) : std::string();
    out.push_back(std::move(m));
  }
  return out;
}

}  // namespace vessel::cgroup::parse
