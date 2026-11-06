// SPDX-License-Identifier: Apache-2.0
#include "vessel/common/strings.h"

#include <cmath>
#include <cstdio>

namespace vessel::str {

std::vector<std::string_view> split(std::string_view s, char sep, bool skip_empty) {
  std::vector<std::string_view> out;
  size_t start = 0;
  while (start <= s.size()) {
    size_t end = s.find(sep, start);
    if (end == std::string_view::npos) end = s.size();
    auto piece = s.substr(start, end - start);
    if (!skip_empty || !piece.empty()) out.push_back(piece);
    start = end + 1;
  }
  return out;
}

std::string_view trim(std::string_view s) noexcept {
  constexpr std::string_view ws = " \t\r\n";
  auto b = s.find_first_not_of(ws);
  if (b == std::string_view::npos) return {};
  auto e = s.find_last_not_of(ws);
  return s.substr(b, e - b + 1);
}

bool startsWith(std::string_view s, std::string_view prefix) noexcept {
  return s.substr(0, prefix.size()) == prefix;
}

namespace {
std::optional<double> parseDouble(std::string_view s) noexcept {
  double v = 0;
  auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
  if (ec != std::errc{} || p != s.data() + s.size() || !std::isfinite(v) || v < 0) return std::nullopt;
  return v;
}
}  // namespace

std::optional<uint64_t> parseBytes(std::string_view s) noexcept {
  s = trim(s);
  if (s.empty()) return std::nullopt;
  uint64_t mult = 1;
  char last = s.back();
  if (last == 'b' || last == 'B') {
    s.remove_suffix(1);
    if (s.empty()) return std::nullopt;
    last = s.back();
  }
  switch (last) {
    case 'k': case 'K': mult = 1ULL << 10; break;
    case 'm': case 'M': mult = 1ULL << 20; break;
    case 'g': case 'G': mult = 1ULL << 30; break;
    case 't': case 'T': mult = 1ULL << 40; break;
    default: break;
  }
  if (mult != 1) s.remove_suffix(1);
  auto v = parseDouble(s);
  if (!v) return std::nullopt;
  double bytes = *v * static_cast<double>(mult);
  if (bytes > 1.8e19) return std::nullopt;
  return static_cast<uint64_t>(bytes);
}

std::optional<uint32_t> parseCpu(std::string_view s) noexcept {
  s = trim(s);
  if (s.empty()) return std::nullopt;
  if (s.back() == 'm') {
    auto v = parseInt<uint32_t>(s.substr(0, s.size() - 1));
    return v;
  }
  auto v = parseDouble(s);
  if (!v || *v > 4'000'000.0) return std::nullopt;
  return static_cast<uint32_t>(std::lround(*v * 1000.0));
}

std::string humanBytes(uint64_t b) {
  static constexpr const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double v = static_cast<double>(b);
  int u = 0;
  while (v >= 1024.0 && u < 4) {
    v /= 1024.0;
    ++u;
  }
  char buf[32];
  std::snprintf(buf, sizeof buf, u == 0 ? "%.0f %s" : "%.1f %s", v, units[u]);
  return buf;
}

}  // namespace vessel::str
