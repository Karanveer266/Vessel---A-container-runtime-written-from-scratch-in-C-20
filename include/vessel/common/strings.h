// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <charconv>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vessel::str {

std::vector<std::string_view> split(std::string_view s, char sep, bool skip_empty = true);
std::string_view trim(std::string_view s) noexcept;
bool startsWith(std::string_view s, std::string_view prefix) noexcept;

template <typename T>
std::optional<T> parseInt(std::string_view s, int base = 10) noexcept {
  s = trim(s);
  T v{};
  auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v, base);
  if (ec != std::errc{} || p != s.data() + s.size()) return std::nullopt;
  return v;
}

// "64M", "1.5G", "4096", "512k" -> bytes. Binary (1024-based) multipliers.
std::optional<uint64_t> parseBytes(std::string_view s) noexcept;
// "1.5" CPUs, "500m" millicores -> millicores.
std::optional<uint32_t> parseCpu(std::string_view s) noexcept;
std::string humanBytes(uint64_t b);

}  // namespace vessel::str
