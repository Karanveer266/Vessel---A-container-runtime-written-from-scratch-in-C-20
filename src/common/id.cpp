// SPDX-License-Identifier: Apache-2.0
#include "vessel/common/id.h"

#include <sys/random.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <random>

namespace vessel {

std::string newContainerId() {
  uint8_t buf[6];
  ssize_t n;
  do {
    n = ::getrandom(buf, sizeof buf, 0);
  } while (n < 0 && errno == EINTR);
  if (n != static_cast<ssize_t>(sizeof buf)) {
    // getrandom can only fail on ancient kernels; fall back to a seeded PRNG.
    std::mt19937_64 rng(static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()));
    for (auto& b : buf) b = static_cast<uint8_t>(rng());
  }
  static constexpr char hex[] = "0123456789abcdef";
  std::string id(12, '0');
  for (size_t i = 0; i < sizeof buf; ++i) {
    id[2 * i] = hex[buf[i] >> 4];
    id[2 * i + 1] = hex[buf[i] & 0xF];
  }
  return id;
}

bool isValidName(const std::string& name) noexcept {
  if (name.empty() || name.size() > 63) return false;
  auto alnum = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'); };
  if (!alnum(name[0])) return false;
  for (char c : name)
    if (!alnum(c) && c != '_' && c != '.' && c != '-') return false;
  return true;
}

}  // namespace vessel
