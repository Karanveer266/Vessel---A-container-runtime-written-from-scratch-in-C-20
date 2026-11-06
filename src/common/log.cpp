// SPDX-License-Identifier: Apache-2.0
#include "vessel/common/log.h"

#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <ctime>
#include <mutex>

#ifdef __ANDROID__
#include <android/log.h>
#endif

namespace vessel::log {
namespace {
std::atomic<int> g_level{static_cast<int>(Level::kInfo)};
std::mutex g_mu;
constexpr const char* kNames[] = {"D", "I", "W", "E"};
}  // namespace

void setLevel(Level lvl) noexcept { g_level.store(static_cast<int>(lvl), std::memory_order_relaxed); }
Level level() noexcept { return static_cast<Level>(g_level.load(std::memory_order_relaxed)); }

bool parseLevel(std::string_view s, Level* out) noexcept {
  if (s == "debug") *out = Level::kDebug;
  else if (s == "info") *out = Level::kInfo;
  else if (s == "warn") *out = Level::kWarn;
  else if (s == "error") *out = Level::kError;
  else return false;
  return true;
}

void write(Level lvl, std::string_view component, std::string_view msg) noexcept {
#ifdef __ANDROID__
  static constexpr int prio[] = {ANDROID_LOG_DEBUG, ANDROID_LOG_INFO, ANDROID_LOG_WARN, ANDROID_LOG_ERROR};
  std::string line = std::string(component) + ": " + std::string(msg);
  __android_log_write(prio[static_cast<int>(lvl)], "vesseld", line.c_str());
#else
  timespec ts{};
  ::clock_gettime(CLOCK_REALTIME, &ts);
  tm t{};
  ::localtime_r(&ts.tv_sec, &t);
  char hdr[64];
  std::snprintf(hdr, sizeof hdr, "%02d:%02d:%02d.%03ld %s ", t.tm_hour, t.tm_min, t.tm_sec,
                ts.tv_nsec / 1'000'000, kNames[static_cast<int>(lvl)]);
  std::lock_guard lk(g_mu);
  std::fprintf(stderr, "%s[%.*s] %.*s\n", hdr, static_cast<int>(component.size()), component.data(),
               static_cast<int>(msg.size()), msg.data());
#endif
}

}  // namespace vessel::log
