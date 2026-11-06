// SPDX-License-Identifier: Apache-2.0
// Tiny leveled logger. Thread-safe, line-buffered to stderr, std::format based.
// On Android builds the same macros route to __android_log_write so the daemon
// shows up in `adb logcat -s vesseld`.
#pragma once

#include <format>
#include <string>
#include <string_view>

namespace vessel::log {

enum class Level : int { kDebug = 0, kInfo = 1, kWarn = 2, kError = 3 };

void setLevel(Level lvl) noexcept;
Level level() noexcept;
bool parseLevel(std::string_view s, Level* out) noexcept;
void write(Level lvl, std::string_view component, std::string_view msg) noexcept;

template <typename... Args>
void logf(Level lvl, std::string_view component, std::format_string<Args...> fmt, Args&&... args) {
  if (static_cast<int>(lvl) < static_cast<int>(level())) return;
  write(lvl, component, std::format(fmt, std::forward<Args>(args)...));
}

}  // namespace vessel::log

#define VLOG_DEBUG(comp, ...) ::vessel::log::logf(::vessel::log::Level::kDebug, comp, __VA_ARGS__)
#define VLOG_INFO(comp, ...) ::vessel::log::logf(::vessel::log::Level::kInfo, comp, __VA_ARGS__)
#define VLOG_WARN(comp, ...) ::vessel::log::logf(::vessel::log::Level::kWarn, comp, __VA_ARGS__)
#define VLOG_ERROR(comp, ...) ::vessel::log::logf(::vessel::log::Level::kError, comp, __VA_ARGS__)
