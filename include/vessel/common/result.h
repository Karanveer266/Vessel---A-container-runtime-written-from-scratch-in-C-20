// SPDX-License-Identifier: Apache-2.0
// Result<T>: a minimal std::expected-like type for C++20.
//
// Why not exceptions? The runtime layer runs in contexts where exceptions are
// either forbidden (the post-clone child, which must stay async-signal-safe)
// or undesirable (hot tracing paths). Every failure carries the errno and a
// static description of the *stage* that failed, which is exactly what an
// operator needs ("pivot_root: EINVAL") and is cheap to build.
//
// Why not std::expected? It is C++23; the project targets C++20 so that it
// builds with the Android NDK's libc++ and older distro toolchains.
#pragma once

#include <cerrno>
#include <cstring>
#include <string>
#include <utility>
#include <variant>

namespace vessel {

struct Error {
  int code = 0;              // errno value (0 = logical error, see message)
  std::string message;       // "<stage>: <detail>"

  [[nodiscard]] std::string str() const {
    if (code == 0) return message;
    return message + ": " + std::strerror(code) + " (errno " + std::to_string(code) + ")";
  }
  static Error fromErrno(std::string what) { return Error{errno, std::move(what)}; }
  static Error logic(std::string what) { return Error{0, std::move(what)}; }
};

template <typename T>
class [[nodiscard]] Result {
 public:
  Result(T value) : v_(std::move(value)) {}            // NOLINT(google-explicit-constructor)
  Result(Error err) : v_(std::move(err)) {}            // NOLINT(google-explicit-constructor)

  [[nodiscard]] bool ok() const noexcept { return v_.index() == 0; }
  explicit operator bool() const noexcept { return ok(); }

  T& value() & { return std::get<0>(v_); }
  const T& value() const& { return std::get<0>(v_); }
  T&& value() && { return std::get<0>(std::move(v_)); }
  T* operator->() { return &value(); }
  const T* operator->() const { return &value(); }
  T& operator*() & { return value(); }

  [[nodiscard]] const Error& error() const { return std::get<1>(v_); }

  template <typename U>
  T valueOr(U&& fallback) const& { return ok() ? value() : static_cast<T>(std::forward<U>(fallback)); }

 private:
  std::variant<T, Error> v_;
};

template <>
class [[nodiscard]] Result<void> {
 public:
  Result() = default;
  Result(Error err) : err_(std::move(err)), ok_(false) {}  // NOLINT(google-explicit-constructor)
  [[nodiscard]] bool ok() const noexcept { return ok_; }
  explicit operator bool() const noexcept { return ok_; }
  [[nodiscard]] const Error& error() const { return err_; }

 private:
  Error err_;
  bool ok_ = true;
};

using Status = Result<void>;
inline Status okStatus() { return {}; }

}  // namespace vessel

// Propagate an error from an expression returning Result<...>/Status.
#define VESSEL_TRY(expr)                              \
  do {                                                \
    auto vessel_try_r_ = (expr);                      \
    if (!vessel_try_r_.ok()) return vessel_try_r_.error(); \
  } while (0)

#define VESSEL_CONCAT_INNER(a, b) a##b
#define VESSEL_CONCAT(a, b) VESSEL_CONCAT_INNER(a, b)
#define VESSEL_ASSIGN_OR_RETURN_IMPL(tmp, lhs, expr) \
  auto tmp = (expr);                                 \
  if (!tmp.ok()) return tmp.error();                 \
  lhs = std::move(tmp).value()
#define VESSEL_ASSIGN_OR_RETURN(lhs, expr) \
  VESSEL_ASSIGN_OR_RETURN_IMPL(VESSEL_CONCAT(vessel_r_, __LINE__), lhs, expr)
