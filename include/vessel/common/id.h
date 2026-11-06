// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <string>

namespace vessel {
// 12 lowercase hex chars from getrandom(2) (48 bits). Collision-checked by the
// registry; 48 bits keeps ids short enough to type while making accidental
// collisions irrelevant at the scale of one host.
std::string newContainerId();
bool isValidName(const std::string& name) noexcept;  // [a-zA-Z0-9][a-zA-Z0-9_.-]{0,62}
}  // namespace vessel
