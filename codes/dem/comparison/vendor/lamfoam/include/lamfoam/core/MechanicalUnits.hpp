// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <string_view>
namespace lamfoam::core {
// These native systems have coherent mechanical force/energy units (ftm2v=1).
inline bool supportedMechanicalUnits(std::string_view style)
{ return style=="si" || style=="cgs" || style=="micro"; }
}
