// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lamfoam::core {
struct VdwInteraction { double normalForce; double potential; };

// Spherical cohesive part of khalifali/LIGGGHTS-PUBLIC hertz_fvdw,
// commit 7163357d2be7ca92c1fc39d1abb70eecb3d6795e.
// Negative force is attractive. Potential is continuous and zero at cutoff;
// unlike the legacy diagnostic, it includes the cohesive contribution.
inline VdwInteraction vanDerWaals(double gap, double effectiveRadius,
                                 double hamaker, double minimumSeparation)
{
    if (!std::isfinite(gap) || !std::isfinite(effectiveRadius) || effectiveRadius<=0
        || !std::isfinite(hamaker) || hamaker<0 || !std::isfinite(minimumSeparation)
        || minimumSeparation<=0)
        throw std::invalid_argument("vdW requires finite gap, R_eff>0, A>=0 and h0>0");
    const double cutoff=0.2*effectiveRadius;
    if (gap>cutoff || hamaker==0) return {0,0};
    const double c=hamaker*effectiveRadius/6;
    const double separation=std::max(gap,0.0)+minimumSeparation;
    const double force=-c/(separation*separation);
    double energy=c*(1/(cutoff+minimumSeparation)-1/separation);
    if (gap<0) energy-=force*gap;
    if (!std::isfinite(force) || !std::isfinite(energy))
        throw std::overflow_error("vdW force or energy overflow");
    return {force,energy};
}
}
