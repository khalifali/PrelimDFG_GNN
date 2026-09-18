# FracVAL measured-dimension pilot

Purpose: find three monodisperse, connected, non-overlapping agglomerates whose
measured finite-scale box-counting exponents approach 1.5, 2.0 and 2.9.
Target tolerance is ±0.05, set before calibration. Calibration changes generation
parameters only; no fitting window is selected to improve agreement with a target.
All candidate settings and failures are retained in measurements.json.

## Provenance

Original FracVAL release by J. Morán, A. Fuentes, F. Liu and J. Yon:
https://doi.org/10.17632/mgf8wdcsfb.1
Paper: https://doi.org/10.1016/j.cpc.2019.01.015
License: GPLv3. Original archive SHA256:
7c743d920a211257468b7d45d1a9a15ca21d2edc3647e87a1564de3ce63457d2

The original Linux Fortran sources are vendored under codes/fracval/vendor, with
individual SHA256 checksums and license. Generation uses the self-hosted runner.
Changes in working copies are input constants and explicit random-seed
initialization. Compilation uses -O2 -ffree-line-length-none -fdefault-real-8
-fdefault-double-8. The aggregation equations are unchanged. This is a
monodisperse pilot (radius 0.5, N=1000); Ext_case=1. Requested Df and kf refer to
the original mass-radius relation and are not box-counting target guarantees.

## Measurement and QA

Use codes/structure/descriptors.py without changes. It counts boxes intersecting
particle bodies, not only centers. Eight logarithmically spaced box sizes range
from one diameter to one-quarter of the maximum span in the principal-axis frame.
The reported exponent averages three orientations and two grid shifts. All six
curves, grid variation, fit R², scale span and endpoint-window sensitivity are
retained. A good fit alone does not establish a true asymptotic fractal dimension.

Geometry QA requires 1000 finite equal-radius particles, one connected contact
component (gap tolerance 2e-6 diameters), and maximum overlap <=2e-6 diameters.
Convex-hull porosity uses the hull of the particle bodies. Coordinates are not
stretched or rearranged after generation. These are pre-DEM geometries; no
relaxation, exposure calculation, or model training is part of this pilot.

## Files and visualization

particles.csv: coordinates and radius in particle-diameter units.
particles.vtp and post_000000000000.txt: SI coordinates, choosing diameter 1 um.
ParaView: open particles.vtp, add a Sphere Glyph, select radius as scale array,
set scale factor 2, and display all points. No geometric-exposure array is present.
The preview uses orthographic projections of sphere silhouettes, with independent
panel scales and depth shading. The preview is not a porosity or exposure map.

## Estimator reference check

A separate diagnostic applies the unchanged counter to simple-cubic arrays of
touching spheres. They have bulk three-dimensional scaling, but finite-scale
estimates for these small samples need not equal 3. The references are not
FracVAL outputs and must not be substituted for a requested pilot agglomerate.
See estimator_references.json. This checks finite-size measurement limitations;
it does not change the estimator or existing dataset descriptors.
