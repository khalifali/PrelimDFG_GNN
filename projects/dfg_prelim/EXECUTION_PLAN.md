# Preliminary fixed-geometry oxygen-uptake study: execution plan

## Purpose and scope

Test whether DEM/GNN geometric exposure is associated with oxygen uptake by BPM-relaxed agglomerates when the structures are held fixed. This is a transport-focused bridge to the later coupled CFD–DEM work. It does not claim that geometric exposure itself represents transport.

All generated cases hold the agglomerate geometry fixed. The LAMMPS input has no integration fix or positive run command; no contact or bond law is active during transport. Oxygen is a passive dissolved scalar with molecular diffusivity and a perfect-absorber particle surface model (`surfaceMarkerExperimental`). The flow is steady laminar crossflow, prescribed at the x-inlet, with fixed pressure at the outlet. Pe=0 is the diffusion-only reference.

## Geometry selection

Use 12 geometries with 50 primary particles each, drawn from `PrelimDFG_GNN/data/generation/study180_v1`. They are evenly spaced through the 45-case N=50 exposure ranking, giving four low-, four middle- and four high-exposure structures. This keeps particle count and primary-particle size matched while sampling the available geometric-exposure range. The three exposure strata are cut at 0.64234 and 0.70381.

Exposure is the final (20,000-step) 512-ray mean geometric-exposure value in `results/study180/exposure_summary_post.json`. It is a geometry-only descriptor. The supplied structures and post-relaxation descriptor come from the GNN study set; each selected geometry has source commit, hash, exposure and DEM provenance recorded alongside the copied CSV.

## Case matrix

For each geometry, prepare Pe = 0, 1, 10 and 50, for 48 fixed-geometry cases total. Pe is defined using inlet speed, molecular diffusivity and the agglomerate streamwise bounding-box diameter: `Pe = U_in L / D`. With D = 2e-9 m²/s, the selected speeds produce Re <= 0.1 using water-like kinematic viscosity 1e-6 m²/s, consistent with laminar flow.

Run in stages:

1. **Numerical gate:** Before the matrix, validate the absorbing-surface implementation against an isolated sphere at multiple mesh and time-step resolutions. Check mass balance and non-negativity. The existing surface-marker prototype has not yet demonstrated certified accuracy, so the inherited discretization is only a starting point.
2. **Pilot (24 cases):** Run two geometries from each exposure stratum at all four Pe values. Review solver stability, oxygen mass balance, mesh sensitivity, and whether uptake curves reach a useful comparison interval.
3. **Full matrix (remaining 24 cases):** Proceed only if the gate and pilot show adequate numerical behavior. Compare initial uptake rate and cumulative uptake normalized by particle surface area and inlet concentration; retain raw oxygen inventory and integrated-flux diagnostics.
4. **Orientation check:** Because the GNN descriptor is orientation-averaged but crossflow transport is directional, repeat selected pilot geometries at controlled rotations at Pe=10. Treat this as a confound check before interpreting exposure as predictive across shapes.

The 24 pilot cases are a staged subset of the 48, not additional cases. Numerical-gate and orientation-check runs are separate validation runs and are not yet generated in this matrix.

## Prepared case inputs

`cases/matrix.csv` indexes every case, parameter and directory. Each case contains a translated copy of the fixed geometry, OpenFOAM and LAMMPS inputs, source provenance, and `run.json`. The output mesh is a finite rectangular channel with no-slip side walls; assess confinement by enlarging the domain in a mesh/domain sensitivity check before production interpretation. The preparation is **not accuracy-certified**.

## Launch procedure

Run the numerical gate first in an activated LAMFOAM environment. Do not launch the 48-case matrix until the absorption model and mass-balance checks pass and the pilot stop/go decision is recorded. For a prepared case, activate the LAMFOAM environment and run `python run.py --run-prepared cases/<geometry-id>/PeXX`. This runs `blockMesh` and `lamfoamIBSolver` in place and updates that case’s manifest. Each case is separate, and preparation refuses to overwrite existing directories. Archive solver version, environment, logs, mesh statistics, and analysis outputs with the case manifest.
