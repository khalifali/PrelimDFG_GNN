# DFG preliminary transport cases — preparation status

## Prepared

- Twelve matched-size geometries (N=50) selected across low, middle and high thirds of the available 45-geometry exposure ranking are copied into `geometries/` with source and DEM/GNN provenance.
- Forty-eight fixed-geometry laminar cases are prepared: each geometry at Pe=0, 1, 10 and 50. The case index is `cases/matrix.csv`; per-case `run.json` records geometry hash, mesh, Pe, speed, Reynolds number and prepared-unrun status.
- Flow uses an x-directed inlet/outlet channel, no-slip transverse walls, and fixed oxygen concentration at the inlet. Particle surfaces use the existing experimental perfect-absorber surface-marker exchange.

## Not done

No simulation has been launched. Inputs are not accuracy-certified. First perform the isolated-sphere mesh/time-step and mass-balance gate, then run a 24-case pilot (six geometries, two from each exposure group, all four Pe). Run the other 24 matrix cases only if the pilot is stable and the uptake metric is useful. The flow-direction/orientation check is also needed because geometric exposure is orientation-averaged.

See `EXECUTION_PLAN.md` for scope, parameter definitions and staged launch conditions.
