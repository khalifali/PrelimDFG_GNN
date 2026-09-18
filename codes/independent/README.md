# Independent BPM tests

This prospective evaluation uses all 179 previously accepted BPM cases as
development data. Previous cross-validation results are development diagnostics;
these fresh holdouts are not an exact replacement for their estimands.

| Test | Particle counts | (Df, kf) pairs | Diameters in µm | Replicates | Cases |
|---|---|---|---|---|---:|
| Parameter interpolation | 75, 150, 350 | (2.0, 1.2), (2.4, 0.95) | 1, 1.5, 2 | 5 | 90 |
| Size extrapolation | 750, 1000 | (1.8, 1.3), (2.2, 1.1), (2.6, 0.8) | 1, 1.5, 2 | 5 | 90 |

Interpolation morphology pairs lie along the segments between existing training
pairs, not independently selected rectangular limits. Extrapolation changes size
only. Both use the existing reconstructed tunable-fractal generator, with fresh
deterministic seeds. Neither test measures transfer to another generator,
experimental aggregates, or an unseen morphology family. Diameter is not counted
as an independent dimensionless structural parameter.

## Freeze before test evaluation

The self-hosted `Independent BPM interpolation and extrapolation` workflow:

1. Downloads the existing 179-case development dataset and existing v9-style
   development inner-validation records. No historical outer scores are used by
   the selection code.
2. Publishes `freeze.json` and the complete new case/seed manifest. For GNN and
   bottleneck ANN separately, chooses the smallest q in 1–4 within 0.02 of the
   highest mean inner-validation R² across the 15 existing fits. Training epochs
   for each architecture are the rounded median of those fits' best epochs.
   The tuned forest uses the most frequent development-selected configuration,
   with lexicographic tie breaking. These are pragmatic development choices,
   not a claim of optimal tuning or strict nested group validation.
3. Fits all models on the full 179 development cases, using seeds 7, 17, 27,
   and publishes checkpoint hashes before generating the test geometries.
4. Generates both test sets, verifies fresh seeds and rotation/scale invariant
   pair-distance fingerprints against original development geometries and each
   other. Fingerprints are a duplicate guard, not proof of distributional independence.
5. Runs BPM only, using the existing material/numerical parameters, dt=1e-10 s,
   160000 steps (16 µs), and three particle/bond snapshots. Applies the existing
   mechanical acceptance criteria without relaxing thresholds for new cases.
6. Computes exposure using 2048 rays and contacts with gap tolerance 1e-6 d.
   Preserves raw dumps plus exposure-colored VTP/PVD files for ParaView.
7. Checks frozen checkpoint hashes and complete test coverage, then reports
   interpolation and extrapolation separately for all prespecified models.

Primary models: internally selected GNN and ANN. Fixed q=1–4, conventional ANN,
linear regression, tuned forest and fixed-v9 forest are prespecified comparisons.
There is no test-based seed or model selection. Scores are un-clipped. Report R²,
RMSE and MAE per set and per parameter combination. Across-training-seed SD is
not a confidence interval; do not interpret three seeds as independent datasets.

## Failures and interpretation

No automatic exclusion or replacement seed is allowed. All diagnostics are
preserved and scoring stops if any case fails geometry or mechanical checks.
Any subsequent amendment must be recorded before inspecting the corresponding
test prediction errors. Numerical failures and coverage must remain visible.
Fixed 16 µs checks do not certify equilibrium, especially for the larger sizes.

The datasets have already informed development decisions, so the previous
cross-validation scores are not a new untouched confirmation. Once these fresh
test scores are inspected, changes motivated by their errors require a new
confirmation set. Retain the present test results rather than overwrite them.

Launch from GitHub Actions on the repository self-hosted runner. Output artifacts
include the frozen protocol, trained models, geometry manifest, BPM diagnostics,
visualization files, raw predictions, and separate final score tables.
