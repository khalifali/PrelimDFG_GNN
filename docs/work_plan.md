# Active work plan: structural coverage and two exposure targets

Updated 19 September 2026. This is the current plan and supersedes earlier “next stages” instructions for the exposure study.

## Objective and boundaries

Predict properties of unseen intact BPM agglomerates across a documented spectrum of measured morphology. Compare normal-ray exposure with surface-averaged cosine-weighted angular accessibility. Each target is predicted independently at aggregate level by GNNs and by ordinary linear regression, random forest and ANN using traditional descriptors.

Required descriptors: finite-scale box-counting fractal dimension and particle-body convex-hull porosity. Requested Df/kf remain generation metadata, never inference inputs. Freeze the exact conventional feature list and model-selection protocol before training. Expanded structural descriptors may be a clearly labelled secondary baseline, not a replacement for the traditional baseline.

No matched-pair objective or pair-preserving folds in the restored study. Existing matched analyses and the structural-comparison runner campaign remain exploratory historical results. FracVAL stays paused. Breakage literature provides motivation only; no overlap with the ongoing DFG deagglomeration study. Claims concern investigated morphology, size and generation families, not universal generalization or unvalidated physical transport effectiveness.

## Ordered stages and decision gates

| Stage | Status | Deliverable / gate |
|---|---|---|
| 1. Audit existing 359 BPM geometries | Initial numerical audit complete | Per-N descriptor ranges, joint occupancy, quality flags and targeted gaps |
| 2. Verify attainable additional coverage | Pilot implemented; runner campaign initiated | Small current-generator feasibility pilot, with box-estimator reference checks |
| 3. Expand development database | Pending stage 2 | Accepted final BPM geometries fill demonstrable gaps; no arbitrary total |
| 4. Freeze evaluation and reserve fresh tests | Pending coverage definition | Separate independent interpolation and extrapolation cohorts with tracked lineages |
| 5. Verify and compute angular accessibility | Verification may proceed alongside stages 2–3 | Reference geometries, resolution/rotation tests, then both targets for accepted database |
| 6. Compare prediction models | Pending data and target gates | Same cases and frozen splits for every model and both targets |
| 7. Interpret and write | Pending results | Coverage maps, region-specific errors, limitations and literature-grounded motivation |

### Stage 1 — completed initial audit

See [coverage results](../results/coverage_audit_20260919/README.md). The input is the committed final-geometry descriptor table. Exposure outcomes were not used to define gaps.

359 cases; 98 box-quality flags; measured Dbox approximately 1.74–2.31 and hull porosity 0.729–0.983. Low-N estimates contain most quality flags. Joint occupancy is reported separately by N, because pooled ranges conceal strong size dependence. These are development geometries: all have previously been examined.

### Stage 2 — immediate next action

Inspect the current generator's supported configuration controls and define a small target-blind pilot across open and compact structures and representative particle counts. Do not assume requested exponent 2.9 produces measured Dbox 2.9. Compare box-counting behaviour on open and compact reference structures across sizes to distinguish finite-scale estimator limitations from generation limitations.

Priorities from the audit: lower porosity at medium/large N; greater open-structure coverage at small/medium N; reliable dimension variation within particle-count classes. Seek anisotropy, branching and local-density variety within accessible dimension/porosity regions. Empty grid cells are candidate questions, not necessarily realizable structures.

Proceed to larger batches only when a small pilot demonstrates useful measured coverage. Use BPM only, then classify by final relaxed geometry. FracVAL is not restarted.

### Stage 3 — targeted database expansion

Generate independent realizations preferentially in sparse, achievable regions. Check non-overlap/connectivity, provenance, numerical/mechanical assessment and final measured descriptors. Define acceptance criteria before reviewing target/prediction results. Retain unsuccessful-attempt records and reasons. Additional seeds quantify realization variability after configuration diversity has been established.

Select case counts from achieved coverage and observed redundancy, not a predetermined thousand-case quota. Track initial-to-final descriptor changes and quality flags. Do not discard inconvenient structures based on ML errors.

### Stage 4 — independent testing

Freeze the development domain and protocol before creating reserved tests. Interpolation means new independent structures spread throughout covered descriptor regions; extrapolation means explicitly specified boundary regions and/or larger N. Do not reuse the 359 previously inspected cases as fresh confirmation.

Keep all derivatives of a parent geometry (rotations, scaled copies, DEM snapshots) within one provenance group. Generation parameters are not inference features. Morphology may balance sampling without making every case in a descriptor bin one group. Split logic must be documented and validated before training. No matched-pair grouping.

Test descriptors can be used for preregistered coverage eligibility; test exposure labels and model scores must not guide generation, model tuning or latent selection. Keep identical case assignments for both targets and all model families.

### Stage 5 — two targets

Keep current normal-ray exposure as the reference definition. Implement angular accessibility as the outward-hemisphere cosine-weighted visibility integral averaged over each spherical surface, then averaged across particles for the monodisperse aggregate.

Verify an isolated sphere, simple sphere arrangements, numerical resolution and rotation invariance within sampling error. Evaluate representative pilot geometries and computation time before full database calculation. Choose resolution from numerical accuracy, not which target produces better GNN scores. Save particle-level values for visualization and aggregate targets for the primary model comparison. Neither metric is automatically a drag or diffusion shielding factor.

### Stage 6 — restored model comparison

Ordinary linear regression, random forest, ANN and geometry/contact GNN. Traditional descriptor list must include measured Dbox and convex-hull porosity and must exclude requested Df/kf. Use fit-only scaling, validation-only hyperparameter/epoch/latent-dimension selection, and a declared seed budget held constant across targets. Compare GNN latent sizes 1–4 without selecting on independent test performance.

Report R², RMSE and MAE for each target, interpolation/extrapolation separately, plus errors by morphology region and N. Treat repeated training seeds as training variability, not independent physical replicates. Report actual computation time if claiming a speed benefit. Do not require a particular score by changing the evaluation after seeing results.

## Writing resources

[Exposure and shielding literature](exposure_shielding_literature.md) supplies motivation and interpretation limits. Existing runner results remain useful methodological evidence, but do not establish the result of this expanded two-target study.

## Stage 2 pilot implementation

`codes/coverage_pilot/run.py` and `.github/workflows/coverage-pilot.yml` define 18 attempts: N=100,350,1000, each with coupled (requested Df,kf)=(1.5,1.3),(1.8,1.3),(2.2,1.1),(2.6,0.8),(2.8,0.8),(2.9,1.0). One independent deterministic seed per setting, 180-second generation budget. All successfully generated candidates proceed to unchanged 160000-step BPM verification; failed assessments are retained and not accepted automatically. Initial/final descriptors are recorded. Chains N=50,200,1000 and simple-cubic arrays N=125,1000,8000 are estimator references only. This feasibility sample estimates neither success probability nor final database size. Review results before expanding batches.
