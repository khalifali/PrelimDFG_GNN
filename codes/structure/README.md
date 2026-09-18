# Geometry-based characterization and development validation

This campaign uses the 179 accepted original BPM cases and the 180 already
inspected independent-test cases. All 359 are now **development data**. It does
not overwrite old datasets, scores, frozen models or mechanical assessments.
No new DEM simulation or vdW campaign is launched. The one recorded fixed-time
kinetic-energy exception in the 180-case set remains visible and included.
A future independent confirmation must use new untouched data.

## Box-counting descriptor

`descriptors.py` uses the same closed sphere–cube intersection criterion as
`fractalagglomerategenration/DimensionAnalysis/testdim.py`, verified numerically
against that implementation. It is intentionally not the legacy global fit:

* An unbounded sparse integer grid covers complete particle bodies. The legacy
  farthest-pair midpoint cube is not guaranteed to contain every sphere.
* Positions are centered and scaled by the primary diameter. A principal-axis
  frame plus two fixed rotations and two grid shifts quantify grid sensitivity.
  Symmetric/near-degenerate PCA axes remain a potential orientation ambiguity.
* Eight logarithmically spaced physical box sizes between d and L/4 are used,
  where L is the maximum particle-body span in the principal-axis frame.
  Subparticle scales are excluded because solid sphere volume then controls
  the slope. No fit window is selected using exposure or prediction performance.
* The descriptor is the mean slope over six orientation/shift fits. Raw counts,
  each fitting scale, minimum fit R², mean regression slope standard error,
  across-grid slope SD, scale span and endpoint-trimming sensitivity are saved.
* Flags: span below 0.5 decades, minimum fit R² below .95, grid SD above .10,
  or maximum trimmed-window deviation above .20. Flags do not remove cases.

This is a finite-scale morphological exponent, not a uniquely true or
asymptotic fractal dimension. Requested mass–radius exponent and measured box
exponent are different quantities. A high fit R² with little scale separation
does not establish fractality. Small aggregates are especially limited.

## Particle-body convex-hull porosity

For equal spheres, the convex hull of the particle bodies is the Minkowski sum
of the centre hull and a sphere of primary radius r. We compute its volume as

    V_body = V_centres + A_centres*r + M*r^2 + 4*pi*r^3/3
    M = 0.5 * sum_edges(length * exterior_dihedral_angle)
    porosity = 1 - sum_particle_volumes / V_body

This avoids approximate sphere tessellation and the centre-only hull's missing
outer shell. Rank-deficient configurations use sphere, capsule and rounded
planar-hull formulas. Analytic cube/sphere/line/square tests cover the formulas.
Hull facets and normals come from scipy.spatial.ConvexHull.

Particle-volume summation neglects overlaps only when the sum of pairwise
sphere-lens volumes divided by hull volume is <=1e-5. That sum bounds the
porosity approximation error even with multiple overlaps. Larger overlaps stop
the calculation. The hull-porosity estimate is a lower bound on union-based
porosity by at most the reported bound. It includes void space between branches
and is not a local pore-network porosity. Polydisperse input is rejected pending
a different enclosing-body calculation.

Both descriptors are measured on initial and final BPM snapshots. Final
measurements are used for prediction. Requested Df/kf remain metadata only.

## Groups versus folds

Independence groups use available root-geometry lineage and an initial-geometry
fingerprint (sorted pair distances in diameter units, rounded to 7 decimals).
Unioning both relations keeps identified related variants together. A distance
fingerprint is a duplicate guard, not a proof against all near duplicates or
unrecorded shared ancestry. Imports should provide root_geometry_id when known.

The primary development folds balance tercile strata of log(N), measured box
dimension and hull porosity across five folds, assigning complete groups only.
These bins distribute morphology; they are not independence groups. Descriptor
values over the full candidate set are used to design folds, but target values
and requested Df/kf are never consulted. This is a retrospective, covariate-
balanced development design, not a prospective random-sampling estimate.

Two separate development stress tests withhold: (a) the highest-porosity 20% of
groups; (b) all groups containing N>=750. Internal validation uses a geometry-
balanced group holdout for interpolation, or the corresponding high-porosity /
high-size group holdout within outer training for the stress tests. All split
roles and their structural ranges are saved. Measured ranges are not evidence
of transfer to another generator; this dataset still comes from one generator.

## Controlled model comparison

* Original six: N, Rg/d, contact count, contact density, mean/max coordination.
* Expanded eight: the six above plus measured box exponent and hull porosity.
* Compare ordinary linear regression, ANN (64,32) and random forest with both
  descriptor sets. Forest leaf size 1/3/5 is selected on internal validation.
* Compare geometry-only GNN q=1–4 on the same cases, with the existing smallest-q
  within .02 internal-validation R² rule. GNN inputs do not include new descriptors.
* Three training seeds: 7,17,27; neural budget 300 epochs, patience 50. ANN and
  GNN use restored best internal-validation checkpoints, not outer-test stopping.
  Linear/forest/ANN/GNN all fit the exact same fitting cases. Feature and target
  scaling use those fitting cases only.
* Report pooled five-fold predictions for balanced development evaluation and
  separate stress-test metrics. Across-seed SD is not a confidence interval.
  Do not select a reported winner and call these data untouched confirmation.

## Running

GitHub Actions: **Measured structure and geometry-based validation** uses the
repository self-hosted runner and downloads the preserved completed artifacts.

    python codes/structure/test_structure.py
    python codes/structure/run.py collect --original ORIGINAL_ARTIFACT --independent SCORED_ARTIFACT --output OUTPUT
    python codes/structure/run.py train --output OUTPUT

Outputs include descriptors.csv, box_counting_curves.csv, audit.md, audit.png,
audit_summary.json, graphs.json, source_hashes.json, splits.csv, fold_summary.csv,
predictions.csv, metrics.csv, selections.csv and results.md.

Source documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
and https://scikit-learn.org/stable/modules/cross_validation.html.
