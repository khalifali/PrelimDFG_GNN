# Active study: tunable-fractal core and N=500 extension

The user stopped the mixed-generator training run (35282022748) and requested
only the historical generation mechanism. No other generation algorithm enters
the active study.

## Design

- Diameter: 1.0, 1.5 and 2.0 micrometres.
- Particle count: 50, 100, 200 and **500**.
- Prescribed (Df,kf): (1.8,1.3), (2.2,1.1), (2.6,0.8), from the old report's
  design table. The intermediate prefactor is **1.1**, not 1.0.
- Five independent realizations per combination, with unique recorded seeds.
- 135 cases matching the historical parameter grid, plus 45 N=500 cases:
  **180 agglomerates total**. These are freshly generated independent structures,
  not copies of historical coordinates or scaled duplicates.
- LAMMPS bonded-particle DEM remains the mechanical check. No legacy LIGGGHTS
  model is reinstated. Geometry validation is complete; the active ML workflow first reruns bonded DEM checks.

## Algorithm

The reconstruction follows the tunable particle/cluster mass-radius formulation
in the old report, associated with Filippov et al. (2000) and Skorupski et al.
(2014), https://doi.org/10.1016/j.physa.2014.02.072 . It is not a recovered copy
of their implementation or the missing historical generator.

All internal lengths are in primary-particle radii. Start with two touching
spheres. Build 25-particle seeds by attaching particles subject to the prescribed
mass-radius relation N=kf*(Rg/r)^Df. For every addition after the initial pair,
and every subsequent cluster merge, compute the required center separation from

R12² = N²/(N1*N2) * [Rg_target² - (N1/N)*Rg1² - (N2/N)*Rg2²].

Rg² includes the individual solid sphere contribution 3r²/5. The initial dimer
is the exact touching pair and is not forced to fit an incompatible two-particle
mass-radius target. Accepted clusters from N=3 onward satisfy the relation.

For each randomly rotated pair of clusters, candidate translations lie on the
intersection of the prescribed center-separation sphere and a radius-2r contact
sphere for a particle pair. Sample the resulting circle, reject every placement
with any overlap, and accept contact-connected geometry. A KD tree accelerates
the all-pair exclusion query; independent direct all-pair checks validate each
accepted structure. Exhausted searches fail explicitly or regenerate subclusters,
never substitute a different generator or rescale overlapping particles.

Hierarchical merging uses equal-sized subclusters for N=50,100,200. For N=500,
20 seeds of 25 particles are merged by recursive splits as equal as possible in
numbers of seed clusters; some merges are necessarily unequal (e.g. 50+75).
All merges use actual subcluster moments and the same mass-radius constraint.

## Reproduce and validate

```bash
python -m pip install numpy==2.2.6 scipy==1.15.3
python codes/generation/test_tunable.py
python codes/generation/build_fractal_study.py --output data/generation/fractal180
```

Outputs: particles.csv in SI units, contacts.csv with one-based IDs,
particles.vtk, metadata.json per case; manifest.csv and summary.json at campaign
level. Existing output directories are never silently overwritten.

Acceptance checks: exact requested N and monodispersity; connected contact graph;
maximum overlap below 1e-9 particle radii; relative mass-radius error below 1e-9;
finite centered coordinates; deterministic regeneration. Inspect representative
three-dimensional particle images for all size/morphology combinations before
training. A prescribed mass-radius relation does not prove self-similarity or
physical representativeness; use structural distributions and morphology checks
as well. Do not claim that changing the generator guarantees the previous R².

Keep all realizations and diameters at each N/Df/kf in one ML group. First compare
the 135-case core, then the extended dataset and an N=500 extrapolation test.
This separates the effects of recovering the historical design and adding size.

## Later extensions

No additional mechanisms are currently approved for the active dataset. Before
adding another generator, validate it independently against its intended growth
rule, numerical geometry checks, representative images and appropriate structural
statistics. Briesen mentioned diffusion/reaction-limited, cluster-cluster,
kinetic and ballistic aggregation; he did not prescribe a single replacement.
His named HEALPix recommendation concerns spherical exposure sampling, not
agglomerate generation.

Within the same mechanism, future useful extensions would independently vary Df
and kf (the original three settings confound them) and add realizations to quantify
sampling variability. No such extra cases are silently included in this 180-case
campaign. Feasibility must be checked for each new Df/kf combination.

## Current generated files

The recovered local `data/generation/fractal180_v2.zip` contains all 180
coordinate/contact/metadata sets. This archive is not published in Git. The runner
reproduces them using the recorded seeds and published generator. The directory name v2 identifies the completed construction;
an interrupted local development attempt was not published. VTK files can be
regenerated from coordinates and are omitted from the compact archive.

All 180 completed cases passed independent checks. The maximum relative error
in the prescribed mass-radius relation was 8.89e-16 (rounded upward). Three
unit tests cover all 15 combinations of seed/output counts and morphology,
repeatability and unsupported counts. Twelve representative projections (one
realization per size/morphology at 1 micrometre) were inspected. These confirm
qualitative open/intermediate/compact differences but are not physical validation
of an aggregation mechanism or evidence of a particular eventual GNN accuracy.

The 180 bonded inputs have been prepared locally. No new DEM completion or final training result is claimed at launch. The old training remains cancelled. The active workflow now reproduces this
design and runs bonded checks before core135, extended180 and size500 ML. The manuscript's old mixed-generator numerical results
must be replaced before the next report is presented as the current study.

Fibonacci sampling remains the active exposure method, following the user's
latest decision. Preserve its Gonzalez (2010) citation and original ray-tracing
TikZ schematic. HEALPix is an alternative, not the current target definition.
