# Generate diverse aggregates

This is the new geometry-only campaign. It does not reproduce or replace the
135 cases in the existing paper. The older scripts remain in
`codes/fractalagglomerategenration/` for reference.

## Start here

From the repository root, use Python 3.10 or later with NumPy installed:

```bash
python -m pip install -r codes/generation/requirements.txt
python codes/generation/test_generation.py
python codes/generation/generate.py --output data/generation/my_campaign
```

The output directory must not exist. A failed run leaves its partial output for
inspection and does not write a successful summary. Use a new directory to retry.
For a quick check use `--counts 20 --replicates 1`. The default campaign has 75 cases.

## Three growth rules

| Method | How it grows | What is controlled |
|---|---|---|
| `bpca` | A single particle travels on a straight line and sticks at the first collision with the aggregate. | Particle count and random seed |
| `bcca` | Two independently constructed clusters are rotated and brought together along a straight trajectory. | Particle count and random seed |
| `local_filling` | A new particle attaches in a random non-overlapping direction to a particle with the lowest local neighbour count. | Particle count, seed, and neighbourhood radius |

The ballistic methods sample isotropic approach directions and impact offsets
uniformly in the area of a bounding disk. Missed trajectories are rejected.
Collision positions are calculated analytically from all cross-cluster sphere
pairs. No finite step can jump through another particle. There is no rolling,
restructuring, Brownian motion, reaction probability, or force calculation.

For BCCA, recursive splits use floor(N/2) and ceil(N/2). Thus it is equal-mass
hierarchical aggregation for powers of two and a **near-equal-mass variant** for
other counts. It is not the tunable fractal generator from the draft, nor a full
simulation of aggregation kinetics. The local-filling algorithm is a geometric
construction rule, not an aggregation kinetics model. These distinctions matter
when describing what physical mechanisms the dataset represents.

The local-filling implementation follows the idea already present in
`Generation/LocalFillingFactor/urbassek3d.py`, based on Ringl and Urbassek,
[Computer Physics Communications (2013)](https://doi.org/10.1016/j.cpc.2013.02.012).
For background on ballistic cluster aggregation and arbitrary particle counts,
see [Ding et al. (2023)](https://doi.org/10.1063/5.0123360). Our recursive split is
explicitly defined above; it does not reproduce their lowbit construction.

## Campaign design

- Counts: 50, 100, 200 particles.
- Five independent seeds per setting.
- Local neighbourhood radii: 3, 6, 12 particle radii (1.5, 3, 6 diameters).
- Total: 3 counts x 5 seeds x (BPCA + BCCA + 3 local-filling settings) = 75.
- Primary-particle diameter: 1 micrometre; all particles are equal spheres.
- Master seed: 18427. Per-case seeds are derived from SHA-256 of the master seed
  and case name; changing campaign ordering does not change a case.

No fractal dimension is prescribed or inferred from the method name. The detector
radius is a construction parameter, not a guaranteed fractal dimension. These
small aggregates need not show a clear self-similar scaling range. Different
algorithms broaden the geometry sample but do not establish generality by themselves.

Only one physical diameter is used because uniform scaling does not add a new
normalised shape. Changing `--diameter-um` rescales the same geometry. Later, size
can be varied for DEM and transport physics. Scaled versions must remain in the
same training/test group to prevent leakage.

All three rules usually make sparse, tree-like contact networks: new spheres or
clusters stop at a first contact. Different spatial shapes do not guarantee a
broad coordination-number range. Dense/restructured structures and loop-rich
contact networks remain a separate extension after this pilot is inspected.

## Outputs and units

Each case contains:

- `particles.csv`: header `x_m,y_m,z_m,radius_m`, SI metres, centred at the centre
  of mass. Particle IDs are the one-based row numbers after the header.
- `contacts.csv`: one-based `id_i,id_j`; undirected geometric contacts, each once.
  These are not yet mechanical bonds. Bond constitutive properties are chosen
  in the later LAMMPS stage.
- `particles.vtk`: ParaView point data with a `radius` field. Apply Glyph, select
  Sphere (default radius 0.5), scale by `radius` with scale factor 2, and display
  all points. VTK files are regenerated rather than committed.
- `metadata.json`: algorithm, settings, exact seed, and geometric measurements.

The root `manifest.csv` joins all cases. `summary.json` records the generator
source hash, Python/NumPy versions, and completed checks. Measurements include:

- Number of particles and geometric contacts.
- Mean coordination: twice the contact count divided by particle count.
- Radius of gyration divided by particle radius, including each solid sphere's
  internal contribution 3r^2/5.
- Relative shape anisotropy from centre-coordinate covariance eigenvalues:
  0 for an isotropic distribution and 1 for collinear centres.
- Largest overlap divided by particle radius and connectedness.

An independent all-pairs check rejects disconnected structures, non-finite
coordinates, or overlap exceeding 1e-9 particle radii. This tolerance permits
floating-point roundoff at exact contact; it is not a DEM overlap allowance.
No exposure, transport target, fitted fractal dimension, or ML results are
claimed at this step.

## Inspect the pilot

```bash
python codes/generation/plot_campaign.py data/generation/my_campaign
```

This writes `representatives.png` and `shape_summary.png`. The first shows one
100-particle aggregate from each family at equal plotting scale; the second
compares radius of gyration and shape anisotropy across all cases. Plotting needs
Matplotlib; generation itself only needs NumPy.

## Self-hosted runner

`.github/workflows/generate-agglomerates.yml` uses the established
`[self-hosted, linux, x64, lamfoam-local]` labels. It runs the geometric tests,
generates the complete campaign, and uploads the outputs as a workflow artifact.
It uses an isolated temporary virtual environment, without altering the existing
LAMMPS/OpenFOAM environment. Workflow artifacts have limited retention; the
committed pilot CSV files and generation commands remain the durable record.
