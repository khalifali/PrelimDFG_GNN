# Bonded-particle DEM with LAMMPS

## Purpose and scope

This stage attaches permanent mechanical bonds to the initial geometric contacts.
It checks the response to a small velocity disturbance and records any change in
shape. It is **not** a calibrated prediction of a particular biological material,
a compression procedure, or evidence that bonds would survive arbitrary loads.
Bonds are deliberately nonbreaking, so their survival alone is not validation.

Bonds start from the generated geometry as their stress-free reference. Without
an applied disturbance the particles would have almost no reason to move. Each
case therefore starts with small reproducible translational and angular velocities,
with total linear and angular momentum removed. Numerical drag drains motion.
This verifies the implemented response around the chosen reference configuration;
it does not independently prove the physical feasibility of the generated shape.

## Model

Native LAMMPS `atom_style bpm/sphere`, `bond_style bpm/rotational` and
`fix nve/bpm/sphere` are used. The BPM, GRANULAR and EXTRA-FIX packages are needed.
The bonds resist extension, shear, twist and bending. A simple central spring alone
would leave branches free to turn at their joints, which is why rotational bonds
are used here. Bond breaking and vdW forces are disabled. Hertz-Mindlin repulsive
contacts act between unbonded neighbours; bonded pairs are not double counted.
No permanent bonds are created later when new contacts occur.

The reference test environment is **LAMMPS 22 July 2025, Update 4**, distributed as
`lammps==2025.7.22.4.0`. Newer LAMMPS changed BPM defaults in July 2026.
`run.py` explicitly selects `frame particle damping dem` on those versions to
retain the legacy model. Both rotational damping coefficients are equal here.
The new-version compatibility branch is documented but not verified by the
local stable-version runs. Do not silently compare campaigns with different BPM
frames or damping definitions.

## Parameters (illustrative, not fitted)

| Parameter | SI value | Role |
|---|---:|---|
| Particle density | 1050 kg/m3 | Translational mass and rotational inertia |
| Bond normal/shear stiffness | 10 / 4 N/m | Resistance to extension and shear |
| Bond twist/bend stiffness | 1e-13 / 1e-13 N m | Resistance to relative rotation |
| Bond normal/shear damping | 1e-8 / 1e-8 kg/s | Dissipation of relative motion |
| Bond twist/bend damping | 1e-22 / 1e-22 kg m2/s | Dissipation of relative rotation |
| Contact Young modulus | 1e9 Pa | Unbonded repulsive contact |
| Poisson ratio | 0.3 | Contact elasticity |
| Restitution / friction | 0.5 / 0.3 | Unbonded contact dissipation and sliding |
| Numerical translational drag | 1e-8 kg/s | Energy removal, not calibrated fluid drag |
| Numerical rotational drag | 1e-22 kg m2/s | Rotational energy removal |
| Initial random velocity scale | 1e-3 m/s per component before momentum correction | Small disturbance |
| Initial random angular velocity scale | 1000 rad/s per component before momentum correction | Small rotational disturbance |
| Time step | 1e-10 s | Integration step |
| Duration | 20000 steps = 2 microseconds | Fixed verification interval |

All parameters are recorded in each `case.json`. Input and dump files use MICRO:
micrometres, microseconds, picograms, nanonewtons and femtojoules. The generated
geometry CSVs and `final_particles.csv` use SI metres. `posttovtk.py` does not
convert units; ParaView coordinates from the dumps therefore remain micrometres.

## Run every case

From the repository root:

```bash
python codes/generation/build_study.py --output data/generation/study180_new
python codes/dem/prepare.py data/generation/study180_new --output data/dem/study180_new
python codes/dem/run.py data/dem/study180_new --lammps /absolute/path/to/lmp --workers 2
python codes/postprocessing/visualize.py data/dem/study180_new
```

Preparation and execution refuse to overwrite earlier results. Each case can also
be run from its own directory with `lmp -in in.lammps`. For LAMMPS >=4 July 2026,
use the campaign runner so the model compatibility options are selected explicitly.
No LAMFOAM plugin or OpenFOAM installation is required for this standalone step.
`--no-kick` prepares a stationary control. It is not a meaningful substitute for
the disturbed verification run.

## Only three time snapshots

Each case writes **initial, midpoint and final** snapshots at 0, 1 and 2 us.
Particle files are `post/post_0.txt`, `post/post_10000.txt`, `post/post_20000.txt`.
The three bond files use the same step numbers. Increasing the number of steps
still writes only three snapshots (an even number of steps is required).
`generated/post_0.txt` is a separate geometry-only input preview, not an additional
DEM time; its positions are the same as `post/post_0.txt`.

Particle dumps contain ID, type, xyz, velocity, angular velocity, radius and total
force. Bond dumps contain the two IDs, separation, normal force and total force
vector. `final.restart` is the binary checkpoint that preserves bond reference
state. `final.data` does **not** preserve this state and must not be used for a
mechanically continuous restart.

## Exposure in ParaView

Open `<case>/post/particles.pvd`, apply a Sphere Glyph, choose `radius` as scale
array, use scale factor 2 for the default sphere source radius 0.5, and show all
points. Colour by **`geometric_exposure`** with a fixed range 0 to 1. Each VTP
contains the matching geometry and exposure, so there is no table join.

```bash
# Default: compact VTP files + one PVD time series per aggregate.
python codes/postprocessing/visualize.py data/dem/study180_new
# Add legacy .vtk files while retaining the VTP/PVD series.
python codes/postprocessing/visualize.py data/dem/study180_new --format both
# Convert raw DEM fields without calculating exposure, using the existing converter.
python codes/postprocessing/posttovtk.py --particles both --domain none CASE/post/post_0.txt
```

The enriched VTP keeps only particle ID, type, radius and geometric exposure as
point arrays. Position is stored once as geometry. Original velocities and forces
remain available in the raw dumps. A small `post/exposure/exposure_<step>.csv`
contains ID and exposure for later analysis; the summary records mean exposure,
ray count, physical time and a hash of the source dump. No duplicate enriched
LAMMPS dump is retained.

Exposure is calculated separately for all three snapshots with 512 approximately
uniform Fibonacci directions. Each ray begins just outside a surface point and
follows its outward normal; another sphere blocking that ray makes the direction
inaccessible. The fraction of unblocked directions is geometric exposure.
This is a line-of-sight measure, not a diffusion or oxygen-uptake calculation.
`--rays 2048` allows a refinement check. Very small apparent time changes can be
ray discretization effects and should not be interpreted as physical transport.

## Assessment

`dem_summary.csv` reports the initial/final translational-plus-rotational kinetic
energy, energy ratio, RMS displacement after rigid alignment, change in radius
of gyration, maximum overlap, and bond counts. A completed run passes the chosen
numerical checks when all output is finite, particle/bond identities are retained,
maximum overlap is below 0.1% of diameter, aligned RMS displacement is below 1%
of diameter, and final kinetic energy is below 0.1% of its initial value.
These are stated numerical acceptance thresholds, not material validation.

Official references:
[bpm/rotational](https://docs.lammps.org/bond_bpm_rotational.html),
[nve/bpm/sphere](https://docs.lammps.org/fix_nve_bpm_sphere.html),
[viscous/sphere](https://docs.lammps.org/fix_viscous_sphere.html).
