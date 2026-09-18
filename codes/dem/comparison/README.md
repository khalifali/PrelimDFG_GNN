# Original case001: BPM versus LAMFOAM vdW

Purpose: isolate mechanical-model and initial-bond-selection effects on one
original N=50, diameter=1 micrometre geometry. This does not change ML datasets.
The supplied particles.sys is the initial state. Its positions and radii are
exactly identical to the CSV described as post-relaxation. Preserve both; this
pair cannot establish historical movement or historical contact interactions.

All arms start from identical positions, zero velocities/spins, periodic
[-100,100] micrometre box, density 1050 kg/m3, E=1e9 Pa, nu=.45, restitution .2,
friction .3, no artificial fluid drag. Duration 15 microseconds, dt=5e-11 s.
These shared material values and duration come from the uploaded old input.
Three particle snapshots per arm; all include Fibonacci exposure for ParaView.

- bpm_strict: permanent rotational BPM bonds on gaps <=1e-6 diameter (3 bonds).
- bpm_near: explicit alternative bond assignment at gaps <=.005 diameter (49
  bonds). Reference bond lengths retain these gaps; coordinates are not moved.
- vdw: native LAMMPS Hertz/Mindlin plus the **unmodified LAMFOAM lamfoam/vdw**
  pair implementation; no permanent bonds. A=3.028e-20 J, h0=4e-10 m.
- vdw_half_dt: same physical duration at half dt for numerical sensitivity.

BPM spring and damping parameters match the active bonded campaign, but this
comparison starts at rest and omits that campaign's imposed kick/artificial drag.
Bonded pairs are excluded from Hertz interactions in BPM, as in the campaign.
The vdW force has shifted positive gap (h+h0), constant contact adhesion and
cutoff h=.2 reduced radius, as documented in LAMFOAM. It is not a complete
reproduction of LIGGGHTS: tangential history differs and old rolling/torsion
resistance is absent. No equilibrium or historical-score reproduction is assumed.

Workflow: **Original case BPM-vdW comparison**, self-hosted. It installs pinned
LAMMPS, compiles a minimal registration wrapper against its matching source
headers and the pinned LAMFOAM pair implementation, then runs all arms.
The force code is fetched from LAMFOAM commit
f61d4f06d0a5f57e7e8f962420cb86a42bdf71ba, not reimplemented here.

Outputs: results.md, comparison.csv, provenance.json, source references, input
scripts, logs, restart files, metrics.json and post/particles.pvd with exposure.
Report contact graph and near-neighbour graph separately. Check KE and changes
between midpoint and end; 15 microseconds is a comparison duration, not a claim
of equilibrium. One case cannot explain the whole ML accuracy difference.
