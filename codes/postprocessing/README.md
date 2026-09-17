# Particle visualization and exposure

`posttovtk.py` is the unchanged converter copied from the LAMFOAM repository
(`tutorials/DEM/hardSphereBox/posttovtk.py`). It writes legacy VTK and/or VTP
referenced by PVD, preserves native dump units, and uses physical ITEM: TIME.
The exact upstream commit is recorded in `UPSTREAM.json`.

`visualize.py` adds particle-level `geometric_exposure` to compact VTP files,
plus a small ID/exposure CSV, without retaining a second enriched raw dump.
See the [DEM guide](../dem/README.md#exposure-in-paraview) for commands.
`test_exposure.py` checks an isolated sphere, the analytic exposure of a tangent
equal-sphere pair, uniform rescaling, and particle-order independence.
