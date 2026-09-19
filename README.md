# Aggregate morphology and graph learning

This project studies how particle arrangement affects aggregate properties and
whether a graph neural network can represent the relevant structure compactly.

**Current step: 180 geometries, bonded LAMMPS verification, and exposure visualization.**
See [the dataset definition](docs/study180.md) and [the DEM guide](codes/dem/README.md).
Start with [the generation guide](codes/generation/README.md).
The committed pilot is in [data/generation/pilot_v1](data/generation/pilot_v1).
Existing paper results have not been recalculated with these new geometries.

## Repository layout

| Location | Purpose |
|---|---|
| `codes/generation/` | New common generation interface, checks and plots |
| `codes/dem/` | Bonded LAMMPS preparation, batch execution and assessment |
| `codes/postprocessing/` | Native dump conversion and exposure-coloured VTP/PVD |
| `data/generation/` | Generated case coordinates, contacts, metadata and summaries |
| `codes/fractalagglomerategenration/` | Original generator and dimension-analysis files, retained for provenance |
| `codes/analysis/` | Uploaded legacy DEM and latent-analysis scripts; incomplete pipeline |
| `docs/` | Development sequence and current limitations |
| `beamer_presentation/` | Existing presentation source and figures |
| `prelim_hbr/` | Existing reviewed paper draft and figures |

The final document layout will have one `beamer/` directory and one `report/`
directory at repository root, outside `codes/`. Consolidation of the existing
LaTeX files is deferred until the modelling review; this generation step does
not change the paper, slides, or their reported results.

## Next stages

1. Inspect the new aggregate families and their measured shape diversity.
2. Prepare LAMMPS cases and decide how geometric contacts become mechanical bonds.
3. Measure changes between generated and mechanically equilibrated geometry.
4. Rebuild graph/exposure preprocessing and compare GNNs with descriptor-based
   neural networks using consistent held-out groups.
5. Consolidate and rewrite the report and slides in plain language.

Missing from the upload: the original hierarchical generator, current GNN
training/graph-building/exposure scripts, and some DEM execution helpers.
The new geometry stage is independent of those missing files.

## Literature motivation

See [geometric exposure and shielding](docs/exposure_shielding_literature.md) for the annotated literature on heat/mass transfer, drag, mobility and breakup, and the distinction between physical shielding and our normal-ray geometric target.
