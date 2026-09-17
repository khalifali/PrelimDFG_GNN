# Current 180-case dataset

The original 75-case pilot remains a separate exploratory dataset. The active
study contains **180 newly generated, independent geometries**:

3 methods x 4 particle counts x 15 realizations = 180.

The methods are ballistic particle-cluster, balanced ballistic cluster-cluster,
and local-filling growth. Counts are 50, 100, 200 and 500. Particle diameter is
1 micrometre in all cases. Local filling uses detector radii of 3, 6 and 12
particle radii, with five independent seeds each. BPCA and BCCA each use fifteen
independent seeds per particle count.

## Meaning of open, intermediate and compact

After generation, measure `compactness_score = N^(1/3)/(Rg/r)` (higher is more
compact). Rg includes the finite sphere contribution 3r^2/5. Within each method
and particle count, sort the fifteen cases and label the lowest five open,
the middle five intermediate, and the highest five compact. Store both the
label and the continuous measurement in every metadata file.

This gives five cases in every method/size/class cell, but the labels are
**relative within that cell's method and size**. They are not a common porosity
threshold, a guarantee of dense packing, or imposed fractal dimensions. There
may be only a narrow compactness range within some ballistic families. The
local-filling detector parameter deliberately broadens the generated range;
the resulting class is assigned from measured geometry, not from detector radius
alone. Across methods, compare the continuous compactness score, not just labels.

This distinction must appear in the eventual paper. If universally matched
open/intermediate/dense classes are required, an additional constrained generation
or restructuring stage will be necessary. Bonding the initial contacts does not
supply that stage or make an open structure dense.

All algorithms grow tree-like networks at first contact. The dataset spans
spatial shapes but not a wide distribution of contact coordination or loops.
It therefore remains a restricted morphology ensemble, not all possible aggregates.

Structural labels are descriptive and should not be supplied as ML input by
default. The exact coordinates and contact graph define the geometry. Each
case's three time snapshots and any future scaled copies must stay in the same
train/test group. Hold out a generation method and/or N=500 separately to assess
stronger generalization than a random split of snapshots.

## Reproduce

```bash
python codes/generation/build_study.py --output data/generation/study180_new
```

The previous draft's 135 cases are not part of this count, and its ML metrics do
not apply to these new geometries. DEM uses the full 180-case manifest; see
[codes/dem/README.md](../codes/dem/README.md).
