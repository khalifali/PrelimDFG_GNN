# Existing BPM database: measured coverage audit

Source: committed final-geometry structural_features.csv from the 359-case development dataset. No exposure labels enter this audit. All cases are previously inspected development data.

| N | Cases | Box-quality flags | Measured Dbox range | Hull porosity range |
|---:|---:|---:|---|---|
| 50 | 45 | 43 | 1.7378–2.1208 | 0.7293–0.8901 |
| 75 | 30 | 23 | 1.8543–2.1095 | 0.7916–0.8822 |
| 100 | 44 | 23 | 1.7927–2.1386 | 0.7749–0.9236 |
| 150 | 30 | 4 | 1.8954–2.1387 | 0.8253–0.9194 |
| 200 | 45 | 5 | 1.7600–2.2146 | 0.8073–0.9539 |
| 350 | 30 | 0 | 1.9168–2.1646 | 0.8630–0.9437 |
| 500 | 45 | 0 | 1.7919–2.2903 | 0.8411–0.9742 |
| 750 | 45 | 0 | 1.7839–2.2902 | 0.8508–0.9797 |
| 1000 | 45 | 0 | 1.7875–2.3103 | 0.8602–0.9833 |

## Findings and generation priorities

98 of 359 estimates have box-quality flags. 89 flags occur at N=50,75,100. Do not treat apparent low-N dimension differences as well-resolved fractal regimes. Retain these geometries but carry their flags into coverage and reporting.

Measured Dbox spans about 1.74–2.31, not the requested-generation exponent range. No current case reaches Dbox 2.5 or higher or 1.5. These gaps do not prove a generator cannot make compact/open structures: finite-size and estimator effects must be separated from geometry.

Coverage depends strongly on size: for N>=350 all existing porosities exceed 0.84, while N=50 includes porosities near 0.73. Prioritize compact, lower-porosity candidates at medium/large N; more open candidates at small/medium N; and additional dimension variation within size classes where the estimator is reliable.

joint_coverage_cells.csv records both empty and occupied fixed descriptor bins by N, including unflagged counts. These bins are descriptive, not split groups, physical feasibility claims, or a demand for a rectangular grid. coverage_by_size.csv additionally records anisotropy, branching and local-density ranges.

Next gate: a small generator-feasibility pilot using the current generator, with reference geometries to test the box estimator. Classify final candidates using BPM-relaxed measured geometry. Increase independent seeds only after candidate configurations demonstrate new reliable coverage. FracVAL stays paused.

Do not set a final database size yet. Freeze accepted coverage and the validation/test construction protocol before generating reserved fresh tests. No matched-pair splits will be used in the restored study.
