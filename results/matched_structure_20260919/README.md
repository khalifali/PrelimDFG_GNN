# Matched-structure analysis — 19 September 2026

[Full report](report.md) | [Run](https://github.com/khalifali/PrelimDFG_GNN/actions/runs/35426026371)

Analysis code and matching protocol: commit 8809f0c533feb91de086692540620d6e9408a8e8. Source: measured-structure run 35346823376, artifact 10550461056. This directory preserves the final output of artifact 10579180777. FracVAL is paused; all geometries are existing BPM development data.

## Findings

167 disjoint pairs (334 of 359 geometries) match particle count exactly, box-counting dimension within 0.03 and particle-body convex-hull porosity within 0.01. Median absolute exposure difference is 0.00335 on the 0–1 scale, with a maximum of 0.01722. Tight matching retains 147 pairs with median 0.00291. Restricting to unflagged box estimates retains 120 pairs with median 0.00346.

Expanded anisotropy, branching and local-density descriptors reduce nested pair-level held-out difference RMSE from 0.005899 to 0.004585 (22.3%) relative to residual box dimension, porosity and log radius of gyration. This exploratory linear result supports investigating additional geometric information; it does not demonstrate GNN superiority, external generalization, or physical transport effectiveness.

All six diagnostic pairs reproduce the archived 2048-ray labels, retain the sign of their differences across the tested settings, and have a 16384-ray gap exceeding the summed observed within-case numerical spreads. Settings: 2048, 8192 and 16384 rays, plus 8192 rays after a fixed rotation. The largest-gap N=100 pair has a refined gap of 0.017083 versus summed spread 0.000532. The selected N=50 pair has flagged box-counting estimates and should not serve as the primary example.

These six pairs comprise three largest observed gaps and three closest descriptor matches, not a random numerical validation sample. Observed numerical spread is not an error bound. Matching is approximate and relies on finite-scale estimated descriptors; residual mismatch and descriptor uncertainty remain. All 359 geometries have previously been examined and are development data. Sensitivity protocols reuse cases and are not independent replications.

## Next scientific comparison

Compare a GNN against a strong expanded-descriptor baseline using identical frozen splits and selection rules. Any final confirmation must use genuinely fresh agglomerates. The present result motivates that comparison but does not establish that a graph model is necessary.
