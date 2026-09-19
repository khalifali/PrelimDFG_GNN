# Matched-structure development analysis

Existing 359 BPM geometries. No new GNN training. FracVAL generation remains paused.
Matching is target-blind, maximum-cardinality then minimum descriptor distance, and disjoint within each protocol.
Primary tolerances: identical N, |delta Dbox| <= 0.03, |delta hull porosity| <= 0.01.
Pairs match estimated descriptors, not proven identical physical morphology. Quality flags and residual mismatch matter.

| Matching | Pairs | Median exposure gap | 90th percentile gap | Maximum gap |
|---|---:|---:|---:|---:|
| primary | 167 | 0.0033471679687499534 | 0.00973339843750001 | 0.017216796874999996 |
| tight | 147 | 0.002911551339285723 | 0.009359375000000036 | 0.017104492187499987 |
| loose | 168 | 0.0031126302083333 | 0.010508658854166622 | 0.017216796874999996 |
| rg_matched | 167 | 0.0033471679687499534 | 0.00973339843750001 | 0.017216796874999996 |
| quality_ok | 120 | 0.0034577636718750027 | 0.009382763671875016 | 0.017216796874999996 |

## Can additional descriptors predict paired exposure differences?

Nested pair-level cross-validation of signed differences; no pair member appears in another fold. Basic includes residual Dbox, porosity and log Rg differences. Expanded adds anisotropy, branching and local-density descriptors.
This is a small exploratory linear comparison, not a definitive test of nonlinear sufficiency or GNN advantage.

| Model | Held-out difference RMSE | MAE |
|---|---:|---:|
| zero | 0.005977 | 0.004502 |
| basic | 0.005899 | 0.004418 |
| expanded | 0.004585 | 0.003478 |

## Numerical check

Diagnostic pairs: three largest observed gaps and three closest descriptor matches. This selection is disclosed and is not a random error sample.
Ray checks use archived float32 graph coordinates. Agreement with archived 2048-ray labels is checked before interpretation.
Observed variation across ray counts/orientation is a sensitivity diagnostic, not a rigorous error bound.

| Pair | 16384-ray absolute gap | Sum of observed numerical spreads | Reproduces archive |
|---|---:|---:|---|
| fractal_dp1_N0100_Df1.8_kf1.3_rep02 / fractal_dp2_N0100_Df1.8_kf1.3_rep03 | 0.017083 | 0.000532 | True |
| fractal_dp2_N0100_Df1.8_kf1.3_rep02 / fractal_dp2_N0100_Df1.8_kf1.3_rep05 | 0.016789 | 0.000638 | True |
| fractal_dp1_N0050_Df2.2_kf1.1_rep04 / fractal_dp2_N0050_Df2.2_kf1.1_rep01 | 0.016433 | 0.000991 | True |
| independent_extrapolation_dp1.5_N0750_Df1.8_kf1.3_rep03 / independent_extrapolation_dp1_N0750_Df1.8_kf1.3_rep04 | 0.002488 | 0.000207 | True |
| independent_extrapolation_dp2_N0750_Df1.8_kf1.3_rep01 / independent_extrapolation_dp2_N0750_Df1.8_kf1.3_rep05 | 0.003309 | 0.000486 | True |
| independent_extrapolation_dp1.5_N1000_Df1.8_kf1.3_rep04 / independent_extrapolation_dp1_N1000_Df1.8_kf1.3_rep01 | 0.005243 | 0.000291 | True |

Raw signed/absolute pair differences, all features, model predictions and exploratory associations are retained as CSV. Associations alone are not causal explanations.
