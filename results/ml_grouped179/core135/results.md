# Held-out model comparison

Each agglomerate is evaluated only when its group is withheld. Each seed has a complete set of held-out predictions. The table reports mean ± sample standard deviation across seeds; this is not a confidence interval.

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| ann | 0.0628 ± 0.0043 | 0.0461 ± 0.0035 | 0.2338 ± 0.1058 |
| ann_q1 | 0.0661 ± 0.0049 | 0.0476 ± 0.0015 | 0.1494 ± 0.1286 |
| ann_q2 | 0.0679 ± 0.0016 | 0.0492 ± 0.0029 | 0.1056 ± 0.0434 |
| ann_q3 | 0.0613 ± 0.0036 | 0.0450 ± 0.0006 | 0.2705 ± 0.0866 |
| ann_q4 | 0.0507 ± 0.0194 | 0.0398 ± 0.0132 | 0.4527 ± 0.3441 |
| ann_selected | 0.0591 ± 0.0022 | 0.0426 ± 0.0041 | 0.3211 ± 0.0510 |
| count_categorical | 0.0890 ± 0.0000 | 0.0738 ± 0.0000 | -0.5350 ± 0.0000 |
| count_quadratic | 0.0890 ± 0.0000 | 0.0738 ± 0.0000 | -0.5350 ± 0.0000 |
| gnn_q1 | 0.0414 ± 0.0111 | 0.0301 ± 0.0068 | 0.6518 ± 0.1715 |
| gnn_q2 | 0.0310 ± 0.0070 | 0.0268 ± 0.0065 | 0.8073 ± 0.0797 |
| gnn_q3 | 0.0431 ± 0.0068 | 0.0336 ± 0.0035 | 0.6342 ± 0.1167 |
| gnn_q4 | 0.0434 ± 0.0189 | 0.0349 ± 0.0137 | 0.5879 ± 0.3488 |
| gnn_selected | 0.0324 ± 0.0090 | 0.0280 ± 0.0086 | 0.7864 ± 0.1074 |
| individual_forest_N | 0.0890 ± 0.0000 | 0.0738 ± 0.0001 | -0.5364 ± 0.0007 |
| individual_forest_Nc | 0.0890 ± 0.0000 | 0.0738 ± 0.0001 | -0.5364 ± 0.0007 |
| individual_forest_Rg_over_d | 0.1162 ± 0.0004 | 0.1002 ± 0.0005 | -1.6171 ± 0.0196 |
| individual_forest_contact_density | 0.0890 ± 0.0000 | 0.0738 ± 0.0001 | -0.5364 ± 0.0007 |
| individual_forest_max_coordination | 0.0698 ± 0.0001 | 0.0576 ± 0.0001 | 0.0555 ± 0.0014 |
| individual_forest_mean_coordination | 0.0890 ± 0.0000 | 0.0738 ± 0.0001 | -0.5364 ± 0.0007 |
| individual_linear_N | 0.0749 ± 0.0000 | 0.0598 ± 0.0000 | -0.0892 ± 0.0000 |
| individual_linear_Nc | 0.0749 ± 0.0000 | 0.0598 ± 0.0000 | -0.0892 ± 0.0000 |
| individual_linear_Rg_over_d | 0.0834 ± 0.0000 | 0.0723 ± 0.0000 | -0.3494 ± 0.0000 |
| individual_linear_contact_density | 0.0696 ± 0.0000 | 0.0593 ± 0.0000 | 0.0595 ± 0.0000 |
| individual_linear_max_coordination | 0.0690 ± 0.0000 | 0.0576 ± 0.0000 | 0.0771 ± 0.0000 |
| individual_linear_mean_coordination | 0.0696 ± 0.0000 | 0.0593 ± 0.0000 | 0.0595 ± 0.0000 |
| linear | 0.0308 ± 0.0000 | 0.0248 ± 0.0000 | 0.8161 ± 0.0000 |
| random_forest | 0.0801 ± 0.0018 | 0.0763 ± 0.0015 | -0.2456 ± 0.0555 |
| random_forest_v9 | 0.0818 ± 0.0007 | 0.0793 ± 0.0007 | -0.2993 ± 0.0234 |
| training_mean | 0.0767 ± 0.0000 | 0.0635 ± 0.0000 | -0.1413 ± 0.0000 |

The six descriptors are particle count, radius of gyration divided by diameter, contact count, contact density, mean coordination and maximum coordination. Some are mathematically dependent; the linear baseline uses a least-squares pseudoinverse.

A latent vector contains learned artificial structural parameters that replace the traditional descriptors at the prediction head. Its axes differ between separately trained models, so axes must not be pooled across folds.

GNN dimension selection and stopping use inner validation only. Fixed-q rows are prespecified comparisons, not a basis for choosing a winner on the test set. Predictions are not clipped to [0,1].

See provenance.json, splits.csv, selections.csv and fold_metrics.csv for provenance and between-fold variability. Historical paper scores are not substituted for these results.
