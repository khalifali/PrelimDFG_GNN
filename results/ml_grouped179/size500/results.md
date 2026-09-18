# Held-out model comparison

Each agglomerate is evaluated only when its group is withheld. Each seed has a complete set of held-out predictions. The table reports mean ± sample standard deviation across seeds; this is not a confidence interval.

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| ann | 0.0720 ± 0.0130 | 0.0647 ± 0.0110 | 0.3018 ± 0.2580 |
| ann_q1 | 0.1020 ± 0.0644 | 0.0917 ± 0.0578 | -0.7351 ± 1.4156 |
| ann_q2 | 0.0855 ± 0.0562 | 0.0791 ± 0.0512 | -0.2399 ± 1.3849 |
| ann_q3 | 0.0831 ± 0.0230 | 0.0715 ± 0.0146 | 0.0437 ± 0.5439 |
| ann_q4 | 0.0424 ± 0.0143 | 0.0353 ± 0.0114 | 0.7452 ± 0.1693 |
| ann_selected | 0.0946 ± 0.0380 | 0.0890 ± 0.0322 | -0.3075 ± 1.0540 |
| count_categorical | 0.1351 ± 0.0000 | 0.1084 ± 0.0000 | -1.4082 ± 0.0000 |
| count_quadratic | 0.3392 ± 0.0000 | 0.3278 ± 0.0000 | -14.1711 ± 0.0000 |
| gnn_q1 | 0.0557 ± 0.0083 | 0.0510 ± 0.0101 | 0.5844 ± 0.1275 |
| gnn_q2 | 0.0522 ± 0.0256 | 0.0490 ± 0.0252 | 0.5840 ± 0.3051 |
| gnn_q3 | 0.0477 ± 0.0191 | 0.0443 ± 0.0214 | 0.6686 ± 0.2673 |
| gnn_q4 | 0.0217 ± 0.0065 | 0.0169 ± 0.0067 | 0.9342 ± 0.0340 |
| gnn_selected | 0.0456 ± 0.0209 | 0.0404 ± 0.0192 | 0.6876 ± 0.2349 |
| individual_forest_N | 0.1028 ± 0.0002 | 0.0915 ± 0.0001 | -0.3940 ± 0.0046 |
| individual_forest_Nc | 0.1028 ± 0.0002 | 0.0915 ± 0.0001 | -0.3940 ± 0.0046 |
| individual_forest_Rg_over_d | 0.1811 ± 0.0003 | 0.1546 ± 0.0002 | -3.3263 ± 0.0148 |
| individual_forest_contact_density | 0.1028 ± 0.0002 | 0.0915 ± 0.0001 | -0.3940 ± 0.0046 |
| individual_forest_max_coordination | 0.1103 ± 0.0002 | 0.0941 ± 0.0002 | -0.6042 ± 0.0051 |
| individual_forest_mean_coordination | 0.1028 ± 0.0002 | 0.0915 ± 0.0001 | -0.3940 ± 0.0046 |
| individual_linear_N | 0.1621 ± 0.0000 | 0.1368 ± 0.0000 | -2.4664 ± 0.0000 |
| individual_linear_Nc | 0.1621 ± 0.0000 | 0.1368 ± 0.0000 | -2.4664 ± 0.0000 |
| individual_linear_Rg_over_d | 0.1527 ± 0.0000 | 0.1385 ± 0.0000 | -2.0726 ± 0.0000 |
| individual_linear_contact_density | 0.0969 ± 0.0000 | 0.0874 ± 0.0000 | -0.2372 ± 0.0000 |
| individual_linear_max_coordination | 0.1063 ± 0.0000 | 0.0895 ± 0.0000 | -0.4891 ± 0.0000 |
| individual_linear_mean_coordination | 0.0969 ± 0.0000 | 0.0874 ± 0.0000 | -0.2372 ± 0.0000 |
| linear | 0.1207 ± 0.0000 | 0.1011 ± 0.0000 | -0.9221 ± 0.0000 |
| random_forest | 0.1536 ± 0.0036 | 0.1339 ± 0.0029 | -2.1107 ± 0.1461 |
| random_forest_v9 | 0.1561 ± 0.0013 | 0.1360 ± 0.0008 | -2.2139 ± 0.0522 |
| training_mean | 0.1352 ± 0.0000 | 0.1084 ± 0.0000 | -1.4084 ± 0.0000 |

The six descriptors are particle count, radius of gyration divided by diameter, contact count, contact density, mean coordination and maximum coordination. Some are mathematically dependent; the linear baseline uses a least-squares pseudoinverse.

A latent vector contains learned artificial structural parameters that replace the traditional descriptors at the prediction head. Its axes differ between separately trained models, so axes must not be pooled across folds.

GNN dimension selection and stopping use inner validation only. Fixed-q rows are prespecified comparisons, not a basis for choosing a winner on the test set. Predictions are not clipped to [0,1].

See provenance.json, splits.csv, selections.csv and fold_metrics.csv for provenance and between-fold variability. Historical paper scores are not substituted for these results.
