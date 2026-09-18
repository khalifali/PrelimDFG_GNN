# Held-out model comparison

Each agglomerate is evaluated only when its group is withheld. Each seed has a complete set of held-out predictions. The table reports mean ± sample standard deviation across seeds; this is not a confidence interval.

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| ann | 0.0458 ± 0.0038 | 0.0385 ± 0.0026 | 0.7296 ± 0.0454 |
| ann_q1 | 0.0526 ± 0.0077 | 0.0442 ± 0.0054 | 0.6386 ± 0.1030 |
| ann_q2 | 0.0465 ± 0.0078 | 0.0395 ± 0.0050 | 0.7167 ± 0.0969 |
| ann_q3 | 0.0424 ± 0.0049 | 0.0365 ± 0.0035 | 0.7668 ± 0.0528 |
| ann_q4 | 0.0480 ± 0.0055 | 0.0402 ± 0.0038 | 0.7014 ± 0.0678 |
| ann_selected | 0.0483 ± 0.0083 | 0.0403 ± 0.0058 | 0.6939 ± 0.1060 |
| count_categorical | 0.1044 ± 0.0000 | 0.0839 ± 0.0000 | -0.4019 ± 0.0000 |
| count_quadratic | 0.1023 ± 0.0000 | 0.0812 ± 0.0000 | -0.3445 ± 0.0000 |
| gnn_q1 | 0.0414 ± 0.0096 | 0.0347 ± 0.0068 | 0.7718 ± 0.1066 |
| gnn_q2 | 0.0364 ± 0.0017 | 0.0307 ± 0.0020 | 0.8298 ± 0.0154 |
| gnn_q3 | 0.0323 ± 0.0053 | 0.0266 ± 0.0032 | 0.8637 ± 0.0446 |
| gnn_q4 | 0.0378 ± 0.0024 | 0.0325 ± 0.0027 | 0.8158 ± 0.0226 |
| gnn_selected | 0.0336 ± 0.0036 | 0.0284 ± 0.0019 | 0.8536 ± 0.0321 |
| individual_forest_N | 0.1044 ± 0.0001 | 0.0840 ± 0.0000 | -0.4021 ± 0.0019 |
| individual_forest_Nc | 0.1044 ± 0.0001 | 0.0840 ± 0.0000 | -0.4021 ± 0.0019 |
| individual_forest_Rg_over_d | 0.1397 ± 0.0007 | 0.1235 ± 0.0005 | -1.5103 ± 0.0261 |
| individual_forest_contact_density | 0.1044 ± 0.0001 | 0.0840 ± 0.0000 | -0.4021 ± 0.0019 |
| individual_forest_max_coordination | 0.0834 ± 0.0001 | 0.0670 ± 0.0001 | 0.1060 ± 0.0018 |
| individual_forest_mean_coordination | 0.1044 ± 0.0001 | 0.0840 ± 0.0000 | -0.4021 ± 0.0019 |
| individual_linear_N | 0.0932 ± 0.0000 | 0.0766 ± 0.0000 | -0.1165 ± 0.0000 |
| individual_linear_Nc | 0.0932 ± 0.0000 | 0.0766 ± 0.0000 | -0.1165 ± 0.0000 |
| individual_linear_Rg_over_d | 0.1027 ± 0.0000 | 0.0897 ± 0.0000 | -0.3562 ± 0.0000 |
| individual_linear_contact_density | 0.0771 ± 0.0000 | 0.0647 ± 0.0000 | 0.2349 ± 0.0000 |
| individual_linear_max_coordination | 0.0805 ± 0.0000 | 0.0655 ± 0.0000 | 0.1677 ± 0.0000 |
| individual_linear_mean_coordination | 0.0771 ± 0.0000 | 0.0647 ± 0.0000 | 0.2349 ± 0.0000 |
| linear | 0.0405 ± 0.0000 | 0.0290 ± 0.0000 | 0.7890 ± 0.0000 |
| random_forest | 0.0966 ± 0.0001 | 0.0854 ± 0.0004 | -0.2011 ± 0.0035 |
| random_forest_v9 | 0.0999 ± 0.0002 | 0.0899 ± 0.0002 | -0.2837 ± 0.0045 |
| training_mean | 0.1005 ± 0.0000 | 0.0851 ± 0.0000 | -0.2980 ± 0.0000 |

The six descriptors are particle count, radius of gyration divided by diameter, contact count, contact density, mean coordination and maximum coordination. Some are mathematically dependent; the linear baseline uses a least-squares pseudoinverse.

A latent vector contains learned artificial structural parameters that replace the traditional descriptors at the prediction head. Its axes differ between separately trained models, so axes must not be pooled across folds.

GNN dimension selection and stopping use inner validation only. Fixed-q rows are prespecified comparisons, not a basis for choosing a winner on the test set. Predictions are not clipped to [0,1].

See provenance.json, splits.csv, selections.csv and fold_metrics.csv for provenance and between-fold variability. Historical paper scores are not substituted for these results.
