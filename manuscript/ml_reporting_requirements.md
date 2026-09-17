# Manuscript requirements after the ML analysis

Recorded from the user's instructions: report model details and justify the smallest adequate learned structural description, with population balance models (PBMs) as a prospective follow-up application.

## Model details to report

Use the completed run's provenance and selected settings, not assumed defaults.

- Random forests: number of trees, maximum depth, minimum samples per leaf, feature subsampling, bootstrap sampling, random seeds, software/version, and the inner-validation search space and selected settings. Distinguish the fixed v9 forest from the tuned forest.
- ANNs: six input descriptors, every hidden-layer width, activation functions, dropout, bottleneck size where used, prediction-head structure, trainable parameter counts, input/target scaling, loss function, optimizer, learning rate, weight decay, batch size, stopping criterion and maximum epochs.
- GNN: matching architecture/training details and explicit distinction between q learned structural parameters and the single predicted mean-exposure output.
- Report shared grouped outer folds, grouped inner validation, repeated seeds, and mean/spread of held-out RMSE, MAE and R². Explain these in plain language.

## Compact latent-vector criterion

Compare q = 1, 2, 3, 4 numerically. Prefer the smallest q that satisfies the declared prediction-error tolerance; do not automatically choose the largest model or the numerically best outer-test score.

The running pipeline already prespecifies this operational rule:
choose the smallest q whose INNER VALIDATION R² is no more than 0.02 below the best candidate. Keep this rule for the current run; do not change it after inspecting held-out results.

An equivalent error formulation on the same validation cases is:
MSE(q) <= min_k MSE(k) + 0.02 * variance(y_validation),
where variance uses the mean squared deviation from the validation mean.
Equivalently, RMSE(q) <= sqrt(min_k MSE(k) + 0.02 * variance(y_validation)).
Thus 0.02 is a tolerance in normalized squared error, NOT 2% relative RMSE or two percentage points of exposure.

This is a declared statistical compactness tolerance, not a demonstrated physical transport-accuracy requirement. Report the actual absolute RMSE/MAE alongside it. If application-specific acceptable error is later available, evaluate that explicitly in a separately specified analysis rather than retroactively changing selection for this run.

Selection occurs independently inside each outer fold and seed. Present:
1. Fixed-q held-out scores and variability for all four dimensions.
2. How frequently each q was selected using inner validation.
3. Held-out performance of that selection procedure.
4. The numerical error increase associated with the compact representation.
Do not portray selection-frequency summaries as an independently validated universal q or choose a final q from the best outer-test results.

## Manuscript interpretation

Explain that the latent vector combines learned artificial structural parameters representing the agglomerate as an alternative to the six traditional descriptors. A compact vector may make later PBM extensions easier by limiting the number of additional structural state variables and their associated computational burden. This is motivation for preferring a smaller adequate vector, not evidence that a PBM closure or evolution law has already been established.

Use actual completed results to support any statement that one, two, three or four variables are sufficient. If none achieves a defensible absolute accuracy, say so; relative closeness to the best candidate does not by itself establish adequate physical accuracy.

The bonded DEM model and the 180-case campaign remain the data source. Do not reinstate the uploaded legacy LIGGGHTS/DEM workflow.
