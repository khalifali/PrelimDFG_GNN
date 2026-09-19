# GNN versus expanded structural descriptors

Development comparison authorized 19 September 2026. FracVAL remains paused; BPM only.

359 archived geometries; 167 primary matched pairs are unioned with provenance groups before splitting. Five morphology-balanced folds, with a separate grouped validation subset inside each outer training fold. Measured log N, box-counting dimension and particle-body hull porosity balance the folds. Requested Df/kf never enter features or splitting. Same frozen partitions for all models and seeds 7, 17, 27. This interpolation analysis is not the old v9 or previous geometry-only split, and is not independent confirmation or an extrapolation assessment.

Basic predictors: log N, measured box dimension, hull porosity, log Rg/d. Expanded predictors add all 11 anisotropy, branching and local-density descriptors from codes/matched_structure/analyze.py. Compare scaled ridge (six alphas), random forest (500 trees; leaves 1, 3, 5), ANN, and geometry-only GNN with q=1,2,3,4. ANN/GNN use up to 300 epochs and validation stopping. Selection uses minimum validation absolute-exposure MSE; q is never chosen from test results. No outer refit. Different model families have finite, declared search spaces rather than an identical number of fitted parameters.

Report pooled out-of-fold absolute exposure R²/RMSE/MAE and held-out matched-pair signed-difference R²/RMSE/MAE. Differences are obtained from individual agglomerate predictions, not direct pair training. The earlier paired ridge result therefore is context, not a like-for-like baseline. Treat matched-difference RMSE as the main diagnostic of additional structural information; large overall R² alone does not establish this. Preserve predictions for paired comparisons across models. Three seeds measure training variability, not confidence intervals. Quality flags remain included and available in the matched source results.

This run addresses whether a GNN captures remaining information beyond expanded descriptors. Any positive finding needs fresh-data confirmation; it does not establish physical transport effectiveness or a speed advantage.
