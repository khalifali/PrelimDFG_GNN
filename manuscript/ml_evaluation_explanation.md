# Evaluation and choosing the number of learned structural parameters

## Two questions, with different held-out groups

The completed study in run 35290323430 withholds all diameters and realizations
at each N/Df/kf combination. This tests transfer to unseen structural conditions.
The v9 comparison uses the uploaded script's default grouping: diameter/N/Df.
Realizations within that group stay together, but other diameters at the same
N/Df may be available in training. The latter is less demanding for a normalized,
scale-independent exposure target. The original 0.954 score cannot be assigned
to an exact protocol without its saved run configuration; the uploaded script
also supports optional random folds. Random-fold diagnostics are not proof of
which option produced that historical score.

The new v9 mode calls the uploaded grouping routine directly, freezes the split
seed at 7 for all models and training seeds, and uses its random 15% internal
validation split. Exact partition equality is tested. Like v9, neural models
are evaluated using the best saved inner-fit checkpoint without refitting.
The v9 mode retains current training implementation details: NumPy batch
shuffling, population target scaling, a gradient-norm limit, and unscaled MSE
improvement checks. It is an evaluation-protocol comparison, not bitwise replay
of the historical training or recovery of the original geometries. Other modes
and previously saved results remain separate.

## Simple explanation of internal evaluation

1. Set aside the outer test cases. Their exposure values do not guide training,
   stopping, hyperparameter selection, or choice of latent size.
2. Split the remaining cases into fitting cases (about 85%) and internal
   validation cases (about 15%). Fit weights on the fitting cases only.
3. Train four models with q=1,2,3,4 learned structural parameters. Each predicts
   the same single quantity: mean geometric exposure. During training, save the
   checkpoint with the lowest internal-validation mean squared error.
4. Compare the four saved models on those same validation cases. Let R²_best
   be the largest validation R². Select the smallest q satisfying
   R²_validation(q) >= R²_best - 0.02.
5. Evaluate the chosen model on the untouched outer test cases. Repeat until
   every case has an out-of-fold prediction (a prediction from a model that
   did not fit that case). Compute pooled R², RMSE and MAE from these predictions.

The 0.02 is an absolute R² difference, not 2% relative error and not an RMSE
limit. It expresses a preference for a compact description. It does not prove
that a model is accurate enough for PBM use. Application-specific acceptable
errors still need defining. If all validation candidates perform poorly, the
rule still ranks candidates but cannot establish adequacy; this is flagged.
Internal-validation scores are model-selection data, not unbiased final scores.

## Numerical records

comparison.csv: held-out performance for every q plus the internally selected
model, reported across three training seeds. latent_selection.csv/md: each
fold/seed's internal R², RMSE/MAE (new runs), stopping epoch, gap from the best,
eligibility under the .02 rule and selected q. Selections can differ by fold;
the selected model row is the performance of this selection procedure, not a
claim that one universal q was chosen. A final all-data model would require its
own internal selection, without choosing q from the reported test scores.

Both GNN and bottleneck ANN get these records. The unrestricted ANN, linear
regression, and forests use the same outer cases. Forest tuning is internal;
fixed descriptor baselines fit the entire outer-training partition. Neural
v9 checkpoint models retain the 15% validation holdout, as in the source.
