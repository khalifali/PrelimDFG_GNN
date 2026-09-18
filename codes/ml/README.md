> Active run: reuse the completed 16-us campaign from Actions run 35288931479.
> One user-approved exclusion: fractal_dp2_N0100_Df2.6_kf0.8_rep04, residual KE
> ratio 0.00104786 above 0.001. There are 134 core + 45 extension = 179 cases.
> The full failed case and its original assessment remain in excluded/; all other
> cases must pass. selection_report.json records the decision before ML.
> Output directory names core135 and extended180 refer to the original design;
> actual training sample counts are 134 and 179. The workflow downloads the saved
> campaign artifact (30-day retention), so it must be retained to rerun later.

# Geometric-exposure ML: active tunable-fractal study

The mixed-generator campaign is superseded. The active workflow reproduces the
validated tunable-fractal design with recorded independent seeds, checks every
geometry with bonded LAMMPS, computes exposure, then trains the comparisons.
No LIGGGHTS inputs or mixed-generator artifacts are used.

## Launch and outputs

GitHub Actions → Exposure ML comparison → Run workflow → main. Changes to this
workflow or codes/ml on main also launch it. Keep the self-hosted runner active.
The job creates an isolated CPU environment and writes to a unique runner-temp
folder. It runs the following experiments sequentially:

1. Historical parameter-grid core: 135 freshly generated geometries, N=50/100/200.
2. Extended study: all 180 geometries, including 45 N=500 cases.
3. Extrapolation: train on N=50/100/200, test on N=500 only.

These are reconstructed geometries, not the old coordinates or a promise of the
historical R². Generation is the tunable particle/cluster mass-radius mechanism;
see codes/generation/FRACTAL_STUDY.md. Three diameters, three prescribed (Df,kf)
pairs and five independent realizations per size are used.

DEM checks must pass before learning. The relaxation interval is 160000 steps
at 1e-10 s (16 microseconds), with the original force parameters and acceptance
thresholds. The earlier 2-microsecond interval left excessive residual kinetic
energy for 1.5 and 2 micrometre particles. Three snapshots per case are retained;
only the final snapshot enters each primary ML experiment. Fibonacci exposure
uses 2048 rays. Representative initial-geometry resolution and rotation results
are in results/fractal180/sampling_checks.csv; these are not a convergence proof
for every individual particle or a transport-model validation.

Artifacts contain geometry, bonded DEM assessments and visualization, portable
final_dataset.json, package versions, and core135/, extended180/, size500/ results.
Each comparison saves predictions, pooled out-of-fold R²/RMSE/MAE, per-fold and
per-seed scores, plots, training/validation curves, selected settings, latent
interpretation diagnostics, and reloadable checkpoints. Core results are also
uploaded immediately after that experiment. Partial results upload on failure.
No final score is claimed before its experiment completes.

## Models and training

| Model | Input and configuration |
|---|---|
| GNN q=1–4 | Uploaded v9 GINE, width 64, six residual layers, dropout 0.1, fixed epsilon 0; mean/max/sum divided by sqrt(N) pooling; projection to q; q→16→1 head |
| ANN | Six standardized descriptors →64→32→1, ReLU, dropout 0.1 |
| Bottleneck ANN q=1–4 | Six standardized descriptors →64→q, then q→16→1 |
| Linear regression | Six standardized descriptors, ordinary least squares |
| Original forest | 500 trees, depth 4, minimum leaf size 3 |
| Tuned forest | 500 trees; leaf size 1/3/5 and feature fraction 1/0.67 selected internally |
| Controls | Training mean, particle-count and individual-descriptor regressions |

Six descriptors: N, center-based Rg/d, contact count Nc, contact density
2Nc/[N(N−1)], mean coordination 2Nc/N, maximum coordination. Several are
mathematically redundant. GNN node features are radial distance/Rg and
coordination; edges carry distance/d and relative position. Generator labels,
exposure and the six aggregate descriptors are not GNN inputs. There is one
output: mean geometric exposure. The q coordinates are learned structural
parameters, not additional targets.

Neural settings: Adam, learning rate .001, weight decay 1e-5, batch size 16,
maximum 300 epochs, patience 50, gradient-norm limit 5, seeds 7/17/27. All
scalers are fitted using training data only. GNN training uses random rotations;
it is not exactly rotation-invariant. Source and dataset hashes are recorded.

## Evaluation

Five outer splits use N/Df/kf groups, assigned round-robin across morphologies
with fixed within-morphology shuffling (seed 701), independently of targets. All diameters and realizations
within a group stay together. These folds measure transfer to unseen parameter
combinations, a stricter task than a random realization split. Every model uses
the same partitions. Out-of-fold means each prediction comes from a model that
never trained on that case's group. The pooled R² is not the average fold R².

For each outer fold, one training group per morphology is held out for inner
validation, chosen deterministically without targets. This ensures all three
morphologies occur in both fitting and validation. Inner validation selects
stopping epochs, forest settings and latent size. The smallest q within .02 of
the best inner-validation R² is selected; absolute errors are also reported.
Models are refitted on the full outer training partition for the selected epoch
count. Outer test data are never used for model selection.

Latent interpretation stays within each fold's model: descriptors reconstruct
its learned coordinates using training cases, then are tested on held-out cases.
Latent axes from different fitted networks are not pooled.

## Local training after the bonded campaign

Install CPU torch==2.5.1 and codes/ml/requirements.txt in an isolated environment.

```bash
python codes/ml/dataset.py BONDED_DIR --output final_dataset.json
python codes/ml/train.py final_dataset.json --particle-counts 50 100 200 --output core135 --epochs 300 --patience 50 --seeds 7 17 27 --q 1 2 3 4
python codes/ml/train.py final_dataset.json --output extended180 --epochs 300 --patience 50 --seeds 7 17 27 --q 1 2 3 4
python codes/ml/train.py final_dataset.json --mode size500 --output size500 --epochs 300 --patience 50 --seeds 7 17 27 --q 1 2 3 4
```

Use --resume with unchanged settings/data/source to reuse completed fold/seed
combinations. Existing results are not silently overwritten. Two-epoch runs are
software checks only. Primary libraries: PyTorch/PyTorch Geometric, NumPy,
SciPy, scikit-learn, Matplotlib, pandas and joblib; versions are pinned and saved.
