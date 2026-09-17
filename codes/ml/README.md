# Geometric-exposure learning on the bonded agglomerate campaign

This pipeline compares the uploaded **v9 GNN architecture** with an ANN,
linear regression and random forests using the six traditional descriptors.
It uses the completed **180-case bonded DEM campaign**. The uploaded legacy
LIGGGHTS inputs and old DEM workflow are not used. Generation and the bonded
contact model are unchanged.

## Run

On GitHub, open **Actions → Exposure ML comparison → Run workflow**. The default
source is successful bonded-campaign run `35279655509`. Keep the self-hosted
runner service/terminal running until the job finishes. A main-branch change to
`codes/ml` also starts this workflow. It downloads the existing campaign; it
does not rerun DEM. The source artifact must still exist and be unexpired.

Locally, from the repository root (Python 3.10–3.12):

```bash
python3 -m venv .venv-ml
.venv-ml/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv-ml/bin/python -m pip install -r codes/ml/requirements.txt
.venv-ml/bin/python codes/ml/test_pipeline.py
.venv-ml/bin/python codes/ml/run_campaign.py data/dem/study180_bonded \
  --output data/ml/study180-final --epochs 300 --patience 50 \
  --seeds 7 17 27 --q 1 2 3 4
```

The input may also be a directory containing an extracted bonded-campaign
artifact. There must be exactly one completed campaign. Each case must pass
its DEM assessment, and each exposure CSV must match the particle IDs, final
step, mean exposure and SHA-256 hash of the geometry used to calculate it.

The default is the final snapshot. For a paired pre/post comparison:

```bash
.venv-ml/bin/python codes/ml/run_campaign.py data/dem/study180_bonded \
  --snapshot initial --output data/ml/study180-initial \
  --verify-splits data/ml/study180-final/training/splits.csv
```

`--mode method` holds out each generation method in turn. `--mode size500`
trains on smaller agglomerates and tests only N=500. These are separate
experiments, never mixed into the main grouped comparison. To repeat the study
at one particle count, use `train.py DATASET --particle-counts 500 --output DIR`.

For a **software check only**, use `--epochs 2 --patience 2 --seeds 7 --q 1`.
Reports with fewer than 50 allowed epochs are marked as smoke tests.

After an interruption, repeat the same command and output directory with
`--resume`. Completed fold/seed combinations are reused; an interrupted
combination is rerun. Data hashes, source-code hashes and training settings
must match. Existing runs are never silently overwritten. Workflow artifacts
contain partial progress even when training fails; download them to resume
locally. GitHub workflow retries start a new output directory.

## Inputs and comparison

The target is the mean particle exposure from 512 Fibonacci surface-normal
rays. It is a geometric line-of-sight quantity, not a resolved transport rate.
One snapshot per agglomerate is one observation; three snapshots are never
counted as three independent training cases. Exposure and generator labels
are excluded from the model inputs.

The six conventional descriptors are N, Rg/d, contact count Nc, contact density
2Nc/[N(N−1)], mean coordination 2Nc/N, and maximum coordination. Rg here is the
root-mean-square distance of **particle centers** from their centroid, as in
the uploaded v9 preprocessing and analysis. It does not include the individual
sphere moment. This differs from some finite-sphere definitions in earlier
report drafts. Nodes use radial distance/Rg and coordination. Positions are
centered and divided by particle diameter. Each contact is represented in
both directions, with distance/d and a relative-position vector.

The geometric contact rule is distance/d ≤ (ri+rj)/d + 1e-6. Graph features and
all six descriptors use this same contact graph, independently of the bonded
mechanical force law. In the checked 180-case campaign every final graph is
connected and retains Nc=N−1. Consequently several of the six descriptors are
mathematically redundant; the linear fit uses a least-squares pseudoinverse.

| Model | Settings |
|---|---|
| GNN q=1,2,3,4 | Uploaded v9 forward architecture: hidden width 64, six residual GINE layers, ReLU/dropout then residual LayerNorm, dropout 0.1, fixed GINE epsilon 0 |
| GNN pooling/head | Mean, maximum, and sum/sqrt(N); linear projection to q learned parameters; q→16→1 head |
| ANN | Standardized six descriptors →64→32→1, ReLU, dropout 0.1 |
| ANN q=1,2,3,4 | Standardized six descriptors →64→q, with the same q→16→1 prediction head as the GNN |
| Linear | Standardized six descriptors, ordinary least squares |
| Random forest (v9) | 500 trees, maximum depth 4, minimum leaf size 3 |
| Random forest (tuned) | 500 trees; leaf size 1/3/5 and feature fraction 1/0.67 selected on inner validation |
| Controls | Training mean, count linear/quadratic/categorical/forest, and individual-descriptor linear/forest models |

Neural models use Adam, learning rate 0.001, weight decay 1e-5, batch size 16,
300 maximum epochs and patience 50. Target scaling is fit on the training
partition. ANN descriptor scaling is also training-only; GNN node scalars
remain unscaled as in v9. GNN training includes random rigid rotations.
Coordinates make this architecture rotation-sensitive at inference; rotation
augmentation does not confer exact rotational invariance. No exposure inputs,
precomputed aggregate descriptors, or direct size bypass enter the GNN head.

A latent vector is a short vector of learned artificial structural parameters.
The network learns these parameters as alternatives to the traditional six
inputs. The ANN bottleneck comparison matches the prediction head, not the
total number of trainable parameters; parameter counts are recorded.

## Evaluation and changes relative to uploaded v9

Five outer folds hold out parameter groups defined by generation method, N,
and local-filling detector radius where applicable. All realizations of one
parameter group stay together. There are 20 groups in the 180-case campaign.
The saved splits are shared by every model and seed. An out-of-fold prediction
is made by a model that has never trained on that agglomerate's group.

Within each outer training partition, a fixed grouped validation split holds
out approximately 15% of its groups. It selects the stopping epoch, forest
settings, and q. After selection, the model is refit on the full outer training
partition for the selected number of epochs. The smallest q within 0.02 of the
best **inner validation** R² is selected independently for GNN and bottleneck
ANN. Fixed-q comparisons remain available. Seeds are 7,17,27; neural
initializations use seed+1000q+fold.

This preserves the uploaded v9 forward model, but improves its evaluation:
v9 selected q using the outer predictions it also reported, and split inner
validation by individual graphs. Here q selection and early stopping occur
inside grouped training data. Outer folds use deterministic GroupKFold on the
new campaign metadata, not the old filename parser. Models are refit after
inner selection. A gradient-norm limit of 5 is added for stability. Therefore
this is not a bitwise reproduction of historical training or its scores.
The test suite proves exact v9 forward equivalence with identical weights.

The new graph features recalculate coordination from geometric contacts.
The old preprocessing initially used reported interaction pairs; those could
include separated adhesive particles. For this bonded dataset all initial and
final contacts satisfy the geometric rule. Exposure targets remain the
already verified bonded-campaign ray-tracing outputs.

## Results

The artifact `exposure-ml-RUN_ID-ATTEMPT` contains the compact graph dataset,
source campaign, installed package versions, training log and `training/`:

- `results.md`, `comparison.csv`: held-out RMSE, MAE and R², mean and standard
  deviation across seeds. Standard deviation is not a confidence interval.
- `predictions.csv`, `metrics_by_seed.csv`, `fold_metrics.csv`: every held-out
  prediction and per-seed/per-fold metrics; no clipping to [0,1].
- `splits.csv`, `selections.csv`, `provenance.json`: cases and groups, selected
  settings, parameter counts, dataset and source hashes, software versions.
- `parity.png`, `model_comparison.png`, `latent_dimension.png`, `curves/`:
  comparison figures and separate inner training/validation histories.
- `fold*/seed*/`: checkpoints, refit histories, fold-specific latent vectors,
  correlations and descriptor-to-latent reconstruction scores.

Latent correlations and reconstruction are calculated **within each fitted
fold model**. Traditional descriptors are fitted to reconstruct its latent
coordinates on outer training data and evaluated on its held-out graphs.
Latent axes from different networks are not pooled or aligned. This is an
interpretation diagnostic, not an independent transport prediction test.
Unlike the old full-data latent analysis, it never trains the encoder on its
own reconstruction test cases. Requested fractal dimension and prefactor
baselines are omitted because those generator parameters do not exist for
this BPCA/BCCA/local-filling campaign; they are not invented.

For checkpoint inference:

```bash
.venv-ml/bin/python codes/ml/predict.py \
  data/ml/study180-final/training/fold0/seed7/gnn_q1.pt \
  data/ml/study180-final/final_dataset.json --output predictions.csv
```

The output flags cases seen during training. Inference on the whole dataset
is not a new out-of-fold test. The existing evaluation checkpoints are saved;
an all-data deployment model is not automatically substituted for them.

Primary dependencies: NumPy for arrays, SciPy for statistics, scikit-learn for
splits and conventional regressors, PyTorch/PyTorch Geometric for ANN/GINE,
Matplotlib for figures, joblib for baseline checkpoints. Exact major-package
versions are pinned and the full installed environment is archived.
