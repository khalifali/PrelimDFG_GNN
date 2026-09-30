# DFG preliminary transport cases

This directory contains the fixed-geometry oxygen-uptake case builder and runner. The case matrix is generated from this repository's checked-in `data/generation/study180_v1` geometries and `results/study180/exposure_summary_post.json`.

The runner workflow prepares the complete matrix (12 N=50 agglomerates at Pe = 0, 1, 10, and 50) on the repository's `lamfoam-local` self-hosted runner. It validates the 48 manifests, then attempts exactly one runtime smoke test: `bpca_N0050_sample09/Pe10`. It does not launch the other cases.

The workflow is intentionally scoped to the `codex/dfg-prelim-run-one-case` branch. It saves the generated project and solver output under `$HOME/dfg_prelim_results/<run-id>-<attempt>` on the runner and uploads a 30-day GitHub Actions artifact. Solver completion only checks workflow execution; it does not certify numerical accuracy of the experimental surface-marker model.

For manual case preparation from the repository root:

```bash
python3 projects/dfg_prelim/build_cases.py --source-repo "$PWD"
```

For manual execution after activating the LAMFOAM solver environment:

```bash
python3 projects/dfg_prelim/run.py --run-prepared projects/dfg_prelim/cases/bpca_N0050_sample09/Pe10
```
