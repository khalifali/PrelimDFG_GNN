#!/usr/bin/env python3
"""Interpret the selected exposure-only GNN latent representation.

This script is deliberately separate from cross-validation and result plotting.
It imports the data loader and exact Model class from the v9 training script, trains
final all-data models for descriptive interpretation, aligns their latent
coordinates across seeds, and tests whether conventional morphology
descriptors can reconstruct each learned coordinate.

The all-data latent values are for interpretation only. They must not be used
as leakage-free test features when claiming target-prediction performance.

USAGE: python analyze_pure_graph_latent_v9.py gnn_dataset/ \
       --training-script train_gnn_pure_graph_latent_v9.py

"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import orthogonal_procrustes
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch_geometric.loader import DataLoader


TARGET_COLUMNS = ["mean_geometric_exposure"]


def load_training_module(path: Path):
    spec = importlib.util.spec_from_file_location("gnn_training", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import training code from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_training_metrics(dataset: Path, feature_set: str) -> dict:
    result = (
        dataset / "additional_analyses" / "all_particle_counts" / feature_set
        / "exposure_only" / "training_metrics.json"
    )
    if not result.is_file():
        raise FileNotFoundError(
            f"Selected-model metadata not found: {result}. Run v9 training first."
        )
    return json.loads(result.read_text(encoding="utf-8"))


def parse_case(case: str) -> dict:
    # Supported examples:
    #   case_001_dp1_N50_Df1.8_rep01
    #   case_001_dp1_N50_open_Df1.8_kf1.3_rep01
    pattern = re.compile(
        r"dp(?P<dp>[0-9.]+)_N(?P<N>[0-9]+)_"
        r"(?:(?P<morphology>[A-Za-z][A-Za-z0-9-]*)_)?"
        r"Df(?P<Df>[0-9.]+)"
        r"(?:_kf(?P<kf>[0-9.]+))?_rep(?P<rep>[0-9]+)",
        flags=re.IGNORECASE,
    )
    match = pattern.search(case)
    if not match:
        return {
            "particle_diameter": np.nan,
            "particle_count_design": np.nan,
            "requested_fractal_dimension": np.nan,
            "fractal_prefactor": np.nan,
            "morphology": "unknown",
            "realization": np.nan,
            "parameter_group": case,
            "case_name_parsed": False,
        }
    values = match.groupdict()
    morphology = values["morphology"] or "unspecified"
    prefactor = float(values["kf"]) if values["kf"] is not None else np.nan
    group_parts = [
        f"dp{values['dp']}",
        f"N{values['N']}",
        morphology,
        f"Df{values['Df']}",
    ]
    if values["kf"] is not None:
        group_parts.append(f"kf{values['kf']}")
    return {
        "particle_diameter": float(values["dp"]),
        "particle_count_design": int(values["N"]),
        "requested_fractal_dimension": float(values["Df"]),
        "fractal_prefactor": prefactor,
        "morphology": morphology,
        "realization": int(values["rep"]),
        "parameter_group": "_".join(group_parts),
        "case_name_parsed": True,
    }


def graph_table(data, metadata) -> pd.DataFrame:
    rows = []
    target_names = metadata["target_names"]
    for graph in data:
        case = str(graph.case)
        design = parse_case(case)
        number = int(graph.num_nodes)
        source = graph.edge_index[0].cpu().numpy()
        degree = np.bincount(source, minlength=number).astype(float)
        positions = graph.pos.cpu().numpy()
        radius_of_gyration = float(
            np.sqrt(np.mean(np.sum(positions**2, axis=1)))
        )
        undirected_edges = int(graph.edge_index.shape[1] // 2)
        possible_edges = number * (number - 1) / 2
        targets = graph.y.view(-1).cpu().numpy()
        row = {
            "case": case,
            **design,
            "particle_count": number,
            "radius_of_gyration_over_dref": radius_of_gyration,
            "mean_coordination": float(degree.mean()),
            "maximum_coordination": float(degree.max()),
            "contact_count": undirected_edges,
            "contact_density": (
                undirected_edges / possible_edges if possible_edges else np.nan
            ),
        }
        row.update({name: float(value) for name, value in zip(target_names, targets)})
        rows.append(row)
    return pd.DataFrame(rows)


def rotate_graph(graph):
    matrix = torch.randn(3, 3, device=graph.pos.device)
    q, _ = torch.linalg.qr(matrix)
    if torch.det(q) < 0:
        q[:, 0] *= -1
    graph.pos = graph.pos @ q
    return graph


def train_final_model(module, data, metadata, args, seed: int, checkpoint: Path,
                      latent_dimension: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(args.device)
    model = module.Model(
        nin=metadata["node_feature_count"],
        ein=metadata["edge_feature_count"],
        nout=metadata["target_count"],
        global_dim=len(metadata["global_feature_names"]),
        hidden=args.hidden,
        latent=latent_dimension,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    targets = torch.cat([graph.y for graph in data], dim=0)
    target_mean = targets.mean(dim=0).to(device)
    target_std = targets.std(dim=0).clamp_min(1.0e-8).to(device)

    checkpoint_signature = {
        "node_feature_names": metadata["node_feature_names"],
        "edge_feature_names": metadata["edge_feature_names"],
        "target_names": metadata["target_names"],
        "feature_set": metadata.get("feature_set", "minimal"),
        "hidden": args.hidden,
        "layers": args.layers,
        "dropout": args.dropout,
        "latent_dimension": latent_dimension,
        "model_class": "train_gnn_pure_graph_latent_v9.Model",
    }
    if checkpoint.exists() and not args.retrain:
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        saved_signature = saved.get("analysis_signature")
        try:
            if saved_signature != checkpoint_signature:
                raise ValueError("checkpoint configuration differs from this analysis")
            model.load_state_dict(saved["state_dict"])
        except (KeyError, RuntimeError, ValueError) as exc:
            print(f"Ignoring incompatible checkpoint {checkpoint.name}: {exc}")
            print("Retraining this final all-data model with the current v9 configuration.")
        else:
            return model, target_mean, target_std

    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function = nn.MSELoss()
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            if not args.no_rotation_augmentation:
                batch = rotate_graph(batch)
            optimizer.zero_grad()
            prediction, _ = model(batch)
            loss = loss_function(
                prediction, (batch.y - target_mean) / target_std
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss) * batch.num_graphs
        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(
                f"seed={seed}, epoch={epoch:4d}, "
                f"standardized MSE={epoch_loss / len(data):.6f}"
            )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "seed": seed,
            "epochs": args.epochs,
            "target_mean": target_mean.cpu(),
            "target_std": target_std.cpu(),
            "analysis_signature": checkpoint_signature,
        },
        checkpoint,
    )
    return model, target_mean, target_std


@torch.no_grad()
def extract_latent(model, data, device):
    model.eval()
    loader = DataLoader(data, batch_size=64, shuffle=False)
    values = []
    for batch in loader:
        _, latent = model(batch.to(device))
        values.append(latent.cpu().numpy())
    return np.concatenate(values, axis=0).astype(float, copy=False)


def standardize_and_align(latent_values: list[np.ndarray]):
    """Z-score and orthogonally align independently trained latent spaces."""
    standardized = []
    reference = None
    for values in latent_values:
        mean = values.mean(axis=0, keepdims=True)
        scale = np.maximum(values.std(axis=0, ddof=1, keepdims=True), 1.0e-12)
        z = (values - mean) / scale
        if reference is None:
            reference = z.copy()
        else:
            rotation, _ = orthogonal_procrustes(z, reference)
            z = z @ rotation
        standardized.append(z)
    stack = np.stack(standardized, axis=0)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(mean)
    return mean, std


def correlations(table: pd.DataFrame, descriptors: list[str], latent_columns,
                 output: Path):
    rows = []
    for latent_column in latent_columns:
      for descriptor in descriptors:
        subset = table[[descriptor, latent_column]].dropna()
        n_valid = len(subset)
        descriptor_is_constant = subset[descriptor].nunique() < 2
        latent_is_constant = subset[latent_column].nunique() < 2
        if n_valid < 2 or descriptor_is_constant or latent_is_constant:
            pearson_r = pearson_p = spearman_rho = spearman_p = np.nan
        else:
            pearson = pearsonr(subset[descriptor], subset[latent_column])
            spearman = spearmanr(subset[descriptor], subset[latent_column])
            pearson_r, pearson_p = pearson.statistic, pearson.pvalue
            spearman_rho, spearman_p = spearman.statistic, spearman.pvalue
        rows.append({
            "latent_coordinate": latent_column,
            "descriptor": descriptor,
            "n_valid": n_valid,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
        })
    result = pd.DataFrame(rows).sort_values("spearman_rho", key=np.abs, ascending=False)
    result.to_csv(output, index=False)
    return result


def reconstruction_models(table, numeric, categorical, latent_columns, output):
    groups = table["parameter_group"].to_numpy()
    splitter = GroupKFold(n_splits=5)
    rows = []
    for latent_column in latent_columns:
      y = table[latent_column].to_numpy()
      for descriptor in numeric:
        x = table[[descriptor]]
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ])
        prediction = cross_val_predict(model, x, y, groups=groups, cv=splitter)
        rows.append({"latent_coordinate": latent_column, "model": f"linear: {descriptor}", "grouped_cv_r2": r2_score(y, prediction)})

      columns = numeric + categorical
      transformer = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
      ])
      linear = Pipeline([("features", transformer), ("model", LinearRegression())])
      pred_linear = cross_val_predict(linear, table[columns], y, groups=groups, cv=splitter)
      rows.append({"latent_coordinate": latent_column, "model": "combined linear", "grouped_cv_r2": r2_score(y, pred_linear)})

      forest = Pipeline([
        ("features", transformer),
        ("model", RandomForestRegressor(
            n_estimators=500, max_depth=4, min_samples_leaf=3,
            random_state=7, n_jobs=-1,
        )),
      ])
      pred_forest = cross_val_predict(
        forest, table[columns], y, groups=groups, cv=splitter
      )
      rows.append({"latent_coordinate": latent_column, "model": "combined random forest", "grouped_cv_r2": r2_score(y, pred_forest)})
    result = pd.DataFrame(rows).sort_values("grouped_cv_r2", ascending=False)
    result.to_csv(output, index=False)
    return result


def make_plots(table, descriptors, latent_columns, output_directory, font_size):
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
    })
    columns = 3
    for latent_column in latent_columns:
      rows = int(np.ceil(len(descriptors) / columns))
      figure, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 3.7 * rows))
      axes = np.atleast_1d(axes).ravel()
      for axis, descriptor in zip(axes, descriptors):
        sns.scatterplot(
            data=table, x=descriptor, y=latent_column,
            hue="requested_fractal_dimension", style="particle_count_design",
            palette="viridis", s=55, ax=axis, legend=False,
        )
        sns.regplot(
            data=table, x=descriptor, y=latent_column, scatter=False,
            color="black", line_kws={"linewidth": 1.2}, ax=axis,
        )
        axis.set_title(descriptor.replace("_", " "))
        axis.set_xlabel("")
        axis.set_ylabel(f"Aligned {latent_column.replace('_', ' ')}" if axis is axes[0] else "")
      for axis in axes[len(descriptors):]:
        axis.remove()
      figure.tight_layout()
      figure.savefig(output_directory / f"{latent_column}_vs_descriptors.png", dpi=300, bbox_inches="tight")
      plt.close(figure)

    correlation_columns = descriptors + TARGET_COLUMNS + list(latent_columns)
    correlation = table[correlation_columns].corr(method="spearman")
    figure, axis = plt.subplots(figsize=(11, 8))
    sns.heatmap(correlation, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", ax=axis)
    figure.tight_layout()
    figure.savefig(output_directory / "descriptor_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    for latent_column in latent_columns:
      figure, axis = plt.subplots(figsize=(7.2, 5.4))
      scatter = axis.scatter(
        table["radius_of_gyration_over_dref"], table["mean_coordination"],
        c=table[latent_column], s=35 + 0.35 * table["particle_count"],
        cmap="coolwarm", edgecolor="black", linewidth=0.35,
    )
      axis.set_xlabel(r"$R_g/d_{\mathrm{ref}}$")
      axis.set_ylabel(r"Mean coordination $\overline{z}$")
      colourbar = figure.colorbar(scatter, ax=axis)
      colourbar.set_label(f"Aligned {latent_column.replace('_', ' ')}")
      figure.tight_layout()
      figure.savefig(output_directory / f"{latent_column}_morphology_map.png", dpi=300, bbox_inches="tight")
      plt.close(figure)

    grouped = table.groupby("parameter_group")[list(latent_columns)].agg(["mean", "std", "count"])
    grouped.to_csv(output_directory / "within_parameter_group_variation.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Directory containing case_*/graph.npz")
    parser.add_argument(
        "--training-script", type=Path,
        default=Path("train_gnn_pure_graph_latent_v9.py"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--contact-overlap-tolerance", type=float, default=1.0e-6)
    parser.add_argument(
        "--feature-set", choices=("minimal", "stored"), default="minimal",
        help="Must match the v9 training configuration being interpreted.",
    )
    parser.add_argument("--no-rotation-augmentation", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--font-size", type=float, default=13.0)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    training_script = args.training_script.resolve()
    output = (
        args.output
        or dataset / "latent_descriptor_analysis" / "exposure_only"
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    module = load_training_module(training_script)
    data, metadata, skipped = module.load_dataset(
        dataset,
        contact_overlap_tolerance=args.contact_overlap_tolerance,
        include_non_touching=False,
        feature_set=args.feature_set,
    )
    if skipped:
        print(f"Warning: skipped {len(skipped)} graphs; see analysis_summary.json")
    if not data:
        raise SystemExit("No valid graphs were loaded")
    selected_metrics = selected_training_metrics(dataset, args.feature_set)
    selected_dimension = selected_metrics.get("selected_latent_dimension")
    if not isinstance(selected_dimension, int) or selected_dimension < 1:
        raise ValueError("training_metrics.json has no valid selected latent dimension")
    if selected_metrics.get("target_names") != metadata.get("target_names"):
        raise ValueError(
            "The selected training result and current graph files have different targets. "
            "Remove stale result directories and rerun v9 training."
        )

    table = graph_table(data, metadata)
    failed_case_names = table.loc[~table["case_name_parsed"], "case"].tolist()
    if failed_case_names:
        examples = ", ".join(failed_case_names[:5])
        raise ValueError(
            f"Could not parse {len(failed_case_names)} case names. "
            f"Examples: {examples}"
        )
    latent_runs = []
    for seed in args.seeds:
        # Use architecture-specific checkpoint names so models containing the
        # former global-feature bypass cannot be reused accidentally.
        checkpoint = output / (
            f"final_pure_graph_latent_v9_{args.feature_set}_"
            f"lambda_{selected_dimension}_seed_{seed}.pt"
        )
        model, _, _ = train_final_model(
            module, data, metadata, args, seed, checkpoint, selected_dimension
        )
        latent_runs.append(extract_latent(model, data, torch.device(args.device)))
    aligned_mean, aligned_std = standardize_and_align(latent_runs)
    latent_columns = [f"lambda_{index + 1}" for index in range(selected_dimension)]
    for index, latent_column in enumerate(latent_columns):
        table[latent_column] = aligned_mean[:, index]
        table[f"{latent_column}_seed_std"] = aligned_std[:, index]
    for seed, values in zip(args.seeds, latent_runs):
        for index, latent_column in enumerate(latent_columns):
            table[f"{latent_column}_raw_seed_{seed}"] = values[:, index]

    numeric = [
        "particle_diameter",
        "requested_fractal_dimension",
        "fractal_prefactor",
        "particle_count",
        "radius_of_gyration_over_dref",
        "mean_coordination",
        "maximum_coordination",
        "contact_count",
        "contact_density",
    ]
    categorical = ["morphology"]
    descriptors = numeric
    table.to_csv(output / "agglomerate_latent_analysis.csv", index=False)
    correlation = correlations(
        table, descriptors, latent_columns,
        output / "lambda_descriptor_correlations.csv"
    )
    reconstruction = reconstruction_models(
        table, numeric, categorical, latent_columns,
        output / "lambda_reconstruction_models.csv"
    )
    make_plots(table, descriptors, latent_columns, output, args.font_size)

    summary = {
        "purpose": "descriptive interpretation of the selected one-dimensional latent coordinate",
        "warning": "All-data latent values are not leakage-free prediction features.",
        "n_graphs": len(data),
        "n_parameter_groups": int(table["parameter_group"].nunique()),
        "realizations_per_parameter_group": table.groupby("parameter_group").size().describe().to_dict(),
        "seeds": args.seeds,
        "edge_selection": "geometrically_touching_pairs",
        "contact_overlap_tolerance_over_d_ref": args.contact_overlap_tolerance,
        "feature_set": args.feature_set,
        "selected_latent_dimension": selected_dimension,
        "training_script": str(training_script),
        "latent_input": metadata.get("latent_input", "unspecified"),
        "global_feature_path": metadata.get("global_feature_path", "unspecified"),
        "prediction_head_input": metadata.get("prediction_head_input", "unspecified"),
        "skipped": skipped,
        "strongest_absolute_spearman_association": correlation.iloc[0].to_dict(),
        "best_reconstruction_model": reconstruction.iloc[0].to_dict(),
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(f"Analysis written to {output}")
    print("\nDescriptor correlations:\n", correlation.to_string(index=False))
    print("\nGrouped-CV reconstruction:\n", reconstruction.to_string(index=False))


if __name__ == "__main__":
    main()
