#!/usr/bin/env python3
"""Interpret the selected one-dimensional GNN morphology descriptor.

This script is deliberately separate from cross-validation and result plotting.
It imports the data loader and exact Model class from train_gnn_3.py, trains
final all-data models for descriptive interpretation, aligns the arbitrary
latent sign/scale across seeds, and tests whether conventional morphology
descriptors can reconstruct the learned coordinate.

The all-data latent values are for interpretation only. They must not be used
as leakage-free test features when claiming target-prediction performance.

USAGE: python analyze_latent_descriptor.py gnn_dataset/  --training-script train_gnn_4.py

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


TARGET_COLUMNS = [
    "mean_graph_depth",
    "shielded_fraction_h_ge_2",
    "accessibility_score",
]


def load_training_module(path: Path):
    spec = importlib.util.spec_from_file_location("gnn_training", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import training code from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_case(case: str) -> dict:
    pattern = re.compile(
        r"dp(?P<dp>[0-9.]+)_N(?P<N>[0-9]+)_Df(?P<Df>[0-9.]+)_rep(?P<rep>[0-9]+)",
        flags=re.IGNORECASE,
    )
    match = pattern.search(case)
    if not match:
        return {
            "particle_diameter": np.nan,
            "particle_count_design": np.nan,
            "requested_fractal_dimension": np.nan,
            "realization": np.nan,
            "parameter_group": case,
        }
    values = match.groupdict()
    return {
        "particle_diameter": float(values["dp"]),
        "particle_count_design": int(values["N"]),
        "requested_fractal_dimension": float(values["Df"]),
        "realization": int(values["rep"]),
        "parameter_group": (
            f"dp{values['dp']}_N{values['N']}_Df{values['Df']}"
        ),
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


def train_final_model(module, data, metadata, args, seed: int, checkpoint: Path):
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
        latent=1,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    targets = torch.cat([graph.y for graph in data], dim=0)
    target_mean = targets.mean(dim=0).to(device)
    target_std = targets.std(dim=0).clamp_min(1.0e-8).to(device)

    if checkpoint.exists() and not args.retrain:
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["state_dict"])
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
        values.extend(latent[:, 0].cpu().numpy().tolist())
    return np.asarray(values, dtype=float)


def standardize_and_align(latent_values: list[np.ndarray]):
    standardized = []
    reference = None
    for values in latent_values:
        z = (values - values.mean()) / max(values.std(ddof=1), 1.0e-12)
        if reference is None:
            reference = z.copy()
        elif np.corrcoef(reference, z)[0, 1] < 0:
            z = -z
        standardized.append(z)
    matrix = np.column_stack(standardized)
    return matrix.mean(axis=1), matrix.std(axis=1, ddof=1) if matrix.shape[1] > 1 else np.zeros(matrix.shape[0])


def correlations(table: pd.DataFrame, descriptors: list[str], output: Path):
    rows = []
    for descriptor in descriptors:
        subset = table[[descriptor, "lambda_1"]].dropna()
        pearson = pearsonr(subset[descriptor], subset["lambda_1"])
        spearman = spearmanr(subset[descriptor], subset["lambda_1"])
        rows.append({
            "descriptor": descriptor,
            "pearson_r": pearson.statistic,
            "pearson_p": pearson.pvalue,
            "spearman_rho": spearman.statistic,
            "spearman_p": spearman.pvalue,
        })
    result = pd.DataFrame(rows).sort_values("spearman_rho", key=np.abs, ascending=False)
    result.to_csv(output, index=False)
    return result


def reconstruction_models(table, numeric, categorical, output):
    groups = table["parameter_group"].to_numpy()
    splitter = GroupKFold(n_splits=5)
    y = table["lambda_1"].to_numpy()
    rows = []

    for descriptor in numeric:
        x = table[[descriptor]]
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ])
        prediction = cross_val_predict(model, x, y, groups=groups, cv=splitter)
        rows.append({"model": f"linear: {descriptor}", "grouped_cv_r2": r2_score(y, prediction)})

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
    rows.append({"model": "combined linear", "grouped_cv_r2": r2_score(y, pred_linear)})

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
    rows.append({"model": "combined random forest", "grouped_cv_r2": r2_score(y, pred_forest)})
    result = pd.DataFrame(rows).sort_values("grouped_cv_r2", ascending=False)
    result.to_csv(output, index=False)
    return result


def make_plots(table, descriptors, output_directory, font_size):
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
    })
    columns = 3
    rows = int(np.ceil(len(descriptors) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 3.7 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, descriptor in zip(axes, descriptors):
        sns.scatterplot(
            data=table, x=descriptor, y="lambda_1",
            hue="requested_fractal_dimension", style="particle_count_design",
            palette="viridis", s=55, ax=axis, legend=False,
        )
        sns.regplot(
            data=table, x=descriptor, y="lambda_1", scatter=False,
            color="black", line_kws={"linewidth": 1.2}, ax=axis,
        )
        axis.set_title(descriptor.replace("_", " "))
        axis.set_xlabel("")
        axis.set_ylabel(r"Aligned descriptor $\lambda_1$" if axis is axes[0] else "")
    for axis in axes[len(descriptors):]:
        axis.remove()
    figure.tight_layout()
    figure.savefig(output_directory / "lambda_vs_descriptors.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    correlation_columns = descriptors + TARGET_COLUMNS + ["lambda_1"]
    correlation = table[correlation_columns].corr(method="spearman")
    figure, axis = plt.subplots(figsize=(11, 8))
    sns.heatmap(correlation, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", ax=axis)
    figure.tight_layout()
    figure.savefig(output_directory / "descriptor_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 5.4))
    scatter = axis.scatter(
        table["radius_of_gyration_over_dref"], table["mean_coordination"],
        c=table["lambda_1"], s=35 + 0.35 * table["particle_count"],
        cmap="coolwarm", edgecolor="black", linewidth=0.35,
    )
    axis.set_xlabel(r"$R_g/d_{\mathrm{ref}}$")
    axis.set_ylabel(r"Mean coordination $\overline{z}$")
    colourbar = figure.colorbar(scatter, ax=axis)
    colourbar.set_label(r"Aligned descriptor $\lambda_1$")
    figure.tight_layout()
    figure.savefig(output_directory / "lambda_morphology_map.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    grouped = table.groupby("parameter_group")["lambda_1"].agg(["mean", "std", "count"])
    grouped.to_csv(output_directory / "within_parameter_group_variation.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Directory containing case_*/graph.npz")
    parser.add_argument("--training-script", type=Path, default=Path("train_gnn_3.py"))
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
    parser.add_argument("--no-rotation-augmentation", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--font-size", type=float, default=13.0)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    training_script = args.training_script.resolve()
    output = (args.output or dataset / "latent_descriptor_analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    module = load_training_module(training_script)
    data, metadata, skipped = module.load_dataset(
        dataset,
        contact_overlap_tolerance=args.contact_overlap_tolerance,
        include_non_touching=False,
    )
    if skipped:
        print(f"Warning: skipped {len(skipped)} graphs; see analysis_summary.json")
    if not data:
        raise SystemExit("No valid graphs were loaded")

    table = graph_table(data, metadata)
    latent_runs = []
    for seed in args.seeds:
        checkpoint = output / f"final_interpretation_model_seed_{seed}.pt"
        model, _, _ = train_final_model(
            module, data, metadata, args, seed, checkpoint
        )
        latent_runs.append(extract_latent(model, data, torch.device(args.device)))
    table["lambda_1"], table["lambda_1_seed_std"] = standardize_and_align(latent_runs)
    for seed, values in zip(args.seeds, latent_runs):
        table[f"lambda_1_raw_seed_{seed}"] = values

    numeric = [
        "particle_count",
        "radius_of_gyration_over_dref",
        "mean_coordination",
        "maximum_coordination",
        "contact_count",
        "contact_density",
    ]
    categorical = ["particle_diameter", "requested_fractal_dimension"]
    descriptors = ["particle_diameter", "requested_fractal_dimension"] + numeric
    table.to_csv(output / "agglomerate_latent_analysis.csv", index=False)
    correlation = correlations(
        table, descriptors, output / "lambda_descriptor_correlations.csv"
    )
    reconstruction = reconstruction_models(
        table, numeric, categorical, output / "lambda_reconstruction_models.csv"
    )
    make_plots(table, descriptors, output, args.font_size)

    summary = {
        "purpose": "descriptive interpretation of the selected one-dimensional latent coordinate",
        "warning": "All-data latent values are not leakage-free prediction features.",
        "n_graphs": len(data),
        "seeds": args.seeds,
        "edge_selection": "geometrically_touching_pairs",
        "contact_overlap_tolerance_over_d_ref": args.contact_overlap_tolerance,
        "skipped": skipped,
        "strongest_absolute_spearman_descriptor": correlation.iloc[0]["descriptor"],
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
