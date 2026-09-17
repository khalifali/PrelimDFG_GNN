#!/usr/bin/env python3
"""Compare classical baselines with exposure-only v9 GNN predictions.

The script reads the predictions.csv written by
train_gnn_pure_graph_latent_v9.py and the agglomerate_latent_analysis.csv
written by analyze_pure_graph_latent_v9.py. Every baseline reuses exactly the
same grouped test folds as the GNN. No graph files or PyTorch installation are
required.

USAGE: python test_combined_descriptor_baselines_v9.py \
  gnn_dataset/additional_analyses/all_particle_counts/minimal/predictions.csv


"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


TARGETS = ["mean_geometric_exposure"]

TARGET_LABELS = {"mean_geometric_exposure": "Mean geometric exposure"}

# Conventional descriptors calculated directly from the relaxed aggregate and
# its contact graph. Generator settings (requested D_f, k_f and morphology
# label) are deliberately excluded.
COMBINED_DESCRIPTORS = [
    "particle_count",
    "radius_of_gyration_over_dref",
    "contact_count",
    "contact_density",
    "mean_coordination",
    "maximum_coordination",
]

# Prescribed agglomerate-generator settings. These are evaluated separately
# from the post-relaxation descriptors because D_f and k_f are inputs to the
# generator rather than measured properties of the relaxed agglomerate.
GENERATOR_PARAMETERS = [
    "requested_fractal_dimension",
    "fractal_prefactor",
]


def particle_count_from_case(case_name: str) -> int:
    match = re.search(r"(?:^|_)N(\d+)(?:_|$)", str(case_name))
    if not match:
        raise ValueError(f"Could not extract particle count from case: {case_name}")
    return int(match.group(1))


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "mae": float(mean_absolute_error(truth, prediction)),
        "r_squared": float(r2_score(truth, prediction)),
    }


def make_model_specs(seed: int) -> dict[str, tuple[object, list[str]]]:
    # Treating N as a category estimates an independent training-set mean for
    # each particle count. This is the most flexible size-class-only baseline.
    categorical = Pipeline([
        ("encode", OneHotEncoder(handle_unknown="ignore")),
        ("model", LinearRegression()),
    ])
    combined_linear = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LinearRegression()),
    ])
    combined_forest = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        )),
    ])
    generator_linear = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LinearRegression()),
    ])
    generator_forest = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        )),
    ])
    models = {
        "count_linear": (LinearRegression(), ["particle_count"]),
        "count_quadratic": (Pipeline([
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", LinearRegression()),
        ]), ["particle_count"]),
        "count_categorical": (categorical, ["particle_count"]),
        "count_random_forest": (RandomForestRegressor(
            n_estimators=500,
            max_depth=3,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        ), ["particle_count"]),
        "combined_descriptors_linear": (combined_linear, COMBINED_DESCRIPTORS),
        "combined_descriptors_random_forest": (
            combined_forest, COMBINED_DESCRIPTORS
        ),
        "generator_df_kf_linear": (generator_linear, GENERATOR_PARAMETERS),
        "generator_df_kf_random_forest": (
            generator_forest, GENERATOR_PARAMETERS
        ),
    }
    # Particle count is already covered by the dedicated count models above.
    for descriptor in COMBINED_DESCRIPTORS[1:]:
        models[f"individual_linear__{descriptor}"] = (
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LinearRegression()),
            ]),
            [descriptor],
        )
        models[f"individual_random_forest__{descriptor}"] = (
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(
                    n_estimators=500, max_depth=3, min_samples_leaf=3,
                    random_state=seed, n_jobs=-1,
                )),
            ]),
            [descriptor],
        )
    return models


def grouped_oof_predictions(table: pd.DataFrame, seed: int) -> pd.DataFrame:
    result = table[
        ["case", "fold"] + COMBINED_DESCRIPTORS + GENERATOR_PARAMETERS
    ].copy()
    y_all = table[[f"true_{target}" for target in TARGETS]].to_numpy()

    for model_name, (model, feature_names) in make_model_specs(seed).items():
        x_all = table[feature_names]
        prediction = np.full_like(y_all, np.nan, dtype=float)
        for fold in sorted(table["fold"].unique()):
            test_mask = table["fold"].to_numpy() == fold
            train_mask = ~test_mask
            fit_target = (
                y_all[train_mask, 0]
                if y_all.shape[1] == 1 else y_all[train_mask]
            )
            model.fit(x_all.loc[train_mask], fit_target)
            fold_prediction = np.asarray(model.predict(x_all.loc[test_mask]))
            if fold_prediction.ndim == 1:
                fold_prediction = fold_prediction[:, None]
            prediction[test_mask] = fold_prediction
        for target_index, target in enumerate(TARGETS):
            result[f"pred_{model_name}_{target}"] = prediction[:, target_index]

    for target in TARGETS:
        result[f"true_{target}"] = table[f"true_{target}"].to_numpy()
        result[f"pred_gnn_{target}"] = table[f"pred_{target}"].to_numpy()
    return result


def summarize(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    within_rows = []
    models = ["gnn", *make_model_specs(seed=7)]

    for model_name in models:
        for target in TARGETS:
            score = metrics(
                oof[f"true_{target}"].to_numpy(),
                oof[f"pred_{model_name}_{target}"].to_numpy(),
            )
            overall_rows.append({"model": model_name, "target": target, **score})

            for particle_count, subset in oof.groupby("particle_count"):
                within_score = metrics(
                    subset[f"true_{target}"].to_numpy(),
                    subset[f"pred_{model_name}_{target}"].to_numpy(),
                )
                within_rows.append({
                    "model": model_name,
                    "particle_count": int(particle_count),
                    "target": target,
                    **within_score,
                })
    return pd.DataFrame(overall_rows), pd.DataFrame(within_rows)


def plot_comparison(summary: pd.DataFrame, output: Path) -> None:
    models = summary["model"].drop_duplicates().tolist()
    labels = {
        "gnn": "Pure-graph GNN",
        "count_linear": "Count: linear",
        "count_quadratic": "Count: quadratic",
        "count_categorical": "Count: categorical",
        "count_random_forest": "Count: random forest",
        "combined_descriptors_linear": "Combined descriptors: linear",
        "combined_descriptors_random_forest": "Combined descriptors: random forest",
        "generator_df_kf_linear": r"Requested $D_f+k_f$: linear",
        "generator_df_kf_random_forest": r"Requested $D_f+k_f$: random forest",
    }
    for model in models:
        if model.startswith("individual_linear__"):
            labels[model] = "Individual linear: " + model.split("__", 1)[1].replace("_", " ")
        elif model.startswith("individual_random_forest__"):
            labels[model] = "Individual RF: " + model.split("__", 1)[1].replace("_", " ")
    colours = plt.cm.tab20(np.linspace(0, 1, len(models)))
    figure, axes = plt.subplots(1, len(TARGETS), figsize=(16, 6.5), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, target in zip(axes, TARGETS):
        subset = summary[summary["target"] == target].set_index("model").loc[models]
        axis.bar(
            np.arange(len(models)), subset["r_squared"],
            color=colours, edgecolor="black", linewidth=0.5,
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(TARGET_LABELS[target])
        axis.set_xticks(np.arange(len(models)))
        axis.set_xticklabels([labels[name] for name in models], rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(r"Combined out-of-fold $R^2$")
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def validate_input(table: pd.DataFrame) -> None:
    required = {"case", "fold"}
    for target in TARGETS:
        required.update({f"true_{target}", f"pred_{target}"})
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Missing columns in predictions CSV: {missing}")
    if table["case"].duplicated().any():
        duplicates = table.loc[table["case"].duplicated(), "case"].tolist()[:5]
        raise ValueError(f"Each case must occur once; duplicates include {duplicates}")
    if table["fold"].nunique() < 2:
        raise ValueError("At least two saved folds are required")


def validate_v9_minimal_result(predictions_path: Path) -> dict:
    """Verify that predictions belong to the selected v9 minimal model."""
    metrics_path = predictions_path.parent / "training_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Training metadata not found beside predictions: {metrics_path}"
        )
    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metadata.get("feature_set") != "minimal":
        raise ValueError(
            "This comparison expects the v9 minimal-feature result, but "
            f"training_metrics.json reports feature_set={metadata.get('feature_set')!r}."
        )
    if "selected_latent_dimension" not in metadata:
        raise ValueError(
            "training_metrics.json does not describe a selected v9 latent model. "
            "Pass the predictions.csv published at the v9 result root."
        )
    return metadata


def find_descriptor_csv(predictions_path: Path) -> Path:
    """Find the dataset-level descriptor table from a nested v9 result path."""
    relative = Path(
        "latent_descriptor_analysis/exposure_only/agglomerate_latent_analysis.csv"
    )
    candidates = [parent / relative for parent in predictions_path.parents]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if not existing:
        searched = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(
            "Could not locate agglomerate_latent_analysis.csv. Searched:\n"
            f"{searched}\nPass it explicitly with --descriptors-csv."
        )
    # The nearest matching ancestor is the intended dataset when result trees
    # happen to be nested inside other projects.
    return existing[0]


def load_and_merge_descriptors(
    predictions: pd.DataFrame, descriptors_path: Path
) -> pd.DataFrame:
    descriptors = pd.read_csv(descriptors_path)
    required = {"case", *COMBINED_DESCRIPTORS, *GENERATOR_PARAMETERS}
    missing = sorted(required.difference(descriptors.columns))
    if missing:
        raise ValueError(f"Missing columns in descriptor CSV: {missing}")
    if descriptors["case"].duplicated().any():
        duplicates = descriptors.loc[
            descriptors["case"].duplicated(), "case"
        ].tolist()[:5]
        raise ValueError(
            f"Each descriptor case must occur once; duplicates include {duplicates}"
        )
    merged = predictions.merge(
        descriptors[["case"] + COMBINED_DESCRIPTORS + GENERATOR_PARAMETERS],
        on="case",
        how="left",
        validate="one_to_one",
    )
    numeric_inputs = COMBINED_DESCRIPTORS + GENERATOR_PARAMETERS
    incomplete = merged[numeric_inputs].isna().any(axis=1)
    missing_cases = merged.loc[incomplete, "case"].tolist()
    if missing_cases:
        raise ValueError(
            f"Missing descriptor values for {len(missing_cases)} prediction cases; "
            f"examples: {missing_cases[:5]}"
        )
    values = merged[numeric_inputs].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Descriptor CSV contains NaN or infinite numeric values")
    parsed_count = merged["case"].map(particle_count_from_case)
    if not np.allclose(parsed_count, merged["particle_count"], equal_nan=False):
        raise ValueError("Particle counts in case names and descriptor CSV disagree")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the GNN with particle-count-only and combined conventional-"
            "individual and combined classical-descriptor models for mean "
            "geometric exposure."
        )
    )
    parser.add_argument(
        "predictions_csv", type=Path,
        help=(
            "Selected predictions.csv produced by the v9 minimal-feature run, "
            "normally under additional_analyses/all_particle_counts/minimal."
        ),
    )
    parser.add_argument(
        "--descriptors-csv", type=Path, default=None,
        help=(
            "agglomerate_latent_analysis.csv produced by the latent analysis "
            "(default: automatically locate the nearest dataset-level "
            "latent_descriptor_analysis/agglomerate_latent_analysis.csv)"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (default: <predictions directory>/descriptor_baselines)",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    predictions_path = args.predictions_csv.resolve()
    training_metadata = validate_v9_minimal_result(predictions_path)
    descriptors_path = (
        args.descriptors_csv.resolve()
        if args.descriptors_csv is not None
        else find_descriptor_csv(predictions_path)
    )
    if not descriptors_path.is_file():
        raise FileNotFoundError(
            f"Descriptor CSV not found: {descriptors_path}\n"
            "Pass it explicitly with --descriptors-csv."
        )
    output = (
        args.output.resolve()
        if args.output is not None
        else predictions_path.parent / "descriptor_baselines"
    )
    output.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(predictions_path)
    validate_input(predictions)
    table = load_and_merge_descriptors(predictions, descriptors_path)

    oof = grouped_oof_predictions(table, args.seed)
    overall, within_count = summarize(oof)
    oof.to_csv(output / "descriptor_baseline_oof_predictions.csv", index=False)
    overall.to_csv(output / "descriptor_model_comparison.csv", index=False)
    within_count.to_csv(output / "performance_within_fixed_particle_count.csv", index=False)
    plot_comparison(overall, output / "descriptor_baseline_comparison.png")

    mean_r2 = (
        overall.groupby("model", sort=False)["r_squared"].mean()
        .sort_values(ascending=False)
    )
    best_combined_name = max(
        [
            "combined_descriptors_linear",
            "combined_descriptors_random_forest",
        ],
        key=lambda name: mean_r2[name],
    )
    best_generator_name = max(
        ["generator_df_kf_linear", "generator_df_kf_random_forest"],
        key=lambda name: mean_r2[name],
    )
    gnn_minus_best_combined = float(mean_r2["gnn"] - mean_r2[best_combined_name])
    gnn_minus_best_generator = float(mean_r2["gnn"] - mean_r2[best_generator_name])
    report = {
        "input": str(predictions_path),
        "descriptor_input": str(descriptors_path),
        "gnn_feature_set": training_metadata["feature_set"],
        "gnn_selected_latent_dimension": training_metadata["selected_latent_dimension"],
        "combined_descriptor_names": COMBINED_DESCRIPTORS,
        "generator_parameter_names": GENERATOR_PARAMETERS,
        "n_cases": int(len(table)),
        "folds_reused_from_gnn_predictions": sorted(
            int(value) for value in table["fold"].unique()
        ),
        "particle_counts": sorted(int(value) for value in table["particle_count"].unique()),
        "mean_r_squared_across_targets": {
            model: float(value) for model, value in mean_r2.items()
        },
        "best_model": str(mean_r2.index[0]),
        "best_combined_descriptor_model": best_combined_name,
        "best_generator_parameter_model": best_generator_name,
        "gnn_minus_best_combined_mean_r_squared": gnn_minus_best_combined,
        "gnn_minus_best_generator_mean_r_squared": gnn_minus_best_generator,
        "interpretation": (
            "Compare the GNN with count-only, combined-descriptor, and paired "
            "requested-D_f-plus-k_f models. "
            "A GNN advantage over the combined models supports predictive information "
            "in the detailed graph beyond these conventional scalar summaries."
        ),
    }
    (output / "descriptor_baseline_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Analysis written to {output}")
    print("Combined descriptors:", ", ".join(COMBINED_DESCRIPTORS))
    print("\nCombined out-of-fold R-squared:")
    display = overall.pivot(index="model", columns="target", values="r_squared")
    display["mean"] = display.mean(axis=1)
    print(display.sort_values("mean", ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))
    print(
        f"\nGNN minus best combined-descriptor mean R-squared: "
        f"{gnn_minus_best_combined:+.4f}"
    )
    print(
        f"GNN minus best requested-D_f-plus-k_f mean R-squared: "
        f"{gnn_minus_best_generator:+.4f}"
    )
    print(
        "\nInterpretation: the combined-descriptor models are the critical baseline. "
        "If they approach or exceed the GNN, the present targets do not establish a "
        "need for detailed graph encoding. If the GNN remains clearly better, it "
        "captures useful information beyond these scalar summaries."
    )


if __name__ == "__main__":
    main()
