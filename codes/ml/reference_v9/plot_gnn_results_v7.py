#!/usr/bin/env python3
"""Create two LaTeX-styled GNN figures for mean geometric exposure."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np


EXPOSURE_TARGET = "mean_geometric_exposure"

TARGET_LABELS = {
    EXPOSURE_TARGET: r"Mean geometric exposure $\overline{a}_{\mathrm{geo}}$",
}

# Identical canvas dimensions keep the three images aligned when placed as
# equal-width LaTeX subfigures.  Saving without bbox_inches="tight" preserves
# this common aspect ratio even though the parity plot contains a color bar.
FIGURE_SIZE = (5.2, 4.4)


def latex_plot_style(font_size):
    """Use Matplotlib's bundled Computer Modern LaTeX-style rendering."""
    return {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
    }


def read_predictions(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No prediction rows found in {path}")
    predictions = []
    for raw in rows:
        row = {"case": raw["case"], "fold": int(float(raw["fold"]))}
        for key, value in raw.items():
            if key not in ("case", "fold") and value != "":
                row[key] = float(value)
        predictions.append(row)
    return predictions


def available_targets(predictions):
    keys = set(predictions[0])
    required = {f"true_{EXPOSURE_TARGET}", f"pred_{EXPOSURE_TARGET}"}
    missing = sorted(required - keys)
    if missing:
        raise ValueError(
            "predictions.csv does not contain the mean geometric exposure result. "
            f"Missing column(s): {', '.join(missing)}"
        )
    return [EXPOSURE_TARGET]


def metrics(truth, predicted):
    error = predicted - truth
    denominator = np.sum((truth - truth.mean()) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "r_squared": float(1.0 - np.sum(error ** 2) / denominator)
        if denominator > 0 else np.nan,
    }


def discrete_fold_colours(folds):
    fold_ids = np.unique(folds)
    base = plt.get_cmap("tab10").colors
    cmap = ListedColormap([base[i % len(base)] for i in range(len(fold_ids))])
    boundaries = np.arange(len(fold_ids) + 1) - 0.5
    norm = BoundaryNorm(boundaries, cmap.N)
    index = {fold_id: i for i, fold_id in enumerate(fold_ids)}
    colours = np.asarray([index[value] for value in folds])
    return fold_ids, colours, cmap, norm, boundaries


def plot_oof(predictions, targets, output, dpi, font_size):
    folds = np.asarray([row["fold"] for row in predictions], dtype=int)
    fold_ids, colours, cmap, norm, boundaries = discrete_fold_colours(folds)
    number = len(targets)

    with plt.rc_context(latex_plot_style(font_size)):
        figure, axes = plt.subplots(1, number, figsize=FIGURE_SIZE)
        axes = np.atleast_1d(axes)
        figure.subplots_adjust(
            left=0.18, right=0.84, bottom=0.17, top=0.97, wspace=0.16
        )

        for panel_index, (axis, target) in enumerate(zip(axes, targets)):
            truth = np.asarray([row[f"true_{target}"] for row in predictions])
            predicted = np.asarray([row[f"pred_{target}"] for row in predictions])
            score = metrics(truth, predicted)
            lower = float(min(truth.min(), predicted.min()))
            upper = float(max(truth.max(), predicted.max()))
            margin = 0.05 * max(upper - lower, 1.0e-8)

            scatter = axis.scatter(
                truth, predicted, c=colours, cmap=cmap, norm=norm,
                s=34, alpha=0.82, edgecolors="black", linewidths=0.3,
            )
            axis.plot(
                [lower - margin, upper + margin],
                [lower - margin, upper + margin],
                color="black", linestyle="--", linewidth=1.1,
            )
            axis.set_xlim(lower - margin, upper + margin)
            axis.set_ylim(lower - margin, upper + margin)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("Reference")
            if panel_index == 0:
                axis.set_ylabel("Out-of-fold prediction")
            axis.text(
                0.04, 0.96,
                rf"$R^2={score['r_squared']:.3f}$" + "\n"
                + rf"$\mathrm{{RMSE}}={score['rmse']:.3f}$" + "\n"
                + rf"$\mathrm{{MAE}}={score['mae']:.3f}$",
                transform=axis.transAxes, va="top", ha="left",
                fontsize=font_size - 1,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
            )
            axis.grid(alpha=0.2)

        colourbar = figure.colorbar(
            scatter, ax=axes.tolist(), ticks=np.arange(len(fold_ids)),
            boundaries=boundaries, spacing="uniform", shrink=0.84, pad=0.015,
        )
        colourbar.set_ticklabels([str(value + 1) for value in fold_ids])
        colourbar.set_label("Held-out test fold")
        # Deliberately no overall figure title for publication use.
        figure.savefig(output, dpi=dpi)
        plt.close(figure)


def plot_fold_rmse(predictions, targets, output, dpi, font_size):
    fold_ids = np.unique([row["fold"] for row in predictions])
    scores = np.empty((len(fold_ids), len(targets)))
    for i, fold_id in enumerate(fold_ids):
        fold_rows = [row for row in predictions if row["fold"] == fold_id]
        for j, target in enumerate(targets):
            truth = np.asarray([row[f"true_{target}"] for row in fold_rows])
            predicted = np.asarray([row[f"pred_{target}"] for row in fold_rows])
            scores[i, j] = metrics(truth, predicted)["rmse"]

    with plt.rc_context(latex_plot_style(font_size)):
        figure, axes = plt.subplots(1, len(targets), figsize=FIGURE_SIZE)
        axes = np.atleast_1d(axes)
        # A deliberately generous left margin prevents the vertical RMSE label
        # from being clipped when the PNG is included at subfigure width.
        figure.subplots_adjust(left=0.21, right=0.97, bottom=0.17, top=0.97, wspace=0.22)
        x = np.arange(1, len(fold_ids) + 1)
        for j, (axis, target) in enumerate(zip(axes, targets)):
            axis.bar(x, scores[:, j], color="#3070B3", edgecolor="black")
            axis.axhline(scores[:, j].mean(), color="#D1495B", linestyle="--")
            axis.set_xticks(x)
            axis.set_xlabel("Test fold")
            if j == 0:
                axis.set_ylabel("RMSE")
            axis.grid(axis="y", alpha=0.2)
        figure.savefig(output, dpi=dpi)
        plt.close(figure)


def plot_latent_comparison(path, targets, output, dpi, font_size):
    if not path.exists():
        print(f"Skipping latent-dimension plot: {path} not found")
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    dimensions = [int(row["latent_dimension"]) for row in rows]
    with plt.rc_context({"font.size": font_size}):
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        figure.subplots_adjust(left=0.16, right=0.97, bottom=0.16, top=0.97)
        for target in targets:
            key = f"r_squared_{target}"
            if key in rows[0]:
                axis.plot(
                    dimensions, [float(row[key]) for row in rows],
                    marker="o", linewidth=1.8,
                )
        axis.set_xticks(dimensions)
        axis.set_xlabel(r"Latent dimension $n_\lambda$")
        axis.set_ylabel(r"Out-of-fold $R^2$")
        axis.grid(alpha=0.25)
        figure.savefig(output, dpi=dpi)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Plot only the mean geometric exposure prediction results."
    )
    parser.add_argument("dataset", type=Path, help="Directory containing predictions.csv")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font-size", type=float, default=15.0)
    args = parser.parse_args()

    root = args.dataset.resolve()
    predictions = read_predictions(root / "predictions.csv")
    targets = available_targets(predictions)
    plot_oof(
        predictions, targets, root / "mean_geometric_exposure_oof_performance.png",
        args.dpi, args.font_size,
    )
    plot_fold_rmse(
        predictions, targets, root / "mean_geometric_exposure_fold_rmse.png",
        args.dpi, args.font_size,
    )
    print(f"Figures written to {root}")


if __name__ == "__main__":
    main()
