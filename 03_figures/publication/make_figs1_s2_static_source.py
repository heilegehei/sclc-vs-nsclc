from __future__ import annotations
import sys as _public_sys
from pathlib import Path as _PublicPath
_public_root = next(p for p in _PublicPath(__file__).resolve().parents if (p / "common" / "public_paths.py").is_file())
if str(_public_root) not in _public_sys.path:
    _public_sys.path.insert(0, str(_public_root))
from common.public_paths import cohort_workbook, code_root, data_root, explainability_results, fusion_results, publication_results

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from nature_style import (
    DATASET_COLORS,
    LAYER_LABELS,
    LAYER_ORDER,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    apply_style,
    panel_label,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent
SUBMISSION_ROOT = OUTPUT_ROOT.parent
V2_ROOT = data_root() / "explainability_results"
V1_ROOT = data_root() / "fusion_results"
SUPP_OUT = publication_results() / "figures" / "supplementary"

PRIMARY = "weighted_voting_cv_auc"
A_DEVELOPMENT = "A_dev_cv_clean"
A_HOLDOUT = "A_holdout_clean"
B_EXTERNAL = "B_external"
C_EXTERNAL = "C_external"
LAYER_EVAL_DATASETS = [A_HOLDOUT, B_EXTERNAL, C_EXTERNAL]
FUSIONS = MODEL_ORDER[:3]
SIX_MODELS = MODEL_ORDER[:]


DATASET_LABELS = {
    A_DEVELOPMENT: "A development (10-fold CV)",
    A_HOLDOUT: "A evaluation set",
    B_EXTERNAL: "B external validation",
    C_EXTERNAL: "C external validation",
}

INPUTS = {

    "smooth": V2_ROOT / "data" / "F6_layer_calibration_lowess_frac075.csv",
    "calibration_bins": V1_ROOT / "data" / "weighted_voting_layer_calibration_plot_data.csv",
    "layer_metrics": V2_ROOT / "tables" / "weighted_voting_layer_metrics_ci_harmonised.csv",
    "model_delong": V1_ROOT / "tables" / "delong_holm_vs_weighted_voting.csv",
}


def _required_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{name} is missing columns: {missing}")


def _load_inputs() -> dict[str, pd.DataFrame]:
    missing = [str(path) for path in INPUTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Archived figure inputs are missing: {missing}")
    frames = {name: pd.read_csv(path) for name, path in INPUTS.items()}
    _required_columns(
        frames["smooth"],
        [
            "dataset",
            "layer",
            "model",
            "point",
            "predicted_probability",
            "observed_probability_lowess",
        ],
        "archived F6 smooths",
    )
    _required_columns(
        frames["calibration_bins"],
        ["layer", "dataset", "series_type", "x", "y", "bin"],
        "archived calibration bins",
    )
    _required_columns(
        frames["layer_metrics"],
        ["layer", "dataset", "model", "brier", "brier_ci_low", "brier_ci_high"],
        "archived layer metrics",
    )
    _required_columns(
        frames["model_delong"],
        [
            "dataset",
            "reference_model",
            "comparator_model",
            "delta_auc",
            "delta_ci_low",
            "delta_ci_high",
            "p_raw",
            "p_holm",
        ],
        "archived DeLong table",
    )
    _validate_archived_scope(frames)
    return frames


def _validate_archived_scope(frames: dict[str, pd.DataFrame]) -> None:
    for layer in LAYER_ORDER:
        for dataset in LAYER_EVAL_DATASETS:
            smooth = frames["smooth"].loc[
                (frames["smooth"]["layer"] == layer)
                & (frames["smooth"]["dataset"] == dataset)
                & (frames["smooth"]["model"] == PRIMARY)
            ]
            if len(smooth) != 300:
                raise RuntimeError(f"Expected 300 archived smooth points for {layer}/{dataset}")
            bins = frames["calibration_bins"].loc[
                (frames["calibration_bins"]["layer"] == layer)
                & (frames["calibration_bins"]["dataset"] == dataset)
                & (frames["calibration_bins"]["series_type"] == "equal_frequency_bin")
            ]
            if len(bins) != 10:
                raise RuntimeError(f"Expected 10 archived calibration bins for {layer}/{dataset}")
            metric = frames["layer_metrics"].loc[
                (frames["layer_metrics"]["layer"] == layer)
                & (frames["layer_metrics"]["dataset"] == dataset)
                & (frames["layer_metrics"]["model"] == PRIMARY)
            ]
            if len(metric) != 1:
                raise RuntimeError(f"Expected one archived Brier row for {layer}/{dataset}")
    for dataset in (A_DEVELOPMENT, A_HOLDOUT, B_EXTERNAL, C_EXTERNAL):
        expected = FUSIONS[1:] if dataset == A_DEVELOPMENT else SIX_MODELS[1:]
        found = frames["model_delong"].loc[
            frames["model_delong"]["dataset"] == dataset, "comparator_model"
        ].tolist()
        if set(found) != set(expected):
            raise RuntimeError(f"Unexpected archived DeLong comparison scope for {dataset}: {found}")


def _setup_axis(ax: plt.Axes, *, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    if grid == "both":
        ax.grid(color="#E9E9E9", linewidth=0.55, alpha=0.85)
    elif grid:
        ax.grid(axis=grid, color="#E9E9E9", linewidth=0.55, alpha=0.85)


def _save_dual(fig: plt.Figure, basename: str) -> tuple[Path, Path]:
    SUPP_OUT.mkdir(parents=True, exist_ok=True)
    pdf_path = SUPP_OUT / f"{basename}.pdf"
    png_path = SUPP_OUT / f"{basename}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, format="png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def _make_s1_layer_calibration(frames: dict[str, pd.DataFrame]) -> tuple[Path, Path]:
    calibration = frames["calibration_bins"]
    smooths = frames["smooth"]
    fig, axes = plt.subplots(1, 3, figsize=(12.9, 4.25), sharex=True, sharey=True, facecolor="white")
    for ax, layer, letter in zip(axes, LAYER_ORDER, "abc"):
        _setup_axis(ax, grid="both")
        ax.plot([0, 1], [0, 1], color="#777777", linewidth=1.0, linestyle=":", zorder=1)
        for dataset in LAYER_EVAL_DATASETS:
            bins = calibration.loc[
                (calibration["layer"] == layer)
                & (calibration["dataset"] == dataset)
                & (calibration["series_type"] == "equal_frequency_bin")
            ].sort_values("bin", kind="mergesort")
            curve = smooths.loc[
                (smooths["layer"] == layer)
                & (smooths["dataset"] == dataset)
                & (smooths["model"] == PRIMARY)
            ].sort_values("point", kind="mergesort")
            color = DATASET_COLORS[dataset]
            ax.plot(
                curve["predicted_probability"],
                curve["observed_probability_lowess"],
                color=color,
                linewidth=1.9,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=2,
            )
            ax.scatter(
                bins["x"],
                bins["y"],
                s=22,
                facecolor="white",
                edgecolor=color,
                linewidth=0.9,
                zorder=3,
            )
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted probability")
        ax.set_aspect("equal", adjustable="box")
        ax.text(0.035, 0.955, LAYER_LABELS[layer], transform=ax.transAxes, va="top", fontweight="bold", fontsize=10)
        panel_label(ax, letter, x=-0.11, y=1.02)
    axes[0].set_ylabel("Observed proportion")
    handles = [
        Line2D(
            [0],
            [0],
            color=DATASET_COLORS[dataset],
            linewidth=1.9,
            marker="o",
            markerfacecolor="white",
            markersize=4.5,
            label=DATASET_LABELS[dataset],
        )
        for dataset in LAYER_EVAL_DATASETS
    ]
    handles.append(Line2D([0], [0], color="#777777", linewidth=1.0, linestyle=":", label="Ideal"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=4, frameon=False, columnspacing=1.6)
    fig.text(
        0.5,
        0.072,
        "Weighted voting only; local-linear LOWESS (tricube), frac=0.75, it=0, central 2.5%-97.5%; open circles: original 10 equal-frequency groups.",
        ha="center",
        va="bottom",
        fontsize=7.1,
        color="#444444",
    )
    fig.subplots_adjust(left=0.065, right=0.995, top=0.95, bottom=0.235, wspace=0.16)
    return _save_dual(fig, "FigS3_layer_calibration_weighted_voting")


def _make_s2_layer_brier(frames: dict[str, pd.DataFrame]) -> tuple[Path, Path]:
    metrics = frames["layer_metrics"]
    scope = metrics.loc[metrics["dataset"].isin(LAYER_EVAL_DATASETS)]
    x_low = float(scope["brier_ci_low"].min())
    x_high = float(scope["brier_ci_high"].max())
    span = max(0.008, x_high - x_low)
    limits = (max(0.0, x_low - 0.10 * span), min(1.0, x_high + 0.30 * span))
    positions = np.arange(len(LAYER_EVAL_DATASETS), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharex=True, sharey=True, facecolor="white")
    for ax, layer, letter in zip(axes, LAYER_ORDER, "abc"):
        _setup_axis(ax, grid="x")
        layer_rows = metrics.loc[
            (metrics["layer"] == layer) & metrics["dataset"].isin(LAYER_EVAL_DATASETS)
        ].set_index("dataset").loc[LAYER_EVAL_DATASETS]
        for position, dataset in zip(positions, LAYER_EVAL_DATASETS):
            row = layer_rows.loc[dataset]
            value = float(row["brier"])
            low = float(row["brier_ci_low"])
            high = float(row["brier_ci_high"])
            ax.errorbar(
                value,
                position,
                xerr=np.array([[value - low], [high - value]]),
                fmt="o",
                color=DATASET_COLORS[dataset],
                markerfacecolor="white",
                markeredgewidth=1.2,
                markersize=6,
                capsize=3,
                elinewidth=1.2,
                zorder=3,
            )
            ax.text(high + 0.008 * span, position, f"{value:.3f}", ha="left", va="center", fontsize=7.0)
        ax.set_xlim(*limits)
        ax.set_ylim(-0.58, len(LAYER_EVAL_DATASETS) - 0.30)
        ax.invert_yaxis()
        ax.set_xlabel("Brier score")
        ax.set_yticks(positions, [DATASET_LABELS[dataset] for dataset in LAYER_EVAL_DATASETS])
        ax.text(0.035, 0.955, LAYER_LABELS[layer], transform=ax.transAxes, va="top", fontweight="bold", fontsize=10)
        ax.annotate(
            "Lower is better",
            xy=(0.06, 0.07),
            xytext=(0.40, 0.07),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.9},
            ha="center",
            va="center",
            fontsize=7.2,
            color="#444444",
        )
        panel_label(ax, letter, x=-0.12, y=1.02)
    fig.text(0.5, 0.035, "Weighted voting only; points and bars show Brier score and 95% bootstrap CI.", ha="center", fontsize=7.3)
    fig.subplots_adjust(left=0.18, right=0.995, top=0.95, bottom=0.16, wspace=0.14)
    return _save_dual(fig, "FigS2_layer_brier_weighted_voting")


def _fmt_ci(point: float, low: float, high: float, decimals: int = 3, sign: bool = False) -> str:
    marker = "+" if sign else ""
    return f"{point:{marker}.{decimals}f} ({low:{marker}.{decimals}f} to {high:{marker}.{decimals}f})"


def _fmt_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _make_s3_delong_holm(frames: dict[str, pd.DataFrame]) -> tuple[Path, Path]:
    delong = frames["model_delong"].copy()
    datasets = [A_DEVELOPMENT, A_HOLDOUT, B_EXTERNAL, C_EXTERNAL]
    global_low = min(0.0, float(delong["delta_ci_low"].min()))
    global_high = max(0.0, float(delong["delta_ci_high"].max()))
    span = max(0.02, global_high - global_low)
    x_limits = (global_low - 0.08 * span, global_high + 0.08 * span)
    fig = plt.figure(figsize=(15.6, 8.8), facecolor="white")
    outer = fig.add_gridspec(2, 2, left=0.055, right=0.995, top=0.96, bottom=0.08, wspace=0.20, hspace=0.30)
    for panel_index, dataset in enumerate(datasets):
        sub = outer[panel_index // 2, panel_index % 2].subgridspec(1, 2, width_ratios=[1.05, 1.55], wspace=0.04)
        ax = fig.add_subplot(sub[0, 0])
        table_ax = fig.add_subplot(sub[0, 1])
        rows = delong.loc[delong["dataset"] == dataset].copy()
        comparators = [model for model in (FUSIONS[1:] if dataset == A_DEVELOPMENT else SIX_MODELS[1:])]
        rows = rows.set_index("comparator_model").loc[comparators].reset_index()
        y = np.arange(len(rows), dtype=float)
        points = rows["delta_auc"].to_numpy(float)
        lows = rows["delta_ci_low"].to_numpy(float)
        highs = rows["delta_ci_high"].to_numpy(float)
        _setup_axis(ax, grid="x")
        ax.axvline(0, color="#777777", linestyle=":", linewidth=1.0)
        for index, comparator in enumerate(comparators):
            ax.errorbar(
                points[index],
                y[index],
                xerr=np.array([[points[index] - lows[index]], [highs[index] - points[index]]]),
                fmt="o",
                color=MODEL_COLORS[comparator],
                ecolor=MODEL_COLORS[comparator],
                markerfacecolor="white",
                markeredgewidth=1.1,
                markersize=5.2,
                capsize=2.8,
                elinewidth=1.2,
            )
        ax.set_xlim(*x_limits)
        ax.set_ylim(-0.78, len(rows) - 0.35)
        ax.invert_yaxis()
        ax.set_yticks(y, [MODEL_LABELS[model].replace(" (Primary)", "") for model in comparators])
        ax.tick_params(axis="y", labelsize=6.9)
        ax.set_xlabel("ΔAUC (Weighted voting - comparator)")
        ax.text(0.02, 0.98, DATASET_LABELS[dataset], transform=ax.transAxes, va="top", fontweight="bold", fontsize=8.4)
        panel_label(ax, "abcd"[panel_index], x=-0.16, y=1.02)

        table_ax.axis("off")
        table_ax.set_ylim(-0.78, len(rows) - 0.35)
        table_ax.invert_yaxis()
        table_ax.text(0.01, -0.58, "ΔAUC (95% CI)", fontsize=7.0, fontweight="bold")
        table_ax.text(0.69, -0.58, "P", fontsize=7.0, fontweight="bold")
        table_ax.text(0.83, -0.58, "Holm P", fontsize=7.0, fontweight="bold")
        for index, row in rows.iterrows():
            table_ax.text(0.01, index, _fmt_ci(float(row.delta_auc), float(row.delta_ci_low), float(row.delta_ci_high), sign=True), fontsize=6.5, va="center")
            table_ax.text(0.69, index, _fmt_p(float(row.p_raw)), fontsize=6.5, va="center")
            table_ax.text(0.83, index, _fmt_p(float(row.p_holm)), fontsize=6.5, va="center")
    fig.text(
        0.5,
        0.02,
        "Positive values favor Weighted voting. Holm correction was applied within each frozen dataset-specific comparison family.",
        ha="center",
        fontsize=7.3,
        color="#444444",
    )
    return _save_dual(fig, "unnumbered_delong_holm_vs_weighted_voting")


def main() -> None:
    apply_style()
    frames = _load_inputs()
    outputs = [
        _make_s1_layer_calibration(frames),
        _make_s2_layer_brier(frames),
        _make_s3_delong_holm(frames),
    ]
    for png_path, pdf_path in outputs:
        print(f"Wrote {png_path}")
        print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
