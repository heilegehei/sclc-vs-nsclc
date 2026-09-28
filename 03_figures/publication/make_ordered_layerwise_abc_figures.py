from __future__ import annotations
import sys as _public_sys
from pathlib import Path as _PublicPath
_public_root = next(p for p in _PublicPath(__file__).resolve().parents if (p / "common" / "public_paths.py").is_file())
if str(_public_root) not in _public_sys.path:
    _public_sys.path.insert(0, str(_public_root))
from common.public_paths import cohort_workbook, code_root, data_root, explainability_results, fusion_results, publication_results

from pathlib import Path
import string

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from nature_style import apply_style, panel_label


HERE = Path(__file__).resolve().parent
V4_ROOT = HERE.parent
FINAL_ROOT = V4_ROOT.parent
ARCHIVE_ROOT = data_root() / "fusion_results"
OUTPUT_DIR = publication_results() / "figures" / "ordered_all"
PREDICTIONS_INPUT = ARCHIVE_ROOT / "data" / "weighted_voting_layer_predictions.csv"


METRICS_INPUT = data_root() / "explainability_results" / "tables" / "weighted_voting_layer_metrics_ci_harmonised.csv"
CALIBRATION_INPUT = ARCHIVE_ROOT / "data" / "weighted_voting_layer_calibration_plot_data.csv"

MODEL = "weighted_voting_cv_auc"
DATASETS = [
    "A_holdout_clean",
    "B_external",
    "C_external",
]
DATASET_LABELS = {
    "A_holdout_clean": "A evaluation set",
    "B_external": "B external",
    "C_external": "C external",
}
LAYERS = ["I", "I_M", "I_M_N"]
LAYER_LABELS = {"I": "I", "I_M": "I + M", "I_M_N": "I + M + N"}

LAYER_COLORS = {"I": "#D55E00", "I_M": "#2F5597", "I_M_N": "#009E73"}
LAYER_LINESTYLES = {"I": "--", "I_M": "-.", "I_M_N": "-"}
THRESHOLDS = np.linspace(0.01, 0.80, 160)


def _save(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUTPUT_DIR / f"{stem}.pdf"
    png = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    return pdf, png


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [str(path) for path in (PREDICTIONS_INPUT, METRICS_INPUT, CALIBRATION_INPUT) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required frozen source file(s) missing:\n" + "\n".join(missing))
    predictions = pd.read_csv(PREDICTIONS_INPUT, usecols=["dataset", "layer", "model", "record_id", "y_SCLC", "probability"])
    predictions = predictions.loc[
        predictions["dataset"].isin(DATASETS) & predictions["layer"].isin(LAYERS) & predictions["model"].eq(MODEL)
    ].copy()
    metrics = pd.read_csv(METRICS_INPUT)
    metrics = metrics.loc[
        metrics["dataset"].isin(DATASETS) & metrics["layer"].isin(LAYERS) & metrics["model"].eq(MODEL)
    ].copy()
    calibration = pd.read_csv(CALIBRATION_INPUT)
    calibration = calibration.loc[calibration["dataset"].isin(DATASETS) & calibration["layer"].isin(LAYERS)].copy()
    if len(metrics) != len(DATASETS) * len(LAYERS) or predictions.empty:
        raise RuntimeError("Unexpected frozen layer source scope.")
    for dataset in DATASETS:
        ids: tuple[str, ...] | None = None
        labels: np.ndarray | None = None
        for layer in LAYERS:
            group = predictions.loc[predictions["dataset"].eq(dataset) & predictions["layer"].eq(layer)].copy()
            if group.empty or group["record_id"].isna().any() or not group["record_id"].astype(str).is_unique:
                raise RuntimeError(f"Incomplete or duplicate frozen records for {dataset}/{layer}.")
            group["probability"] = pd.to_numeric(group["probability"], errors="coerce")
            group["y_SCLC"] = pd.to_numeric(group["y_SCLC"], errors="coerce")
            if group[["probability", "y_SCLC"]].isna().any().any() or not group["probability"].between(0, 1).all():
                raise RuntimeError(f"Invalid frozen probability or outcome for {dataset}/{layer}.")
            group["record_id"] = group["record_id"].astype(str)
            group = group.sort_values("record_id", kind="mergesort")
            current_ids = tuple(group["record_id"].astype(str))
            current_labels = group["y_SCLC"].to_numpy(dtype=int)
            if ids is None:
                ids, labels = current_ids, current_labels
            elif current_ids != ids or not np.array_equal(current_labels, labels):
                raise RuntimeError(f"Layer records/outcomes are not aligned for {dataset}.")
    type_sets = calibration.groupby(["dataset", "layer"])["series_type"].agg(set)
    if len(type_sets) != 9 or not type_sets.map(lambda value: {"LOWESS", "equal_frequency_bin"}.issubset(value)).all():
        raise RuntimeError("Incomplete archived calibration display data.")
    return predictions, metrics, calibration


def _prediction_group(predictions: pd.DataFrame, dataset: str, layer: str) -> pd.DataFrame:
    return predictions.loc[predictions["dataset"].eq(dataset) & predictions["layer"].eq(layer)].sort_values("record_id", kind="mergesort")


def _metric_row(metrics: pd.DataFrame, dataset: str, layer: str) -> pd.Series:
    rows = metrics.loc[metrics["dataset"].eq(dataset) & metrics["layer"].eq(layer)]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one metrics row for {dataset}/{layer}.")
    return rows.iloc[0]


def _metric_label(row: pd.Series, layer: str, metric: str) -> str:
    name = "AUC" if metric == "auc" else "AUPRC"
    return f"{LAYER_LABELS[layer]}  {name} {row[metric]:.3f} ({row[f'{metric}_ci_low']:.3f}–{row[f'{metric}_ci_high']:.3f})"


def _layer_width(layer: str) -> float:
    return 2.25 if layer == "I_M_N" else 1.35


def _layer_handles() -> list[Line2D]:
    return [
        Line2D(
            [0], [0], color=LAYER_COLORS[layer], linestyle=LAYER_LINESTYLES[layer],
            linewidth=_layer_width(layer), label=LAYER_LABELS[layer],
        )
        for layer in LAYERS
    ]


def _base_figure() -> tuple[plt.Figure, list[tuple[plt.Axes, plt.Axes]]]:
    fig = plt.figure(figsize=(16.8, 7.15), facecolor="white")
    outer = fig.add_gridspec(1, 3, left=0.055, right=0.992, top=0.945, bottom=0.060, wspace=0.28)
    panels: list[tuple[plt.Axes, plt.Axes]] = []
    for subspec in outer:
        inner = subspec.subgridspec(2, 1, height_ratios=[4.25, 1.75], hspace=0.24)
        axis = fig.add_subplot(inner[0, 0])
        legend = fig.add_subplot(inner[1, 0])
        legend.axis("off")
        panels.append((axis, legend))
    return fig, panels


def _setup_axis(axis: plt.Axes, *, grid: str | None = None, box_aspect: float | None = None) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out")
    if grid == "both":
        axis.grid(color="#E9E9E9", linewidth=0.55, alpha=0.85)
    elif grid:
        axis.grid(axis=grid, color="#E9E9E9", linewidth=0.55, alpha=0.85)
    if box_aspect is not None:
        axis.set_box_aspect(box_aspect)


def _annotate_panel(axis: plt.Axes, dataset: str, letter: str) -> None:
    panel_label(axis, letter, x=-0.14, y=1.025, fontsize=12)
    axis.text(
        0.02, 0.98, DATASET_LABELS[dataset], transform=axis.transAxes,
        ha="left", va="top", fontsize=8.5, fontweight="bold", zorder=5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 1.1},
    )


def _legend_below(
    legend: plt.Axes,
    handles: list[Line2D],
    labels: list[str],
    *,
    columns: int = 1,
) -> None:
    legend.legend(
        handles, labels, loc="center", ncol=columns, frameon=False,
        fontsize=6.20, handlelength=2.25, columnspacing=0.82,
        labelspacing=0.22, borderaxespad=0.0,
    )


def make_roc(predictions: pd.DataFrame, metrics: pd.DataFrame) -> tuple[Path, Path]:
    fig, panels = _base_figure()
    for index, ((axis, legend), dataset) in enumerate(zip(panels, DATASETS, strict=True)):
        _setup_axis(axis)
        handles: list[Line2D] = []
        labels: list[str] = []
        for layer in LAYERS:
            group = _prediction_group(predictions, dataset, layer)
            y = group["y_SCLC"].to_numpy(dtype=int)
            probability = group["probability"].to_numpy(dtype=float)
            fpr, tpr, _ = roc_curve(y, probability)
            row = _metric_row(metrics, dataset, layer)
            if not np.isclose(roc_auc_score(y, probability), float(row["auc"]), atol=1e-12):
                raise RuntimeError(f"AUC disagreement for {dataset}/{layer}.")
            axis.plot(
                fpr, tpr, color=LAYER_COLORS[layer], linestyle=LAYER_LINESTYLES[layer],
                linewidth=_layer_width(layer), zorder=3 if layer == "I_M_N" else 2,
                solid_capstyle="round", solid_joinstyle="round",
            )
            handles.append(Line2D([0], [0], color=LAYER_COLORS[layer], linestyle=LAYER_LINESTYLES[layer], linewidth=_layer_width(layer)))
            labels.append(_metric_label(row, layer, "auc"))
        axis.plot([0, 1], [0, 1], color="#707070", linestyle=":", linewidth=0.9, zorder=0)
        axis.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="1 - Specificity", ylabel="Sensitivity")
        axis.set_aspect("equal", adjustable="box")
        _annotate_panel(axis, dataset, string.ascii_lowercase[index])
        _legend_below(legend, handles, labels)
    result = _save(fig, "Fig6_layerwise_ABC_roc")
    plt.close(fig)
    return result


def make_pr(predictions: pd.DataFrame, metrics: pd.DataFrame) -> tuple[Path, Path]:
    fig, panels = _base_figure()
    for index, ((axis, legend), dataset) in enumerate(zip(panels, DATASETS, strict=True)):
        _setup_axis(axis)
        handles: list[Line2D] = []
        labels: list[str] = []
        prevalence: float | None = None
        for layer in LAYERS:
            group = _prediction_group(predictions, dataset, layer)
            y = group["y_SCLC"].to_numpy(dtype=int)
            probability = group["probability"].to_numpy(dtype=float)
            precision, recall, _ = precision_recall_curve(y, probability)
            row = _metric_row(metrics, dataset, layer)
            if not np.isclose(average_precision_score(y, probability), float(row["auprc"]), atol=1e-12):
                raise RuntimeError(f"AUPRC disagreement for {dataset}/{layer}.")
            axis.plot(
                recall, precision, color=LAYER_COLORS[layer], linestyle=LAYER_LINESTYLES[layer],
                linewidth=_layer_width(layer), zorder=3 if layer == "I_M_N" else 2,
                solid_capstyle="round", solid_joinstyle="round",
            )
            handles.append(Line2D([0], [0], color=LAYER_COLORS[layer], linestyle=LAYER_LINESTYLES[layer], linewidth=_layer_width(layer)))
            labels.append(_metric_label(row, layer, "auprc"))
            prevalence = float(y.mean())
        if prevalence is not None:
            axis.axhline(prevalence, color="#707070", linestyle=":", linewidth=0.9, zorder=0)
        axis.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Recall", ylabel="Precision")
        axis.set_aspect("equal", adjustable="box")
        _annotate_panel(axis, dataset, string.ascii_lowercase[index])
        _legend_below(legend, handles, labels)
    result = _save(fig, "FigS1_layerwise_ABC_precision_recall")
    plt.close(fig)
    return result


def make_calibration(calibration: pd.DataFrame) -> tuple[Path, Path]:
    fig, panels = _base_figure()
    for index, ((axis, legend), dataset) in enumerate(zip(panels, DATASETS, strict=True)):
        _setup_axis(axis, grid="both")
        for layer in LAYERS:
            group = calibration.loc[calibration["dataset"].eq(dataset) & calibration["layer"].eq(layer)]
            smooth = group.loc[group["series_type"].eq("LOWESS")].sort_values("x", kind="mergesort")
            bins = group.loc[group["series_type"].eq("equal_frequency_bin")].sort_values("bin", kind="mergesort")
            axis.plot(
                smooth["x"], smooth["y"], color=LAYER_COLORS[layer], linestyle=LAYER_LINESTYLES[layer],
                linewidth=_layer_width(layer), zorder=3 if layer == "I_M_N" else 2,
                solid_capstyle="round", solid_joinstyle="round",
            )
            axis.scatter(bins["x"], bins["y"], s=20 if layer == "I_M_N" else 15,
                         facecolors="white", edgecolors=LAYER_COLORS[layer], linewidths=0.75, zorder=4)
        axis.plot([0, 1], [0, 1], color="#707070", linestyle=":", linewidth=0.9, zorder=0)
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed SCLC fraction")
        axis.set_aspect("equal", adjustable="box")
        _annotate_panel(axis, dataset, string.ascii_lowercase[index])
        ideal = Line2D([0], [0], color="#707070", linestyle=":", linewidth=0.9)
        _legend_below(
            legend, _layer_handles() + [ideal], [LAYER_LABELS[layer] for layer in LAYERS] + ["Ideal"],
        )
    result = _save(fig, "FigS3_layerwise_ABC_calibration")
    plt.close(fig)
    return result


def _net_benefit(y: np.ndarray, probability: np.ndarray) -> np.ndarray:
    predicted_positive = probability[:, None] >= THRESHOLDS[None, :]
    tp = ((y[:, None] == 1) & predicted_positive).sum(axis=0)
    fp = ((y[:, None] == 0) & predicted_positive).sum(axis=0)
    return tp / len(y) - fp / len(y) * THRESHOLDS / (1.0 - THRESHOLDS)


def _dca_figure() -> tuple[plt.Figure, list[plt.Axes]]:
    fig = plt.figure(figsize=(16.8, 5.80), facecolor="white")
    grid = fig.add_gridspec(
        1, 3, left=0.055, right=0.992, top=0.945, bottom=0.205, wspace=0.28
    )
    return fig, [fig.add_subplot(grid[0, index]) for index in range(3)]


def _annotate_dca_panel(axis: plt.Axes, dataset: str, letter: str) -> None:
    panel_label(axis, letter, x=-0.14, y=1.025, fontsize=12)
    axis.text(
        0.98, 0.975, DATASET_LABELS[dataset], transform=axis.transAxes,
        ha="right", va="top", fontsize=8.5, fontweight="bold", zorder=5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 1.1},
    )


def make_dca(predictions: pd.DataFrame) -> tuple[Path, Path]:
    fig, axes = _dca_figure()
    for index, (axis, dataset) in enumerate(zip(axes, DATASETS, strict=True)):
        _setup_axis(axis, box_aspect=0.68)
        reference = _prediction_group(predictions, dataset, "I")
        y = reference["y_SCLC"].to_numpy(dtype=int)
        for layer in LAYERS:
            probability = _prediction_group(predictions, dataset, layer)["probability"].to_numpy(dtype=float)
            axis.plot(
                THRESHOLDS, _net_benefit(y, probability), color=LAYER_COLORS[layer],
                linestyle=LAYER_LINESTYLES[layer], linewidth=_layer_width(layer),
                zorder=4 if layer == "I_M_N" else 3, solid_capstyle="round", solid_joinstyle="round",
            )
        prevalence = float(y.mean())
        treat_all = prevalence - (1.0 - prevalence) * THRESHOLDS / (1.0 - THRESHOLDS)

        treat_all = np.where(treat_all >= -0.05, treat_all, np.nan)
        axis.plot(THRESHOLDS, treat_all, color="#202020", linestyle="--", linewidth=0.95, zorder=1)
        axis.axhline(0, color="#9A9A9A", linestyle=":", linewidth=0.95, zorder=1)
        axis.set(xlim=(0.01, 0.80), ylim=(-0.05, 0.20), xlabel="Threshold probability", ylabel="Net benefit")
        axis.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
        _annotate_dca_panel(axis, dataset, string.ascii_lowercase[index])
    references = [
        Line2D([0], [0], color="#202020", linestyle="--", linewidth=0.95),
        Line2D([0], [0], color="#9A9A9A", linestyle=":", linewidth=0.95),
    ]
    fig.legend(
        _layer_handles() + references,
        [LAYER_LABELS[layer] for layer in LAYERS] + ["Treat all", "Treat none"],
        loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=5, frameon=False,
        fontsize=7.4, handlelength=2.35, columnspacing=1.05, borderaxespad=0.0,
    )
    result = _save(fig, "FigS4_layerwise_ABC_decision_curve")
    plt.close(fig)
    return result


def make_brier(metrics: pd.DataFrame) -> tuple[Path, Path]:

    fig = plt.figure(figsize=(16.8, 5.30), facecolor="white")
    outer = fig.add_gridspec(1, 3, left=0.055, right=0.992, top=0.945, bottom=0.120, wspace=0.28)
    panels = [(fig.add_subplot(subspec), None) for subspec in outer]
    low = float(pd.to_numeric(metrics["brier_ci_low"], errors="raise").min())
    high = float(pd.to_numeric(metrics["brier_ci_high"], errors="raise").max())
    margin = max(0.008, 0.10 * (high - low))
    for index, ((axis, _), dataset) in enumerate(zip(panels, DATASETS, strict=True)):
        _setup_axis(axis, grid="x", box_aspect=0.68)
        positions = np.arange(len(LAYERS))
        for position, layer in enumerate(LAYERS):
            row = _metric_row(metrics, dataset, layer)
            value, ci_low, ci_high = (float(row[column]) for column in ("brier", "brier_ci_low", "brier_ci_high"))
            axis.errorbar(value, position, xerr=np.array([[value - ci_low], [ci_high - value]]), fmt="o", color=LAYER_COLORS[layer], ecolor=LAYER_COLORS[layer], markersize=5.0, markeredgecolor="white", markeredgewidth=0.42, elinewidth=1.05, capsize=2.0, zorder=3)
            axis.text(ci_high + 0.0010, position, f"{value:.3f}", va="center", ha="left", fontsize=7.1, color=LAYER_COLORS[layer])
        axis.set(yticks=positions, yticklabels=[LAYER_LABELS[layer] for layer in LAYERS], xlim=(max(0.0, low - margin), high + margin), xlabel="Brier score (95% CI)")
        axis.invert_yaxis()
        _annotate_panel(axis, dataset, string.ascii_lowercase[index])
    result = _save(fig, "FigS2_layerwise_ABC_brier_score")
    plt.close(fig)
    return result


def main() -> None:
    predictions, metrics, calibration = _load_inputs()
    apply_style()
    outputs = [
        make_roc(predictions, metrics),
        make_pr(predictions, metrics),
        make_calibration(calibration),
        make_dca(predictions),
        make_brier(metrics),
    ]
    print(f"SOURCE_ROWS={len(predictions)}; METRIC_ROWS={len(metrics)}; CALIBRATION_ROWS={len(calibration)}")
    for pdf, png in outputs:
        print(f"WROTE {pdf}")
        print(f"WROTE {png}")


if __name__ == "__main__":
    main()
