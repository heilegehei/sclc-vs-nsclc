from __future__ import annotations
import sys as _public_sys
from pathlib import Path as _PublicPath
_public_root = next(p for p in _PublicPath(__file__).resolve().parents if (p / "common" / "public_paths.py").is_file())
if str(_public_root) not in _public_sys.path:
    _public_sys.path.insert(0, str(_public_root))
from common.public_paths import cohort_workbook, code_root, data_root, explainability_results, fusion_results, publication_results

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from nature_style import apply_style, panel_label


ANALYSIS_DIR = Path(__file__).resolve().parent.parent
FINAL_ROOT = ANALYSIS_DIR.parent
SOURCE_V1 = data_root() / "fusion_results"
SOURCE_V2 = data_root() / "explainability_results"
MAIN_OUT = publication_results() / "figures" / "main"

ROC_PATH = SOURCE_V1 / "data" / "roc_curve_points.csv"
PR_PATH = SOURCE_V1 / "data" / "pr_curve_points.csv"
DCA_PATH = SOURCE_V1 / "data" / "dca_curve_points.csv"
METRICS_PATH = SOURCE_V1 / "tables" / "model_metrics_with_ci.csv"
CALIBRATION_PATH = SOURCE_V2 / "data" / "F9_F10_calibration_lowess_frac075.csv"

FUSIONS: tuple[str, ...] = (
    "weighted_voting_cv_auc",
    "soft_voting_native_probability",
    "stacking_in_sample_A_train",
)
PRIMARY = FUSIONS[0]
MODEL_LABELS = {
    "weighted_voting_cv_auc": "Weighted voting (Primary)",
    "soft_voting_native_probability": "Soft voting",
    "stacking_in_sample_A_train": "Stacking",
}
MODEL_COLORS = {
    "weighted_voting_cv_auc": "#2F5597",
    "soft_voting_native_probability": "#D55E00",
    "stacking_in_sample_A_train": "#009E73",
}
MODEL_LINESTYLES = {
    "weighted_voting_cv_auc": "-",
    "soft_voting_native_probability": "--",
    "stacking_in_sample_A_train": "-.",
}


def configure_rcparams() -> None:
    apply_style()

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 7.5,
        }
    )


def load_archived_frames(datasets: Sequence[str]) -> dict[str, pd.DataFrame]:
    source_paths = {
        "roc": ROC_PATH,
        "pr": PR_PATH,
        "dca": DCA_PATH,
        "metrics": METRICS_PATH,
        "calibration": CALIBRATION_PATH,
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required archived figure input(s) missing:\n" + "\n".join(missing))

    frames = {name: pd.read_csv(path) for name, path in source_paths.items()}
    required_datasets = set(datasets)
    for frame_name in ("roc", "pr", "dca", "metrics", "calibration"):
        observed = set(frames[frame_name]["dataset"].dropna().astype(str))
        absent = sorted(required_datasets - observed)
        if absent:
            raise RuntimeError(f"{frame_name} is missing frozen dataset(s): {absent}")

    for dataset in datasets:
        for model in FUSIONS:
            metric = frames["metrics"].loc[
                (frames["metrics"]["dataset"] == dataset)
                & (frames["metrics"]["model"] == model)
            ]
            if len(metric) != 1:
                raise RuntimeError(f"Expected one archived metric row for {dataset}/{model}; found {len(metric)}")
            for frame_name, point_column in (("roc", "point"), ("pr", "point")):
                point_rows = frames[frame_name].loc[
                    (frames[frame_name]["dataset"] == dataset)
                    & (frames[frame_name]["model"] == model)
                ]
                if point_rows.empty:
                    raise RuntimeError(f"{frame_name} has no archived points for {dataset}/{model}")
                if point_rows[point_column].duplicated().any():
                    raise RuntimeError(f"{frame_name} has duplicate point identifiers for {dataset}/{model}")

            dca = frames["dca"].loc[
                (frames["dca"]["dataset"] == dataset)
                & (frames["dca"]["curve_type"] == "model")
                & (frames["dca"]["model"] == model)
            ]
            if len(dca) != 160 or dca["threshold_grid_n"].nunique() != 1 or int(dca["threshold_grid_n"].iloc[0]) != 160:
                raise RuntimeError(f"Expected frozen 160-point DCA grid for {dataset}/{model}")

            calibration = frames["calibration"].loc[
                (frames["calibration"]["dataset"] == dataset)
                & (frames["calibration"]["model"] == model)
            ]
            kind_counts = calibration.groupby("point_kind").size().to_dict()
            if kind_counts.get("LOWESS") != 300 or kind_counts.get("equal-frequency bin") != 10:
                raise RuntimeError(
                    f"Expected archived 300-point LOWESS plus 10-bin calibration for {dataset}/{model}; got {kind_counts}"
                )

        for curve_type in ("treat_all", "treat_none"):
            baseline = frames["dca"].loc[
                (frames["dca"]["dataset"] == dataset)
                & (frames["dca"]["curve_type"] == curve_type)
            ]
            if len(baseline) != 160:
                raise RuntimeError(f"Expected 160 archived {curve_type} DCA points for {dataset}")
    return frames


def metric_row(metrics: pd.DataFrame, dataset: str, model: str) -> pd.Series:
    row = metrics.loc[(metrics["dataset"] == dataset) & (metrics["model"] == model)]
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one metric row for {dataset}/{model}; found {len(row)}")
    return row.iloc[0]


def setup_axis(ax: plt.Axes, *, grid: str | None = None) -> None:
    ax.tick_params(direction="out", length=3.0, width=0.8, pad=2.5)
    if grid is None:
        ax.grid(False)
    else:
        ax.grid(True, axis=grid, color="#E2E2E2", linewidth=0.55, zorder=0)
    ax.set_facecolor("white")


def model_handle(model: str) -> Line2D:
    return Line2D(
        [0],
        [0],
        color=MODEL_COLORS[model],
        linestyle=MODEL_LINESTYLES[model],
        linewidth=2.3 if model == PRIMARY else 1.45,
        solid_capstyle="round",
    )


def draw_model_line(ax: plt.Axes, model: str, x: Iterable[float], y: Iterable[float]) -> None:
    ax.plot(
        x,
        y,
        color=MODEL_COLORS[model],
        linestyle=MODEL_LINESTYLES[model],
        linewidth=2.3 if model == PRIMARY else 1.45,
        zorder=3 if model == PRIMARY else 2,
        solid_capstyle="round",
        solid_joinstyle="round",
    )


def legend_axis(fig: plt.Figure, subspec, *, legend_rows: float = 1.85) -> tuple[plt.Axes, plt.Axes]:
    inner = subspec.subgridspec(2, 1, height_ratios=[4.15, legend_rows], hspace=0.28)
    return fig.add_subplot(inner[0, 0]), fig.add_subplot(inner[1, 0])


def add_roc(ax: plt.Axes, legend_ax: plt.Axes, *, dataset: str, frames: dict[str, pd.DataFrame]) -> None:
    setup_axis(ax)
    ax.plot([0, 1], [0, 1], color="#777777", linestyle=":", linewidth=0.9, zorder=1)
    handles: list[Line2D] = []
    labels: list[str] = []
    for model in FUSIONS:
        curve = frames["roc"].loc[
            (frames["roc"]["dataset"] == dataset) & (frames["roc"]["model"] == model)
        ].sort_values("point", kind="mergesort")
        draw_model_line(ax, model, curve["fpr"], curve["tpr"])
        row = metric_row(frames["metrics"], dataset, model)
        handles.append(model_handle(model))
        labels.append(f"{MODEL_LABELS[model]}: AUC {float(row.auc):.3f} ({float(row.auc_ci_low):.3f}-{float(row.auc_ci_high):.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="1 - Specificity", ylabel="Sensitivity")
    ax.set_aspect("equal", adjustable="box")
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center", frameon=False, fontsize=6.25, handlelength=2.25, borderaxespad=0, labelspacing=0.18)


def add_pr(ax: plt.Axes, legend_ax: plt.Axes, *, dataset: str, frames: dict[str, pd.DataFrame]) -> None:
    setup_axis(ax)
    prevalence = float(metric_row(frames["metrics"], dataset, PRIMARY)["prevalence"])
    handles: list[Line2D] = []
    labels: list[str] = []
    for model in FUSIONS:
        curve = frames["pr"].loc[
            (frames["pr"]["dataset"] == dataset) & (frames["pr"]["model"] == model)
        ].sort_values("point", kind="mergesort")
        draw_model_line(ax, model, curve["recall"], curve["precision"])
        row = metric_row(frames["metrics"], dataset, model)
        handles.append(model_handle(model))
        labels.append(f"{MODEL_LABELS[model]}: AUPRC {float(row.auprc):.3f} ({float(row.auprc_ci_low):.3f}-{float(row.auprc_ci_high):.3f})")
    ax.axhline(prevalence, color="#777777", linestyle=":", linewidth=0.9, zorder=1)
    handles.append(Line2D([0], [0], color="#777777", linestyle=":", linewidth=0.9))
    labels.append(f"Prevalence: {prevalence:.3f}")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
    ax.set_aspect("equal", adjustable="box")
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center", frameon=False, fontsize=6.25, handlelength=2.25, borderaxespad=0, labelspacing=0.18)


def add_calibration(ax: plt.Axes, legend_ax: plt.Axes, *, dataset: str, frames: dict[str, pd.DataFrame]) -> None:
    setup_axis(ax, grid="both")
    ax.plot([0, 1], [0, 1], color="#6F6F6F", linestyle=":", linewidth=1.0, zorder=1)
    handles: list[Line2D] = []
    labels: list[str] = []
    for model in FUSIONS:
        base = frames["calibration"].loc[
            (frames["calibration"]["dataset"] == dataset) & (frames["calibration"]["model"] == model)
        ]
        curve = base.loc[base["point_kind"] == "LOWESS"].sort_values("point", kind="mergesort")
        bins = base.loc[base["point_kind"] == "equal-frequency bin"].sort_values("point", kind="mergesort")
        draw_model_line(ax, model, curve["mean_predicted"], curve["observed_fraction"])
        ax.scatter(
            bins["mean_predicted"],
            bins["observed_fraction"],
            s=14 if model == PRIMARY else 9,
            facecolor=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.82,
            zorder=4,
        )
        handles.append(model_handle(model))
        labels.append(MODEL_LABELS[model])
    handles.append(Line2D([0], [0], color="#6F6F6F", linestyle=":", linewidth=1.0))
    labels.append("Ideal")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted probability", ylabel="Observed proportion")
    ax.set_aspect("equal", adjustable="box")
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center", ncol=2, frameon=False, fontsize=6.15, handlelength=2.15, borderaxespad=0, columnspacing=0.55, labelspacing=0.18)


def add_dca(ax: plt.Axes, legend_ax: plt.Axes, *, dataset: str, frames: dict[str, pd.DataFrame]) -> None:
    setup_axis(ax)
    handles: list[Line2D] = []
    labels: list[str] = []
    model_values: list[np.ndarray] = []
    for model in FUSIONS:
        curve = frames["dca"].loc[
            (frames["dca"]["dataset"] == dataset)
            & (frames["dca"]["curve_type"] == "model")
            & (frames["dca"]["model"] == model)
        ].sort_values("threshold", kind="mergesort")
        draw_model_line(ax, model, curve["threshold"], curve["net_benefit"])
        model_values.append(curve["net_benefit"].to_numpy(float))
        handles.append(model_handle(model))
        labels.append(MODEL_LABELS[model])
    treat_all = frames["dca"].loc[
        (frames["dca"]["dataset"] == dataset) & (frames["dca"]["curve_type"] == "treat_all")
    ].sort_values("threshold", kind="mergesort")
    treat_none = frames["dca"].loc[
        (frames["dca"]["dataset"] == dataset) & (frames["dca"]["curve_type"] == "treat_none")
    ].sort_values("threshold", kind="mergesort")
    ax.plot(treat_all["threshold"], treat_all["net_benefit"], color="#777777", linestyle="--", linewidth=1.0, zorder=1)
    ax.plot(treat_none["threshold"], treat_none["net_benefit"], color="#111111", linestyle=":", linewidth=1.0, zorder=1)
    all_values = np.concatenate(model_values + [treat_all["net_benefit"].to_numpy(float)])
    lower = max(-0.18, float(np.quantile(all_values, 0.02)) - 0.015)
    upper = min(0.45, max(float(np.quantile(all_values, 0.99)) + 0.02, float(metric_row(frames["metrics"], dataset, PRIMARY)["prevalence"]) + 0.04))
    ax.set(xlim=(0.01, 0.80), ylim=(lower, upper), xlabel="Threshold probability", ylabel="Net benefit")
    handles.extend([Line2D([0], [0], color="#777777", linestyle="--", linewidth=1.0), Line2D([0], [0], color="#111111", linestyle=":", linewidth=1.0)])
    labels.extend(["Treat all", "Treat none"])
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center", ncol=2, frameon=False, fontsize=6.15, handlelength=2.15, borderaxespad=0, columnspacing=0.55, labelspacing=0.18)


def add_brier(ax: plt.Axes, legend_ax: plt.Axes, *, dataset: str, frames: dict[str, pd.DataFrame]) -> None:
    setup_axis(ax, grid="x")
    rows = [metric_row(frames["metrics"], dataset, model) for model in FUSIONS]
    values = np.asarray([float(row["brier"]) for row in rows])
    lows = np.asarray([float(row["brier_ci_low"]) for row in rows])
    highs = np.asarray([float(row["brier_ci_high"]) for row in rows])
    positions = np.arange(len(FUSIONS), dtype=float)
    for position, model, value, low, high in zip(positions, FUSIONS, values, lows, highs):
        ax.errorbar(
            value,
            position,
            xerr=np.asarray([[value - low], [high - value]]),
            fmt="o",
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model],
            elinewidth=1.5 if model == PRIMARY else 1.0,
            capsize=2.5,
            markersize=5.2 if model == PRIMARY else 4.2,
            markeredgecolor="white",
            markeredgewidth=0.4,
            zorder=3,
        )
    ax.set_yticks(positions, [MODEL_LABELS[model] for model in FUSIONS])
    ax.invert_yaxis()
    for tick, model in zip(ax.get_yticklabels(), FUSIONS):
        tick.set_fontsize(6.4)
        if model == PRIMARY:
            tick.set_fontweight("bold")
    spread = max(0.006, float(highs.max() - lows.min()))
    ax.set_xlim(max(0.0, float(lows.min() - 0.14 * spread)), min(1.0, float(highs.max() + 0.42 * spread)))
    ax.set_xlabel("Brier score")
    ax.set_ylabel("")
    for position, value, high in zip(positions, values, highs):
        ax.text(float(high) + 0.035 * spread, position, f"{value:.3f}", va="center", ha="left", fontsize=6.15)
    legend_ax.axis("off")


def add_measure_panel(
    fig: plt.Figure,
    subspec,
    *,
    letter: str,
    measure: str,
    dataset: str,
    frames: dict[str, pd.DataFrame],
) -> None:
    ax, legend_ax = legend_axis(fig, subspec, legend_rows=0.12 if measure == "Brier" else 1.85)
    panel_label(ax, letter, x=-0.16, y=1.02, fontsize=11)
    if measure == "ROC":
        add_roc(ax, legend_ax, dataset=dataset, frames=frames)
    elif measure == "PR":
        add_pr(ax, legend_ax, dataset=dataset, frames=frames)
    elif measure == "Calibration":
        add_calibration(ax, legend_ax, dataset=dataset, frames=frames)
    elif measure == "DCA":
        add_dca(ax, legend_ax, dataset=dataset, frames=frames)
    elif measure == "Brier":
        add_brier(ax, legend_ax, dataset=dataset, frames=frames)
    else:
        raise ValueError(f"Unsupported measure: {measure}")


def save_vector_pair(fig: plt.Figure, basename: str) -> tuple[Path, Path]:
    MAIN_OUT.mkdir(parents=True, exist_ok=True)
    pdf_path = MAIN_OUT / f"{basename}.pdf"
    png_path = MAIN_OUT / f"{basename}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, format="png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if not pdf_path.is_file() or not png_path.is_file():
        raise RuntimeError(f"Did not create both outputs for {basename}")
    return pdf_path, png_path


def source_paths_text() -> str:
    return "\n".join(str(path) for path in (ROC_PATH, PR_PATH, DCA_PATH, METRICS_PATH, CALIBRATION_PATH))
