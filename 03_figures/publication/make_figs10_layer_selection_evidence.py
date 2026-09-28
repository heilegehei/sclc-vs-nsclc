from __future__ import annotations
import sys as _public_sys
from pathlib import Path as _PublicPath
_public_root = next(p for p in _PublicPath(__file__).resolve().parents if (p / "common" / "public_paths.py").is_file())
if str(_public_root) not in _public_sys.path:
    _public_sys.path.insert(0, str(_public_root))
from common.public_paths import cohort_workbook, code_root, data_root, explainability_results, fusion_results, publication_results


from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


HERE = Path(__file__).resolve().parent
V4_ROOT = HERE.parent
SOURCE_TABLES = fusion_results() / "tables"
OUT_DIR = publication_results() / "figures" / "supplementary"

METRICS_PATH = SOURCE_TABLES / "weighted_voting_layer_metrics_ci.csv"
INCREMENTS_PATH = SOURCE_TABLES / "weighted_voting_layer_increment_ci.csv"
DELONG_PATH = SOURCE_TABLES / "weighted_voting_layer_delong_holm.csv"

DATASET = "A_dev_cv_clean"
MODEL = "weighted_voting_cv_auc"
LAYERS = ["I", "I_M", "I_M_N"]
LAYER_LABELS = {"I": "I", "I_M": "I + M", "I_M_N": "I + M + N"}
COMPARISONS = ["I → I + M", "I + M → I + M + N", "I → I + M + N"]

AUC_COLOR = "#2F5597"
AUPRC_COLOR = "#D55E00"
FOREST_COLOR = "#2F5597"
ZERO_COLOR = "#7A7A7A"
GRID_COLOR = "#D7D7D7"


def _require_times_new_roman() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    if "Times New Roman" not in available:
        raise RuntimeError("Times New Roman is required but was not found.")


def _load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for source in (METRICS_PATH, INCREMENTS_PATH, DELONG_PATH):
        if not source.is_file():
            raise FileNotFoundError(f"Required frozen source table is missing: {source}")

    metrics = pd.read_csv(METRICS_PATH)
    increments = pd.read_csv(INCREMENTS_PATH)
    delong = pd.read_csv(DELONG_PATH)

    metrics = metrics.loc[
        (metrics["dataset"] == DATASET) & (metrics["model"] == MODEL)
    ].copy()
    metrics["layer"] = pd.Categorical(metrics["layer"], LAYERS, ordered=True)
    metrics = metrics.sort_values("layer").reset_index(drop=True)
    if metrics["layer"].astype(str).tolist() != LAYERS or len(metrics) != len(LAYERS):
        raise ValueError("Frozen metrics table does not contain exactly I, I + M, and I + M + N.")
    required_metric_columns = {
        "auc", "auc_ci_low", "auc_ci_high", "auprc", "auprc_ci_low", "auprc_ci_high"
    }
    if missing := required_metric_columns.difference(metrics.columns):
        raise ValueError(f"Frozen metrics table lacks: {sorted(missing)}")
    if metrics[list(required_metric_columns)].isna().any().any():
        raise ValueError("Frozen metric values or confidence intervals contain missing values.")

    increments = increments.loc[
        (increments["dataset"] == DATASET)
        & (increments["model"] == MODEL)
        & (increments["comparison"].isin(COMPARISONS))
        & (increments["metric"].isin(["auc", "auprc"]))
    ].copy()
    increments["comparison"] = pd.Categorical(
        increments["comparison"], COMPARISONS, ordered=True
    )
    increments = increments.sort_values(["comparison", "metric"]).reset_index(drop=True)
    expected_increment_pairs = {(comparison, metric) for comparison in COMPARISONS for metric in ("auc", "auprc")}
    observed_increment_pairs = set(zip(increments["comparison"].astype(str), increments["metric"]))
    if observed_increment_pairs != expected_increment_pairs or len(increments) != 6:
        raise ValueError("Frozen increment table does not contain the six expected AUC/AUPRC entries.")

    delong = delong.loc[
        (delong["dataset"] == DATASET)
        & (delong["model"] == MODEL)
        & (delong["comparison"].isin(COMPARISONS))
    ].copy()
    delong["comparison"] = pd.Categorical(delong["comparison"], COMPARISONS, ordered=True)
    delong = delong.sort_values("comparison").reset_index(drop=True)
    if delong["comparison"].astype(str).tolist() != COMPARISONS or len(delong) != 3:
        raise ValueError("Frozen DeLong table does not contain exactly the three pre-specified comparisons.")
    if (delong["family_n"].astype(int) != 3).any():
        raise ValueError("The DeLong/Holm family must contain the three pre-specified comparisons.")
    required_delong_columns = {
        "delta_auc", "ci_low", "ci_high", "p_raw", "p_holm"
    }
    if missing := required_delong_columns.difference(delong.columns):
        raise ValueError(f"Frozen DeLong table lacks: {sorted(missing)}")
    if delong[list(required_delong_columns)].isna().any().any():
        raise ValueError("Frozen DeLong values contain missing data.")

    return metrics, increments, delong


def _fmt_delta(value: float) -> str:
    return f"{value:+.3f}"


def _fmt_ci(low: float, high: float) -> str:
    return f"[{low:+.3f}, {high:+.3f}]"


def _fmt_p(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _make_increment_table(ax: plt.Axes, increments: pd.DataFrame) -> None:
    ax.axis("off")
    rows: list[list[str]] = []
    for comparison in COMPARISONS:
        subset = increments.loc[increments["comparison"].astype(str) == comparison].set_index("metric")
        auc = subset.loc["auc"]
        auprc = subset.loc["auprc"]
        rows.append(
            [
                comparison,
                f"{_fmt_delta(float(auc['increment']))} {_fmt_ci(float(auc['ci_low']), float(auc['ci_high']))}",
                f"{_fmt_delta(float(auprc['increment']))} {_fmt_ci(float(auprc['ci_low']), float(auprc['ci_high']))}",
            ]
        )

    table = ax.table(
        cellText=rows,
        colLabels=["Stored increment", "Δ AUC (95% CI)", "Δ AUPRC (95% CI)"],
        colWidths=[0.30, 0.35, 0.35],
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.0, 0.00, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.7)
    for (row, _column), cell in table.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor("#BFBFBF")
        if row == 0:
            cell.set_facecolor("#EEF2F7")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#FFFFFF")


def _make_panel_a(ax: plt.Axes, table_ax: plt.Axes, metrics: pd.DataFrame, increments: pd.DataFrame) -> None:
    x = np.arange(len(LAYERS), dtype=float)
    offsets = {"auc": -0.12, "auprc": 0.12}
    styles = {
        "auc": (AUC_COLOR, "AUC"),
        "auprc": (AUPRC_COLOR, "AUPRC"),
    }

    for metric, (color, display) in styles.items():
        values = metrics[metric].astype(float).to_numpy()
        low = metrics[f"{metric}_ci_low"].astype(float).to_numpy()
        high = metrics[f"{metric}_ci_high"].astype(float).to_numpy()
        x_plot = x + offsets[metric]
        ax.plot(x_plot, values, color=color, linewidth=1.25, alpha=0.82, zorder=2)
        ax.errorbar(
            x_plot,
            values,
            yerr=np.vstack([values - low, high - values]),
            fmt="o",
            color=color,
            markerfacecolor="white",
            markeredgewidth=1.3,
            markersize=5.2,
            capsize=2.6,
            elinewidth=1.05,
            label=display,
            zorder=3,
        )
        for x_pos, value in zip(x_plot, values, strict=True):
            ax.text(
                x_pos,
                value + (0.020 if metric == "auc" else -0.028),
                f"{value:.3f}",
                ha="center",
                va="bottom" if metric == "auc" else "top",
                color=color,
                fontsize=7.1,
                fontweight="bold",
            )

    ax.set_xlim(-0.50, 2.50)
    ax.set_ylim(0.25, 0.86)
    ax.set_xticks(x, [LAYER_LABELS[layer] for layer in LAYERS])
    ax.set_ylabel("Discrimination metric")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.55, alpha=0.75)
    ax.legend(loc="upper left", frameon=False, handlelength=1.3, borderaxespad=0.25)
    ax.text(
        -0.15,
        1.04,
        "a",
        transform=ax.transAxes,
        fontsize=12,
        fontstyle="italic",
        fontweight="bold",
        va="bottom",
    )
    _make_increment_table(table_ax, increments)


def _make_panel_b(forest_ax: plt.Axes, stats_ax: plt.Axes, delong: pd.DataFrame) -> None:
    y = np.arange(len(COMPARISONS) - 1, -1, -1, dtype=float)
    values = delong["delta_auc"].astype(float).to_numpy()
    low = delong["ci_low"].astype(float).to_numpy()
    high = delong["ci_high"].astype(float).to_numpy()

    forest_ax.axvline(0.0, color=ZERO_COLOR, linewidth=0.9, linestyle="--", zorder=0)
    forest_ax.errorbar(
        values,
        y,
        xerr=np.vstack([values - low, high - values]),
        fmt="o",
        color=FOREST_COLOR,
        markerfacecolor="white",
        markeredgewidth=1.35,
        markersize=5.6,
        capsize=2.8,
        elinewidth=1.2,
        zorder=3,
    )
    forest_ax.set_yticks(y, COMPARISONS)
    forest_ax.set_ylim(-0.65, 2.65)
    forest_ax.set_xlim(-0.060, 0.160)
    forest_ax.set_xlabel("Δ AUC (upper − lower)")
    forest_ax.grid(axis="x", color=GRID_COLOR, linewidth=0.55, alpha=0.75)
    forest_ax.text(
        -0.18,
        1.04,
        "b",
        transform=forest_ax.transAxes,
        fontsize=12,
        fontstyle="italic",
        fontweight="bold",
        va="bottom",
    )
    stats_ax.axis("off")
    stats_ax.set_xlim(0, 1)
    stats_ax.set_ylim(-0.65, 2.65)
    stats_ax.text(0.02, 2.46, "Δ AUC (95% CI)", fontsize=7.2, fontweight="bold", va="center")
    stats_ax.text(0.61, 2.46, "P raw", fontsize=7.2, fontweight="bold", va="center", ha="center")
    stats_ax.text(0.88, 2.46, "P Holm", fontsize=7.2, fontweight="bold", va="center", ha="center")
    stats_ax.hlines(2.27, 0.02, 0.99, color="#BFBFBF", linewidth=0.55)
    for y_pos, (_, row) in zip(y, delong.iterrows(), strict=True):
        stats_ax.text(
            0.02,
            y_pos,
            f"{_fmt_delta(float(row['delta_auc']))} {_fmt_ci(float(row['ci_low']), float(row['ci_high']))}",
            fontsize=7.0,
            va="center",
        )
        stats_ax.text(0.61, y_pos, _fmt_p(float(row["p_raw"])), fontsize=7.0, va="center", ha="center")
        stats_ax.text(0.88, y_pos, _fmt_p(float(row["p_holm"])), fontsize=7.0, va="center", ha="center")
        stats_ax.hlines(y_pos - 0.40, 0.02, 0.99, color="#E2E2E2", linewidth=0.45)


def main() -> tuple[Path, Path]:
    _require_times_new_roman()
    metrics, increments, delong = _load_and_validate()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12.2, 4.35))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.18, 1.60],
        wspace=0.42,
        left=0.065,
        right=0.985,
        top=0.88,
        bottom=0.14,
    )
    left = outer[0].subgridspec(2, 1, height_ratios=[0.70, 0.30], hspace=0.16)
    ax_a = fig.add_subplot(left[0])
    ax_increment_table = fig.add_subplot(left[1])
    right = outer[1].subgridspec(1, 2, width_ratios=[0.63, 0.77], wspace=0.05)
    ax_b = fig.add_subplot(right[0])
    ax_b_stats = fig.add_subplot(right[1])

    _make_panel_a(ax_a, ax_increment_table, metrics, increments)
    _make_panel_b(ax_b, ax_b_stats, delong)
    fig.text(
        0.065,
        0.025,
        "A development (10-fold CV): selection evidence. Whiskers and tabulated intervals report stored 95% confidence intervals.\n"
        "Not a performance-validation display.",
        fontsize=7.6,
        fontstyle="italic",
        ha="left",
        va="bottom",
    )

    pdf_path = OUT_DIR / "unnumbered_layer_selection_evidence.pdf"
    png_path = OUT_DIR / "unnumbered_layer_selection_evidence.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    return pdf_path, png_path


if __name__ == "__main__":
    pdf, png = main()
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
