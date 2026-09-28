from __future__ import annotations
import sys as _public_sys
from pathlib import Path as _PublicPath
_public_root = next(p for p in _PublicPath(__file__).resolve().parents if (p / "common" / "public_paths.py").is_file())
if str(_public_root) not in _public_sys.path:
    _public_sys.path.insert(0, str(_public_root))
from common.public_paths import cohort_workbook, code_root, data_root, explainability_results, fusion_results, publication_results


import sys as _cohort_sys
from pathlib import Path as _CohortPath
_cohort_root = next(p for p in _CohortPath(__file__).resolve().parents if (p / "common" / "manuscript_cohorts.py").is_file())
if str(_cohort_root) not in _cohort_sys.path:
    _cohort_sys.path.insert(0, str(_cohort_root))
from common.manuscript_cohorts import split_center_a_indices, validate_cohort, validate_base_cohorts, split_metadata

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from nature_style import FEATURE_GROUP_COLORS, apply_style, panel_label, save_fig


HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parent
SUBMISSION_ROOT = OUT_ROOT.parent
PROJECT_ROOT = data_root()

WORKBOOK = cohort_workbook()
SPLIT_RECORDS = (
    PROJECT_ROOT
    / "artifacts"
    / "full"
    / "split_records.csv"
)
def _research_root() -> Path:
    return data_root()

DIRECT_CANDIDATES = [
    "WBC", "NEUT#", "NEUT%", "MONO#", "MONO%", "PLT", "RDW-CV", "CRP", "FIB", "LDH",
    "LYMPH#", "LYMPH%", "EO#", "EO%", "BASO#", "BASO%", "HGB", "ALB", "TP", "GLB",
    "PA", "GLU", "TG", "TC", "HDL-C", "LDL-C", "UA", "MCH", "MCHC", "PCT", "Cl", "Na",
]
FROZEN_FEATURES = [
    "WBC", "PLT", "LDH", "PCT", "EO#", "EO%", "ALB", "TP", "GLB", "TC", "LDL-C", "MCH", "MCHC",
    "SIRI", "LMR", "GAR", "PNI", "HALP",
]

FEATURE_GROUPS = {
    "I": {"WBC", "PLT", "LDH", "PCT", "SIRI"},
    "M": {"EO#", "EO%", "LMR", "GAR"},
    "N": {"ALB", "TP", "GLB", "TC", "LDL-C", "MCH", "MCHC", "PNI", "HALP"},
}
GROUP_LABELS = {
    "I": "Inflammation (I)",
    "M": "Immune (M)",
    "N": "Nutrition / metabolism (N)",
}


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required frozen source file(s) missing:\n" + "\n".join(missing))


def configure_publication_style() -> None:
    apply_style()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def feature_group(feature: str) -> str:
    for group, members in FEATURE_GROUPS.items():
        if feature in members:
            return group
    raise KeyError(f"No frozen layer mapping for {feature!r}")


def group_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=FEATURE_GROUP_COLORS[group],
            markeredgecolor="none",
            markersize=6,
            label=GROUP_LABELS[group],
        )
        for group in ("I", "M", "N")
    ]


def read_cohort_table() -> pd.DataFrame:
    workbook = cohort_workbook()
    a_base = validate_cohort(pd.read_excel(workbook, sheet_name="A_base"), 'A')
    y = pd.to_numeric(a_base["y_SCLC"], errors="raise").to_numpy(dtype=int)
    train_idx, holdout_idx = split_center_a_indices(a_base)
    frames = {
        "A_development": a_base.iloc[np.sort(train_idx)],
        "A_holdout_clean": a_base.iloc[np.sort(holdout_idx)],
        "B_external": validate_cohort(pd.read_excel(workbook, sheet_name="B_base"), 'B'),
        "C_external": validate_cohort(pd.read_excel(workbook, sheet_name="C_base"), 'C'),
    }
    table = pd.DataFrame([
        {"dataset": name, "n": len(frame), "positive_n": int(pd.to_numeric(frame["y_SCLC"], errors="raise").sum())}
        for name, frame in frames.items()
    ]).set_index("dataset")
    if set(train_idx) & set(holdout_idx):
        raise RuntimeError("A development and holdout indices overlap")
    return table


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 9.0,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.15,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.25,
    )


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.0,
            color="#606060",
        )
    )


def make_fig1() -> tuple[Path, Path]:
    table = read_cohort_table()
    a_development = table.loc["A_development"]
    a_evaluation = table.loc["A_holdout_clean"]
    center_b = table.loc["B_external"]
    center_c = table.loc["C_external"]

    fig, ax = plt.subplots(figsize=(12.9, 7.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "a", x=-0.012, y=0.984)

    add_box(
        ax,
        0.05,
        0.78,
        0.26,
        0.13,
        "Center A cohort\nIndependent development/evaluation allocation",
        "#EAF0F8",
        "#2F5597",
    )
    add_box(
        ax,
        0.37,
        0.78,
        0.26,
        0.13,
        f"Center B external cohort\nn = {int(center_b['n'])}; SCLC = {int(center_b['positive_n'])}",
        "#E7F5EF",
        "#009E73",
    )
    add_box(
        ax,
        0.69,
        0.78,
        0.26,
        0.13,
        f"Center C external cohort\nn = {int(center_c['n'])}; SCLC = {int(center_c['positive_n'])}",
        "#F1ECF7",
        "#7B61A8",
    )

    add_box(
        ax,
        0.04,
        0.53,
        0.27,
        0.13,
        f"A development set\n10-fold CV: selection evidence\nn = {int(a_development['n'])}; SCLC = {int(a_development['positive_n'])}",
        "#EAF0F8",
        "#2F5597",
    )

    add_box(
        ax,
        0.34,
        0.53,
        0.27,
        0.13,
        f"A evaluation set\nLocked evaluation\nn = {int(a_evaluation['n'])}; SCLC = {int(a_evaluation['positive_n'])}",
        "#FCEFE7",
        "#D55E00",
    )
    add_box(
        ax,
        0.67,
        0.53,
        0.28,
        0.13,
        f"External evaluation\nB: n = {int(center_b['n'])}; C: n = {int(center_c['n'])}",
        "#EBF5EE",
        "#4D9221",
    )

    add_arrow(ax, (0.18, 0.78), (0.18, 0.66))
    add_arrow(ax, (0.24, 0.78), (0.47, 0.66))
    add_arrow(ax, (0.50, 0.78), (0.75, 0.66))
    add_arrow(ax, (0.82, 0.78), (0.86, 0.66))

    add_box(
        ax,
        0.08,
        0.27,
        0.25,
        0.12,
        "Frozen input variables\n13 direct biomarkers\n+ 5 composite indices",
        "#F7F7F7",
        "#666666",
    )
    add_box(
        ax,
        0.375,
        0.27,
        0.25,
        0.12,
        "Cumulative layers\nI  →  I + M  →  I + M + N",
        "#F7F7F7",
        "#666666",
    )
    add_box(
        ax,
        0.67,
        0.27,
        0.25,
        0.12,
        "Candidate models\n14 individual models\n+ 3 fusion strategies",
        "#F7F7F7",
        "#666666",
    )
    for x in (0.175, 0.475, 0.81):
        add_arrow(ax, (x, 0.53), (x, 0.39))

    add_box(
        ax,
        0.26,
        0.06,
        0.48,
        0.11,
        "Frozen analytical specification\nApplied without re-selection to A evaluation and external cohorts",
        "#F1F4F8",
        "#526A85",
        fontsize=8.9,
    )
    add_arrow(ax, (0.205, 0.27), (0.39, 0.17))
    add_arrow(ax, (0.50, 0.27), (0.50, 0.17))
    add_arrow(ax, (0.795, 0.27), (0.61, 0.17))

    outputs = save_fig(fig, "Fig1_cohort_and_neutral_analysis_flow")
    plt.close(fig)
    return outputs


def load_final_a_analytic_cohort(a_base: pd.DataFrame) -> pd.DataFrame:
    final_a = validate_cohort(a_base.copy(), "A")
    if final_a["record_id"].astype(str).duplicated().any():
        raise ValueError("A analytical cohort must retain distinct delivered record_id values")
    final_a["center"] = "A"
    return final_a


def load_quality_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a_base = validate_cohort(pd.read_excel(WORKBOOK, sheet_name="A_base"), 'A')
    a_final = load_final_a_analytic_cohort(a_base)

    x = a_final[FROZEN_FEATURES].apply(pd.to_numeric, errors="coerce")
    x_imputed = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(x),
        columns=FROZEN_FEATURES,
    )
    correlation = x_imputed.corr(method="spearman")
    standardised = StandardScaler().fit_transform(x_imputed)
    vif_values: list[float] = []
    for index in range(len(FROZEN_FEATURES)):
        try:
            vif_values.append(float(variance_inflation_factor(standardised, index)))
        except Exception:
            vif_values.append(np.nan)
    vif = pd.DataFrame({"feature": FROZEN_FEATURES, "vif": vif_values}).sort_values(
        "vif", ascending=True, kind="mergesort"
    )

    full_frames: list[pd.DataFrame] = [a_final]
    for sheet in ("B_base", "C_base"):
        part = validate_cohort(pd.read_excel(WORKBOOK, sheet_name=sheet), sheet[0])
        part["center"] = sheet[0]
        full_frames.append(part)
    all_centers = pd.concat(full_frames, ignore_index=True)
    missingness = pd.DataFrame(
        {
            center: all_centers.loc[all_centers["center"] == center, DIRECT_CANDIDATES]
            .apply(pd.to_numeric, errors="coerce")
            .isna()
            .mean()
            * 100
            for center in ("A", "B", "C")
        }
    ).T
    medians = pd.DataFrame(
        {
            center: all_centers.loc[all_centers["center"] == center, FROZEN_FEATURES]
            .apply(pd.to_numeric, errors="coerce")
            .median()
            for center in ("A", "B", "C")
        }
    ).T
    a_values = all_centers.loc[all_centers["center"] == "A", FROZEN_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    a_iqr = (a_values.quantile(0.75) - a_values.quantile(0.25)).replace(0, np.nan).fillna(1.0)
    drift = (medians.loc[["B", "C"]] - medians.loc["A"]) / a_iqr
    return correlation, vif, missingness, drift


def make_fig2() -> tuple[Path, Path]:
    correlation, vif, missingness, drift = load_quality_inputs()
    display_vif = vif["vif"].replace([np.inf, -np.inf], 50).fillna(50).clip(upper=50)


    fig = plt.figure(figsize=(15.4, 12.1))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.42, 1.0],
        height_ratios=[1.0, 0.98],
        left=0.055,
        right=0.875,
        bottom=0.075,
        top=0.972,
        wspace=0.38,
        hspace=0.36,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    heat_a = sns.heatmap(
        correlation,
        ax=ax_a,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.12,
        cbar_kws={"label": "Spearman correlation", "shrink": 0.74, "pad": 0.025},
    )
    heat_a.collections[0].colorbar.solids.set_rasterized(False)
    ax_a.set_xlabel("")
    ax_a.set_ylabel("")
    ax_a.tick_params(axis="x", rotation=65, labelsize=6.7)
    ax_a.tick_params(axis="y", rotation=0, labelsize=6.7)
    panel_label(ax_a, "a", x=-0.10, y=1.02)

    vif_colors = [FEATURE_GROUP_COLORS[feature_group(item)] for item in vif["feature"]]
    ax_b.barh(
        np.arange(len(vif)),
        display_vif,
        color=vif_colors,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    ax_b.axvline(5, color="#555555", linestyle="--", linewidth=0.9, label="VIF = 5", zorder=4)
    ax_b.axvline(10, color="#111111", linestyle=":", linewidth=0.9, label="VIF = 10", zorder=4)
    ax_b.set_yticks(np.arange(len(vif)), vif["feature"].tolist())
    ax_b.set_xlim(0, max(12.0, float(display_vif.max()) * 1.09))
    ax_b.set_xlabel("Variance inflation factor (capped at 50)")
    ax_b.grid(axis="x", color="#D9D9D9", linewidth=0.45, zorder=0)
    threshold_legend = ax_b.legend(frameon=False, loc="lower right", fontsize=7.2)
    ax_b.add_artist(threshold_legend)
    ax_b.legend(
        handles=group_handles(),
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7.1,
        handletextpad=0.45,
        borderaxespad=0,
    )
    panel_label(ax_b, "b", x=-0.17, y=1.02)

    heat_c = sns.heatmap(
        missingness,
        ax=ax_c,
        cmap="YlOrRd",
        annot=True,
        fmt=".1f",
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Missing values (%)", "shrink": 0.74, "pad": 0.025},
        annot_kws={"fontsize": 5.8},
    )
    heat_c.collections[0].colorbar.solids.set_rasterized(False)
    ax_c.set_xlabel("Direct biomarker candidate")
    ax_c.set_ylabel("Center")
    ax_c.tick_params(axis="x", rotation=65, labelsize=6.0)
    ax_c.tick_params(axis="y", rotation=0)
    panel_label(ax_c, "c", x=-0.10, y=1.02)

    y = np.arange(len(FROZEN_FEATURES))
    ax_d.axvline(0, color="#555555", linestyle=":", linewidth=0.9, zorder=0)
    ax_d.scatter(drift.loc["B"], y, color="#009E73", s=30, label="Center B", zorder=4)
    ax_d.scatter(drift.loc["C"], y, color="#7B61A8", s=30, marker="s", label="Center C", zorder=4)
    for yi, feature in enumerate(FROZEN_FEATURES):
        ax_d.plot(
            [drift.loc["B", feature], drift.loc["C", feature]],
            [yi, yi],
            color="#B8B8B8",
            linewidth=0.75,
            zorder=1,
        )
    ax_d.set_yticks(y, FROZEN_FEATURES)
    ax_d.set_xlabel("Median shift from Center A (Center A IQR units)")
    ax_d.grid(axis="x", color="#D9D9D9", linewidth=0.45, zorder=0)
    ax_d.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        fontsize=7.3,
    )
    panel_label(ax_d, "d", x=-0.17, y=1.02)

    outputs = save_fig(fig, "Fig2_data_quality_and_center_comparability")
    plt.close(fig)
    return outputs


def main() -> None:
    require_files([WORKBOOK, SPLIT_RECORDS])
    configure_publication_style()
    fig1 = make_fig1()
    fig2 = make_fig2()
    print("Created v4 descriptive figures:")
    for output in (*fig1, *fig2):
        print(output)
    print(f"Descriptive sources: workbook split; {WORKBOOK}; {SPLIT_RECORDS}")


if __name__ == "__main__":
    main()
