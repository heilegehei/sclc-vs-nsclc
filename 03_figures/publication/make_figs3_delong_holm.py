from __future__ import annotations
import sys as _public_sys
from pathlib import Path as _PublicPath
_public_root = next(p for p in _PublicPath(__file__).resolve().parents if (p / "common" / "public_paths.py").is_file())
if str(_public_root) not in _public_sys.path:
    _public_sys.path.insert(0, str(_public_root))
from common.public_paths import cohort_workbook, code_root, data_root, explainability_results, fusion_results, publication_results


import csv
from decimal import Decimal
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


SOURCE_ROOT = data_root() / "fusion_results"
OUTPUT_ROOT = data_root() / "publication_results"
TABLE_OUT = OUTPUT_ROOT / "tables"
FIGURE_OUT = OUTPUT_ROOT / "figures" / "supplementary"

DELONG_SOURCE = SOURCE_ROOT / "tables" / "delong_holm_vs_weighted_voting.csv"
METRICS_SOURCE = SOURCE_ROOT / "tables" / "model_metrics_with_ci.csv"
WEIGHTS_SOURCE = SOURCE_ROOT / "tables" / "weighted_voting_fixed_weights.csv"

FIGURE_BASENAME = "unnumbered_delong_holm_A_evaluation_B_C"
TABLE_S2_CSV = TABLE_OUT / "TableS8_development_cv_selection_evidence.csv"
TABLE_S2_MD = TABLE_OUT / "TableS8_development_cv_selection_evidence.md"
TABLE_S3_CSV = TABLE_OUT / "TableS7_fusion_membership_and_weights.csv"
TABLE_S3_MD = TABLE_OUT / "TableS7_fusion_membership_and_weights.md"

A_DEVELOPMENT = "A_dev_cv_clean"
A_EVALUATION = "A_holdout_clean"
B_EXTERNAL = "B_external"
C_EXTERNAL = "C_external"
PLOT_DATASETS = [A_EVALUATION, B_EXTERNAL, C_EXTERNAL]
DATASET_LABELS = {
    A_EVALUATION: "A evaluation set",
    B_EXTERNAL: "B external validation",
    C_EXTERNAL: "C external validation",
}

WEIGHTED = "weighted_voting_cv_auc"
FUSIONS = [
    "weighted_voting_cv_auc",
    "soft_voting_native_probability",
    "stacking_in_sample_A_train",
]
COMPARATOR_ORDER = [
    "soft_voting_native_probability",
    "stacking_in_sample_A_train",
    "extra_trees",
    "rotation_forest",
    "random_forest",
]
MODEL_LABELS = {
    "weighted_voting_cv_auc": "Weighted voting",
    "soft_voting_native_probability": "Soft voting",
    "stacking_in_sample_A_train": "Stacking",
    "extra_trees": "Extra Trees",
    "rotation_forest": "Rotation Forest",
    "random_forest": "Random Forest",
}
MODEL_COLORS = {
    "weighted_voting_cv_auc": "#2F5597",
    "soft_voting_native_probability": "#D55E00",
    "stacking_in_sample_A_train": "#009E73",
    "extra_trees": "#CC79A7",
    "rotation_forest": "#7B61A8",
    "random_forest": "#E69F00",
}


def apply_style() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    if "Times New Roman" not in available:
        raise RuntimeError("Times New Roman is required but was not found.")
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
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.16,
        1.02,
        letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{name} is missing columns: {missing}")


def _read_required_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (DELONG_SOURCE, METRICS_SOURCE, WEIGHTS_SOURCE):
        if not path.exists():
            raise FileNotFoundError(f"Required frozen source is missing: {path}")
    delong = pd.read_csv(DELONG_SOURCE)


    metrics = pd.read_csv(METRICS_SOURCE, dtype=str, keep_default_na=False)
    weights = pd.read_csv(WEIGHTS_SOURCE, dtype=str, keep_default_na=False)
    _require_columns(
        delong,
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
        "DeLong/Holm archive",
    )
    _require_columns(
        metrics,
        [
            "dataset",
            "model",
            "role",
            "n",
            "events",
            "non_events",
            "prevalence",
            "auc",
            "auc_ci_low",
            "auc_ci_high",
            "auprc",
            "auprc_ci_low",
            "auprc_ci_high",
            "brier",
            "brier_ci_low",
            "brier_ci_high",
            "bootstrap_n",
            "bootstrap_method",
            "evaluation_estimand",
        ],
        "model-metric archive",
    )
    _require_columns(
        weights,
        ["model", "source_cv_auc", "normalized_weight", "weight_source"],
        "fixed-weight archive",
    )
    if len(weights) != 13 or not weights["model"].is_unique:
        raise RuntimeError("Expected exactly 13 unique native-probability base-model weights.")
    weight_total = float(pd.to_numeric(weights["normalized_weight"]).sum())
    if not np.isclose(weight_total, 1.0, atol=1e-10):
        raise RuntimeError(f"Stored normalized weights do not sum to one: {weight_total}")
    return delong, metrics, weights


def _scope_delong(delong: pd.DataFrame) -> pd.DataFrame:
    selected = delong.loc[
        delong["dataset"].isin(PLOT_DATASETS)
        & (delong["reference_model"] == WEIGHTED)
        & delong["comparator_model"].isin(COMPARATOR_ORDER)
    ].copy()
    expected_rows = len(PLOT_DATASETS) * len(COMPARATOR_ORDER)
    if len(selected) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} stored evaluation/external DeLong rows; found {len(selected)}."
        )
    if A_DEVELOPMENT in set(selected["dataset"]):
        raise RuntimeError("Development-CV rows must never appear in the DeLong/Holm figure.")
    for dataset in PLOT_DATASETS:
        rows = selected.loc[selected["dataset"] == dataset]
        if set(rows["comparator_model"]) != set(COMPARATOR_ORDER):
            raise RuntimeError(f"Unexpected comparator scope for {dataset}.")
    return selected


def _fmt_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _fmt_ci(point: float, low: float, high: float) -> str:
    return f"{point:+.3f} ({low:+.3f} to {high:+.3f})"


def _setup_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.grid(axis="x", color="#E9E9E9", linewidth=0.55, alpha=0.9, zorder=0)


def make_figure(delong: pd.DataFrame) -> tuple[Path, Path]:
    scoped = _scope_delong(delong)
    global_low = min(0.0, float(scoped["delta_ci_low"].min()))
    global_high = max(0.0, float(scoped["delta_ci_high"].max()))
    span = max(0.02, global_high - global_low)
    x_limits = (global_low - 0.09 * span, global_high + 0.09 * span)

    fig = plt.figure(figsize=(18.5, 5.45), facecolor="white")
    outer = fig.add_gridspec(
        1, 3, left=0.040, right=0.998, top=0.94, bottom=0.095, wspace=0.19
    )
    for index, dataset in enumerate(PLOT_DATASETS):
        panel = outer[0, index].subgridspec(1, 2, width_ratios=[1.06, 1.72], wspace=0.035)
        ax = fig.add_subplot(panel[0, 0])
        table_ax = fig.add_subplot(panel[0, 1])
        rows = (
            scoped.loc[scoped["dataset"] == dataset]
            .set_index("comparator_model")
            .loc[COMPARATOR_ORDER]
            .reset_index()
        )
        y = np.arange(len(rows), dtype=float)
        points = rows["delta_auc"].to_numpy(float)
        lows = rows["delta_ci_low"].to_numpy(float)
        highs = rows["delta_ci_high"].to_numpy(float)
        _setup_axis(ax)
        ax.axvline(0, color="#6F6F6F", linestyle=":", linewidth=1.0, zorder=1)
        for row_index, comparator in enumerate(COMPARATOR_ORDER):
            color = MODEL_COLORS[comparator]
            ax.errorbar(
                points[row_index],
                y[row_index],
                xerr=np.array(
                    [[points[row_index] - lows[row_index]], [highs[row_index] - points[row_index]]]
                ),
                fmt="o",
                color=color,
                ecolor=color,
                markerfacecolor="white",
                markeredgewidth=1.15,
                markersize=5.4,
                capsize=2.7,
                elinewidth=1.2,
                zorder=3,
            )
        ax.set_xlim(*x_limits)
        ax.set_ylim(-0.84, len(rows) - 0.37)
        ax.invert_yaxis()
        ax.set_yticks(y, [MODEL_LABELS[model] for model in COMPARATOR_ORDER])
        ax.tick_params(axis="y", labelsize=7.2, pad=2)
        ax.set_xlabel("ΔAUC (Weighted voting − comparator)")
        ax.text(
            0.015,
            0.985,
            f"{DATASET_LABELS[dataset]}\nReference: Weighted voting",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
            fontsize=8.6,
            linespacing=1.28,
        )
        panel_label(ax, "abc"[index])

        table_ax.axis("off")
        table_ax.set_xlim(0, 1)
        table_ax.set_ylim(-0.84, len(rows) - 0.37)
        table_ax.invert_yaxis()
        table_ax.text(0.005, -0.60, "ΔAUC (95% CI)", fontsize=7.1, fontweight="bold")
        table_ax.text(0.724, -0.60, "Raw P", fontsize=7.1, fontweight="bold")
        table_ax.text(0.870, -0.60, "Holm P", fontsize=7.1, fontweight="bold")
        for row_index, row in rows.iterrows():
            table_ax.text(
                0.005,
                row_index,
                _fmt_ci(float(row.delta_auc), float(row.delta_ci_low), float(row.delta_ci_high)),
                fontsize=6.55,
                va="center",
                ha="left",
            )
            table_ax.text(
                0.738,
                row_index,
                _fmt_p(float(row.p_raw)),
                fontsize=6.55,
                va="center",
                ha="left",
            )
            table_ax.text(
                0.883,
                row_index,
                _fmt_p(float(row.p_holm)),
                fontsize=6.55,
                va="center",
                ha="left",
            )
    FIGURE_OUT.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_OUT / f"{FIGURE_BASENAME}.pdf"
    png_path = FIGURE_OUT / f"{FIGURE_BASENAME}.png"
    fig.savefig(pdf_path, format="pdf", facecolor="white")
    fig.savefig(png_path, format="png", dpi=600, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def _exact_string(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _md_table(headers: list[str], body: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _display_decimal(value: str) -> str:
    return f"{Decimal(value):.3f}" if value else ""


def _display_interval(low: str, high: str) -> str:
    return f"[{_display_decimal(low)}, {_display_decimal(high)}]"


def make_selection_table(metrics: pd.DataFrame) -> None:
    source = metrics.loc[
        (metrics["dataset"] == A_DEVELOPMENT) & metrics["model"].isin(FUSIONS)
    ].copy()
    if len(source) != len(FUSIONS):
        raise RuntimeError("Expected exactly three frozen development-CV fusion metric rows.")
    source = source.set_index("model").loc[FUSIONS].reset_index()
    fields = [
        "selection_order",
        "evidence_scope",
        "dataset",
        "model_code",
        "model",
        "role",
        "n",
        "events",
        "non_events",
        "prevalence",
        "evaluation_estimand",
        "auc",
        "auc_ci_low",
        "auc_ci_high",
        "auprc",
        "auprc_ci_low",
        "auprc_ci_high",
        "brier",
        "brier_ci_low",
        "brier_ci_high",
        "bootstrap_n",
        "bootstrap_method",
        "source_file",
    ]
    rows: list[dict[str, str]] = []
    for order, row in enumerate(source.itertuples(index=False), start=1):
        data = row._asdict()
        rows.append(
            {
                "selection_order": str(order),
                "evidence_scope": "Development 10-fold CV selection evidence only; not final validation",
                "dataset": "A development (10-fold CV)",
                "model_code": data["model"],
                "model": MODEL_LABELS[data["model"]],
                "role": _exact_string(data["role"]),
                "n": _exact_string(data["n"]),
                "events": _exact_string(data["events"]),
                "non_events": _exact_string(data["non_events"]),
                "prevalence": _exact_string(data["prevalence"]),
                "evaluation_estimand": _exact_string(data["evaluation_estimand"]),
                "auc": _exact_string(data["auc"]),
                "auc_ci_low": _exact_string(data["auc_ci_low"]),
                "auc_ci_high": _exact_string(data["auc_ci_high"]),
                "auprc": _exact_string(data["auprc"]),
                "auprc_ci_low": _exact_string(data["auprc_ci_low"]),
                "auprc_ci_high": _exact_string(data["auprc_ci_high"]),
                "brier": _exact_string(data["brier"]),
                "brier_ci_low": _exact_string(data["brier_ci_low"]),
                "brier_ci_high": _exact_string(data["brier_ci_high"]),
                "bootstrap_n": _exact_string(data["bootstrap_n"]),
                "bootstrap_method": _exact_string(data["bootstrap_method"]),
                "source_file": METRICS_SOURCE.as_posix(),
            }
        )
    _write_csv(TABLE_S2_CSV, fields, rows)
    md_body = [
        [
            row["model"],
            row["role"],
            f"{_display_decimal(row['auc'])} {_display_interval(row['auc_ci_low'], row['auc_ci_high'])}",
            f"{_display_decimal(row['auprc'])} {_display_interval(row['auprc_ci_low'], row['auprc_ci_high'])}",
            f"{_display_decimal(row['brier'])} {_display_interval(row['brier_ci_low'], row['brier_ci_high'])}",
        ]
        for row in rows
    ]
    table = _md_table(
        ["Fusion candidate", "Frozen role", "AUC (stored 95% CI)", "AUPRC (stored 95% CI)", "Brier (stored 95% CI)"],
        md_body,
    )
    TABLE_S2_MD.write_text(
        "# Supplementary Table S8. Development 10-fold CV fusion selection evidence\n\n"
        "Dataset: **A development (10-fold CV)**. This table is selection evidence only, "
        "not a final-validation performance display. All point estimates and 95% confidence "
        "intervals are copied directly from the frozen archived metric table; no model was "
        "refitted and no performance measure was recomputed. The Markdown display is rounded "
        "to three decimals; the companion CSV retains the stored fields.\n\n"
        f"{table}\n\n"
        "## Archived source\n\n"
        f"- `{METRICS_SOURCE.as_posix()}`\n",
        encoding="utf-8",
    )


def make_membership_table(weights: pd.DataFrame) -> None:
    fields = [
        "record_type",
        "model_code",
        "model",
        "native_probability_member",
        "source_cv_auc",
        "soft_voting_weight",
        "weighted_voting_normalized_weight",
        "weighted_weight_source",
        "stacking_input",
        "notes",
    ]
    display = {
        "logistic_regression": "Logistic regression",
        "gam": "GAM",
        "knn": "KNN",
        "gaussian_nb": "Gaussian NB",
        "decision_tree": "Decision tree",
        "random_forest": "Random forest",
        "extra_trees": "Extra Trees",
        "gbdt": "GBDT",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "adaboost": "AdaBoost",
        "rotation_forest": "Rotation Forest",
        "mlp": "MLP",
        "rbf_svm": "RBF-SVM",
        "soft_voting_native_probability": "Soft voting",
        "weighted_voting_cv_auc": "Weighted voting",
        "stacking_in_sample_A_train": "Stacking",
    }
    rows: list[dict[str, str]] = []
    for row in weights.itertuples(index=False):
        data = row._asdict()
        rows.append(
            {
                "record_type": "base_model_member",
                "model_code": data["model"],
                "model": display[data["model"]],
                "native_probability_member": "Yes",
                "source_cv_auc": _exact_string(data["source_cv_auc"]),
                "soft_voting_weight": "1/13 (equal)",
                "weighted_voting_normalized_weight": _exact_string(data["normalized_weight"]),
                "weighted_weight_source": _exact_string(data["weight_source"]),
                "stacking_input": "Yes",
                "notes": "Included as a native-probability base-model output.",
            }
        )
    rows.extend(
        [
            {
                "record_type": "excluded_base_model",
                "model_code": "rbf_svm",
                "model": display["rbf_svm"],
                "native_probability_member": "No",
                "source_cv_auc": "",
                "soft_voting_weight": "",
                "weighted_voting_normalized_weight": "",
                "weighted_weight_source": "",
                "stacking_input": "No",
                "notes": "Excluded because the archived native output is a decision function rather than a native probability.",
            },
            {
                "record_type": "fusion_definition",
                "model_code": "soft_voting_native_probability",
                "model": display["soft_voting_native_probability"],
                "native_probability_member": "13 base models",
                "source_cv_auc": "",
                "soft_voting_weight": "Equal mean across all 13 listed base models",
                "weighted_voting_normalized_weight": "",
                "weighted_weight_source": "",
                "stacking_input": "",
                "notes": "Native-probability outputs only.",
            },
            {
                "record_type": "fusion_definition",
                "model_code": "weighted_voting_cv_auc",
                "model": display["weighted_voting_cv_auc"],
                "native_probability_member": "13 base models",
                "source_cv_auc": "A development 10-fold CV AUC",
                "soft_voting_weight": "",
                "weighted_voting_normalized_weight": "Stored normalized weights above",
                "weighted_weight_source": "Stored A-development-CV-AUC normalization",
                "stacking_input": "",
                "notes": "Fixed weights were retained from the archived weight table.",
            },
            {
                "record_type": "fusion_definition",
                "model_code": "stacking_in_sample_A_train",
                "model": display["stacking_in_sample_A_train"],
                "native_probability_member": "13 base models",
                "source_cv_auc": "",
                "soft_voting_weight": "",
                "weighted_voting_normalized_weight": "",
                "weighted_weight_source": "",
                "stacking_input": "All 13 listed base-model outputs",
                "notes": "Logistic-regression meta layer on the same 13 native-probability base-model outputs.",
            },
        ]
    )
    _write_csv(TABLE_S3_CSV, fields, rows)
    member_rows = [row for row in rows if row["record_type"] == "base_model_member"]
    display_rows = [
        [
            row["model"],
            _display_decimal(row["source_cv_auc"]),
            row["soft_voting_weight"],
            _display_decimal(row["weighted_voting_normalized_weight"]),
            row["stacking_input"],
        ]
        for row in member_rows
    ]
    table = _md_table(
        ["Base model", "Stored CV AUC", "Soft-voting weight", "Weighted-voting weight", "Stacking input"],
        display_rows,
    )
    TABLE_S3_MD.write_text(
        "# Supplementary Table S7. Frozen fusion membership and weights\n\n"
        "All three fusion strategies use the same 13 archived native-probability base-model outputs. "
        "Soft voting uses their equal mean. Weighted voting uses the stored normalized weights derived "
        "from A-development 10-fold-CV AUC. Stacking uses a logistic-regression meta layer on those same "
        "13 outputs. RBF-SVM is excluded because its archived native output is a decision function rather "
        "than a native probability. This table documents the frozen construction only; it does not fit or "
        "re-estimate any model.\n\n"
        f"{table}\n\n"
        "## Archived source\n\n"
        f"- `{WEIGHTS_SOURCE.as_posix()}`\n",
        encoding="utf-8",
    )


def main() -> None:
    apply_style()
    delong, metrics, weights = _read_required_inputs()
    png_path, pdf_path = make_figure(delong)
    make_selection_table(metrics)
    make_membership_table(weights)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {TABLE_S2_CSV}")
    print(f"Wrote {TABLE_S2_MD}")
    print(f"Wrote {TABLE_S3_CSV}")
    print(f"Wrote {TABLE_S3_MD}")


if __name__ == "__main__":
    main()
