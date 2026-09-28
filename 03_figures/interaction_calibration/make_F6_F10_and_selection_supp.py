from __future__ import annotations
import sys as _public_sys
from pathlib import Path as _PublicPath
_public_root = next(p for p in _PublicPath(__file__).resolve().parents if (p / "common" / "public_paths.py").is_file())
if str(_public_root) not in _public_sys.path:
    _public_sys.path.insert(0, str(_public_root))
from common.public_paths import cohort_workbook, code_root, data_root, explainability_results, fusion_results, publication_results


import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from statsmodels.nonparametric.smoothers_lowess import lowess

from nature_style import (
    DATASET_COLORS,
    DATASET_LABELS,
    LAYER_LABELS,
    LAYER_ORDER,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_LINESTYLES,
    MODEL_ORDER,
    apply_style,
    panel_label,
)


_BOOTSTRAP_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "manuscript_bootstrap.py").is_file())
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))
from common.manuscript_bootstrap import BOOTSTRAP_PROTOCOL

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = explainability_results()
SUBMISSION_ROOT = OUTPUT_ROOT.parent
SOURCE_ROOT = data_root() / "fusion_results"

DATA_OUT = OUTPUT_ROOT / "data"
TABLE_OUT = OUTPUT_ROOT / "tables"
MAIN_OUT = publication_results() / "figures" / "main"
SUPP_OUT = publication_results() / "figures" / "supplementary"
MANUSCRIPT_OUT = OUTPUT_ROOT / "manuscript"
QC_OUT = OUTPUT_ROOT / "qc"

PRIMARY = "weighted_voting_cv_auc"
FUSIONS = MODEL_ORDER[:3]
SIX_MODELS = MODEL_ORDER[:]
LAYER_EVAL_DATASETS = [
    "A_holdout_clean",
    "B_external",
    "C_external",
]
A_DEVELOPMENT = "A_dev_cv_clean"
A_HOLDOUT = "A_holdout_clean"
B_EXTERNAL = "B_external"
C_EXTERNAL = "C_external"
F6_LOWESS_FRAC = 0.75
F6_LOWESS_IT = 0
EVALUATION_LOWESS_FRAC = 0.75
EVALUATION_LOWESS_IT = 0

A_CV_FUSION_PREDICTIONS = data_root() / "fusion_results" / "A_fusions_CV_holdout" / "fusion_predictions_A_CV.csv"
A_HOLDOUT_FUSION_PREDICTIONS = data_root() / "fusion_results" / "A_fusions_CV_holdout" / "fusion_predictions_A_holdout_clean.csv"
NATIVE_PREDICTIONS = data_root() / "fusion_results" / "native_predictions_all_eval.csv"
BC_FUSION_PREDICTIONS = data_root() / "fusion_results" / "three_fusions_BC" / "fusion_predictions_B_C.csv"

SOURCE_FILES = {
    "layer_predictions": SOURCE_ROOT / "data" / "weighted_voting_layer_predictions.csv",
    "layer_calibration": SOURCE_ROOT / "data" / "weighted_voting_layer_calibration_plot_data.csv",
    "layer_metrics": SOURCE_ROOT / "tables" / "weighted_voting_layer_metrics_ci.csv",
    "layer_increments": SOURCE_ROOT / "tables" / "weighted_voting_layer_increment_ci.csv",
    "layer_delong": SOURCE_ROOT / "tables" / "weighted_voting_layer_delong_holm.csv",
    "metrics": SOURCE_ROOT / "tables" / "model_metrics_with_ci.csv",
    "roc": SOURCE_ROOT / "data" / "roc_curve_points.csv",
    "pr": SOURCE_ROOT / "data" / "pr_curve_points.csv",
    "calibration": SOURCE_ROOT / "data" / "calibration_curve_points.csv",
    "dca": SOURCE_ROOT / "data" / "dca_curve_points.csv",
    "model_delong": SOURCE_ROOT / "tables" / "delong_holm_vs_weighted_voting.csv",
    "A_CV_fusion_predictions": A_CV_FUSION_PREDICTIONS,
    "A_holdout_fusion_predictions": A_HOLDOUT_FUSION_PREDICTIONS,
    "native_predictions": NATIVE_PREDICTIONS,
    "BC_fusion_predictions": BC_FUSION_PREDICTIONS,
}

EVALUATION_CALIBRATION_V2 = DATA_OUT / "F9_F10_calibration_lowess_frac075.csv"
HARMONISED_LAYER_METRICS = TABLE_OUT / "weighted_voting_layer_metrics_ci_harmonised.csv"
CI_HARMONISATION_AUDIT = TABLE_OUT / "weighted_voting_full_layer_ci_source_audit.csv"

FIGURE_BASENAMES = {
    "F6": "FigS3_layer_calibration_weighted_voting",
    "F7": "FigS2_layer_brier_weighted_voting",
    "F8": "unnumbered_layer_increment_delong",
    "F9": "unnumbered_A_model_comparison_not_manuscript_fig4",
    "F10": "unnumbered_BC_six_model_comparison_not_manuscript_fig5",
    "S1": "unnumbered_A_development_fusion_selection",
    "S6": "unnumbered_delong_holm_vs_weighted_voting",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{name} lacks required columns: {missing}")


def _normalise_saved_predictions(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    a_cv_raw = frames["A_CV_fusion_predictions"]
    a_fusion_raw = frames["A_holdout_fusion_predictions"]
    native_raw = frames["native_predictions"]
    bc_raw = frames["BC_fusion_predictions"]
    _required_columns(
        a_cv_raw,
        ["dataset", "fusion", "record_id", "source_record_id", "y_SCLC", "score"],
        "A_CV_fusion_predictions",
    )
    _required_columns(
        a_fusion_raw,
        ["dataset", "fusion", "record_id", "source_record_id", "y_SCLC", "score"],
        "A_holdout_fusion_predictions",
    )
    _required_columns(
        native_raw,
        ["dataset", "model", "record_id", "source_record_id", "y_SCLC", "native_probability"],
        "native_predictions",
    )
    _required_columns(
        bc_raw,
        ["dataset", "model", "record_id", "source_record_id", "y_SCLC", "native_probability"],
        "BC_fusion_predictions",
    )

    a_cv = a_cv_raw.loc[
        (a_cv_raw["dataset"].astype(str) == A_DEVELOPMENT)
        & a_cv_raw["fusion"].astype(str).isin(FUSIONS)
    ].rename(columns={"fusion": "model", "score": "probability"})
    a_fusions = a_fusion_raw.loc[
        (a_fusion_raw["dataset"].astype(str) == A_HOLDOUT)
        & a_fusion_raw["fusion"].astype(str).isin(FUSIONS)
    ].rename(columns={"fusion": "model", "score": "probability"})
    a_baselines = native_raw.loc[
        (native_raw["dataset"].astype(str) == A_HOLDOUT)
        & native_raw["model"].astype(str).isin(SIX_MODELS[3:])
    ].rename(columns={"native_probability": "probability"})
    bc = bc_raw.loc[
        bc_raw["dataset"].astype(str).isin([B_EXTERNAL, C_EXTERNAL])
        & bc_raw["model"].astype(str).isin(SIX_MODELS)
    ].rename(columns={"native_probability": "probability"})
    columns = ["dataset", "model", "record_id", "source_record_id", "y_SCLC", "probability"]
    out = pd.concat(
        [a_cv[columns], a_fusions[columns], a_baselines[columns], bc[columns]],
        ignore_index=True,
    )
    out["dataset"] = out["dataset"].astype(str)
    out["model"] = out["model"].astype(str)
    out["record_id"] = out["record_id"].astype(str)
    out["source_record_id"] = out["source_record_id"].astype(str)
    out["y_SCLC"] = pd.to_numeric(out["y_SCLC"], errors="raise").astype(int)
    out["probability"] = pd.to_numeric(out["probability"], errors="raise").astype(float)
    return out


def load_frozen_data() -> dict[str, pd.DataFrame]:
    missing = [str(path) for path in SOURCE_FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Frozen plotting inputs are missing: {missing}")
    frames = {name: pd.read_csv(path) for name, path in SOURCE_FILES.items()}

    _required_columns(
        frames["layer_predictions"],
        ["dataset", "layer", "model", "record_id", "y_SCLC", "probability"],
        "layer_predictions",
    )
    _required_columns(
        frames["layer_calibration"],
        ["layer", "dataset", "series_type", "x", "y", "bin", "lowess_frac", "lowess_it"],
        "layer_calibration",
    )
    _required_columns(
        frames["layer_metrics"],
        [
            "layer",
            "dataset",
            "model",
            "auc",
            "auc_ci_low",
            "auc_ci_high",
            "auprc",
            "auprc_ci_low",
            "auprc_ci_high",
            "brier",
            "brier_ci_low",
            "brier_ci_high",
        ],
        "layer_metrics",
    )
    _required_columns(
        frames["layer_increments"],
        ["comparison", "metric", "increment", "ci_low", "ci_high"],
        "layer_increments",
    )
    _required_columns(
        frames["layer_delong"],
        ["comparison", "delta_auc", "ci_low", "ci_high", "p_raw", "p_holm"],
        "layer_delong",
    )
    _required_columns(
        frames["metrics"],
        [
            "dataset",
            "model",
            "auc",
            "auc_ci_low",
            "auc_ci_high",
            "auprc",
            "auprc_ci_low",
            "auprc_ci_high",
            "brier",
            "brier_ci_low",
            "brier_ci_high",
        ],
        "metrics",
    )
    _required_columns(frames["roc"], ["dataset", "model", "point", "fpr", "tpr"], "roc")
    _required_columns(frames["pr"], ["dataset", "model", "point", "recall", "precision"], "pr")
    _required_columns(
        frames["calibration"],
        ["dataset", "model", "point_kind", "point", "mean_predicted", "observed_fraction"],
        "calibration",
    )
    _required_columns(
        frames["dca"],
        ["dataset", "curve_type", "model", "threshold", "net_benefit", "threshold_grid_n"],
        "dca",
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
        "model_delong",
    )
    frames["patient_predictions"] = _normalise_saved_predictions(frames)
    validate_frozen_data(frames)
    return frames


def validate_frozen_data(frames: dict[str, pd.DataFrame]) -> None:
    layer_predictions = frames["layer_predictions"]
    patient_predictions = frames["patient_predictions"]
    reference_cases = {}
    for dataset in (A_DEVELOPMENT, A_HOLDOUT, B_EXTERNAL, C_EXTERNAL):
        reference = patient_predictions.loc[(patient_predictions["dataset"] == dataset) & (patient_predictions["model"] == PRIMARY), ["record_id", "y_SCLC"]].copy()
        if reference.empty or reference["record_id"].isna().any() or not reference["record_id"].astype(str).is_unique:
            raise RuntimeError(f"Invalid locked primary-model case identities for {dataset}")
        reference["record_id"] = reference["record_id"].astype(str)
        reference_cases[dataset] = reference.sort_values("record_id", kind="mergesort").reset_index(drop=True)
    expected_layer_n = {dataset: len(reference) for dataset, reference in reference_cases.items()}
    for dataset, n_expected in expected_layer_n.items():
        for layer in LAYER_ORDER:
            group = layer_predictions.loc[
                (layer_predictions["dataset"] == dataset)
                & (layer_predictions["layer"] == layer)
                & (layer_predictions["model"] == PRIMARY)
            ]
            if len(group) != n_expected or group["record_id"].astype(str).nunique() != n_expected:
                raise RuntimeError(f"Unexpected layer-prediction scope for {dataset}, {layer}")
            if group[["y_SCLC", "probability"]].isna().any().any():
                raise RuntimeError(f"Missing layer prediction value for {dataset}, {layer}")
            identities = group[["record_id", "y_SCLC"]].copy()
            identities["record_id"] = identities["record_id"].astype(str)
            identities = identities.sort_values("record_id", kind="mergesort").reset_index(drop=True)
            if not identities.equals(reference_cases[dataset]):
                raise RuntimeError(f"Layer cases/outcomes differ from locked primary predictions: {dataset}/{layer}")

    metrics = frames["metrics"]
    expected_models = {
        A_DEVELOPMENT: FUSIONS,
        A_HOLDOUT: SIX_MODELS,
        B_EXTERNAL: SIX_MODELS,
        C_EXTERNAL: SIX_MODELS,
    }
    for dataset, models in expected_models.items():
        observed = metrics.loc[metrics["dataset"] == dataset, "model"].tolist()
        if set(observed) != set(models) or len(observed) != len(models):
            raise RuntimeError(f"Frozen model scope mismatch for {dataset}: {observed}")
    ci_columns = [
        "auc",
        "auc_ci_low",
        "auc_ci_high",
        "auprc",
        "auprc_ci_low",
        "auprc_ci_high",
        "brier",
        "brier_ci_low",
        "brier_ci_high",
    ]
    if metrics[ci_columns].isna().any().any():
        raise RuntimeError("Frozen model-metric table contains missing estimates or confidence limits")

    for name in ("roc", "pr", "calibration"):
        frame = frames[name]
        for dataset, models in expected_models.items():
            for model in models:
                if frame.loc[(frame["dataset"] == dataset) & (frame["model"] == model)].empty:
                    raise RuntimeError(f"{name} curve is missing for {dataset}, {model}")
    dca = frames["dca"]
    for dataset, models in expected_models.items():
        for model in models:
            group = dca.loc[
                (dca["dataset"] == dataset)
                & (dca["curve_type"] == "model")
                & (dca["model"] == model)
            ]
            if len(group) != 160 or group["threshold"].nunique() != 160:
                raise RuntimeError(f"DCA grid is not the frozen 160-point grid for {dataset}, {model}")
            diffs = np.diff(np.sort(group["threshold"].to_numpy(float)))
            if not np.allclose(diffs, diffs[0], rtol=0, atol=1e-12):
                raise RuntimeError(f"DCA grid is not equally spaced for {dataset}, {model}")

    patient_predictions = frames["patient_predictions"]
    evaluation_n = expected_layer_n
    for dataset, n_expected in evaluation_n.items():
        reference_keys: list[str] | None = None
        models = FUSIONS if dataset == A_DEVELOPMENT else SIX_MODELS
        for model in models:
            group = patient_predictions.loc[
                (patient_predictions["dataset"] == dataset)
                & (patient_predictions["model"] == model)
            ].sort_values("record_id", kind="mergesort")
            if len(group) != n_expected or group["record_id"].nunique() != n_expected:
                raise RuntimeError(f"Saved patient-prediction scope mismatch for {dataset}, {model}")
            if group[["y_SCLC", "probability"]].isna().any().any():
                raise RuntimeError(f"Missing saved patient prediction for {dataset}, {model}")
            if not group["probability"].between(0, 1, inclusive="both").all():
                raise RuntimeError(f"Saved probability outside [0, 1] for {dataset}, {model}")
            keys = (
                group["record_id"].astype(str)
                + "|"
                + group["source_record_id"].astype(str)
                + "|"
                + group["y_SCLC"].astype(str)
            ).tolist()
            if reference_keys is None:
                reference_keys = keys
            elif keys != reference_keys:
                raise RuntimeError(f"Patient IDs or outcomes differ across models for {dataset}")
            y = group["y_SCLC"].to_numpy(int)
            probability = group["probability"].to_numpy(float)
            frozen = metric_row(metrics, dataset, model)
            checks = {
                "AUC": (roc_auc_score(y, probability), float(frozen["auc"])),
                "AUPRC": (average_precision_score(y, probability), float(frozen["auprc"])),
                "Brier": (brier_score_loss(y, probability), float(frozen["brier"])),
            }
            for metric_name, (recomputed, stored) in checks.items():
                if not np.isclose(recomputed, stored, rtol=0, atol=5e-12):
                    raise RuntimeError(
                        f"Saved predictions do not reproduce frozen {metric_name} for "
                        f"{dataset}, {model}: {recomputed} versus {stored}"
                    )


def harmonise_full_layer_ci(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    for name in ("metrics", "layer_metrics"):
        source = frames[name]
        required = {"bootstrap_protocol", "bootstrap_case_sha256", "bootstrap_n"}
        if (not required.issubset(source.columns)
                or not source["bootstrap_protocol"].eq(BOOTSTRAP_PROTOCOL).all()
                or not source["bootstrap_n"].eq(2000).all()
                or source["bootstrap_case_sha256"].isna().any()):
            raise RuntimeError(f"{name}: old CI results must be regenerated with {BOOTSTRAP_PROTOCOL}")
    result = frames["layer_metrics"].copy()
    result["display_ci_source"] = "layer-analysis bootstrap archive"
    audit_rows: list[dict[str, object]] = []
    ci_columns = [
        "auc_ci_low",
        "auc_ci_high",
        "auprc_ci_low",
        "auprc_ci_high",
        "brier_ci_low",
        "brier_ci_high",
    ]
    for dataset in (A_DEVELOPMENT, A_HOLDOUT, B_EXTERNAL, C_EXTERNAL):
        layer_mask = (
            (result["dataset"] == dataset)
            & (result["layer"] == "I_M_N")
            & (result["model"] == PRIMARY)
        )
        model_mask = (
            (frames["metrics"]["dataset"] == dataset)
            & (frames["metrics"]["model"] == PRIMARY)
        )
        if int(layer_mask.sum()) != 1 or int(model_mask.sum()) != 1:
            raise RuntimeError(f"Cannot harmonise the full-layer CI for {dataset}")
        layer_row = result.loc[layer_mask].iloc[0]
        model_row = frames["metrics"].loc[model_mask].iloc[0]
        point_differences = {
            metric: abs(float(layer_row[metric]) - float(model_row[metric]))
            for metric in ("auc", "auprc", "brier")
        }
        if max(point_differences.values()) > 5e-12:
            raise RuntimeError(
                f"Full-layer point estimate differs between frozen archives for {dataset}: "
                f"{point_differences}"
            )
        audit_row: dict[str, object] = {
            "dataset": dataset,
            **{f"point_abs_difference_{key}": value for key, value in point_differences.items()},
            "selected_ci_source": str(SOURCE_FILES["metrics"]),
            "superseded_monte_carlo_ci_source": str(SOURCE_FILES["layer_metrics"]),
        }
        for column in ci_columns:
            audit_row[f"layer_archive_{column}"] = float(layer_row[column])
            audit_row[f"canonical_{column}"] = float(model_row[column])
            if not np.isclose(float(layer_row[column]), float(model_row[column]), rtol=0, atol=5e-10):
                raise RuntimeError(f"Shared-draw CI mismatch for {dataset}/{column}; regenerate both analyses")
        if layer_row.get("bootstrap_case_sha256") != model_row.get("bootstrap_case_sha256"):
            raise RuntimeError(f"Bootstrap case identities differ for {dataset}")
        result.loc[layer_mask, "display_ci_source"] = BOOTSTRAP_PROTOCOL
        audit_rows.append(audit_row)
    HARMONISED_LAYER_METRICS.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(HARMONISED_LAYER_METRICS, index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(CI_HARMONISATION_AUDIT, index=False, encoding="utf-8-sig")
    return result


def setup_axis(ax: plt.Axes, *, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    if grid == "both":
        ax.grid(color="#E9E9E9", linewidth=0.55, alpha=0.85)
    elif grid:
        ax.grid(axis=grid, color="#E9E9E9", linewidth=0.55, alpha=0.85)


def metric_row(metrics: pd.DataFrame, dataset: str, model: str) -> pd.Series:
    rows = metrics.loc[(metrics["dataset"] == dataset) & (metrics["model"] == model)]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one metric row for {dataset}, {model}; found {len(rows)}")
    return rows.iloc[0]


def model_handle(model: str, linewidth: float | None = None) -> Line2D:
    return Line2D(
        [0],
        [0],
        color=MODEL_COLORS[model],
        linestyle=MODEL_LINESTYLES[model],
        linewidth=linewidth or (2.25 if model == PRIMARY else 1.45),
    )


def save_dual(fig: plt.Figure, figure: str, *, supplementary: bool = False) -> tuple[Path, Path]:
    destination = SUPP_OUT if supplementary else MAIN_OUT
    destination.mkdir(parents=True, exist_ok=True)
    basename = FIGURE_BASENAMES[figure]
    pdf_path = destination / f"{basename}.pdf"
    png_path = destination / f"{basename}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, format="png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def _legend_axis(fig: plt.Figure, subspec, plot_ratio: float = 4.0, legend_ratio: float = 1.75):
    sub = subspec.subgridspec(2, 1, height_ratios=[plot_ratio, legend_ratio], hspace=0.30)
    ax = fig.add_subplot(sub[0, 0])
    legend_ax = fig.add_subplot(sub[1, 0])
    legend_ax.axis("off")
    return ax, legend_ax


def plot_roc(
    ax: plt.Axes,
    legend_ax: plt.Axes,
    dataset: str,
    models: Sequence[str],
    roc_points: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    setup_axis(ax)
    ax.plot([0, 1], [0, 1], color="#858585", linestyle=":", linewidth=0.9, zorder=1)
    handles: list[Line2D] = []
    labels: list[str] = []
    for model in models:
        curve = roc_points.loc[
            (roc_points["dataset"] == dataset) & (roc_points["model"] == model)
        ].sort_values("point", kind="mergesort")
        ax.plot(
            curve["fpr"],
            curve["tpr"],
            color=MODEL_COLORS[model],
            linestyle=MODEL_LINESTYLES[model],
            linewidth=2.25 if model == PRIMARY else 1.35,
            zorder=3 if model == PRIMARY else 2,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        row = metric_row(metrics, dataset, model)
        handles.append(model_handle(model))
        labels.append(
            f"{MODEL_LABELS[model]}: AUC {row.auc:.3f} "
            f"({row.auc_ci_low:.3f}-{row.auc_ci_high:.3f})"
        )
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="1 - Specificity", ylabel="Sensitivity")
    ax.set_aspect("equal", adjustable="box")
    legend_ax.legend(
        handles,
        labels,
        loc="upper left",
        frameon=False,
        fontsize=6.2,
        handlelength=2.25,
        borderaxespad=0,
        labelspacing=0.18,
    )
    legend_ax.text(
        0,
        0.01,
        "95% CI: outcome-stratified bootstrap (2,000 resamples).",
        fontsize=6.2,
        va="bottom",
    )


def plot_pr(
    ax: plt.Axes,
    legend_ax: plt.Axes,
    dataset: str,
    models: Sequence[str],
    pr_points: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    setup_axis(ax)
    prevalence = float(metric_row(metrics, dataset, PRIMARY)["prevalence"])
    handles: list[Line2D] = []
    labels: list[str] = []
    for model in models:
        curve = pr_points.loc[
            (pr_points["dataset"] == dataset) & (pr_points["model"] == model)
        ].sort_values("point", kind="mergesort")
        ax.plot(
            curve["recall"],
            curve["precision"],
            color=MODEL_COLORS[model],
            linestyle=MODEL_LINESTYLES[model],
            linewidth=2.25 if model == PRIMARY else 1.35,
            zorder=3 if model == PRIMARY else 2,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        row = metric_row(metrics, dataset, model)
        handles.append(model_handle(model))
        labels.append(
            f"{MODEL_LABELS[model]}: AUPRC {row.auprc:.3f} "
            f"({row.auprc_ci_low:.3f}-{row.auprc_ci_high:.3f})"
        )
    ax.axhline(prevalence, color="#858585", linestyle=":", linewidth=0.9, zorder=1)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
    ax.set_aspect("equal", adjustable="box")
    handles.append(Line2D([0], [0], color="#858585", linestyle=":", linewidth=0.9))
    labels.append(f"Prevalence: {prevalence:.3f}")
    legend_ax.legend(
        handles,
        labels,
        loc="upper left",
        frameon=False,
        fontsize=6.2,
        handlelength=2.25,
        borderaxespad=0,
        labelspacing=0.18,
    )
    legend_ax.text(
        0,
        0.01,
        "95% CI: outcome-stratified bootstrap (2,000 resamples).",
        fontsize=6.2,
        va="bottom",
    )


def build_evaluation_calibration_v2(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    patient_predictions = frames["patient_predictions"]
    frozen_calibration = frames["calibration"]
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    model_scope = {
        A_DEVELOPMENT: FUSIONS,
        A_HOLDOUT: SIX_MODELS,
        B_EXTERNAL: SIX_MODELS,
        C_EXTERNAL: SIX_MODELS,
    }
    for dataset, models in model_scope.items():
        for model in models:
            group = patient_predictions.loc[
                (patient_predictions["dataset"] == dataset)
                & (patient_predictions["model"] == model)
            ]
            probability = group["probability"].to_numpy(float)
            outcome = group["y_SCLC"].to_numpy(float)
            frozen_metric = metric_row(frames["metrics"], dataset, model)
            recomputed_values = {
                "auc": float(roc_auc_score(outcome.astype(int), probability)),
                "auprc": float(average_precision_score(outcome.astype(int), probability)),
                "brier": float(brier_score_loss(outcome.astype(int), probability)),
            }
            audit_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "n": len(group),
                    **{
                        f"recomputed_{metric}": value
                        for metric, value in recomputed_values.items()
                    },
                    **{
                        f"frozen_{metric}": float(frozen_metric[metric])
                        for metric in ("auc", "auprc", "brier")
                    },
                    **{
                        f"abs_difference_{metric}": abs(value - float(frozen_metric[metric]))
                        for metric, value in recomputed_values.items()
                    },
                    "tolerance": 5e-12,
                    "within_tolerance": all(
                        abs(value - float(frozen_metric[metric])) <= 5e-12
                        for metric, value in recomputed_values.items()
                    ),
                }
            )
            lower, upper = np.quantile(probability, [0.025, 0.975])
            fit = lowess(
                endog=outcome,
                exog=probability,
                frac=EVALUATION_LOWESS_FRAC,
                it=EVALUATION_LOWESS_IT,
                delta=0.0,
                is_sorted=False,
                return_sorted=True,
            )
            fit = fit[
                np.isfinite(fit).all(axis=1)
                & (fit[:, 0] >= lower)
                & (fit[:, 0] <= upper)
            ]
            unique_fit = (
                pd.DataFrame({"x": fit[:, 0], "y": fit[:, 1]})
                .groupby("x", as_index=False)["y"]
                .mean()
                .sort_values("x", kind="mergesort")
            )
            dense_x = np.linspace(float(unique_fit["x"].min()), float(unique_fit["x"].max()), 300)
            dense_y = np.clip(
                np.interp(dense_x, unique_fit["x"].to_numpy(float), unique_fit["y"].to_numpy(float)),
                0.0,
                1.0,
            )
            for point, (x_value, y_value) in enumerate(zip(dense_x, dense_y)):
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "point_kind": "LOWESS",
                        "point": point,
                        "mean_predicted": float(x_value),
                        "observed_fraction": float(y_value),
                        "n_bin": np.nan,
                        "lowess_range_low": float(lower),
                        "lowess_range_high": float(upper),
                        "lowess_frac": EVALUATION_LOWESS_FRAC,
                        "lowess_it": EVALUATION_LOWESS_IT,
                        "source_component": "deterministic LOWESS from saved individual predictions",
                    }
                )
            bins = frozen_calibration.loc[
                (frozen_calibration["dataset"] == dataset)
                & (frozen_calibration["model"] == model)
                & (frozen_calibration["point_kind"] == "equal-frequency bin")
            ].sort_values("point", kind="mergesort")
            if len(bins) != 10:
                raise RuntimeError(f"Expected 10 frozen calibration groups for {dataset}, {model}")
            for _, bin_row in bins.iterrows():
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "point_kind": "equal-frequency bin",
                        "point": int(bin_row["point"]),
                        "mean_predicted": float(bin_row["mean_predicted"]),
                        "observed_fraction": float(bin_row["observed_fraction"]),
                        "n_bin": int(bin_row["n_bin"]),
                        "lowess_range_low": float(lower),
                        "lowess_range_high": float(upper),
                        "lowess_frac": EVALUATION_LOWESS_FRAC,
                        "lowess_it": EVALUATION_LOWESS_IT,
                        "source_component": "unchanged frozen equal-frequency point",
                    }
                )
    result = pd.DataFrame(rows)
    expected_rows = sum(len(models) for models in model_scope.values()) * (300 + 10)
    if len(result) != expected_rows:
        raise RuntimeError(f"Unexpected evaluation calibration output size: {len(result)}")
    EVALUATION_CALIBRATION_V2.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(EVALUATION_CALIBRATION_V2, index=False, encoding="utf-8-sig")
    audit_path = TABLE_OUT / "F9_F10_patient_prediction_metric_reproduction_audit.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False, encoding="utf-8-sig")
    return result


def plot_calibration(
    ax: plt.Axes,
    legend_ax: plt.Axes,
    dataset: str,
    models: Sequence[str],
    calibration_points: pd.DataFrame,
) -> None:
    setup_axis(ax, grid="both")
    ax.plot([0, 1], [0, 1], color="#6F6F6F", linestyle=":", linewidth=1.0, zorder=1)
    handles: list[Line2D] = []
    labels: list[str] = []
    for model in models:
        base = calibration_points.loc[
            (calibration_points["dataset"] == dataset)
            & (calibration_points["model"] == model)
        ]
        curve = base.loc[base["point_kind"] == "LOWESS"].sort_values("point", kind="mergesort")
        bins = base.loc[base["point_kind"] == "equal-frequency bin"].sort_values(
            "point", kind="mergesort"
        )
        ax.plot(
            curve["mean_predicted"],
            curve["observed_fraction"],
            color=MODEL_COLORS[model],
            linestyle=MODEL_LINESTYLES[model],
            linewidth=2.25 if model == PRIMARY else 1.35,
            zorder=3 if model == PRIMARY else 2,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
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
    legend_ax.legend(
        handles,
        labels,
        loc="upper left",
        ncol=2,
        frameon=False,
        fontsize=6.2,
        handlelength=2.2,
        borderaxespad=0,
        columnspacing=0.7,
        labelspacing=0.18,
    )
    legend_ax.text(
        0,
        0.01,
        "Local-linear LOWESS (tricube), frac=0.75, it=0; central 2.5%-97.5%;\n"
        "circles: unchanged 10 equal-frequency groups.",
        fontsize=6.2,
        va="bottom",
    )


def plot_dca(
    ax: plt.Axes,
    legend_ax: plt.Axes,
    dataset: str,
    models: Sequence[str],
    dca_points: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    setup_axis(ax)
    handles: list[Line2D] = []
    labels: list[str] = []
    for model in models:
        curve = dca_points.loc[
            (dca_points["dataset"] == dataset)
            & (dca_points["curve_type"] == "model")
            & (dca_points["model"] == model)
        ].sort_values("threshold", kind="mergesort")
        ax.plot(
            curve["threshold"],
            curve["net_benefit"],
            color=MODEL_COLORS[model],
            linestyle=MODEL_LINESTYLES[model],
            linewidth=2.25 if model == PRIMARY else 1.35,
            zorder=3 if model == PRIMARY else 2,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        handles.append(model_handle(model))
        labels.append(MODEL_LABELS[model])
    treat_all = dca_points.loc[
        (dca_points["dataset"] == dataset) & (dca_points["curve_type"] == "treat_all")
    ].sort_values("threshold", kind="mergesort")
    treat_none = dca_points.loc[
        (dca_points["dataset"] == dataset) & (dca_points["curve_type"] == "treat_none")
    ].sort_values("threshold", kind="mergesort")
    ax.plot(treat_all["threshold"], treat_all["net_benefit"], color="#777777", linestyle="--", linewidth=1.0)
    ax.plot(treat_none["threshold"], treat_none["net_benefit"], color="#111111", linestyle=":", linewidth=1.0)
    model_values = dca_points.loc[
        (dca_points["dataset"] == dataset) & (dca_points["curve_type"] == "model"),
        "net_benefit",
    ].to_numpy(float)
    prevalence = float(metric_row(metrics, dataset, PRIMARY)["prevalence"])
    lower = max(-0.18, min(-0.04, float(np.quantile(model_values, 0.04)) - 0.015))
    upper = min(0.45, max(prevalence + 0.055, float(np.quantile(model_values, 0.99)) + 0.02))
    ax.set(xlim=(0.01, 0.80), ylim=(lower, upper), xlabel="Threshold probability", ylabel="Net benefit")
    handles.extend(
        [
            Line2D([0], [0], color="#777777", linestyle="--", linewidth=1.0),
            Line2D([0], [0], color="#111111", linestyle=":", linewidth=1.0),
        ]
    )
    labels.extend(["Treat all", "Treat none"])
    legend_ax.legend(
        handles,
        labels,
        loc="upper left",
        ncol=2,
        frameon=False,
        fontsize=6.2,
        handlelength=2.2,
        borderaxespad=0,
        columnspacing=0.7,
        labelspacing=0.18,
    )
    legend_ax.text(
        0,
        0.01,
        "160 equally spaced thresholds (0.01-0.80); no Youden line.",
        fontsize=6.2,
        va="bottom",
    )


def plot_brier(
    ax: plt.Axes,
    legend_ax: plt.Axes,
    dataset: str,
    models: Sequence[str],
    metrics: pd.DataFrame,
) -> None:
    setup_axis(ax, grid="x")
    rows = [metric_row(metrics, dataset, model) for model in models]
    values = np.asarray([float(row["brier"]) for row in rows])
    lows = np.asarray([float(row["brier_ci_low"]) for row in rows])
    highs = np.asarray([float(row["brier_ci_high"]) for row in rows])
    positions = np.arange(len(models), dtype=float)
    for position, model, value, low, high in zip(positions, models, values, lows, highs):
        ax.errorbar(
            value,
            position,
            xerr=np.array([[value - low], [high - value]]),
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
        ax.text(high + 0.002, position, f"{value:.3f}", va="center", ha="left", fontsize=6.5)
    labels = [MODEL_LABELS[model] for model in models]
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    for tick, model in zip(ax.get_yticklabels(), models):
        tick.set_fontsize(6.5)
        if model == PRIMARY:
            tick.set_fontweight("bold")
    spread = max(0.006, float(highs.max() - lows.min()))
    ax.set_xlim(max(0.0, float(lows.min() - 0.12 * spread)), min(1.0, float(highs.max() + 0.30 * spread)))
    ax.set_xlabel("Brier score")
    ax.set_ylabel("")
    legend_ax.text(0, 0.80, "Point estimate and 95% CI", fontsize=6.5, fontweight="bold", va="top")
    legend_ax.text(
        0,
        0.45,
        "Outcome-stratified bootstrap; 2,000 resamples.\nLower values indicate better probabilistic accuracy.",
        fontsize=6.2,
        va="top",
    )


def make_f6(frames: dict[str, pd.DataFrame]) -> tuple[Path, Path]:
    calibration = frames["layer_calibration"]
    predictions = frames["layer_predictions"]
    smooth_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 3, figsize=(12.9, 4.25), sharex=True, sharey=True, facecolor="white")
    for ax, layer, letter in zip(axes, LAYER_ORDER, "abc"):
        setup_axis(ax, grid="both")
        ax.plot([0, 1], [0, 1], color="#777777", linewidth=1.0, linestyle=":", zorder=1)
        for dataset in LAYER_EVAL_DATASETS:
            subset = calibration.loc[
                (calibration["layer"] == layer) & (calibration["dataset"] == dataset)
            ]
            bins = subset.loc[subset["series_type"] == "equal_frequency_bin"].sort_values(
                "bin", kind="mergesort"
            )
            individual = predictions.loc[
                (predictions["layer"] == layer)
                & (predictions["dataset"] == dataset)
                & (predictions["model"] == PRIMARY)
            ]
            probability = individual["probability"].to_numpy(float)
            outcome = individual["y_SCLC"].to_numpy(float)
            lower, upper = np.quantile(probability, [0.025, 0.975])
            fit = lowess(
                endog=outcome,
                exog=probability,
                frac=F6_LOWESS_FRAC,
                it=F6_LOWESS_IT,
                delta=0.0,
                is_sorted=False,
                return_sorted=True,
            )
            fit = fit[
                np.isfinite(fit).all(axis=1)
                & (fit[:, 0] >= lower)
                & (fit[:, 0] <= upper)
            ]
            unique_fit = (
                pd.DataFrame({"x": fit[:, 0], "y": fit[:, 1]})
                .groupby("x", as_index=False)["y"]
                .mean()
                .sort_values("x", kind="mergesort")
            )
            dense_x = np.linspace(float(unique_fit["x"].min()), float(unique_fit["x"].max()), 300)
            dense_y = np.clip(
                np.interp(dense_x, unique_fit["x"].to_numpy(float), unique_fit["y"].to_numpy(float)),
                0.0,
                1.0,
            )
            curve = pd.DataFrame({"x": dense_x, "y": dense_y})
            for point, (x_value, y_value) in enumerate(zip(dense_x, dense_y)):
                smooth_rows.append(
                    {
                        "dataset": dataset,
                        "layer": layer,
                        "model": PRIMARY,
                        "point": point,
                        "predicted_probability": float(x_value),
                        "observed_probability_lowess": float(y_value),
                        "lowess_frac": F6_LOWESS_FRAC,
                        "lowess_it": F6_LOWESS_IT,
                        "prediction_percentile_low": float(lower),
                        "prediction_percentile_high": float(upper),
                        "source": str(SOURCE_FILES["layer_predictions"]),
                    }
                )
            color = DATASET_COLORS[dataset]
            ax.plot(
                curve["x"],
                curve["y"],
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
    smooth_path = DATA_OUT / "F6_layer_calibration_lowess_frac075.csv"
    smooth_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(smooth_rows).to_csv(smooth_path, index=False, encoding="utf-8-sig")
    return save_dual(fig, "F6")


def make_f7(frames: dict[str, pd.DataFrame]) -> tuple[Path, Path]:
    metrics = frames["layer_metrics"]
    scope = metrics.loc[metrics["dataset"].isin(LAYER_EVAL_DATASETS)]
    x_low = float(scope["brier_ci_low"].min())
    x_high = float(scope["brier_ci_high"].max())
    span = max(0.008, x_high - x_low)
    limits = (max(0.0, x_low - 0.10 * span), min(1.0, x_high + 0.30 * span))
    positions = np.arange(len(LAYER_EVAL_DATASETS), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharex=True, sharey=True, facecolor="white")
    for ax, layer, letter in zip(axes, LAYER_ORDER, "abc"):
        setup_axis(ax, grid="x")
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
    return save_dual(fig, "F7")


def _fmt_ci(point: float, low: float, high: float, decimals: int = 3, sign: bool = False) -> str:
    marker = "+" if sign else ""
    return f"{point:{marker}.{decimals}f} ({low:{marker}.{decimals}f} to {high:{marker}.{decimals}f})"


def _fmt_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def make_f8(frames: dict[str, pd.DataFrame]) -> tuple[Path, Path]:
    metrics = frames["layer_metrics"]
    increments = frames["layer_increments"]
    delong = frames["layer_delong"]
    fig = plt.figure(figsize=(14.8, 6.0), facecolor="white")
    outer = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.55], wspace=0.24)
    left = outer[0, 0].subgridspec(2, 1, height_ratios=[3.35, 1.80], hspace=0.34)
    ax_a = fig.add_subplot(left[0, 0])
    ax_inc = fig.add_subplot(left[1, 0])
    right = outer[0, 1].subgridspec(1, 2, width_ratios=[1.05, 1.45], wspace=0.06)
    ax_b = fig.add_subplot(right[0, 0])
    ax_table = fig.add_subplot(right[0, 1])

    data = metrics.loc[metrics["dataset"] == A_DEVELOPMENT].set_index("layer").loc[LAYER_ORDER]
    x = np.arange(len(LAYER_ORDER), dtype=float)
    styles = {
        "auc": ("ROC AUC", "#2F5597", "o", -0.04),
        "auprc": ("AUPRC", "#D55E00", "s", 0.04),
    }
    lows_all: list[float] = []
    highs_all: list[float] = []
    for metric, (label, color, marker, offset) in styles.items():
        points = data[metric].to_numpy(float)
        lows = data[f"{metric}_ci_low"].to_numpy(float)
        highs = data[f"{metric}_ci_high"].to_numpy(float)
        lows_all.extend(lows.tolist())
        highs_all.extend(highs.tolist())
        ax_a.errorbar(
            x + offset,
            points,
            yerr=np.vstack([points - lows, highs - points]),
            color=color,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=6.2,
            linewidth=1.55,
            capsize=3.0,
            label=label,
        )
    setup_axis(ax_a, grid="y")
    y_min = max(0, min(lows_all) - 0.075)
    y_max = min(1, max(highs_all) + 0.045)
    ax_a.set(xlim=(-0.28, 2.28), ylim=(y_min, y_max), xlabel="Cumulative layer", ylabel="Discrimination")
    ax_a.set_xticks(x, [LAYER_LABELS[layer] for layer in LAYER_ORDER])
    ax_a.legend(loc="upper left", frameon=False)
    panel_label(ax_a, "a", x=-0.13, y=1.02)

    ax_inc.axis("off")
    comparisons = ["I → I + M", "I + M → I + M + N", "I → I + M + N"]
    header_y = 0.90
    ax_inc.text(0.00, header_y, "Prespecified increment", fontweight="bold", fontsize=7.5)
    ax_inc.text(0.37, header_y, "ΔAUC (95% CI)", fontweight="bold", fontsize=7.5)
    ax_inc.text(0.70, header_y, "ΔAUPRC (95% CI)", fontweight="bold", fontsize=7.5)
    for index, comparison in enumerate(comparisons):
        y = 0.68 - index * 0.25
        rows = increments.loc[(increments["comparison"] == comparison) & increments["metric"].isin(["auc", "auprc"])].set_index("metric")
        ax_inc.text(0.00, y, comparison.replace("→", "to"), fontsize=7.2)
        for xpos, metric in ((0.37, "auc"), (0.70, "auprc")):
            row = rows.loc[metric]
            ax_inc.text(
                xpos,
                y,
                _fmt_ci(float(row["increment"]), float(row["ci_low"]), float(row["ci_high"]), sign=True),
                fontsize=7.0,
            )
    ax_inc.text(0.00, -0.04, "Increment CIs: paired outcome-stratified bootstrap (2,000 resamples).", fontsize=6.7, color="#444444")

    ordered = delong.set_index("comparison").loc[comparisons].reset_index()
    y = np.arange(len(ordered), dtype=float)
    point = ordered["delta_auc"].to_numpy(float)
    low = ordered["ci_low"].to_numpy(float)
    high = ordered["ci_high"].to_numpy(float)
    setup_axis(ax_b, grid="x")
    ax_b.axvline(0, color="#777777", linestyle=":", linewidth=1.0)
    ax_b.errorbar(
        point,
        y,
        xerr=np.vstack([point - low, high - point]),
        fmt="o",
        color="#2F5597",
        markerfacecolor="white",
        markeredgewidth=1.2,
        markersize=6.2,
        capsize=3.2,
        elinewidth=1.25,
    )
    span = max(0.02, max(0, float(high.max())) - min(0, float(low.min())))
    ax_b.set_xlim(min(0, float(low.min())) - 0.13 * span, max(0, float(high.max())) + 0.13 * span)
    ax_b.set_ylim(-0.65, len(ordered) - 0.35)
    ax_b.invert_yaxis()
    ax_b.set_yticks(y, [value.replace("→", "to") for value in comparisons])
    ax_b.set_xlabel("ΔAUC (upper - lower)")
    panel_label(ax_b, "b", x=-0.22, y=1.02)

    ax_table.axis("off")
    ax_table.set_ylim(-0.65, len(ordered) - 0.35)
    ax_table.invert_yaxis()
    ax_table.text(0.01, -0.48, "ΔAUC (95% CI)", fontweight="bold", fontsize=7.6)
    ax_table.text(0.68, -0.48, "P", fontweight="bold", fontsize=7.6)
    ax_table.text(0.83, -0.48, "Holm P", fontweight="bold", fontsize=7.6)
    for index, row in ordered.iterrows():
        ax_table.text(0.01, index, _fmt_ci(float(row.delta_auc), float(row.ci_low), float(row.ci_high), sign=True), fontsize=7.2, va="center")
        ax_table.text(0.68, index, _fmt_p(float(row.p_raw)), fontsize=7.2, va="center")
        ax_table.text(0.83, index, _fmt_p(float(row.p_holm)), fontsize=7.2, va="center")
    ax_table.text(
        0.01,
        1.02,
        "Paired DeLong; upper minus lower; one prespecified three-comparison Holm family.",
        transform=ax_table.transAxes,
        fontsize=6.8,
        color="#444444",
        va="bottom",
    )
    fig.subplots_adjust(left=0.065, right=0.995, top=0.94, bottom=0.11)
    return save_dual(fig, "F8")


def make_f9(frames: dict[str, pd.DataFrame]) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(20.0, 10.8), facecolor="white")
    outer = fig.add_gridspec(
        2,
        5,
        left=0.060,
        right=0.995,
        top=0.98,
        bottom=0.025,
        wspace=0.32,
        hspace=0.18,
    )
    names = ["ROC", "PR", "Calibration", "DCA", "Brier"]
    rows = [
        (A_DEVELOPMENT, FUSIONS, frames["evaluation_calibration_v2"]),
        (A_HOLDOUT, SIX_MODELS, frames["evaluation_calibration_v2"]),
    ]
    letters = "abcdefghij"
    for row_index, (dataset, models, calibration_frame) in enumerate(rows):
        for column, name in enumerate(names):
            ax, legend_ax = _legend_axis(
                fig,
                outer[row_index, column],
                plot_ratio=4.1,
                legend_ratio=2.05,
            )
            panel_label(ax, letters[row_index * 5 + column], x=-0.17, y=1.02)
            if name == "ROC":
                plot_roc(ax, legend_ax, dataset, models, frames["roc"], frames["metrics"])
            elif name == "PR":
                plot_pr(ax, legend_ax, dataset, models, frames["pr"], frames["metrics"])
            elif name == "Calibration":
                plot_calibration(ax, legend_ax, dataset, models, calibration_frame)
            elif name == "DCA":
                plot_dca(ax, legend_ax, dataset, models, frames["dca"], frames["metrics"])
            else:
                plot_brier(ax, legend_ax, dataset, models, frames["metrics"])
        row_text = (
            "A development (10-fold CV)\nThree fusion models"
            if dataset == A_DEVELOPMENT
            else "A locked evaluation\nSix frozen models"
        )
        fig.text(
            0.012,
            0.745 if row_index == 0 else 0.265,
            row_text,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9.6,
            fontweight="bold",
        )
    return save_dual(fig, "F9")


def make_f10(frames: dict[str, pd.DataFrame]) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(18.0, 10.4), facecolor="white")
    outer = fig.add_gridspec(2, 5, left=0.06, right=0.995, top=0.98, bottom=0.025, wspace=0.32, hspace=0.18)
    names = ["ROC", "PR", "Calibration", "DCA", "Brier"]
    datasets = [B_EXTERNAL, C_EXTERNAL]
    letters = "abcdefghij"
    for row, dataset in enumerate(datasets):
        for column, name in enumerate(names):
            ax, legend_ax = _legend_axis(fig, outer[row, column], plot_ratio=4.1, legend_ratio=2.05)
            panel_label(ax, letters[row * 5 + column], x=-0.17, y=1.02)
            if name == "ROC":
                plot_roc(ax, legend_ax, dataset, SIX_MODELS, frames["roc"], frames["metrics"])
            elif name == "PR":
                plot_pr(ax, legend_ax, dataset, SIX_MODELS, frames["pr"], frames["metrics"])
            elif name == "Calibration":
                plot_calibration(ax, legend_ax, dataset, SIX_MODELS, frames["evaluation_calibration_v2"])
            elif name == "DCA":
                plot_dca(ax, legend_ax, dataset, SIX_MODELS, frames["dca"], frames["metrics"])
            else:
                plot_brier(ax, legend_ax, dataset, SIX_MODELS, frames["metrics"])
        fig.text(
            0.012,
            0.745 if row == 0 else 0.265,
            DATASET_LABELS[dataset],
            rotation=90,
            ha="center",
            va="center",
            fontsize=10.0,
            fontweight="bold",
        )
    return save_dual(fig, "F10")


def make_s1(frames: dict[str, pd.DataFrame]) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(13.4, 5.2), facecolor="white")
    outer = fig.add_gridspec(1, 3, left=0.055, right=0.99, top=0.95, bottom=0.04, wspace=0.30)
    ax_a, legend_a = _legend_axis(fig, outer[0, 0], legend_ratio=1.55)
    ax_b, legend_b = _legend_axis(fig, outer[0, 1], legend_ratio=1.55)
    plot_roc(ax_a, legend_a, A_DEVELOPMENT, FUSIONS, frames["roc"], frames["metrics"])
    plot_pr(ax_b, legend_b, A_DEVELOPMENT, FUSIONS, frames["pr"], frames["metrics"])
    panel_label(ax_a, "a", x=-0.14, y=1.02)
    panel_label(ax_b, "b", x=-0.14, y=1.02)

    sub = outer[0, 2].subgridspec(2, 1, height_ratios=[3.45, 1.65], hspace=0.34)
    ax_c = fig.add_subplot(sub[0, 0])
    note_ax = fig.add_subplot(sub[1, 0])
    setup_axis(ax_c, grid="x")
    rows = [metric_row(frames["metrics"], A_DEVELOPMENT, model) for model in FUSIONS]
    values = np.asarray([float(row.auc) for row in rows])
    lows = np.asarray([float(row.auc_ci_low) for row in rows])
    highs = np.asarray([float(row.auc_ci_high) for row in rows])
    positions = np.arange(len(FUSIONS), dtype=float)
    for position, model, value, low, high in zip(positions, FUSIONS, values, lows, highs):
        ax_c.errorbar(
            value,
            position,
            xerr=np.array([[value - low], [high - value]]),
            fmt="o",
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model],
            capsize=3,
            elinewidth=1.35,
            markersize=5.5 if model == PRIMARY else 4.8,
            markeredgecolor="white",
            markeredgewidth=0.4,
        )
        ax_c.text(high + 0.003, position, f"{value:.3f}", fontsize=6.8, va="center")
    ax_c.set_yticks(positions, [MODEL_LABELS[model] for model in FUSIONS])
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Pooled OOF ROC AUC")
    ax_c.set_xlim(max(0, float(lows.min()) - 0.015), min(1, float(highs.max()) + 0.035))
    for tick, model in zip(ax_c.get_yticklabels(), FUSIONS):
        tick.set_fontsize(7.0)
        if model == PRIMARY:
            tick.set_fontweight("bold")
    panel_label(ax_c, "c", x=-0.16, y=1.02)
    weighted_layer = frames["layer_metrics"].loc[
        (frames["layer_metrics"]["dataset"] == A_DEVELOPMENT)
        & (frames["layer_metrics"]["layer"] == "I_M_N")
        & (frames["layer_metrics"]["model"] == PRIMARY)
    ]
    if len(weighted_layer) != 1:
        raise RuntimeError("The archived Weighted-voting fold-mean row is missing")
    fold_mean = float(weighted_layer.iloc[0]["fold_auc_mean"])
    fold_sd = float(weighted_layer.iloc[0]["fold_auc_sd"])
    note_ax.axis("off")
    note_ax.text(
        0,
        0.98,
        "Curves and CIs use pooled patient-level out-of-fold predictions.\n"
        f"Archived Weighted-voting mean of 10 fold AUCs: {fold_mean:.3f} ± {fold_sd:.3f}.\n"
        "The fold mean and pooled OOF AUC are distinct estimands and are not interchangeable.",
        fontsize=6.8,
        va="top",
        color="#333333",
    )
    return save_dual(fig, "S1", supplementary=True)


def make_s6(frames: dict[str, pd.DataFrame]) -> tuple[Path, Path]:
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
        setup_axis(ax, grid="x")
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
    return save_dual(fig, "S6", supplementary=True)


def build_manifest(outputs: dict[str, tuple[Path, Path]]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    metric_sources = {
        "ROC": "roc | metrics",
        "PR": "pr | metrics",
        "Calibration": "evaluation_calibration_v2 | calibration",
        "DCA": "dca | metrics",
        "Brier score": "metrics",
    }
    for index, layer in enumerate(LAYER_ORDER):
        for figure, metric in (("F6", "Calibration"), ("F7", "Brier score")):
            rows.append(
                {
                    "figure": figure,
                    "panel": "abc"[index],
                    "metric": metric,
                    "dataset": " | ".join(LAYER_EVAL_DATASETS),
                    "layer": layer,
                    "models": PRIMARY,
                    "data_source": "layer_predictions | layer_calibration" if figure == "F6" else "layer_metrics",
                    "output_png": str(outputs[figure][0]),
                    "output_pdf": str(outputs[figure][1]),
                }
            )
    rows.extend(
        [
            {
                "figure": "F8",
                "panel": "a",
                "metric": "AUC, AUPRC, and paired increments with 95% CI",
                "dataset": A_DEVELOPMENT,
                "layer": "I | I_M | I_M_N",
                "models": PRIMARY,
                "data_source": "layer_metrics | layer_increments",
                "output_png": str(outputs["F8"][0]),
                "output_pdf": str(outputs["F8"][1]),
            },
            {
                "figure": "F8",
                "panel": "b",
                "metric": "Paired DeLong and Holm correction",
                "dataset": A_DEVELOPMENT,
                "layer": "I | I_M | I_M_N",
                "models": PRIMARY,
                "data_source": "layer_delong",
                "output_png": str(outputs["F8"][0]),
                "output_pdf": str(outputs["F8"][1]),
            },
        ]
    )
    metrics = ["ROC", "PR", "Calibration", "DCA", "Brier score"]
    for row_index, (dataset, models) in enumerate(
        [(A_DEVELOPMENT, FUSIONS), (A_HOLDOUT, SIX_MODELS)]
    ):
        for column, metric in enumerate(metrics):
            calibration_source = (
                "calibration"
                if dataset == A_DEVELOPMENT and metric == "Calibration"
                else metric_sources[metric]
            )
            rows.append(
                {
                    "figure": "F9",
                    "panel": "abcdefghij"[row_index * 5 + column],
                    "metric": metric,
                    "dataset": dataset,
                    "layer": "I_M_N",
                    "models": " | ".join(models),
                    "data_source": calibration_source,
                    "output_png": str(outputs["F9"][0]),
                    "output_pdf": str(outputs["F9"][1]),
                }
            )
    for row_index, dataset in enumerate([B_EXTERNAL, C_EXTERNAL]):
        for column, metric in enumerate(metrics):
            rows.append(
                {
                    "figure": "F10",
                    "panel": "abcdefghij"[row_index * 5 + column],
                    "metric": metric,
                    "dataset": dataset,
                    "layer": "I_M_N",
                    "models": " | ".join(SIX_MODELS),
                    "data_source": metric_sources[metric],
                    "output_png": str(outputs["F10"][0]),
                    "output_pdf": str(outputs["F10"][1]),
                }
            )
    for index, metric in enumerate(["ROC", "PR", "Pooled OOF AUC with CI and fold-mean estimand note"]):
        rows.append(
            {
                "figure": "S1",
                "panel": "abc"[index],
                "metric": metric,
                "dataset": A_DEVELOPMENT,
                "layer": "I_M_N",
                "models": " | ".join(FUSIONS),
                "data_source": "roc | metrics" if index == 0 else ("pr | metrics" if index == 1 else "metrics | layer_metrics"),
                "output_png": str(outputs["S1"][0]),
                "output_pdf": str(outputs["S1"][1]),
            }
        )
    for index, dataset in enumerate([A_DEVELOPMENT, A_HOLDOUT, B_EXTERNAL, C_EXTERNAL]):
        rows.append(
            {
                "figure": "S6",
                "panel": "abcd"[index],
                "metric": "DeLong ΔAUC with raw and Holm-adjusted P",
                "dataset": dataset,
                "layer": "I_M_N",
                "models": "Weighted voting versus frozen comparators",
                "data_source": "model_delong",
                "output_png": str(outputs["S6"][0]),
                "output_pdf": str(outputs["S6"][1]),
            }
        )
    manifest = pd.DataFrame(rows)
    source_lookup = {name: str(path) for name, path in SOURCE_FILES.items()}
    source_lookup["layer_metrics"] = (
        f"{SOURCE_FILES['layer_metrics']} | {HARMONISED_LAYER_METRICS}"
    )
    source_lookup["evaluation_calibration_v2"] = str(EVALUATION_CALIBRATION_V2)
    manifest["data_source"] = manifest["data_source"].map(
        lambda value: " | ".join(source_lookup.get(part.strip(), part.strip()) for part in value.split("|"))
    )
    return manifest


def write_figure_legends() -> Path:
    text = """# Figure legends

**Supplementary Figure S3. Layer-specific calibration of Weighted voting.** Panels a-c show the I, I + M, and I + M + N cumulative layers, respectively, in the A locked evaluation set and the B and C external-validation sets. Only the frozen primary model is displayed. Solid curves are deterministically recalculated local-linear LOWESS smooths from the frozen individual predictions, using tricube neighbourhood weights (fraction 0.75, zero robustifying iterations) and restricted to the central 2.5th-97.5th percentile of predicted probability; open circles preserve the original ten equal-frequency groups. The dotted diagonal denotes ideal calibration. A-development predictions are not displayed as evaluation curves.

**Supplementary Figure S2. Layer-specific Brier scores of Weighted voting.** Panels a-c show the same cumulative layers and evaluation datasets as Supplementary Figure S3. Points are the unchanged Brier scores and bars are 95% percentile confidence intervals from 2,000 outcome-stratified bootstrap resamples. Lower values indicate better probabilistic accuracy.

**Unnumbered layer-increment display.** Manuscript Figure 8 is the online tool. This display shows the incremental value of cumulative biological layers for Weighted voting. Panel a shows pooled A-development out-of-fold ROC AUC and AUPRC with 95% bootstrap confidence intervals, together with the three prespecified paired increments and their 95% confidence intervals. Panel b shows the three prespecified paired DeLong comparisons in the direction upper layer minus lower layer, including Delta AUC, 95% confidence interval, raw P value, and Holm-adjusted P value.

**Unnumbered A-centre comparison after model and layer freezing.** Manuscript Figure 4 shows only the three fusion models on the evaluation set. Panels a-e show ROC, precision-recall, calibration, decision-curve, and Brier analyses for the three fusion candidates in A-development 10-fold cross-validation. Panels f-j show the same analyses in the A-centre locked evaluation set for the three fusion models plus Extra Trees, Rotation Forest, and Random Forest. ROC and precision-recall legends show AUC/AUPRC and 95% bootstrap confidence intervals. Calibration curves use deterministic local-linear LOWESS smoothing with tricube neighbourhood weights (fraction 0.75, zero robustifying iterations); the smooths are restricted to the central 2.5th-97.5th percentile of predicted probability, and circles preserve the original ten equal-frequency groups. Decision curves use 160 equally spaced thresholds from 0.01 to 0.80 and do not show a Youden line. Weighted voting is first and marked as the primary model; model and layer selection were based only on A-development cross-validation.

**Unnumbered external comparison of the frozen models.** Manuscript Figure 5 shows the three fusion models. Panels a-e show centre B and panels f-j show centre C. The datasets, six-model order, colors, line styles, calibration method (local-linear LOWESS, frac=0.75, it=0, central 2.5th-97.5th percentile, with unchanged equal-frequency points), confidence-interval method, and decision-curve threshold grid are fixed from the unnumbered A-centre comparison. No model, threshold, or layer was selected using either external cohort.

**Unnumbered A-development fusion-model selection display.** Manuscript Supplementary Figure S1 is the layer precision-recall figure. Panels a and b show ROC and precision-recall curves from pooled patient-level 10-fold out-of-fold predictions for the three frozen fusion candidates. Panel c shows pooled out-of-fold AUCs and bootstrap confidence intervals. Pooled out-of-fold metrics are explicitly distinguished from the archived unweighted mean of the ten fold-specific AUCs; these estimands are not interchangeable.

**Unnumbered DeLong display referenced to Weighted voting.** Panels a-d show A development, A locked evaluation, B external validation, and C external validation, respectively. Points and bars show the paired AUC difference (Weighted voting minus comparator) and its 95% DeLong confidence interval. Raw and Holm-adjusted P values are reported; multiplicity correction is applied within each frozen dataset-specific family. Positive values favor Weighted voting.
"""
    path = MANUSCRIPT_OUT / "figure_legends_F6_F10_S1_S6.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_run_notes(outputs: dict[str, tuple[Path, Path]], manifest_path: Path, legends_path: Path) -> Path:
    source_lines = "\n".join(f"- `{name}`: `{path}` (SHA256 `{_sha256(path)}`)" for name, path in SOURCE_FILES.items())
    output_lines = "\n".join(
        f"- {figure}: `{png}` and `{pdf}`" for figure, (png, pdf) in outputs.items()
    )
    text = f"""# F6-F10 and selection supplements: frozen-data plotting run

This plotting run reads frozen result CSV files from `{SOURCE_ROOT}` and the exact saved patient-level prediction CSVs enumerated below solely to recompute the F9/F10 display smooths. It does not import a model factory, run preprocessing, fit an estimator, alter predictions, or overwrite any earlier figure. Before smoothing, the saved individual scores are required to reproduce every frozen AUC, AUPRC, and Brier value within numerical tolerance.

## Source files

{source_lines}

## Outputs

{output_lines}

- Panel manifest: `{manifest_path}`
- Figure legends: `{legends_path}`
- F9/F10 metric-reproduction audit: `{TABLE_OUT / "F9_F10_patient_prediction_metric_reproduction_audit.csv"}`

## Statistical display contract

- Weighted voting is the primary model and is listed first.
- F6 and F7 exclude A-development evaluation curves and retain A holdout plus B/C external validation.
- F8 alone retains the A-development layer-selection evidence in the main sequence.
- The unnumbered A-centre comparison contains development 10-fold-CV evidence for the three fusion candidates and the evaluation comparison. It is not manuscript Figure 4 or Supplementary Figure S1.
- AUC, AUPRC, Brier scores, confidence limits, DeLong estimates, and Holm P values are read without numerical modification from the frozen tables.
- Where the identical full-layer Weighted-voting estimand appeared in two archives with slightly different Monte-Carlo bootstrap limits, the model-comparison CI archive is reused consistently in every display. Point-estimate identity is verified in `{CI_HARMONISATION_AUDIT}`; no confidence interval is recomputed.
- F6, F9, and F10 LOWESS curves are deterministically recalculated from unchanged individual probabilities and outcomes using frac=0.75 and it=0; the original equal-frequency points are retained unchanged. All DCA curves are read from the frozen curve-point file and use the existing 160-point equally spaced grid with no Youden line.
- PNG output is 600 dpi; PDF output is editable vector artwork with embedded TrueType text.
"""
    path = OUTPUT_ROOT / "README_F6_F10_S1_S6.md"
    path.write_text(text, encoding="utf-8")
    return path


def run_basic_qc(outputs: dict[str, tuple[Path, Path]]) -> Path:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:
        from PyPDF2 import PdfReader
    import fitz

    rows: list[dict[str, object]] = []
    for figure, (png_path, pdf_path) in outputs.items():
        with Image.open(png_path) as image:
            width, height = image.size
            dpi = image.info.get("dpi", (np.nan, np.nan))
        reader = PdfReader(str(pdf_path))
        fitz_doc = fitz.open(pdf_path)
        pdf_fonts = sorted({font[3] for page in fitz_doc for font in page.get_fonts(full=True)})
        fitz_doc.close()
        image_xobjects = 0
        for page in reader.pages:
            resources = page.get("/Resources")
            if resources:
                resources = resources.get_object()
            if resources and resources.get("/XObject"):
                xobjects = resources["/XObject"].get_object()
                for obj in xobjects.values():
                    if obj.get_object().get("/Subtype") == "/Image":
                        image_xobjects += 1
        rows.append(
            {
                "figure": figure,
                "png_exists": png_path.exists(),
                "png_size_bytes": png_path.stat().st_size,
                "png_width_px": width,
                "png_height_px": height,
                "png_dpi_x": float(dpi[0]),
                "png_dpi_y": float(dpi[1]),
                "pdf_exists": pdf_path.exists(),
                "pdf_size_bytes": pdf_path.stat().st_size,
                "pdf_pages": len(reader.pages),
                "pdf_embedded_image_xobjects": image_xobjects,
                "pdf_vector_only_pass": image_xobjects == 0,
                "pdf_fonts": " | ".join(pdf_fonts),
                "times_new_roman_only_pass": bool(pdf_fonts)
                and all("TimesNewRoman" in font for font in pdf_fonts),
                "png_sha256": _sha256(png_path),
                "pdf_sha256": _sha256(pdf_path),
            }
        )
    frame = pd.DataFrame(rows)
    path = QC_OUT / "F6_F10_S1_S6_basic_qc.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    apply_style()
    for directory in (DATA_OUT, TABLE_OUT, MAIN_OUT, SUPP_OUT, MANUSCRIPT_OUT, QC_OUT):
        directory.mkdir(parents=True, exist_ok=True)
    frames = load_frozen_data()
    frames["layer_metrics"] = harmonise_full_layer_ci(frames)
    frames["evaluation_calibration_v2"] = build_evaluation_calibration_v2(frames)
    outputs = {
        "F6": make_f6(frames),
        "F7": make_f7(frames),
        "F8": make_f8(frames),
        "F9": make_f9(frames),
        "F10": make_f10(frames),
        "S1": make_s1(frames),
        "S6": make_s6(frames),
    }
    manifest = build_manifest(outputs)
    manifest_path = TABLE_OUT / "F6_F10_S1_S6_panel_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    legends_path = write_figure_legends()
    notes_path = write_run_notes(outputs, manifest_path, legends_path)
    qc_path = run_basic_qc(outputs)
    run_manifest = {
        "analysis": "Frozen-data plot-only F6-F10, S1, and S6 revision",
        "primary_model": PRIMARY,
        "main_figures": [],
        "manuscript_figures_produced_elsewhere": ["Fig3", "Fig4", "Fig5", "Fig6", "Fig7", "FigS1", "FigS2", "FigS3", "FigS4", "FigS5", "FigS6", "FigS7", "FigS8"],
        "supplementary_figures": ["S1", "S6"],
        "source_root": str(SOURCE_ROOT),
        "source_files": [
            {"name": name, "path": str(path), "sha256": _sha256(path)}
            for name, path in SOURCE_FILES.items()
        ],
        "model_refit": False,
        "prediction_recalculation": False,
        "statistical_recalculation": False,
        "deterministic_display_recalculation": {
            "figures": ["F6", "F9", "F10"],
            "method": "local-linear LOWESS on unchanged saved individual outcomes and probabilities",
            "frac": EVALUATION_LOWESS_FRAC,
            "it": EVALUATION_LOWESS_IT,
            "display_range": "central 2.5th-97.5th percentile of predicted probability",
            "equal_frequency_points_modified": False,
            "metric_reproduction_tolerance": 5e-12,
            "metric_reproduction_audit": str(
                TABLE_OUT / "F9_F10_patient_prediction_metric_reproduction_audit.csv"
            ),
            "derived_curve_files": [
                str(DATA_OUT / "F6_layer_calibration_lowess_frac075.csv"),
                str(EVALUATION_CALIBRATION_V2),
            ],
        },
        "output_files": [
            {"figure": figure, "png": str(paths[0]), "pdf": str(paths[1])}
            for figure, paths in outputs.items()
        ],
        "panel_manifest": str(manifest_path),
        "figure_legends": str(legends_path),
        "run_notes": str(notes_path),
        "basic_qc": str(qc_path),
        "style": {
            "language": "English",
            "font": "Times New Roman",
            "panel_labels": "lower-case bold",
            "figure_titles": False,
            "png_dpi": 600,
            "pdf_fonttype": 42,
        },
    }
    manifest_json = DATA_OUT / "run_manifest_F6_F10_S1_S6.json"
    manifest_json.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated {len(outputs)} figure pairs from frozen CSV inputs only.")
    print(f"Panel manifest: {manifest_path}")
    print(f"Figure legends: {legends_path}")
    print(f"Basic QC: {qc_path}")
    print(f"Run manifest: {manifest_json}")


if __name__ == "__main__":
    main()
