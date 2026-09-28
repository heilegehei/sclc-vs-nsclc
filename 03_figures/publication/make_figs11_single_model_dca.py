from __future__ import annotations
import sys as _public_sys
from pathlib import Path as _PublicPath
_public_root = next(p for p in _PublicPath(__file__).resolve().parents if (p / "common" / "public_paths.py").is_file())
if str(_public_root) not in _public_sys.path:
    _public_sys.path.insert(0, str(_public_root))
from common.public_paths import code_root, data_root, explainability_results, fusion_results, publication_results

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from nature_style import apply_style, save_supp_fig


HERE = Path(__file__).resolve().parent
PREDICTIONS_INPUT = data_root() / "fusion_results" / "native_predictions_all_eval.csv"

DATASET = "A_holdout_clean"
DATASET_LABEL = "A evaluation set"
THRESHOLDS = np.linspace(0.01, 0.80, 160)


SINGLE_MODELS = [
    "logistic_regression",
    "gam",
    "knn",
    "gaussian_nb",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "gbdt",
    "xgboost",
    "lightgbm",
    "adaboost",
    "rotation_forest",
    "mlp",
]
DISPLAY_NAME = {
    "logistic_regression": "Logistic regression",
    "gam": "GAM",
    "knn": "KNN",
    "gaussian_nb": "Gaussian NB",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "gbdt": "GBDT",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "adaboost": "AdaBoost",
    "rotation_forest": "Rotation Forest",
    "mlp": "MLP",
}
MODEL_COLOR = {
    "logistic_regression": "#0072B2",
    "gam": "#E69F00",
    "knn": "#009E73",
    "rbf_svm": "#CC79A7",
    "gaussian_nb": "#56B4E9",
    "decision_tree": "#D55E00",
    "random_forest": "#6A3D9A",
    "extra_trees": "#A6761D",
    "gbdt": "#4D9221",
    "xgboost": "#B2182B",
    "lightgbm": "#777777",
    "adaboost": "#8C6D31",
    "rotation_forest": "#17A2B8",
    "mlp": "#5C5C5C",
}
_ALL_SINGLE_ORDER = [
    "logistic_regression", "gam", "knn", "rbf_svm", "gaussian_nb",
    "decision_tree", "random_forest", "extra_trees", "gbdt", "xgboost",
    "lightgbm", "adaboost", "rotation_forest", "mlp",
]
MODEL_LINESTYLE = {
    model: ["-", "--", "-.", (0, (5, 2)), (0, (1, 1))][index % 5]
    for index, model in enumerate(_ALL_SINGLE_ORDER)
}


def _load_and_validate_predictions() -> tuple[dict[str, pd.DataFrame], int, int]:
    required = ["dataset", "model", "record_id", "y_SCLC", "native_probability", "score_kind"]
    if not PREDICTIONS_INPUT.is_file():
        raise FileNotFoundError(f"Required archived prediction file is missing: {PREDICTIONS_INPUT}")
    frame = pd.read_csv(PREDICTIONS_INPUT, usecols=required)
    subset = frame.loc[
        (frame["dataset"] == DATASET) & frame["model"].isin(SINGLE_MODELS)
    ].copy()

    if set(subset["model"].unique()) != set(SINGLE_MODELS):
        raise RuntimeError("The archived file does not contain exactly the 13 prespecified individual models.")
    if not subset["score_kind"].eq("probability").all():
        bad = sorted(subset.loc[~subset["score_kind"].eq("probability"), "model"].unique())
        raise RuntimeError(f"Every included model must have score_kind='probability'; failed: {bad}")
    if subset[["record_id", "y_SCLC", "native_probability"]].isna().any().any():
        raise RuntimeError("Missing record identifier, outcome, or native probability in the frozen input.")

    output: dict[str, pd.DataFrame] = {}
    expected_ids: tuple[str, ...] | None = None
    expected_y: np.ndarray | None = None
    expected_n: int | None = None
    for model in SINGLE_MODELS:
        group = subset.loc[subset["model"] == model].copy()
        if not group["record_id"].is_unique:
            raise RuntimeError(f"Duplicate record_id values found for {model}.")
        group["native_probability"] = pd.to_numeric(group["native_probability"], errors="coerce")
        group["y_SCLC"] = pd.to_numeric(group["y_SCLC"], errors="coerce")
        if group[["native_probability", "y_SCLC"]].isna().any().any():
            raise RuntimeError(f"Non-numeric native probability or label found for {model}.")
        if not group["native_probability"].between(0.0, 1.0).all():
            raise RuntimeError(f"Native probability outside [0, 1] found for {model}.")
        if not set(group["y_SCLC"].unique()).issubset({0, 1}):
            raise RuntimeError(f"Binary y_SCLC labels are invalid for {model}.")

        group = group.sort_values("record_id", kind="mergesort").reset_index(drop=True)
        ids = tuple(group["record_id"].astype(str))
        y = group["y_SCLC"].to_numpy(dtype=int)
        if expected_ids is None:
            expected_ids, expected_y, expected_n = ids, y, len(group)
        elif ids != expected_ids or not np.array_equal(y, expected_y) or len(group) != expected_n:
            raise RuntimeError(f"Archived records/outcomes do not match across included models ({model}).")
        output[model] = group

    if expected_y is None or expected_n is None or expected_y.sum() in {0, expected_n}:
        raise RuntimeError("The frozen input has no valid binary-outcome comparison set.")
    return output, expected_n, int(expected_y.sum())


def _net_benefit(y_true: np.ndarray, probability: np.ndarray) -> np.ndarray:
    n = len(y_true)
    predicted_positive = probability[:, None] >= THRESHOLDS[None, :]
    true_positive = ((y_true[:, None] == 1) & predicted_positive).sum(axis=0)
    false_positive = ((y_true[:, None] == 0) & predicted_positive).sum(axis=0)
    odds = THRESHOLDS / (1.0 - THRESHOLDS)
    return true_positive / n - false_positive / n * odds


def draw_dca(axis: plt.Axes, predictions: dict[str, pd.DataFrame], n: int, events: int) -> None:
    y = predictions[SINGLE_MODELS[0]]["y_SCLC"].to_numpy(dtype=int)
    model_nb = {
        model: _net_benefit(y, predictions[model]["native_probability"].to_numpy(dtype=float))
        for model in SINGLE_MODELS
    }
    prevalence = float(events / n)
    treat_all = prevalence - (1.0 - prevalence) * THRESHOLDS / (1.0 - THRESHOLDS)
    treat_none = np.zeros_like(THRESHOLDS)


    y_min = -0.05
    y_max = max(0.05, float(np.nanmax(np.concatenate([treat_all, *model_nb.values()]))) + 0.015)

    handles: list[Line2D] = []
    for model in SINGLE_MODELS:
        axis.plot(
            THRESHOLDS,
            model_nb[model],
            color=MODEL_COLOR[model],
            linestyle=MODEL_LINESTYLE[model],
            linewidth=1.10,
            alpha=0.91,
            zorder=3,
        )
        handles.append(
            Line2D(
                [0], [0], color=MODEL_COLOR[model], linestyle=MODEL_LINESTYLE[model],
                linewidth=1.35, label=DISPLAY_NAME[model]
            )
        )

    axis.plot(THRESHOLDS, treat_all, color="#202020", linestyle="--", linewidth=1.05, zorder=1)
    axis.plot(THRESHOLDS, treat_none, color="#9A9A9A", linestyle=":", linewidth=1.05, zorder=1)
    handles.extend([
        Line2D([0], [0], color="#202020", linestyle="--", linewidth=1.15, label="Treat all"),
        Line2D([0], [0], color="#9A9A9A", linestyle=":", linewidth=1.15, label="Treat none"),
    ])
    axis.set_xlim(float(THRESHOLDS.min()), float(THRESHOLDS.max()))
    axis.set_ylim(y_min, y_max)
    axis.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    axis.tick_params(direction="out")
    axis.set_box_aspect(0.62)
    axis.set_xlabel("Threshold probability")
    axis.set_ylabel("Net benefit")
    axis.text(0.02, 0.98, DATASET_LABEL, transform=axis.transAxes, ha="left", va="top", fontsize=8.5, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 1.3})

    axis.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#C8C8C8",
        framealpha=0.94,
        fontsize=5.80,
        handlelength=1.75,
        labelspacing=0.12,
        columnspacing=0.75,
        ncol=2,
        borderpad=0.34,
        borderaxespad=0.38,
    )


def main() -> None:
    from make_v4_single_model_figures import main as write_figure3
    write_figure3()


if __name__ == "__main__":
    main()
