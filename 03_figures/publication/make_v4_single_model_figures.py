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
from statsmodels.nonparametric.smoothers_lowess import lowess

from nature_style import apply_style, save_fig, save_supp_fig


HERE = Path(__file__).resolve().parent
RESULTS = publication_results()
TABLES = RESULTS / "tables"
ROC_INPUT = RESULTS / "curves" / "roc_curve_points.csv"
PR_INPUT = RESULTS / "curves" / "pr_curve_points.csv"
CALIBRATION_INPUT = RESULTS / "curves" / "calibration_curve_points.csv"
METRICS_INPUT = TABLES / "table2_model_performance_ci.csv"
PREDICTIONS_INPUT = data_root() / "fusion_results" / "native_predictions_all_eval.csv"

DATASET = "A_holdout_clean"
DATASET_LABEL = "A evaluation set"


SINGLE_MODELS = [
    "logistic_regression",
    "gam",
    "knn",
    "rbf_svm",
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
NATIVE_PROBABILITY_MODELS = [model for model in SINGLE_MODELS if model != "rbf_svm"]

DISPLAY_NAME = {
    "logistic_regression": "Logistic regression",
    "gam": "GAM",
    "knn": "KNN",
    "rbf_svm": "RBF-SVM",
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
MODEL_LINESTYLE = {
    model: ["-", "--", "-.", (0, (5, 2)), (0, (1, 1))][index % 5]
    for index, model in enumerate(SINGLE_MODELS)
}


LOWESS_FRAC = 0.75
LOWESS_IT = 0
LOWESS_LOWER_Q = 0.025
LOWESS_UPPER_Q = 0.975


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen input(s):\n" + "\n".join(missing))


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan).dropna(subset=columns)


def _validate_prediction_inventory() -> None:
    columns = ["dataset", "model", "record_id", "y_SCLC", "raw_score", "score_kind"]
    prediction = pd.read_csv(PREDICTIONS_INPUT, usecols=columns)
    subset = prediction.loc[
        (prediction["dataset"] == DATASET) & prediction["model"].isin(SINGLE_MODELS)
    ].copy()
    counts = subset.groupby("model", sort=False).size().reindex(SINGLE_MODELS)
    if counts.isna().any() or counts.nunique() != 1:
        raise RuntimeError("Archived A-evaluation single-model prediction inventory is incomplete.")
    if subset["record_id"].isna().any() or subset["y_SCLC"].isna().any() or subset["raw_score"].isna().any():
        raise RuntimeError("Archived A-evaluation prediction inventory contains missing identifier, label, or score.")
    rbf_kind = set(subset.loc[subset["model"] == "rbf_svm", "score_kind"].dropna().unique())
    if rbf_kind != {"decision_function"}:
        raise RuntimeError(f"RBF-SVM score kind is not the expected decision_function: {rbf_kind}")


def _load_metrics() -> pd.DataFrame:
    required = [
        "dataset", "model", "score_kind", "n", "positive_n", "negative_n",
        "auc", "auc_ci_low", "auc_ci_high", "auprc", "auprc_ci_low", "auprc_ci_high",
        "brier", "brier_ci_low", "brier_ci_high",
    ]
    metrics = pd.read_csv(METRICS_INPUT, usecols=required)
    metrics = metrics.loc[
        (metrics["dataset"] == DATASET) & metrics["model"].isin(SINGLE_MODELS)
    ].copy()
    metrics = metrics.set_index("model").reindex(SINGLE_MODELS).reset_index()
    if metrics["model"].isna().any() or len(metrics) != len(SINGLE_MODELS):
        raise RuntimeError("Archived A-evaluation metrics are incomplete for the 14 prespecified single models.")
    metrics = _numeric(
        metrics,
        ["n", "positive_n", "negative_n", "auc", "auc_ci_low", "auc_ci_high", "auprc", "auprc_ci_low", "auprc_ci_high"],
    )
    if len(metrics) != len(SINGLE_MODELS):
        raise RuntimeError("Archived AUC or AUPRC confidence intervals are missing.")
    return metrics.set_index("model", drop=False)


def _load_curve(path: Path, kind: str) -> pd.DataFrame:
    if kind == "roc":
        coordinate_columns = ["fpr", "tpr"]
    elif kind == "pr":
        coordinate_columns = ["recall", "precision"]
    else:
        raise ValueError(kind)
    frame = pd.read_csv(path)
    required = ["dataset", "model", "point", *coordinate_columns]
    if not set(required).issubset(frame.columns):
        raise RuntimeError(f"Archived {kind.upper()} coordinate file lacks required columns.")
    frame = frame.loc[(frame["dataset"] == DATASET) & frame["model"].isin(SINGLE_MODELS)].copy()
    frame = _numeric(frame, ["point", *coordinate_columns])
    present = set(frame["model"])
    missing = [model for model in SINGLE_MODELS if model not in present]
    if missing:
        raise RuntimeError(f"Archived {kind.upper()} coordinates missing models: {missing}")
    return frame.sort_values(["model", "point"], kind="mergesort")


def _equal_frequency_bins(y: np.ndarray, probability: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    order = np.argsort(probability, kind="mergesort")
    groups = np.array_split(order, n_bins)
    rows: list[dict[str, float | int]] = []
    for index, group in enumerate(groups, start=1):
        if len(group) == 0:
            continue
        rows.append(
            {
                "bin": index,
                "mean_predicted": float(np.mean(probability[group])),
                "observed_fraction": float(np.mean(y[group])),
                "n_bin": int(len(group)),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != n_bins:
        raise RuntimeError("Expected ten non-empty equal-frequency calibration groups.")
    return result


def _lowess_display_curve(y: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    lower, upper = np.quantile(probability, [LOWESS_LOWER_Q, LOWESS_UPPER_Q])
    if np.isclose(lower, upper):
        raise RuntimeError("Insufficient prediction range for LOWESS calibration display.")
    fit = lowess(
        endog=y.astype(float),
        exog=probability.astype(float),
        frac=LOWESS_FRAC,
        it=LOWESS_IT,
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
        pd.DataFrame({"mean_predicted": fit[:, 0], "observed_fraction": fit[:, 1]})
        .groupby("mean_predicted", as_index=False)["observed_fraction"]
        .mean()
        .sort_values("mean_predicted", kind="mergesort")
    )
    if len(unique_fit) < 2:
        raise RuntimeError("Insufficient distinct central prediction values for LOWESS display.")
    dense_x = np.linspace(
        float(unique_fit["mean_predicted"].min()),
        float(unique_fit["mean_predicted"].max()),
        300,
    )
    dense_y = np.clip(
        np.interp(
            dense_x,
            unique_fit["mean_predicted"].to_numpy(float),
            unique_fit["observed_fraction"].to_numpy(float),
        ),
        0.0,
        1.0,
    )
    return pd.DataFrame({"mean_predicted": dense_x, "observed_fraction": dense_y})


def _load_lowess_calibration_display() -> dict[str, dict[str, pd.DataFrame]]:
    required = ["dataset", "model", "record_id", "y_SCLC", "native_probability", "score_kind"]
    predictions = pd.read_csv(PREDICTIONS_INPUT, usecols=required)
    subset = predictions.loc[
        (predictions["dataset"] == DATASET) & predictions["model"].isin(NATIVE_PROBABILITY_MODELS)
    ].copy()
    output: dict[str, dict[str, pd.DataFrame]] = {}
    for model in NATIVE_PROBABILITY_MODELS:
        group = subset.loc[subset["model"] == model].copy()
        if len(group) == 0 or group["record_id"].isna().any() or not group["record_id"].is_unique:
            raise RuntimeError(f"Saved individual predictions are incomplete for {model}.")
        if set(group["score_kind"].dropna().unique()) != {"probability"}:
            raise RuntimeError(f"{model} is not a native-probability model.")
        probability = pd.to_numeric(group["native_probability"], errors="coerce").to_numpy(float)
        outcome = pd.to_numeric(group["y_SCLC"], errors="coerce").to_numpy(float)
        if not (np.isfinite(probability).all() and np.isfinite(outcome).all()):
            raise RuntimeError(f"Saved individual probability or outcome is missing for {model}.")
        if (probability < 0).any() or (probability > 1).any() or not set(np.unique(outcome)).issubset({0.0, 1.0}):
            raise RuntimeError(f"Saved probability or binary outcome is outside the expected range for {model}.")
        output[model] = {
            "smooth": _lowess_display_curve(outcome, probability),
            "bins": _equal_frequency_bins(outcome, probability),
        }
    if set(output) != set(NATIVE_PROBABILITY_MODELS):
        raise RuntimeError("LOWESS calibration display scope is incomplete.")
    return output


def _curve_label(metrics: pd.DataFrame, model: str, metric: str) -> str:
    row = metrics.loc[model]
    statistic = "AUC" if metric == "auc" else "AUPRC"
    return (
        f"{DISPLAY_NAME[model]}  {statistic} {row[metric]:.3f} "
        f"({row[f'{metric}_ci_low']:.3f}\N{EN DASH}{row[f'{metric}_ci_high']:.3f})"
    )


def _models_by_metric(metrics: pd.DataFrame, metric: str) -> list[str]:
    return sorted(
        SINGLE_MODELS,
        key=lambda model: (-float(metrics.loc[model, metric]), SINGLE_MODELS.index(model)),
    )


def _curve_axis(
    ax: plt.Axes,
    curves: pd.DataFrame,
    metrics: pd.DataFrame,
    kind: str,
) -> None:
    metric = "auc" if kind == "roc" else "auprc"
    display_order = _models_by_metric(metrics, metric)

    for model in reversed(display_order):
        subset = curves.loc[curves["model"] == model]
        if kind == "roc":
            x, y = subset["fpr"], subset["tpr"]
        else:
            x, y = subset["recall"], subset["precision"]
        ax.plot(
            x, y, color=MODEL_COLOR[model], linestyle=MODEL_LINESTYLE[model], linewidth=1.18,
            alpha=0.94, solid_capstyle="round", solid_joinstyle="round",
            zorder=3 if model == display_order[0] else 2,
        )
    handles = [
        Line2D([0], [0], color=MODEL_COLOR[model], linestyle=MODEL_LINESTYLE[model], linewidth=1.35)
        for model in display_order
    ]
    labels = [_curve_label(metrics, model, metric) for model in display_order]
    if kind == "roc":
        ax.plot([0, 1], [0, 1], color="#606060", linestyle=":", linewidth=0.9, zorder=0)
        ax.set_xlabel("1 - Specificity")
        ax.set_ylabel("Sensitivity")
    else:
        prevalence = float(metrics.iloc[0]["positive_n"] / metrics.iloc[0]["n"])
        ax.axhline(prevalence, color="#606060", linestyle=":", linewidth=0.9, zorder=0)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="out")
    ax.legend(
        handles,
        labels,
        loc="lower right" if kind == "roc" else "upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#C8C8C8",
        framealpha=0.94,
        fontsize=5.15,
        handlelength=1.72,
        labelspacing=0.08,
        borderpad=0.34,
        borderaxespad=0.38,
    )


def _draw_calibration(axis: plt.Axes, calibration: dict[str, dict[str, pd.DataFrame]]) -> None:
    handles: list[Line2D] = []
    for model in NATIVE_PROBABILITY_MODELS:
        smooth = calibration[model]["smooth"]
        bins = calibration[model]["bins"]
        axis.plot(
            smooth["mean_predicted"],
            smooth["observed_fraction"],
            color=MODEL_COLOR[model],
            linestyle=MODEL_LINESTYLE[model],
            linewidth=1.10,
            alpha=0.91,
            zorder=2,
        )
        axis.scatter(
            bins["mean_predicted"],
            bins["observed_fraction"],
            facecolors="white",
            edgecolors=MODEL_COLOR[model],
            linewidths=0.70,
            s=15,
            alpha=0.92,
            zorder=3,
        )
        handles.append(Line2D([0], [0], color=MODEL_COLOR[model], linestyle=MODEL_LINESTYLE[model], linewidth=1.35, label=DISPLAY_NAME[model]))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed SCLC fraction")
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(direction="out")
    axis.grid(color="#E9E9E9", linewidth=0.55, alpha=0.85)
    axis.text(0.02, 0.98, DATASET_LABEL, transform=axis.transAxes, ha="left", va="top", fontsize=8.5, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 1.3})
    axis.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#C8C8C8",
        framealpha=0.94,
        fontsize=5.25,
        handlelength=1.72,
        labelspacing=0.10,
        columnspacing=0.72,
        ncol=2,
        borderpad=0.34,
        borderaxespad=0.38,
    )


def _draw_brier(axis: plt.Axes, metrics: pd.DataFrame) -> None:
    brier = metrics.loc[NATIVE_PROBABILITY_MODELS].copy()
    brier = _numeric(brier, ["brier", "brier_ci_low", "brier_ci_high"])
    if len(brier) != len(NATIVE_PROBABILITY_MODELS):
        raise RuntimeError("Archived Brier confidence intervals are incomplete for native-probability single models.")
    y = np.arange(len(NATIVE_PROBABILITY_MODELS))
    for yi, model in enumerate(NATIVE_PROBABILITY_MODELS):
        row = brier.loc[model]
        value, low, high = (float(row[column]) for column in ("brier", "brier_ci_low", "brier_ci_high"))
        axis.errorbar(
            value,
            yi,
            xerr=np.array([[value - low], [high - value]]),
            fmt="o",
            color=MODEL_COLOR[model],
            ecolor=MODEL_COLOR[model],
            markersize=4.8,
            markeredgecolor="white",
            markeredgewidth=0.4,
            elinewidth=1.05,
            capsize=2.0,
            zorder=3,
        )
        axis.text(high + 0.0035, yi, f"{value:.3f}", va="center", ha="left", fontsize=7.0, color=MODEL_COLOR[model])
    axis.set_yticks(y, [DISPLAY_NAME[model] for model in NATIVE_PROBABILITY_MODELS])
    axis.invert_yaxis()
    axis.set_xlim(0.10, 0.37)
    axis.set_xlabel("Brier score (95% CI)")
    axis.grid(axis="x", color="#E9E9E9", linewidth=0.55, alpha=0.85)
    axis.tick_params(direction="out")
    axis.text(0.02, 0.98, DATASET_LABEL, transform=axis.transAxes, ha="left", va="top", fontsize=8.5, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 1.3})


def make_main_figure(metrics: pd.DataFrame, roc: pd.DataFrame, pr: pd.DataFrame, calibration: dict[str, dict[str, pd.DataFrame]]) -> tuple[Path, Path]:
    from make_figs11_single_model_dca import _load_and_validate_predictions, draw_dca

    predictions, n_cases, events = _load_and_validate_predictions()
    fig = plt.figure(figsize=(16.6, 10.8), facecolor="white")
    grid = fig.add_gridspec(2, 6, left=0.045, right=0.99, top=0.96, bottom=0.07, wspace=0.85, hspace=0.38)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 0:3]),
        fig.add_subplot(grid[1, 3:6]),
    ]
    _curve_axis(axes[0], roc, metrics, "roc")
    _curve_axis(axes[1], pr, metrics, "pr")
    _draw_calibration(axes[2], calibration)
    draw_dca(axes[3], predictions, n_cases, events)
    _draw_brier(axes[4], metrics)
    for axis, letter in zip(axes, "abcde"):
        axis.text(-0.12, 1.06, letter, transform=axis.transAxes, fontsize=12, fontweight="bold", ha="left", va="bottom")
    output = save_fig(fig, "Fig3_single_model_A_evaluation")
    plt.close(fig)
    return output


def main() -> None:
    _require_files([ROC_INPUT, PR_INPUT, METRICS_INPUT, PREDICTIONS_INPUT])
    _validate_prediction_inventory()
    metrics = _load_metrics()
    roc = _load_curve(ROC_INPUT, "roc")
    pr = _load_curve(PR_INPUT, "pr")
    calibration = _load_lowess_calibration_display()
    apply_style()
    outputs = [make_main_figure(metrics, roc, pr, calibration)]
    for pdf, png in outputs:
        print(f"WROTE {pdf}")
        print(f"WROTE {png}")


if __name__ == "__main__":
    main()
