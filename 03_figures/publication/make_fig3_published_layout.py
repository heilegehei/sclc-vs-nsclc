from __future__ import annotations
import sys as _public_sys
from pathlib import Path as _PublicPath
_public_root = next(p for p in _PublicPath(__file__).resolve().parents if (p / "common" / "public_paths.py").is_file())
if str(_public_root) not in _public_sys.path:
    _public_sys.path.insert(0, str(_public_root))
from common.public_paths import code_root, data_root, explainability_results, fusion_results, publication_results

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from nature_style import apply_style, save_fig
from make_v4_single_model_figures import (
    DATASET,
    DATASET_LABEL,
    DISPLAY_NAME,
    METRICS_INPUT,
    MODEL_COLOR,
    MODEL_LINESTYLE,
    NATIVE_PROBABILITY_MODELS,
    PR_INPUT,
    ROC_INPUT,
    SINGLE_MODELS,
    _curve_axis,
    _draw_brier,
    _load_curve,
    _load_metrics,
    _require_files,
    _validate_prediction_inventory,
)


PREDICTIONS_INPUT = fusion_results() / "native_predictions_all_eval.csv"
FIGURE_DIR = publication_results() / "figures" / "main"

LOWESS_FRAC = 0.75
LOWESS_IT = 0
LOWESS_LOWER_Q = 0.025
LOWESS_UPPER_Q = 0.975
THRESHOLDS = np.linspace(0.01, 0.80, 160)

LEGEND_SCALE = 2.0
ROC_PR_LEGEND_SIZE = 5.15 * LEGEND_SCALE
CALIBRATION_LEGEND_SIZE = 5.35 * LEGEND_SCALE
DCA_LEGEND_SIZE = 5.80 * LEGEND_SCALE
BRIER_TICK_SIZE = 17.0
BRIER_VALUE_SIZE = 14.0

F02_NAME = "F02_individual_model_A_evaluation_v2"
F03_NAME = "F03_individual_model_calibration_v2"
F04_NAME = "F04_individual_model_brier_score_v2"
F05_NAME = "F05_individual_model_decision_curve_v2"
COMBINED_NAME = "Fig3_single_model_A_evaluation_published_layout"

PANEL_ORDER = [F02_NAME, F03_NAME, F05_NAME, F04_NAME]
DPI = 600
GAP_PX = 150
SIDE_PX = 60
TOP_PX = 200
BOTTOM_PX = 40
LETTER_PX = 120
LETTER_BASELINE_PX = 149
LETTER_DX_PX = 10
PANEL_B_FRACTION = 4451 / 7956


def _load_probabilities() -> tuple[dict[str, np.ndarray], np.ndarray]:
    required = ["dataset", "model", "record_id", "y_SCLC", "native_probability", "score_kind"]
    frame = pd.read_csv(PREDICTIONS_INPUT, usecols=required)
    frame = frame.loc[(frame["dataset"] == DATASET) & frame["model"].isin(SINGLE_MODELS)].copy()
    rbf_kind = set(frame.loc[frame["model"] == "rbf_svm", "score_kind"].dropna().astype(str))
    if rbf_kind != {"decision_function"}:
        raise RuntimeError(f"Unexpected RBF-SVM score kind: {rbf_kind}")
    probabilities: dict[str, np.ndarray] = {}
    expected_ids: tuple[str, ...] | None = None
    expected_y: np.ndarray | None = None
    for model in NATIVE_PROBABILITY_MODELS:
        part = frame.loc[frame["model"] == model].copy()
        if len(part) == 0 or part["record_id"].isna().any() or not part["record_id"].is_unique:
            raise RuntimeError(f"Saved individual predictions are incomplete for {model}.")
        if not part["score_kind"].eq("probability").all():
            raise RuntimeError(f"{model} is not a native-probability model.")
        part = part.sort_values("record_id", kind="mergesort")
        probability = pd.to_numeric(part["native_probability"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(part["y_SCLC"], errors="coerce").to_numpy(float)
        if not (np.isfinite(probability).all() and np.isfinite(y).all()):
            raise RuntimeError(f"Missing probability or outcome for {model}.")
        if (probability < 0).any() or (probability > 1).any() or not set(np.unique(y)).issubset({0.0, 1.0}):
            raise RuntimeError(f"Probability or outcome outside the expected range for {model}.")
        ids = tuple(part["record_id"].astype(str))
        if expected_ids is None:
            expected_ids, expected_y = ids, y.astype(int)
        elif ids != expected_ids or not np.array_equal(expected_y, y.astype(int)):
            raise RuntimeError(f"Records or outcomes do not match across models ({model}).")
        probabilities[model] = probability
    if expected_y is None or expected_y.sum() in {0, len(expected_y)}:
        raise RuntimeError("No valid binary-outcome evaluation set.")
    return probabilities, expected_y


def _equal_frequency_bins(y: np.ndarray, probability: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    order = np.argsort(probability, kind="mergesort")
    rows = [
        {
            "mean_predicted": float(np.mean(probability[group])),
            "observed_fraction": float(np.mean(y[group])),
        }
        for group in np.array_split(order, n_bins)
        if len(group)
    ]
    return pd.DataFrame(rows)


def _lowess_display_curve(y: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    lower, upper = np.quantile(probability, [LOWESS_LOWER_Q, LOWESS_UPPER_Q])
    if np.isclose(lower, upper):
        raise RuntimeError("Insufficient probability range for LOWESS display.")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
        fit = lowess(
            endog=y.astype(float), exog=probability.astype(float), frac=LOWESS_FRAC,
            it=LOWESS_IT, delta=0.0, is_sorted=False, return_sorted=True,
        )
    fit = fit[np.isfinite(fit).all(axis=1) & (fit[:, 0] >= lower) & (fit[:, 0] <= upper)]
    unique_fit = (
        pd.DataFrame({"mean_predicted": fit[:, 0], "observed_fraction": fit[:, 1]})
        .groupby("mean_predicted", as_index=False)["observed_fraction"].mean()
        .sort_values("mean_predicted", kind="mergesort")
    )
    if len(unique_fit) < 2:
        raise RuntimeError("Insufficient distinct predictions for LOWESS display.")
    dense_x = np.linspace(float(unique_fit["mean_predicted"].min()), float(unique_fit["mean_predicted"].max()), 300)
    dense_y = np.clip(
        np.interp(dense_x, unique_fit["mean_predicted"].to_numpy(float), unique_fit["observed_fraction"].to_numpy(float)),
        0.0, 1.0,
    )
    return pd.DataFrame({"mean_predicted": dense_x, "observed_fraction": dense_y})


def _net_benefit(y: np.ndarray, probability: np.ndarray) -> np.ndarray:
    predicted_positive = probability[:, None] >= THRESHOLDS[None, :]
    true_positive = ((y[:, None] == 1) & predicted_positive).sum(axis=0)
    false_positive = ((y[:, None] == 0) & predicted_positive).sum(axis=0)
    odds = THRESHOLDS / (1.0 - THRESHOLDS)
    return true_positive / len(y) - false_positive / len(y) * odds


def make_f02(metrics: pd.DataFrame, roc: pd.DataFrame, pr: pd.DataFrame) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(13.4, 6.45), facecolor="white")
    outer = fig.add_gridspec(1, 2, left=0.075, right=0.995, top=0.950, bottom=0.115, wspace=0.28)
    roc_ax = fig.add_subplot(outer[0, 0])
    pr_ax = fig.add_subplot(outer[0, 1])
    _curve_axis(roc_ax, roc, metrics, "roc")
    _curve_axis(pr_ax, pr, metrics, "pr")
    for axis in (roc_ax, pr_ax):
        for text in axis.get_legend().get_texts():
            text.set_fontsize(ROC_PR_LEGEND_SIZE)
    fig.text(0.018, 0.53, DATASET_LABEL, rotation=90, ha="center", va="center", fontsize=8.5, fontweight="bold")
    output = save_fig(fig, F02_NAME)
    plt.close(fig)
    return output


def make_f03(probabilities: dict[str, np.ndarray], y: np.ndarray) -> tuple[Path, Path]:
    fig, axis = plt.subplots(figsize=(8.65, 7.15), facecolor="white")
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.125, top=0.950)
    handles: list[Line2D] = []
    for model in NATIVE_PROBABILITY_MODELS:
        smooth = _lowess_display_curve(y, probabilities[model])
        bins = _equal_frequency_bins(y, probabilities[model])
        axis.plot(
            smooth["mean_predicted"], smooth["observed_fraction"], color=MODEL_COLOR[model],
            linestyle=MODEL_LINESTYLE[model], linewidth=1.18, alpha=0.92, zorder=2,
            solid_capstyle="round", solid_joinstyle="round",
        )
        axis.scatter(
            bins["mean_predicted"], bins["observed_fraction"], facecolors="white",
            edgecolors=MODEL_COLOR[model], linewidths=0.72, s=19, alpha=0.94, zorder=3,
        )
        handles.append(Line2D([0], [0], color=MODEL_COLOR[model], linestyle=MODEL_LINESTYLE[model], linewidth=1.35, label=DISPLAY_NAME[model]))
    ideal = Line2D([0], [0], color="#555555", linestyle=":", linewidth=1.0, label="Ideal")
    axis.plot([0, 1], [0, 1], color="#555555", linestyle=":", linewidth=1.0, zorder=1)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed SCLC fraction")
    axis.tick_params(direction="out")
    axis.grid(color="#E5E5E5", linewidth=0.55, alpha=0.82)
    axis.legend(
        handles=handles + [ideal], loc="upper left", ncol=2, frameon=True, facecolor="white",
        edgecolor="#C8C8C8", framealpha=0.94, fontsize=CALIBRATION_LEGEND_SIZE, handlelength=1.70,
        labelspacing=0.10, columnspacing=0.72, borderpad=0.34, borderaxespad=0.38,
    )
    output = save_fig(fig, F03_NAME)
    plt.close(fig)
    return output


def make_f05(probabilities: dict[str, np.ndarray], y: np.ndarray) -> tuple[Path, Path]:
    fig, axis = plt.subplots(figsize=(10.6, 6.60), facecolor="white")
    fig.subplots_adjust(left=0.125, right=0.975, bottom=0.120, top=0.950)
    handles: list[Line2D] = []
    net_benefits: list[np.ndarray] = []
    for model in NATIVE_PROBABILITY_MODELS:
        values = _net_benefit(y, probabilities[model])
        net_benefits.append(values)
        axis.plot(THRESHOLDS, values, color=MODEL_COLOR[model], linestyle=MODEL_LINESTYLE[model], linewidth=1.12, alpha=0.92, zorder=3, solid_capstyle="round", solid_joinstyle="round")
        handles.append(Line2D([0], [0], color=MODEL_COLOR[model], linestyle=MODEL_LINESTYLE[model], linewidth=1.35, label=DISPLAY_NAME[model]))
    prevalence = float(y.mean())
    treat_all = prevalence - (1.0 - prevalence) * THRESHOLDS / (1.0 - THRESHOLDS)
    axis.plot(THRESHOLDS, treat_all, color="#202020", linestyle="--", linewidth=1.05, zorder=1)
    axis.axhline(0.0, color="#969696", linestyle=":", linewidth=1.05, zorder=1)
    handles.extend([
        Line2D([0], [0], color="#202020", linestyle="--", linewidth=1.15, label="Treat all"),
        Line2D([0], [0], color="#969696", linestyle=":", linewidth=1.15, label="Treat none"),
    ])
    axis.set_xlim(float(THRESHOLDS.min()), float(THRESHOLDS.max()))
    axis.set_ylim(-0.05, max(0.05, float(np.nanmax(np.concatenate(net_benefits))) + 0.015))
    axis.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    axis.set_xlabel("Threshold probability")
    axis.set_ylabel("Net benefit")
    axis.tick_params(direction="out")
    axis.grid(color="#E5E5E5", linewidth=0.55, alpha=0.82)
    axis.legend(
        handles=handles, loc="upper right", ncol=2, frameon=True, facecolor="white",
        edgecolor="#C8C8C8", framealpha=0.94, fontsize=DCA_LEGEND_SIZE, handlelength=1.75,
        labelspacing=0.12, columnspacing=0.75, borderpad=0.34, borderaxespad=0.38,
    )
    output = save_fig(fig, F05_NAME)
    plt.close(fig)
    return output


def make_f04(metrics: pd.DataFrame) -> tuple[Path, Path]:
    fig, axis = plt.subplots(figsize=(8.8, 6.85), facecolor="white")
    fig.subplots_adjust(left=0.300, right=0.980, top=0.955, bottom=0.135)
    _draw_brier(axis, metrics)
    axis.tick_params(axis="y", labelsize=BRIER_TICK_SIZE)
    for text in axis.texts:
        if text.get_text() != DATASET_LABEL:
            text.set_fontsize(BRIER_VALUE_SIZE)
    output = save_fig(fig, F04_NAME)
    plt.close(fig)
    return output


def _arial_bold() -> str:
    return font_manager.findfont(font_manager.FontProperties(family="Arial", weight="bold"), fallback_to_default=False)


def _letter_offsets(widths: list[float]) -> list[float]:
    offsets: list[float] = []
    x = float(SIDE_PX)
    for index, width in enumerate(widths):
        offsets.append(x)
        if index == 0:
            offsets.append(x + width * PANEL_B_FRACTION)
        x += width + GAP_PX
    return offsets


def combine_png(sources: list[Path]) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    Image.MAX_IMAGE_PIXELS = None
    images = [Image.open(path).convert("RGB") for path in sources]
    height = images[0].height
    scaled = [
        image if image.height == height
        else image.resize((round(image.width * height / image.height), height), Image.LANCZOS)
        for image in images
    ]
    width = 2 * SIDE_PX + sum(image.width for image in scaled) + GAP_PX * (len(scaled) - 1)
    canvas = Image.new("RGB", (width, TOP_PX + height + BOTTOM_PX), "white")
    x = SIDE_PX
    for image in scaled:
        canvas.paste(image, (x, TOP_PX))
        x += image.width + GAP_PX
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(_arial_bold(), LETTER_PX)
    for letter, offset in zip("abcde", _letter_offsets([image.width for image in scaled])):
        draw.text((round(offset) + LETTER_DX_PX, LETTER_BASELINE_PX), letter, fill="black", font=font, anchor="ls")
    output = FIGURE_DIR / f"{COMBINED_NAME}.png"
    canvas.save(output, dpi=(DPI, DPI))
    return output


def combine_pdf(sources: list[Path]) -> Path:
    import fitz

    px = 72.0 / DPI
    documents = [fitz.open(path) for path in sources]
    rects = [document[0].rect for document in documents]
    height = rects[0].height
    widths = [rect.width * height / rect.height for rect in rects]
    width = 2 * SIDE_PX * px + sum(widths) + GAP_PX * px * (len(widths) - 1)
    output_document = fitz.open()
    page = output_document.new_page(width=width, height=TOP_PX * px + height + BOTTOM_PX * px)
    x = SIDE_PX * px
    for document, panel_width in zip(documents, widths):
        page.show_pdf_page(fitz.Rect(x, TOP_PX * px, x + panel_width, TOP_PX * px + height), document, 0)
        x += panel_width + GAP_PX * px
    offsets = [SIDE_PX * px]
    offsets.append(offsets[0] + widths[0] * PANEL_B_FRACTION)
    x = SIDE_PX * px + widths[0] + GAP_PX * px
    for panel_width in widths[1:]:
        offsets.append(x)
        x += panel_width + GAP_PX * px
    for letter, offset in zip("abcde", offsets):
        page.insert_text((offset + LETTER_DX_PX * px, LETTER_BASELINE_PX * px), letter, fontname="hebo", fontsize=LETTER_PX * px, color=(0, 0, 0))
    output = FIGURE_DIR / f"{COMBINED_NAME}.pdf"
    output_document.save(output, garbage=4, deflate=True)
    for document in documents:
        document.close()
    return output


def main() -> None:
    _require_files([ROC_INPUT, PR_INPUT, METRICS_INPUT, PREDICTIONS_INPUT])
    _validate_prediction_inventory()
    metrics = _load_metrics()
    roc = _load_curve(ROC_INPUT, "roc")
    pr = _load_curve(PR_INPUT, "pr")
    probabilities, y = _load_probabilities()
    apply_style()
    outputs = {
        F02_NAME: make_f02(metrics, roc, pr),
        F03_NAME: make_f03(probabilities, y),
        F04_NAME: make_f04(metrics),
        F05_NAME: make_f05(probabilities, y),
    }
    for pdf, png in outputs.values():
        print(f"WROTE {pdf}")
        print(f"WROTE {png}")
    print(f"WROTE {combine_png([outputs[name][1] for name in PANEL_ORDER])}")
    print(f"WROTE {combine_pdf([outputs[name][0] for name in PANEL_ORDER])}")


if __name__ == "__main__":
    main()
