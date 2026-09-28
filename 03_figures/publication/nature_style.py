from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager


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
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "figure.titlesize": 0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


CATEGORICAL_12 = [
    "#2F5597",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#7B61A8",
    "#E69F00",
    "#0072B2",
    "#6B6B6B",
    "#B2182B",
    "#4D9221",
    "#56B4E9",
    "#999999",
]

MODEL_ORDER = [
    "weighted_voting_cv_auc",
    "soft_voting_native_probability",
    "stacking_in_sample_A_train",
    "extra_trees",
    "rotation_forest",
    "random_forest",
]

MODEL_LABELS = {
    "weighted_voting_cv_auc": "Weighted voting (Primary)",
    "soft_voting_native_probability": "Soft voting",
    "stacking_in_sample_A_train": "Stacking",
    "extra_trees": "Extra Trees",
    "rotation_forest": "Rotation Forest",
    "random_forest": "Random Forest",
}

MODEL_COLORS = dict(zip(MODEL_ORDER, CATEGORICAL_12[: len(MODEL_ORDER)]))
MODEL_LINESTYLES = {
    "weighted_voting_cv_auc": "-",
    "soft_voting_native_probability": "--",
    "stacking_in_sample_A_train": "-.",
    "extra_trees": (0, (5, 2)),
    "rotation_forest": (0, (3, 1, 1, 1)),
    "random_forest": (0, (1, 1)),
}

DATASET_ORDER = [
    "A_dev_cv_clean",
    "A_holdout_clean",
    "B_external",
    "C_external",
]

DATASET_LABELS = {
    "A_dev_cv_clean": "A development (10-fold CV; selection evidence)",
    "A_holdout_clean": "A evaluation set",
    "B_external": "B external",
    "C_external": "C external",
}

DATASET_COLORS = {
    "A_dev_cv_clean": "#2F5597",
    "A_holdout_clean": "#D55E00",
    "B_external": "#009E73",
    "C_external": "#7B61A8",
}

LAYER_ORDER = ["I", "I_M", "I_M_N"]
LAYER_LABELS = {
    "I": "I",
    "I_M": "I + M",
    "I_M_N": "I + M + N",
}

FEATURE_GROUP_COLORS = {
    "I": "#D55E00",
    "M": "#0072B2",
    "N": "#009E73",
}


def cell_text_color(rgba) -> str:
    r, g, b = rgba[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.55 else "black"


def panel_label(ax, letter: str, x: float = -0.10, y: float = 1.04, fontsize: int = 12) -> None:
    text_method = ax.text2D if hasattr(ax, "text2D") else ax.text
    text_method(
        x,
        y,
        str(letter).lower(),
        transform=ax.transAxes,
        fontsize=fontsize,
        fontstyle="italic",
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def short_combo_label(value: str) -> str:
    return MODEL_LABELS.get(value, DATASET_LABELS.get(value, LAYER_LABELS.get(value, value)))


def _figure_dir(kind: str) -> Path:
    import sys
    root = next(p for p in Path(__file__).resolve().parents if (p / "common" / "public_paths.py").is_file())
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from common.public_paths import publication_results
    directory = publication_results() / "figures" / kind
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_fig(fig, basename: str):
    pdf = _figure_dir("main") / f"{basename}.pdf"
    png = _figure_dir("main") / f"{basename}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    return pdf, png


def save_supp_fig(fig, basename: str):
    pdf = _figure_dir("supplementary") / f"{basename}.pdf"
    png = _figure_dir("supplementary") / f"{basename}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    return pdf, png
