from __future__ import annotations
import sys as _public_sys
from pathlib import Path as _PublicPath
_public_root = next(p for p in _PublicPath(__file__).resolve().parents if (p / "common" / "public_paths.py").is_file())
if str(_public_root) not in _public_sys.path:
    _public_sys.path.insert(0, str(_public_root))
from common.public_paths import cohort_workbook, code_root, data_root, explainability_results, fusion_results, publication_results

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from nature_style import DATASET_COLORS, apply_style, panel_label, save_supp_fig


HERE = Path(__file__).resolve().parent
WEIGHTED_V1 = data_root() / "fusion_results"
FUSION_A_CV = WEIGHTED_V1 / "A_fusions_CV_holdout" / "fusion_predictions_A_CV.csv"
FUSION_A_EVALUATION = WEIGHTED_V1 / "A_fusions_CV_holdout" / "fusion_predictions_A_holdout_clean.csv"
FUSION_BC = WEIGHTED_V1 / "three_fusions_BC" / "fusion_predictions_B_C.csv"
FIXED_THRESHOLD = WEIGHTED_V1 / "tables" / "primary_threshold_metrics.csv"

WEIGHTED_MODEL = "weighted_voting_cv_auc"
A_DEVELOPMENT = "A_dev_cv_clean"
A_EVALUATION = "A_holdout_clean"
B_EXTERNAL = "B_external"
C_EXTERNAL = "C_external"


DISPLAY_LABELS = {
    A_DEVELOPMENT: "A development (10-fold CV)",
    A_EVALUATION: "A evaluation set",
    B_EXTERNAL: "B external validation",
    C_EXTERNAL: "C external validation",
    "A_development": "A development (5-fold calibrated)",
    "A_holdout_clean": "A evaluation set",
    "B_external": "B external validation",
    "C_external": "C external validation",
}


def _require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen source file is missing: {path}")


def _finite(frame: pd.DataFrame, columns: list[str], source_name: str) -> pd.DataFrame:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")
    out = frame.loc[:, columns].copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out.isna().any().any():
        raise ValueError(f"{source_name} contains missing/non-numeric frozen values")
    return out


def _weighted_predictions() -> dict[str, pd.DataFrame]:
    records: dict[str, pd.DataFrame] = {}

    a_cv = pd.read_csv(FUSION_A_CV)
    part = a_cv.loc[
        (a_cv["dataset"] == A_DEVELOPMENT) & (a_cv["fusion"] == WEIGHTED_MODEL),
        ["y_SCLC", "score"],
    ].rename(columns={"score": "probability"})
    records[A_DEVELOPMENT] = _finite(part, ["y_SCLC", "probability"], "A development saved scores")

    a_eval = pd.read_csv(FUSION_A_EVALUATION)
    part = a_eval.loc[
        (a_eval["dataset"] == A_EVALUATION) & (a_eval["fusion"] == WEIGHTED_MODEL),
        ["y_SCLC", "score"],
    ].rename(columns={"score": "probability"})
    records[A_EVALUATION] = _finite(part, ["y_SCLC", "probability"], "A evaluation saved scores")

    bc = pd.read_csv(FUSION_BC)
    for dataset in (B_EXTERNAL, C_EXTERNAL):
        part = bc.loc[
            (bc["dataset"] == dataset) & (bc["model"] == WEIGHTED_MODEL),
            ["y_SCLC", "native_probability"],
        ].rename(columns={"native_probability": "probability"})
        records[dataset] = _finite(part, ["y_SCLC", "probability"], f"{dataset} saved scores")

    return records


def make_s9() -> tuple[Path, Path]:
    predictions = _weighted_predictions()
    threshold_table = pd.read_csv(FIXED_THRESHOLD)
    if "threshold" not in threshold_table.columns or threshold_table.empty:
        raise ValueError("Fixed threshold source contains no threshold value")
    threshold = float(threshold_table["threshold"].iloc[0])

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.1))
    cmap = sns.light_palette("#2F5597", as_cmap=True)
    matrices: dict[str, np.ndarray] = {}
    for index, (ax, dataset) in enumerate(
        zip(axes.ravel(), (A_DEVELOPMENT, A_EVALUATION, B_EXTERNAL, C_EXTERNAL), strict=True)
    ):
        scores = predictions[dataset]
        matrix = confusion_matrix(
            scores["y_SCLC"].astype(int),
            (scores["probability"] >= threshold).astype(int),
            labels=[0, 1],
        )
        matrices[dataset] = matrix
        norm = mpl_colors.Normalize(vmin=0, vmax=max(1, int(matrix.max())))
        for row in range(2):
            for column in range(2):
                value = int(matrix[row, column])
                rgba = cmap(norm(value))

                ax.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        facecolor=rgba,
                        edgecolor="white",
                        linewidth=0.8,
                    )
                )
                text_color = "white" if np.mean(rgba[:3]) < 0.55 else "black"
                ax.text(
                    column,
                    row,
                    f"{value}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    color=text_color,
                    fontweight="bold",
                )
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(1.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([0, 1], ["NSCLC", "SCLC"])
        ax.set_yticks([0, 1], ["NSCLC", "SCLC"])
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Observed class")
        ax.text(0.5, 1.03, DISPLAY_LABELS[dataset], transform=ax.transAxes, ha="center", va="bottom", fontsize=9.0)
        panel_label(ax, chr(ord("a") + index), x=-0.20, y=1.02)

    fig.text(
        0.5,
        0.005,
        f"Weighted-voting threshold = {threshold:.4f}, the frozen manuscript threshold 0.2076.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.09, wspace=0.34, hspace=0.40)
    outputs = save_supp_fig(fig, "unnumbered_weighted_fixed_threshold_confusion")
    plt.close(fig)
    print("S9 frozen confusion matrices:", {key: matrix.tolist() for key, matrix in matrices.items()})
    return outputs


def main() -> None:
    for source in (FUSION_A_CV, FUSION_A_EVALUATION, FUSION_BC, FIXED_THRESHOLD):
        _require(source)
    apply_style()
    outputs = [*make_s9()]
    print("Refreshed outputs:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
