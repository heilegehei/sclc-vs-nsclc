from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from nature_style import _figure_dir, apply_style, panel_label, save_fig


HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parent
OUT_PDF = _figure_dir("main") / "unnumbered_clinical_pathway.pdf"
OUT_PNG = _figure_dir("main") / "unnumbered_clinical_pathway.png"


def _rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.014",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.25,
        mutation_aspect=1,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=9.2,
        linespacing=1.24,
        zorder=3,
    )


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.1,
            color="#666666",
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )


def make_figure() -> tuple[Path, Path]:
    apply_style()
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.105, top=0.94)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "a", x=-0.010, y=0.975)


    _rounded_box(
        ax,
        0.035,
        0.355,
        0.180,
        0.295,
        "Laboratory input\n13 selected direct predictors\n+ NEUT#, MONO#, LYMPH#, GLU, HGB\n\nI / M / N groups",
        "#F7F7F7",
        "#666666",
    )
    _rounded_box(
        ax,
        0.280,
        0.355,
        0.185,
        0.295,
        "Frozen preprocessing\nDevelopment-derived medians\nPrespecified clipping\nPrespecified log1p",
        "#FFF2E8",
        "#D55E00",
    )
    _rounded_box(
        ax,
        0.530,
        0.355,
        0.205,
        0.295,
        "Frozen full-layer model\n13 fitted member models\nAUC-derived fixed weights\n\nWeighted voting",
        "#EAF0F8",
        "#2F5597",
    )
    _rounded_box(
        ax,
        0.800,
        0.355,
        0.165,
        0.295,
        "Patient-level output\nSCLC probability\nFixed classification threshold\nPredicted class",
        "#E7F5EF",
        "#009E73",
    )

    _arrow(ax, (0.215, 0.503), (0.280, 0.503))
    _arrow(ax, (0.465, 0.503), (0.530, 0.503))
    _arrow(ax, (0.735, 0.503), (0.800, 0.503))
    ax.text(
        0.50,
        0.165,
        "The preprocessing transform, member models, voting weights, and classification threshold "
        "were fixed before evaluation or external application.",
        ha="center",
        va="center",
        fontsize=8.7,
    )

    outputs = save_fig(fig, "unnumbered_clinical_pathway")
    plt.close(fig)
    return outputs


if __name__ == "__main__":
    pdf_path, png_path = make_figure()
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
