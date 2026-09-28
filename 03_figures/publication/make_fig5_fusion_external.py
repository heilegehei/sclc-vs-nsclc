from __future__ import annotations

import matplotlib.pyplot as plt

from frozen_fusion_plotting import (
    add_measure_panel,
    configure_rcparams,
    load_archived_frames,
    save_vector_pair,
    source_paths_text,
)


DATASETS = (
    ("B_external", "B external"),
    ("C_external", "C external"),
)
MEASURES = ("ROC", "PR", "Calibration", "DCA", "Brier")


def main() -> None:
    configure_rcparams()
    frames = load_archived_frames([dataset for dataset, _ in DATASETS])
    fig = plt.figure(figsize=(19.2, 10.65), facecolor="white")
    outer = fig.add_gridspec(
        2,
        5,
        left=0.055,
        right=0.995,
        top=0.982,
        bottom=0.028,
        wspace=0.34,
        hspace=0.18,
    )
    letters = "abcdefghij"
    for row_index, (dataset, label) in enumerate(DATASETS):
        for column_index, measure in enumerate(MEASURES):
            add_measure_panel(
                fig,
                outer[row_index, column_index],
                letter=letters[row_index * len(MEASURES) + column_index],
                measure=measure,
                dataset=dataset,
                frames=frames,
            )

        fig.text(
            0.015,
            0.745 if row_index == 0 else 0.260,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9.6,
            fontweight="bold",
        )
    pdf_path, png_path = save_vector_pair(fig, "Fig5_fusion_BC_external")
    print("Created:")
    print(pdf_path)
    print(png_path)
    print("Archived inputs:")
    print(source_paths_text())


if __name__ == "__main__":
    main()
