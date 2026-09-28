from __future__ import annotations

import matplotlib.pyplot as plt

from frozen_fusion_plotting import (
    add_measure_panel,
    configure_rcparams,
    load_archived_frames,
    save_vector_pair,
    source_paths_text,
)


DATASET = "A_holdout_clean"
MEASURES = ("ROC", "PR", "Calibration", "DCA", "Brier")


def main() -> None:
    configure_rcparams()
    frames = load_archived_frames([DATASET])
    fig = plt.figure(figsize=(19.2, 5.35), facecolor="white")
    outer = fig.add_gridspec(
        1,
        5,
        left=0.045,
        right=0.995,
        top=0.975,
        bottom=0.035,
        wspace=0.34,
    )
    for index, measure in enumerate(MEASURES):
        add_measure_panel(
            fig,
            outer[0, index],
            letter="abcde"[index],
            measure=measure,
            dataset=DATASET,
            frames=frames,
        )

    fig.text(0.012, 0.57, "A evaluation set", rotation=90, ha="center", va="center", fontsize=9.5, fontweight="bold")
    pdf_path, png_path = save_vector_pair(fig, "Fig4_fusion_A_evaluation")
    print("Created:")
    print(pdf_path)
    print(png_path)
    print("Archived inputs:")
    print(source_paths_text())


if __name__ == "__main__":
    main()
