from __future__ import annotations

import os
from pathlib import Path


def code_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    value = os.environ.get("SCLC_DATA_ROOT", "").strip()
    if not value:
        raise FileNotFoundError(
            "Individual-level data are not part of this repository. "
            "Set SCLC_DATA_ROOT to a directory you are authorised to analyse."
        )
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("SCLC_DATA_ROOT is not a directory.")
    return root


def cohort_workbook() -> Path:
    return data_root() / "cohort_workbook.xlsx"


def fusion_results() -> Path:
    return data_root() / "fusion_results"


def explainability_results() -> Path:
    return data_root() / "explainability_results"


def publication_results() -> Path:
    return data_root() / "publication_results"
