from pathlib import Path
import sys


def activate_archive_sources(anchor):
    root = next((parent for parent in Path(anchor).resolve().parents
                 if (parent / "common" / "manuscript_source_paths.py").is_file()), None)
    if root is None:
        raise RuntimeError("Keep the revised code archive together when running analyses")
    relative_paths = (
        "01_methods/preprocessing/src",
        "01_methods/models/src",
        "01_methods/training",
        "02_analysis/fusion",
    )
    paths = [str(root / relative) for relative in relative_paths]
    missing = [path for path in paths if not Path(path).is_dir()]
    if missing:
        raise FileNotFoundError(f"Required local source directories missing: {missing}")
    for path in paths:
        if path in sys.path:
            sys.path.remove(path)
    sys.path[:0] = paths
    return root
