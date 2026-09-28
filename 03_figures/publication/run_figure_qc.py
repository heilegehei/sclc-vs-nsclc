from __future__ import annotations


import csv
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


_CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "public_paths.py").is_file())
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
from common.public_paths import publication_results

ROOT = publication_results()
FIG_ROOT = ROOT / "figures"
QC_ROOT = ROOT / "qc"
RENDER_ROOT = QC_ROOT / "rendered_pdf"
QC_ROOT.mkdir(parents=True, exist_ok=True)
RENDER_ROOT.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tool(name: str) -> str:
    value = shutil.which(name)
    if value is None:
        raise RuntimeError(f"Required PDF utility is unavailable: {name}")
    return value


def run_text(args: list[str]) -> str:
    completed = subprocess.run(args, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return completed.stdout


def font_audit(path: Path) -> tuple[bool, str, int]:
    output = run_text([tool("pdffonts"), str(path)])
    rows = [line.strip() for line in output.splitlines()[2:] if line.strip()]
    names = [line.split()[0] for line in rows]
    allowed = all("Times" in name or "NimbusRoman" in name for name in names)
    embedded = all("yes" in line.lower() for line in rows)
    return bool(rows and allowed and embedded), " | ".join(names), len(rows)


def raster_audit(path: Path) -> tuple[bool, int]:
    output = run_text([tool("pdfimages"), "-list", str(path)])
    data_rows = [line for line in output.splitlines() if re.match(r"\s*\d+\s+\d+\s+", line)]
    return len(data_rows) == 0, len(data_rows)


def text_audit(path: Path) -> tuple[bool, int]:
    completed = subprocess.run([tool("pdftotext"), str(path), "-"], check=True, capture_output=True)
    text = completed.stdout.decode("utf-8", errors="replace").strip()
    return len(text) >= 8, len(text)


def render_pdf(path: Path) -> tuple[Path, bool, float, float]:
    rel = path.relative_to(FIG_ROOT).with_suffix("")
    flat = "__".join(rel.parts)
    prefix = RENDER_ROOT / flat
    subprocess.run(
        [tool("pdftoppm"), "-f", "1", "-singlefile", "-png", "-r", "150", str(path), str(prefix)],
        check=True,
        capture_output=True,
    )
    rendered = prefix.with_suffix(".png")
    with Image.open(rendered) as image:
        gray = np.asarray(image.convert("L"), dtype=float)
    nonwhite = float(np.mean(gray < 250))
    contrast = float(np.std(gray))
    return rendered, bool(nonwhite > 0.003 and contrast > 2.0), nonwhite, contrast


def png_audit(path: Path) -> tuple[bool, int, int, float, float]:
    with Image.open(path) as image:
        width, height = image.size
        dpi = image.info.get("dpi", (0.0, 0.0))
    xdpi = float(dpi[0] or 0.0)
    ydpi = float(dpi[1] or 0.0)
    return bool(width >= 1800 and height >= 1000 and xdpi >= 590 and ydpi >= 590), width, height, xdpi, ydpi


def main() -> None:
    pdfs = sorted(FIG_ROOT.rglob("*.pdf"))
    pngs = sorted(FIG_ROOT.rglob("*.png"))
    pdf_stems = {path.relative_to(FIG_ROOT).with_suffix(""): path for path in pdfs}
    png_stems = {path.relative_to(FIG_ROOT).with_suffix(""): path for path in pngs}
    all_stems = sorted(set(pdf_stems) | set(png_stems), key=lambda item: str(item).lower())
    rows: list[dict[str, object]] = []
    for stem in all_stems:
        pdf = pdf_stems.get(stem)
        png = png_stems.get(stem)
        row: dict[str, object] = {
            "figure": str(stem).replace("\\", "/"),
            "pdf_exists": pdf is not None,
            "png_exists": png is not None,
            "paired": pdf is not None and png is not None,
        }
        if pdf is not None:
            row["pdf_bytes"] = pdf.stat().st_size
            row["pdf_sha256"] = sha256(pdf)
            row["font_pass"], row["pdf_fonts"], row["font_object_n"] = font_audit(pdf)
            row["vector_pass"], row["pdf_raster_object_n"] = raster_audit(pdf)
            row["text_pass"], row["extracted_text_char_n"] = text_audit(pdf)
            rendered, render_pass, nonwhite, contrast = render_pdf(pdf)
            row["rendered_png"] = str(rendered)
            row["render_pass"] = render_pass
            row["render_nonwhite_fraction"] = nonwhite
            row["render_contrast_sd"] = contrast
        if png is not None:
            row["png_bytes"] = png.stat().st_size
            row["png_sha256"] = sha256(png)
            (
                row["png_600dpi_pass"],
                row["png_width_px"],
                row["png_height_px"],
                row["png_xdpi"],
                row["png_ydpi"],
            ) = png_audit(png)
        checks = [
            bool(row.get("paired")),
            bool(row.get("font_pass")),
            bool(row.get("vector_pass")),
            bool(row.get("text_pass")),
            bool(row.get("render_pass")),
            bool(row.get("png_600dpi_pass")),
            int(row.get("pdf_bytes", 0)) > 0,
            int(row.get("png_bytes", 0)) > 0,
        ]
        row["overall_pass"] = all(checks)
        rows.append(row)

    csv_path = QC_ROOT / "figure_file_qc.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    failures = [row for row in rows if not row["overall_pass"]]
    report = [
        "# Figure QC report",
        "",
        f"- Figure pairs inspected: {len(rows)}",
        f"- PDF files: {len(pdfs)}",
        f"- PNG files: {len(pngs)}",
        f"- Automated file-level pass: {len(rows) - len(failures)}/{len(rows)}",
        "- PDF criteria: embedded Times New Roman text, extractable text, and zero raster image objects.",
        "- PNG criteria: at least 1,800 x 1,000 pixels and at least 590 dpi metadata on both axes.",
        "- Every PDF first page was rendered at 150 dpi for independent visual inspection.",
        "",
        "## Failed automated checks",
        "",
    ]
    if failures:
        for row in failures:
            failed = [
                key
                for key in ("paired", "font_pass", "vector_pass", "text_pass", "render_pass", "png_600dpi_pass")
                if not bool(row.get(key))
            ]
            report.append(f"- `{row['figure']}`: {', '.join(failed)}")
    else:
        report.append("- None.")
    report.extend(
        [
            "",
            "## Manual visual review checklist",
            "",
            "- [ ] No clipped labels, legends, confidence intervals, or panel letters.",
            "- [ ] No unintended figure titles.",
            "- [ ] Lower-case bold panel labels are positioned consistently.",
            "- [ ] ROC legends show AUC and 95% CI; PR legends show AUPRC and 95% CI.",
            "- [ ] Calibration smoothing is documented and original grouped points remain visible.",
            "- [ ] DCA contains no Youden line and includes treat-all/treat-none references.",
            "- [ ] Weighted voting appears first and is identified as the primary model where applicable.",
            "- [ ] Model colors, line styles, cohort names, and layer names are consistent.",
            "- [ ] SHAP and SII panels refer only to the complete Weighted voting model.",
            "- [ ] The interaction 3D z-axis is estimated pairwise SII, not PDP or ALE.",
        ]
    )
    (QC_ROOT / "figure_qc_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"QC pairs={len(rows)} failures={len(failures)} csv={csv_path}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

