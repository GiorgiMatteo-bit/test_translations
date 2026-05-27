"""Assemble a local IAM dataset from Kaggle mirrors into the layout htr_bench expects.

No single Kaggle dataset bundles full-page scans with the canonical transcripts,
so we stitch two:
  - bustergone/iam-handwriting-dataset      -> ascii.tgz (lines.txt + friends)
  - naderabdelghany/iam-handwritten-forms-dataset -> full-page PNGs (~4.6 GB)

Result:
  <out>/lines.txt
  <out>/forms/<form_id>.png

Requires Kaggle credentials at ~/.kaggle/kaggle.json (chmod 600).
The full forms download is large; use --forms-only-sample for a quick smoke set.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ASCII_DS = "bustergone/iam-handwriting-dataset"
FORMS_DS = "naderabdelghany/iam-handwritten-forms-dataset"

# A few forms with known transcripts, handy for offline smoke tests.
SAMPLE_FORMS = ["data/000/a01-000u.png", "data/000/a01-003u.png", "data/000/a01-007u.png"]


def _kaggle(*args: str) -> None:
    subprocess.run(["kaggle", *args], check=True)


def fetch_ascii(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    _kaggle("datasets", "download", ASCII_DS, "-f", "ascii.tgz", "-p", str(out))
    tgz = out / "ascii.tgz"
    with tarfile.open(tgz) as t:
        t.extractall(out)  # noqa: S202 - trusted dataset archive
    tgz.unlink(missing_ok=True)


def fetch_forms(out: Path, sample: bool) -> None:
    forms_dir = out / "forms"
    forms_dir.mkdir(parents=True, exist_ok=True)
    files = SAMPLE_FORMS if sample else [None]  # None => whole dataset
    for f in files:
        cmd = ["datasets", "download", FORMS_DS, "-p", str(forms_dir)]
        if f is not None:
            cmd += ["-f", f]
        else:
            cmd += ["--unzip"]
        _kaggle(*cmd)
    # single-file pulls arrive as <name>.png.zip; unzip + flatten
    for z in forms_dir.rglob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(forms_dir)
        z.unlink()
    for png in list(forms_dir.rglob("*.png")):
        target = forms_dir / png.name
        if png != target:
            png.replace(target)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/iam"))
    ap.add_argument(
        "--forms-only-sample",
        action="store_true",
        help="download only a handful of pages instead of the full ~4.6 GB set",
    )
    args = ap.parse_args()

    if not (Path.home() / ".kaggle" / "kaggle.json").exists():
        sys.exit("missing ~/.kaggle/kaggle.json — create a Kaggle API token first")

    fetch_ascii(args.out)
    fetch_forms(args.out, sample=args.forms_only_sample)
    print(f"IAM assembled under {args.out} (lines.txt + forms/*.png)")


if __name__ == "__main__":
    main()
