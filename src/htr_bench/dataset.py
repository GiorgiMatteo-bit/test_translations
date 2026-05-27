"""IAM Handwriting Database loader.

Works against the canonical IAM ground-truth files (as shipped in the FKI
`ascii` bundle, also mirrored on Kaggle). Expected under `root`:

  lines.txt            transcripts, one row per text line (see format below)
  forms/**/*.png       full-page scans, named <form_id>.png

`lines.txt` (and `forms/` images) may sit at the root or one level down; both
are located by search, so Kaggle mirrors that nest pages under data/NNN/ work
without reshuffling files.

lines.txt row format:
    a01-000u-00 ok 154 19 408 746 1661 89 A|MOVE|to|stop|Mr.|Gaitskell|from
    └ line id  └status        └ bbox x y w h  └ transcript ('|' = word break)

The form id is the line id minus its trailing segment: a01-000u-00 -> a01-000u.
Per-form transcripts are rebuilt by joining lines in file order with '\n' and
replacing '|' with a single space.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Page:
    page_id: str
    image_path: Path
    transcript: str


SPLIT_FILES = {
    "train": "largeWriterIndependentTextLineRecognitionTask/trainset.txt",
    "val": "largeWriterIndependentTextLineRecognitionTask/validationset1.txt",
    "test": "largeWriterIndependentTextLineRecognitionTask/testset.txt",
}


def _line_id_to_form_id(line_id: str) -> str:
    # a01-000u-00 -> a01-000u  (drop the trailing line segment)
    return "-".join(line_id.split("-")[:-1])


def _find_file(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.exists():
        return direct
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


def _read_split_ids(root: Path, split: str) -> set[str] | None:
    if split == "all":
        return None
    rel = SPLIT_FILES.get(split)
    if rel is None:
        raise ValueError(f"unknown split: {split!r}")
    path = _find_file(root, Path(rel).name)
    if path is None:
        raise FileNotFoundError(
            f"split file {Path(rel).name!r} not found under {root}. Download the "
            "IAM task split files or use split='all'."
        )
    page_ids: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        page_ids.add(_line_id_to_form_id(line.split()[0]))
    return page_ids


def _parse_lines_txt(lines_txt: Path) -> dict[str, str]:
    """Aggregate IAM lines.txt into {form_id: transcript}."""
    per_form: dict[str, list[str]] = {}
    for raw in lines_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split(" ", 8)
        if len(parts) < 9:
            continue  # malformed / missing transcript column
        line_id, transcript = parts[0], parts[8]
        form_id = _line_id_to_form_id(line_id)
        per_form.setdefault(form_id, []).append(transcript.replace("|", " "))
    return {fid: "\n".join(lines) for fid, lines in per_form.items()}


def _index_images(root: Path) -> dict[str, Path]:
    """Map {form_id: image_path} from all PNGs found under root."""
    index: dict[str, Path] = {}
    for img in sorted(root.rglob("*.png")):
        index.setdefault(img.stem, img)
    return index


def load_iam(root: str | os.PathLike, split: str = "all", limit: int | None = None) -> list[Page]:
    """Load IAM pages + ground-truth transcripts.

    Parameters
    ----------
    root  : IAM dataset root (contains lines.txt and page PNGs, possibly nested).
    split : 'train' | 'val' | 'test' | 'all' (non-'all' needs the task split files).
    limit : optional cap (useful for smoke tests).
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"IAM root not found: {root_path}")

    lines_txt = _find_file(root_path, "lines.txt")
    if lines_txt is None:
        raise FileNotFoundError(f"lines.txt not found under {root_path}")

    keep = _read_split_ids(root_path, split)
    transcripts = _parse_lines_txt(lines_txt)
    images = _index_images(root_path)

    pages: list[Page] = []
    for form_id in sorted(transcripts):
        if keep is not None and form_id not in keep:
            continue
        img = images.get(form_id)
        if img is None:
            continue  # transcript present but page image not downloaded
        pages.append(Page(form_id, img, transcripts[form_id]))
        if limit is not None and len(pages) >= limit:
            break
    return pages
