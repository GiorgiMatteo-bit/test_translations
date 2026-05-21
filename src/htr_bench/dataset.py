"""IAM Handwriting Database loader.

Expects an IAM root containing at least:
  forms/                 PNG scans, one per page (e.g. a01-000u.png)
  ascii/forms.txt        ground-truth metadata + line transcripts

The standard IAM split files (largeWriterIndependentTextLineRecognitionTask)
are honored if present; otherwise `split="all"` returns every page.
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


def _read_split_ids(root: Path, split: str) -> set[str] | None:
    if split == "all":
        return None
    rel = SPLIT_FILES.get(split)
    if rel is None:
        raise ValueError(f"unknown split: {split!r}")
    path = root / rel
    if not path.exists():
        raise FileNotFoundError(
            f"split file missing: {path}. Download the IAM task split files "
            "or use split='all'."
        )
    # line IDs look like a01-000u-00-00 — page id is the first two dash-joined parts
    page_ids: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("-")
        if len(parts) >= 2:
            page_ids.add(f"{parts[0]}-{parts[1]}")
    return page_ids


def _parse_forms_txt(forms_txt: Path) -> dict[str, str]:
    """Return {page_id: transcript} from ascii/forms.txt.

    Each form is referenced indirectly; the actual transcripts live in
    ascii/lines.txt keyed by line id. We aggregate lines per page.
    """
    raise NotImplementedError(
        "Implement once we pin the exact IAM ascii/ layout being used "
        "(forms.txt vs lines.txt vs xml/)."
    )


def load_iam(root: str | os.PathLike, split: str = "test", limit: int | None = None) -> list[Page]:
    """Load IAM pages + ground-truth transcripts.

    Parameters
    ----------
    root  : IAM dataset root.
    split : 'train' | 'val' | 'test' | 'all'.
    limit : optional cap (useful for smoke tests).
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"IAM root not found: {root_path}")

    forms_dir = root_path / "forms"
    ascii_dir = root_path / "ascii"
    if not forms_dir.is_dir():
        raise FileNotFoundError(f"expected forms/ under {root_path}")
    if not ascii_dir.is_dir():
        raise FileNotFoundError(f"expected ascii/ under {root_path}")

    keep = _read_split_ids(root_path, split)
    transcripts = _parse_forms_txt(ascii_dir / "lines.txt")

    pages: list[Page] = []
    for img in sorted(forms_dir.glob("*.png")):
        page_id = img.stem
        if keep is not None and page_id not in keep:
            continue
        gt = transcripts.get(page_id)
        if gt is None:
            continue
        pages.append(Page(page_id=page_id, image_path=img, transcript=gt))
        if limit is not None and len(pages) >= limit:
            break
    return pages
