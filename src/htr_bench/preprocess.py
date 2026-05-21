"""Light page preprocessing: deskew + denoise.

Kept intentionally minimal — the point of evaluating a VLM end-to-end is to
expose its robustness to raw scans. Heavy preprocessing here would mask that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    deskew: bool = True
    denoise: bool = False  # off by default; non-local means is slow per page


def _estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate skew in degrees via the bounding box of dark pixels."""
    inv = cv2.bitwise_not(gray)
    _, bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if coords.size == 0:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # cv2 returns angle in [-90, 0); fold into ~[-45, 45]
    if angle < -45:
        angle += 90
    return float(angle)


def _deskew(gray: np.ndarray) -> np.ndarray:
    angle = _estimate_skew_angle(gray)
    if abs(angle) < 0.1:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def preprocess_image(path: Path, cfg: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    """Load a page image and apply minimal cleanup. Returns a grayscale ndarray."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    if cfg.deskew:
        img = _deskew(img)
    if cfg.denoise:
        img = cv2.fastNlMeansDenoising(img, h=10)
    return img
