"""Transcribe a folder of page images via a running vLLM endpoint.

Outputs one `.txt` per input image, into --out-dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from htr_bench.inference import InferenceConfig, VLMTranscriber


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--prompt", default="default")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    items = [
        (p.stem, p)
        for p in sorted(args.in_dir.iterdir())
        if p.suffix.lower() in IMG_EXTS
    ]
    if not items:
        raise SystemExit(f"no images in {args.in_dir}")

    cfg = InferenceConfig(
        endpoint=args.endpoint,
        model=args.model,
        concurrency=args.concurrency,
        prompt=args.prompt,
    )
    tx = VLMTranscriber(cfg)
    results = tx.transcribe_many(items)
    for r in results:
        (args.out_dir / f"{r.page_id}.txt").write_text(r.text, encoding="utf-8")
    print(f"wrote {len(results)} transcripts to {args.out_dir}")


if __name__ == "__main__":
    main()
