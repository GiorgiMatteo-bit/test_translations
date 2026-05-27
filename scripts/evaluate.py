"""End-to-end IAM × VLM benchmark.

Loads IAM pages, transcribes them via a running vLLM endpoint, and writes a
JSON report with CER/WER and throughput.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from htr_bench.dataset import load_iam
from htr_bench.eval import EvalConfig, score_run
from htr_bench.inference import InferenceConfig, VLMTranscriber


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iam-root", required=True, type=Path)
    ap.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--prompt", default="default")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    pages = load_iam(args.iam_root, split=args.split, limit=args.limit)
    if not pages:
        raise SystemExit("no pages loaded — check --iam-root and --split")

    tx = VLMTranscriber(
        InferenceConfig(
            endpoint=args.endpoint,
            model=args.model,
            concurrency=args.concurrency,
            prompt=args.prompt,
        )
    )
    results = tx.transcribe_many([(p.page_id, p.image_path) for p in pages])

    refs = {p.page_id: p.transcript for p in pages}
    hyps = {r.page_id: r.text for r in results}
    lats = {r.page_id: r.latency_s for r in results}

    report = score_run(refs, hyps, lats, EvalConfig())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "split": args.split,
        "prompt": args.prompt,
        "concurrency": args.concurrency,
        **report.to_dict(),
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"n={report.n_pages}  CER={report.cer:.4f}  WER={report.wer:.4f}  "
        f"mean_latency={report.mean_latency_s:.2f}s  pages/s={report.pages_per_s:.2f}"
    )


if __name__ == "__main__":
    main()
