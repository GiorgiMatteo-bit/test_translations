"""CER / WER evaluation plus throughput stats."""

from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass

import jiwer


@dataclass(frozen=True)
class EvalConfig:
    normalize: bool = True


@dataclass(frozen=True)
class PageScore:
    page_id: str
    cer: float
    wer: float
    ref_chars: int
    hyp_chars: int


@dataclass(frozen=True)
class RunReport:
    n_pages: int
    cer: float           # corpus-level (micro-averaged via jiwer)
    wer: float
    mean_latency_s: float
    pages_per_s: float
    per_page: list[PageScore]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["per_page"] = [asdict(p) for p in self.per_page]
        return d


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    # collapse all whitespace runs; HTR refs/hyps differ in line breaks far more
    # than they differ in actual content.
    return " ".join(s.split())


def score_page(reference: str, hypothesis: str, cfg: EvalConfig) -> PageScore:
    if cfg.normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    return PageScore(
        page_id="",  # filled in by caller
        cer=jiwer.cer(reference, hypothesis),
        wer=jiwer.wer(reference, hypothesis),
        ref_chars=len(reference),
        hyp_chars=len(hypothesis),
    )


def score_run(
    refs: dict[str, str],
    hyps: dict[str, str],
    latencies: dict[str, float],
    cfg: EvalConfig,
) -> RunReport:
    """Aggregate per-page scores into a corpus report.

    Pages present in `refs` but missing from `hyps` are scored against an empty
    hypothesis (counted as full deletion); this keeps the comparison honest
    when a model refuses or times out.
    """
    page_ids = sorted(refs.keys())
    norm_refs: list[str] = []
    norm_hyps: list[str] = []
    per_page: list[PageScore] = []
    for pid in page_ids:
        r = refs[pid]
        h = hyps.get(pid, "")
        s = score_page(r, h, cfg)
        per_page.append(PageScore(pid, s.cer, s.wer, s.ref_chars, s.hyp_chars))
        norm_refs.append(normalize_text(r) if cfg.normalize else r)
        norm_hyps.append(normalize_text(h) if cfg.normalize else h)

    corpus_cer = jiwer.cer(norm_refs, norm_hyps)
    corpus_wer = jiwer.wer(norm_refs, norm_hyps)

    lat_values = [latencies[pid] for pid in page_ids if pid in latencies]
    mean_lat = sum(lat_values) / len(lat_values) if lat_values else 0.0
    pps = (1.0 / mean_lat) if mean_lat > 0 else 0.0

    return RunReport(
        n_pages=len(page_ids),
        cer=corpus_cer,
        wer=corpus_wer,
        mean_latency_s=mean_lat,
        pages_per_s=pps,
        per_page=per_page,
    )
