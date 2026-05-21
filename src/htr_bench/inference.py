"""vLLM (OpenAI-compatible) inference client for Qwen-VL transcription.

Assumes a `vllm serve` process is running locally and reachable at the given
base URL. Images are sent as base64 data URIs so nothing leaves the host.
"""

from __future__ import annotations

import base64
import concurrent.futures as cf
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from .prompts import get_prompt


@dataclass(frozen=True)
class InferenceConfig:
    endpoint: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    api_key: str = "EMPTY"  # vLLM ignores it, but openai-py requires a value
    temperature: float = 0.0
    max_tokens: int = 2048
    concurrency: int = 4
    prompt: str = "default"


@dataclass(frozen=True)
class InferenceResult:
    page_id: str
    text: str
    latency_s: float


def _image_to_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


class VLMTranscriber:
    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.endpoint, api_key=cfg.api_key)
        self.prompt = get_prompt(cfg.prompt)

    def transcribe_one(self, page_id: str, image_path: Path) -> InferenceResult:
        data_uri = _image_to_data_uri(image_path)
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
        )
        latency = time.perf_counter() - t0
        text = (resp.choices[0].message.content or "").strip()
        return InferenceResult(page_id=page_id, text=text, latency_s=latency)

    def transcribe_many(
        self, items: list[tuple[str, Path]]
    ) -> list[InferenceResult]:
        if self.cfg.concurrency <= 1:
            return [self.transcribe_one(pid, p) for pid, p in items]
        out: list[InferenceResult] = []
        with cf.ThreadPoolExecutor(max_workers=self.cfg.concurrency) as ex:
            futs = [ex.submit(self.transcribe_one, pid, p) for pid, p in items]
            for fut in cf.as_completed(futs):
                out.append(fut.result())
        # restore input order
        order = {pid: i for i, (pid, _) in enumerate(items)}
        out.sort(key=lambda r: order[r.page_id])
        return out
