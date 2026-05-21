"""Transcription prompts. Pinned by key so eval runs are reproducible."""

from __future__ import annotations

PROMPTS: dict[str, str] = {
    "default": (
        "You are transcribing a scanned handwritten English document.\n"
        "Return ONLY the transcription as plain UTF-8 text.\n"
        "Preserve the original line breaks. Do not translate, summarize, or "
        "add commentary. Do not include headers, footers, or page numbers "
        "unless they appear in the handwriting itself.\n"
        "If a word is illegible, write [illegible]."
    ),
}


def get_prompt(name: str) -> str:
    try:
        return PROMPTS[name]
    except KeyError as e:
        raise KeyError(f"unknown prompt {name!r}; known: {sorted(PROMPTS)}") from e
