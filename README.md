# HTR Bench

Local, offline benchmark for handwritten document transcription using open-weights
vision-language models. Built around the IAM Handwriting Database and the Qwen3-VL
family, served via vLLM.

## Why this exists

We need to transcribe scanned, multi-writer handwritten documents into `.txt`.
The data is sensitive, so the pipeline must run entirely on in-house GPUs — no
hosted APIs, open-weights models only. This repo is the eval harness used to
pick a model size and prompt strategy before committing to a production run.

## Pipeline

1. **Dataset** — IAM pages + ground-truth transcripts (`htr_bench.dataset`).
2. **Preprocess** — light deskew/denoise (`htr_bench.preprocess`); kept minimal
   to expose raw VLM robustness.
3. **Inference** — vLLM OpenAI-compatible endpoint, batch transcription
   (`htr_bench.inference`).
4. **Eval** — CER/WER via `jiwer`, plus latency and pages/s
   (`htr_bench.eval`).
5. **Compare** — same harness across Qwen2.5-VL-7B (speed baseline),
   Qwen3-VL-32B, and Qwen3-VL-235B-A22B.

## Quickstart

```bash
# Install
pip install -e .

# 1. Assemble IAM locally from Kaggle (needs ~/.kaggle/kaggle.json).
#    Drop --forms-only-sample for the full ~4.6 GB page set.
python scripts/download_iam.py --out data/iam --forms-only-sample

# 2. Serve the model with vLLM on a GPU box, in another shell
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8000 \
  --max-model-len 8192 \
  --limit-mm-per-prompt image=1

# 3. Run the benchmark
python -m scripts.evaluate \
  --iam-root data/iam \
  --endpoint http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --limit 50 \
  --out results/qwen2.5-vl-7b.json
```

### Dataset layout

`download_iam.py` assembles the canonical IAM layout the loader expects:

```
data/iam/
  lines.txt           # transcripts, '|'-separated word tokens, per text line
  forms/<form_id>.png # full-page scans
```

The loader (`htr_bench.dataset.load_iam`) finds `lines.txt` and the page PNGs by
search, so nested Kaggle mirror layouts also work without reshuffling. The
`train`/`val`/`test` splits need the separate IAM task split files
(`largeWriterIndependentTextLineRecognitionTask`); without them, use
`--split all` (the default).

## Models

Initial speed baseline is `Qwen/Qwen2.5-VL-7B-Instruct` — well-supported by vLLM
today. Once the Qwen3-VL serving path is stable we swap in `Qwen3-VL-8B`,
`Qwen3-VL-32B`, and `Qwen3-VL-235B-A22B-Instruct` using the same harness.

## Layout

```
src/htr_bench/
  dataset.py     # IAM page+transcript loader
  preprocess.py  # deskew, denoise
  inference.py   # vLLM OpenAI-compatible client
  eval.py        # CER/WER, latency, pages/s
  prompts.py     # transcription prompt(s)
scripts/
  download_iam.py # assemble IAM from Kaggle mirrors
  transcribe.py   # transcribe a folder of images
  evaluate.py     # end-to-end eval against IAM
configs/
  default.yaml   # default run config
```

## Constraints

- 100% local, no external APIs.
- Open weights only (Apache-2.0 preferred).
- Reproducibility matters more than peak score: fix seeds, pin prompts, log
  versions.
