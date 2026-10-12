"""Time the fine-tuned model one message at a time, the way the API calls it.

    python -m evals.latency

Compares the add-on merged into the base weights with the add-on kept as separate layers,
and writes the result into evals/report.json under "latency".
"""
from __future__ import annotations

import json
import statistics
import time

import torch

from evals.run import REPORT, SETS
from triage.config import get_settings
from triage.data import read_jsonl
from triage.model import generate, load_model, load_tokenizer, set_threads
from triage.prompts import ft_prompt


def time_model(model, tokenizer, prompts: list[str], max_new_tokens: int) -> dict:
    generate(model, tokenizer, prompts[:2], max_new_tokens, batch_size=1)  # warm-up, not timed
    times = []
    for prompt in prompts:
        started = time.perf_counter()
        generate(model, tokenizer, [prompt], max_new_tokens, batch_size=1)
        times.append(1000 * (time.perf_counter() - started))
    times.sort()
    return {
        "messages": len(times),
        "median_ms": round(statistics.median(times)),
        "p95_ms": round(times[int(0.95 * (len(times) - 1))]),
    }


def main() -> None:
    settings = get_settings()
    set_threads(settings.threads)
    tokenizer = load_tokenizer(settings.base_model)
    prompts = [ft_prompt(ex["message"]) for ex in read_jsonl(SETS["challenge"])[:40]]
    result = {"cpu_threads": torch.get_num_threads()}
    for name, merge in [("merged", True), ("unmerged", False)]:
        model = load_model(settings.base_model, settings.adapter_dir, merge=merge)
        result[name] = time_model(model, tokenizer, prompts, settings.max_new_tokens)
        print(name, result[name])
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    report["latency"] = result
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
