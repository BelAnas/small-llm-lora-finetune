"""Score the systems on the same messages and write evals/report.json.

    python -m evals.run --systems classic lora --set test
    python -m evals.run --systems base-prompt base-examples --set test --limit 100
    python -m evals.run --systems classic lora mock --set challenge

Results are merged into the report under "<system>/<set>", so slow systems can be run on
their own. Every entry records how many messages it was scored on.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from triage.config import DATA_DIR, ROOT, get_settings
from triage.data import read_jsonl
from triage.metrics import score

REPORT = ROOT / "evals" / "report.json"
SETS = {"test": DATA_DIR / "test.jsonl", "challenge": ROOT / "evals" / "challenge.jsonl"}
SYSTEMS = ["classic", "mock", "base-prompt", "base-examples", "lora", "hosted"]


class _Lazy:
    """Loads the tokenizer and each model once, however many systems are scored."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._cache: dict[str, object] = {}

    def get(self, key: str, factory):
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    def provider(self, system: str):
        from triage.providers import LocalProvider, MockProvider, OpenAIProvider, pick_examples

        s = self.settings
        if system == "mock":
            return MockProvider()
        if system == "hosted":
            return self.get("hosted", lambda: OpenAIProvider(s.openai_model))
        if system == "classic":
            from triage.baseline import predict_raw, train_classifier

            model = self.get("classic", lambda: train_classifier(read_jsonl(DATA_DIR / "train.jsonl")))

            class Classic:
                name = "classic"

                def complete(self, messages):
                    return predict_raw(model, messages)

            return Classic()

        from triage.model import load_model, load_tokenizer, set_threads

        set_threads(s.threads)
        tokenizer = self.get("tokenizer", lambda: load_tokenizer(s.base_model))
        if system == "lora":
            model = self.get("lora", lambda: load_model(s.base_model, s.adapter_dir, merge=True))
            return LocalProvider(model, tokenizer, "lora", max_new_tokens=s.max_new_tokens)
        base = self.get("base", lambda: load_model(s.base_model))
        if system == "base-prompt":
            return LocalProvider(base, tokenizer, "zero", max_new_tokens=s.max_new_tokens)
        examples = pick_examples(read_jsonl(DATA_DIR / "train.jsonl"))
        # Long prompts with worked examples need a smaller batch to stay within memory and time.
        return LocalProvider(base, tokenizer, "few", examples, s.max_new_tokens, batch_size=8)


def run(systems: list[str], set_name: str, limit: int) -> dict:
    examples = read_jsonl(SETS[set_name])[: limit or None]
    messages = [ex["message"] for ex in examples]
    lazy = _Lazy()
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    for system in systems:
        provider = lazy.provider(system)
        started = time.perf_counter()
        raws = provider.complete(messages)
        seconds = time.perf_counter() - started
        result = score(examples, raws)
        result["ms_per_message_batched"] = round(1000 * seconds / len(messages), 2)
        if system == "hosted":
            result["model"] = provider.model
            result["input_tokens"], result["output_tokens"] = provider.input_tokens, provider.output_tokens
        # A few raw outputs are kept so a reader can see what each system actually wrote.
        result["samples"] = [
            {"message": m, "expected": ex["target"], "output": raw[:200]}
            for m, ex, raw in list(zip(messages, examples, raws))[:5]
        ]
        report[f"{system}/{set_name}"] = result
        REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(
            f"{system:14s} {set_name:9s} n={result['n']:<5d} valid={result['valid_record_pct']:5.1f}% "
            f"intent={result['intent_accuracy_pct']:5.1f}% exact={result['exact_record_pct']:5.1f}% "
            f"review={result['sent_to_review_pct']:5.1f}% {result['ms_per_message_batched']:8.2f} ms/msg"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=["classic", "lora"])
    parser.add_argument("--set", dest="set_name", choices=list(SETS), default="test")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N messages")
    args = parser.parse_args()
    run(args.systems, args.set_name, args.limit)


if __name__ == "__main__":
    main()
