"""Print evals/report.json as the Markdown table used in the README: python -m evals.table"""
from __future__ import annotations

import json

from evals.run import REPORT

NAMES = {
    "base-prompt": "Base model, instructions only",
    "base-examples": "Base model, instructions + 8 examples",
    "lora": "Fine-tuned with LoRA",
    "classic": "Classic classifier + regex",
    "mock": "Offline keyword rules (test stand-in)",
    "hosted": "Hosted model, instructions only",
}


def main() -> None:
    report = json.loads(REPORT.read_text())
    for set_name, title in [("test", "Held-out test set"), ("challenge", "Hand-written set")]:
        print(f"\n**{title}**\n")
        print("| System | Messages | Valid record | Intent correct | Order number correct | Whole record correct | ms per message |")
        print("|---|---|---|---|---|---|---|")
        for system, label in NAMES.items():
            r = report.get(f"{system}/{set_name}")
            if r:
                print(
                    f"| {label} | {r['n']} | {r['valid_record_pct']}% | {r['intent_accuracy_pct']}% | "
                    f"{r['order_number_accuracy_pct']}% | {r['exact_record_pct']}% | {r['ms_per_message_batched']} |"
                )


if __name__ == "__main__":
    main()
