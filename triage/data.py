"""Turns the public Bitext customer-support dataset into message -> JSON training pairs.

Dataset: bitext/Bitext-customer-support-llm-chatbot-training-dataset on the Hugging Face Hub
(26,872 messages, 27 intents, licence CDLA-Sharing-1.0). The messages contain placeholders such
as {{Order Number}}; this script fills them with invented values so the model has something
real to copy, which is what makes the order-number field a genuine extraction task.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .config import DATA_DIR
from .labels import INTENT_TO_CATEGORY
from .schema import target_json

DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

_NAMES = ["Sam Carter", "Lina Haddad", "Omar Reyes", "Mei Tanaka", "Jon Peters", "Aya Nasser"]
_ACCOUNTS = ["Premium", "Free", "Pro", "Gold", "Standard", "Business", "Platinum"]
_CITIES = ["Lyon", "Toronto", "Leeds", "Denver", "Casablanca", "Osaka", "Porto"]
_COUNTRIES = ["France", "Canada", "Spain", "Morocco", "Japan", "Germany", "Brazil"]
_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"


def make_order_number(rng: random.Random) -> str:
    """Order numbers in several shapes, so the model learns to copy and not to memorise one."""
    digits = lambda n: "".join(rng.choice("0123456789") for _ in range(n))  # noqa: E731
    shape = rng.randrange(6)
    if shape == 0:
        return "#" + digits(rng.choice([5, 6]))
    if shape == 1:
        return "ORD-" + digits(5)
    if shape == 2:
        return digits(8)
    if shape == 3:
        return rng.choice(_LETTERS) + rng.choice(_LETTERS) + digits(6)
    if shape == 4:
        return f"SO-{digits(4)}-{digits(3)}"
    return digits(3) + "-" + digits(7)


def fill_placeholders(text: str, rng: random.Random) -> tuple[str, str | None]:
    """Replace every {{Placeholder}} with an invented value. Returns the text and the order number."""
    order_number: str | None = None

    def replace(match: re.Match) -> str:
        nonlocal order_number
        name = match.group(1).strip().lower()
        if name == "order number":
            # The same number is reused if the placeholder appears twice in one message.
            order_number = order_number or make_order_number(rng)
            return order_number
        if name == "invoice number":
            return "INV-" + str(rng.randrange(10000, 99999))
        if name in ("account type", "account category"):
            return rng.choice(_ACCOUNTS)
        if name == "person name":
            return rng.choice(_NAMES)
        if name == "currency symbol":
            return rng.choice(["$", "€", "£"])
        if name == "refund amount":
            return str(rng.randrange(10, 900))
        if name == "delivery city":
            return rng.choice(_CITIES)
        if name == "delivery country":
            return rng.choice(_COUNTRIES)
        return name

    return re.sub(r"\{\{(.*?)\}\}", replace, text), order_number


def make_example(message: str, intent: str, order_number: str | None) -> dict:
    category = INTENT_TO_CATEGORY[intent]
    return {
        "message": message,
        "category": category,
        "intent": intent,
        "order_number": order_number,
        "target": target_json(category, intent, order_number),
    }


def build_examples(rows: list[dict], seed: int = 7) -> list[dict]:
    """Fill placeholders and drop exact duplicate messages (the dataset repeats some)."""
    rng = random.Random(seed)
    seen: set[str] = set()
    examples: list[dict] = []
    for row in rows:
        key = re.sub(r"\s+", " ", row["instruction"].strip().lower())
        if key in seen or row["intent"] not in INTENT_TO_CATEGORY:
            continue
        seen.add(key)
        message, order_number = fill_placeholders(row["instruction"].strip(), rng)
        examples.append(make_example(message, row["intent"], order_number))
    return examples


def split_examples(
    examples: list[dict], n_train: int, n_val: int, n_test: int, seed: int = 7
) -> dict[str, list[dict]]:
    """Shuffle once, then cut. Test is cut first so its content never depends on n_train."""
    rng = random.Random(seed)
    pool = examples[:]
    rng.shuffle(pool)
    test, val = pool[:n_test], pool[n_test : n_test + n_val]
    train = pool[n_test + n_val : n_test + n_val + n_train]
    return {"train": train, "val": val, "test": test}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def prepare(n_train: int = 4000, n_val: int = 300, n_test: int = 1000, seed: int = 7) -> dict[str, int]:
    """Download the dataset and write data/train.jsonl, val.jsonl and test.jsonl."""
    from datasets import load_dataset  # imported here so tests never need the network

    rows = list(load_dataset(DATASET, split="train"))
    examples = build_examples(rows, seed)
    splits = split_examples(examples, n_train, n_val, n_test, seed)
    for name, part in splits.items():
        write_jsonl(DATA_DIR / f"{name}.jsonl", part)
    counts = {name: len(part) for name, part in splits.items()}
    counts["unique_messages"] = len(examples)
    counts["source_rows"] = len(rows)
    return counts
