"""A classic classifier, as a reference point for the language model.

TF-IDF plus logistic regression predicts the intent in microseconds. It cannot write a record
or copy an order number by itself, so a regular expression does that part. If this simple
pipeline matches the fine-tuned model, a client should know.
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from .labels import INTENT_TO_CATEGORY
from .schema import target_json

# Anything with at least four digits, optionally with a prefix and dashes. Deliberately simple:
# it is what a developer would write in ten minutes, and it has no idea what the number means.
_NUMBER = re.compile(r"#?\b(?:[A-Z]{2,3}-?)?\d[\d-]{3,}\d\b|#\d{4,}")


def train_classifier(train: list[dict]) -> Pipeline:
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True),
        LogisticRegression(max_iter=2000, C=10.0),
    )
    model.fit([ex["message"] for ex in train], [ex["intent"] for ex in train])
    return model


def regex_order_number(message: str) -> str | None:
    match = _NUMBER.search(message)
    return match.group(0) if match else None


def predict_raw(model: Pipeline, messages: list[str]) -> list[str]:
    """Same output format as the language model, so one scorer handles every system."""
    intents = model.predict(messages)
    return [
        target_json(INTENT_TO_CATEGORY[intent], intent, regex_order_number(message))
        for intent, message in zip(intents, messages)
    ]
