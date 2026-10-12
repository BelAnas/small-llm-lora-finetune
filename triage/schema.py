"""The record the model must write, and the checks applied to it before anyone trusts it.

A language model can write anything. These checks are what make its output safe to act on:
a record that fails any of them is not used, and the message goes to a person instead.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from .labels import CATEGORIES, INTENT_TO_CATEGORY


class Record(BaseModel):
    category: str
    intent: str
    order_number: str | None = None


@dataclass
class Checked:
    """A parsed record plus the reasons it should not be trusted, if any."""

    record: Record | None
    problems: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return bool(self.problems)


def target_json(category: str, intent: str, order_number: str | None) -> str:
    """The exact text the model is trained to write. One fixed key order keeps it easy to learn."""
    return json.dumps({"category": category, "intent": intent, "order_number": order_number})


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def check_output(raw: str, message: str) -> Checked:
    """Parse the model's text and list every reason not to trust it."""
    # Small models sometimes keep writing after the record, so take the first {...} only.
    match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
    if not match:
        return Checked(None, ["no JSON object in the output"])
    try:
        record = Record.model_validate(json.loads(match.group(0)))
    except (json.JSONDecodeError, ValidationError):
        return Checked(None, ["output is not a valid record"])

    problems: list[str] = []
    if record.category not in CATEGORIES:
        problems.append(f"unknown category '{record.category}'")
    if record.intent not in INTENT_TO_CATEGORY:
        problems.append(f"unknown intent '{record.intent}'")
    elif INTENT_TO_CATEGORY[record.intent] != record.category:
        problems.append("intent does not belong to the category")
    if record.order_number is not None:
        # An order number that is not in the message was invented by the model.
        if not record.order_number.strip() or _squash(record.order_number) not in _squash(message):
            problems.append("order number does not appear in the message")
    return Checked(record, problems)
