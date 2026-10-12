"""Route one message: ask the provider, check the answer, decide whether a person must look."""
from __future__ import annotations

import time
from dataclasses import dataclass

from .providers import Provider
from .schema import check_output


@dataclass
class Routed:
    message: str
    category: str | None
    intent: str | None
    order_number: str | None
    needs_review: bool
    problems: list[str]
    raw_output: str
    provider: str
    latency_ms: int


def route(provider: Provider, message: str) -> Routed:
    started = time.perf_counter()
    raw = provider.complete([message])[0]
    latency = int(1000 * (time.perf_counter() - started))
    checked = check_output(raw, message)
    record = checked.record
    # A record that failed a check is still shown, so the reviewer sees what the model
    # proposed, but needs_review tells the caller not to act on it.
    return Routed(
        message=message,
        category=record.category if record else None,
        intent=record.intent if record else None,
        order_number=record.order_number if record else None,
        needs_review=checked.needs_review,
        problems=checked.problems,
        raw_output=raw,
        provider=provider.name,
        latency_ms=latency,
    )
