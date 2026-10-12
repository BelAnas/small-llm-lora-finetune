"""Scoring. Every system writes the same kind of text, so one scorer is fair to all of them."""
from __future__ import annotations

from collections import Counter

from .schema import check_output


def score(examples: list[dict], raws: list[str]) -> dict:
    """Compare generated records with the labels. An unparseable output counts as wrong."""
    n = len(examples)
    valid = intent_ok = category_ok = order_ok = exact = 0
    with_order = with_order_ok = no_order = invented = 0
    review = accepted = accepted_exact = 0
    confusions: Counter[tuple[str, str]] = Counter()

    for ex, raw in zip(examples, raws):
        checked = check_output(raw, ex["message"])
        rec = checked.record
        got_intent = rec.intent if rec else None
        got_order = rec.order_number if rec else None
        i_ok = got_intent == ex["intent"]
        c_ok = bool(rec) and rec.category == ex["category"]
        o_ok = bool(rec) and got_order == ex["order_number"]
        all_ok = i_ok and c_ok and o_ok

        valid += rec is not None
        intent_ok += i_ok
        category_ok += c_ok
        order_ok += o_ok
        exact += all_ok
        if ex["order_number"] is not None:
            with_order += 1
            with_order_ok += o_ok
        else:
            no_order += 1
            invented += bool(rec) and got_order is not None
        if not i_ok:
            confusions[(ex["intent"], got_intent or "(no valid record)")] += 1
        # The checks send some messages to a person; what is left is acted on automatically.
        if checked.needs_review:
            review += 1
        else:
            accepted += 1
            accepted_exact += all_ok

    pct = lambda a, b: round(100.0 * a / b, 1) if b else None  # noqa: E731
    return {
        "n": n,
        "valid_record_pct": pct(valid, n),
        "intent_accuracy_pct": pct(intent_ok, n),
        "category_accuracy_pct": pct(category_ok, n),
        "order_number_accuracy_pct": pct(order_ok, n),
        "order_number_found_pct": pct(with_order_ok, with_order),
        "order_number_invented_pct": pct(invented, no_order),
        "exact_record_pct": pct(exact, n),
        "sent_to_review_pct": pct(review, n),
        "accepted_exact_pct": pct(accepted_exact, accepted),
        "n_with_order_number": with_order,
        "top_confusions": [
            {"expected": e, "got": g, "count": c} for (e, g), c in confusions.most_common(8)
        ],
    }
