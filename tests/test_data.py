import json
import random
import re

from triage.data import build_examples, fill_placeholders, make_order_number, split_examples
from triage.labels import CATEGORIES, INTENT_TO_CATEGORY, INTENTS


def test_label_list_shape():
    assert len(INTENTS) == 27 and len(CATEGORIES) == 11
    assert set(INTENT_TO_CATEGORY.values()) == set(CATEGORIES)


def test_order_number_is_filled_and_returned():
    text, number = fill_placeholders("cancel order {{Order Number}} now", random.Random(1))
    assert number and number in text and "{{" not in text


def test_same_order_number_reused_within_a_message():
    text, number = fill_placeholders("{{Order Number}} and again {{Order Number}}", random.Random(1))
    assert text.count(number) == 2


def test_invoice_number_is_not_an_order_number():
    text, number = fill_placeholders("invoice {{Invoice Number}} is wrong", random.Random(1))
    assert number is None and "INV-" in text


def test_no_placeholder_left_behind():
    raw = "{{Person Name}} {{Account Type}} {{Currency Symbol}}{{Refund Amount}} {{Delivery City}} {{Delivery Country}}"
    text, _ = fill_placeholders(raw, random.Random(3))
    assert "{{" not in text


def test_order_numbers_come_in_several_shapes():
    rng = random.Random(0)
    shapes = {re.sub(r"\d", "9", re.sub(r"[A-Z]", "A", make_order_number(rng))) for _ in range(200)}
    assert len(shapes) >= 6


ROWS = [
    {"instruction": "cancel order {{Order Number}}", "intent": "cancel_order"},
    {"instruction": "Cancel  order {{Order Number}}", "intent": "cancel_order"},  # duplicate
    {"instruction": "i forgot my password", "intent": "recover_password"},
    {"instruction": "something else", "intent": "not_a_known_intent"},
]


def test_build_examples_drops_duplicates_and_unknown_intents():
    examples = build_examples(ROWS)
    assert [e["intent"] for e in examples] == ["cancel_order", "recover_password"]


def test_target_is_json_that_matches_the_fields():
    for ex in build_examples(ROWS):
        target = json.loads(ex["target"])
        assert target == {"category": ex["category"], "intent": ex["intent"], "order_number": ex["order_number"]}


def test_build_is_reproducible():
    assert build_examples(ROWS, seed=5) == build_examples(ROWS, seed=5)


def test_splits_do_not_overlap():
    examples = [{"message": str(i)} for i in range(100)]
    parts = split_examples(examples, n_train=50, n_val=10, n_test=20)
    ids = [e["message"] for part in parts.values() for e in part]
    assert len(ids) == len(set(ids)) == 80


def test_test_split_does_not_depend_on_train_size():
    examples = [{"message": str(i)} for i in range(100)]
    a = split_examples(examples, n_train=10, n_val=10, n_test=20)["test"]
    b = split_examples(examples, n_train=60, n_val=10, n_test=20)["test"]
    assert a == b


def _plain(n: int) -> list[dict]:
    from triage.data import make_example

    return [make_example(f"i forgot my password {i}x", "recover_password", None) for i in range(n)]


def test_augment_adds_order_numbers_that_are_in_the_message():
    from triage.data import augment

    out = augment(_plain(400))
    with_number = [ex for ex in out if ex["order_number"]]
    assert 30 <= len(with_number) <= 90  # about 15% of 400
    for ex in with_number:
        assert ex["order_number"] in ex["message"]
        assert json.loads(ex["target"])["order_number"] == ex["order_number"]
        assert ex["intent"] == "recover_password"  # the intent never changes


def test_augment_adds_numbers_that_are_not_order_numbers():
    from triage.data import augment

    originals = {ex["message"] for ex in _plain(400)}
    changed = [ex for ex in augment(_plain(400)) if ex["message"] not in originals and not ex["order_number"]]
    assert changed and all(re.search(r"\d{3,}| \d times", ex["message"]) for ex in changed)


def test_augment_leaves_messages_with_an_order_number_alone():
    from triage.data import augment, make_example

    rows = [make_example(f"cancel order ORD-{10000 + i}", "cancel_order", f"ORD-{10000 + i}") for i in range(50)]
    assert augment(rows) == rows


def test_augment_is_reproducible():
    from triage.data import augment

    assert augment(_plain(100), seed=3) == augment(_plain(100), seed=3)
