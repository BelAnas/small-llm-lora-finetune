from triage.metrics import score
from triage.schema import target_json


def test_perfect_outputs_score_100(examples):
    result = score(examples, [e["target"] for e in examples])
    assert result["exact_record_pct"] == 100.0
    assert result["sent_to_review_pct"] == 0.0
    assert result["order_number_found_pct"] == 100.0
    assert result["top_confusions"] == []


def test_garbage_scores_zero_and_goes_to_review(examples):
    result = score(examples, ["sorry, I cannot help"] * len(examples))
    assert result["valid_record_pct"] == 0.0
    assert result["intent_accuracy_pct"] == 0.0
    assert result["sent_to_review_pct"] == 100.0
    assert result["accepted_exact_pct"] is None


def test_wrong_intent_is_counted_as_a_confusion(examples):
    raws = [e["target"] for e in examples]
    raws[2] = target_json("REFUND", "get_refund", None)  # expected recover_password
    result = score(examples, raws)
    assert result["intent_accuracy_pct"] == round(100 * 5 / 6, 1)
    assert result["top_confusions"][0] == {"expected": "recover_password", "got": "get_refund", "count": 1}


def test_invented_order_number_is_measured_and_reviewed(examples):
    raws = [e["target"] for e in examples]
    raws[2] = target_json("ACCOUNT", "recover_password", "ORD-00000")
    result = score(examples, raws)
    assert result["order_number_invented_pct"] == 25.0  # 1 of the 4 messages with no number
    assert result["sent_to_review_pct"] == round(100 / 6, 1)
    # The check caught the only wrong record, so everything that was accepted is right.
    assert result["accepted_exact_pct"] == 100.0


def test_missed_order_number(examples):
    raws = [e["target"] for e in examples]
    raws[0] = target_json("ORDER", "cancel_order", None)
    assert score(examples, raws)["order_number_found_pct"] == 50.0
