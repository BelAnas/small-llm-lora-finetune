from triage.schema import check_output, target_json

MSG = "Please cancel order ORD-12345, thanks"


def test_good_record_passes():
    checked = check_output(target_json("ORDER", "cancel_order", "ORD-12345"), MSG)
    assert not checked.needs_review
    assert checked.record.intent == "cancel_order"


def test_text_after_the_record_is_ignored():
    raw = target_json("ORDER", "cancel_order", None) + "\n### Message\nmore text {"
    assert not check_output(raw, MSG).needs_review


def test_no_json_is_flagged():
    checked = check_output("I'm sorry, as an AI I cannot help.", MSG)
    assert checked.record is None and checked.needs_review


def test_broken_json_is_flagged():
    assert check_output('{"category": "ORDER", "intent": }', MSG).needs_review


def test_wrong_types_are_flagged():
    assert check_output('{"category": 3, "intent": ["x"]}', MSG).record is None


def test_unknown_intent_is_flagged():
    checked = check_output(target_json("ORDER", "destroy_order", None), MSG)
    assert "unknown intent 'destroy_order'" in checked.problems


def test_unknown_category_is_flagged():
    assert check_output(target_json("WEATHER", "cancel_order", None), MSG).needs_review


def test_intent_must_belong_to_category():
    checked = check_output(target_json("REFUND", "cancel_order", None), MSG)
    assert checked.problems == ["intent does not belong to the category"]


def test_invented_order_number_is_flagged():
    checked = check_output(target_json("ORDER", "cancel_order", "ORD-99999"), MSG)
    assert checked.problems == ["order number does not appear in the message"]


def test_order_number_match_ignores_spaces_and_case():
    raw = target_json("ORDER", "cancel_order", "ord-12345")
    assert not check_output(raw, "cancel ORD- 12345").needs_review


def test_empty_order_number_is_flagged():
    assert check_output(target_json("ORDER", "cancel_order", " "), MSG).needs_review
