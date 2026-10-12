from types import SimpleNamespace

from triage.baseline import predict_raw, regex_order_number, train_classifier
from triage.prompts import SYSTEM_PROMPT, chat_messages, ft_prompt
from triage.providers import MockProvider, OpenAIProvider, pick_examples
from triage.schema import check_output


def test_mock_provider_writes_valid_records(examples):
    provider = MockProvider()
    for ex, raw in zip(examples, provider.complete([e["message"] for e in examples])):
        checked = check_output(raw, ex["message"])
        assert not checked.needs_review


def test_mock_provider_is_deterministic():
    provider = MockProvider()
    assert provider.complete(["cancel order #12345"]) == provider.complete(["cancel order #12345"])


def test_mock_provider_known_cases():
    raw = MockProvider().complete(["I forgot my password", "please cancel order ORD-55555"])
    assert '"recover_password"' in raw[0] and '"order_number": null' in raw[0]
    assert '"cancel_order"' in raw[1] and "ORD-55555" in raw[1]


def test_regex_order_number():
    assert regex_order_number("order #48213 is late") == "#48213"
    assert regex_order_number("no number here") is None


def test_classic_baseline_learns_the_fixture(examples):
    model = train_classifier(examples * 3)
    raws = predict_raw(model, [e["message"] for e in examples])
    assert [check_output(r, e["message"]).record.intent for r, e in zip(raws, examples)] == [
        e["intent"] for e in examples
    ]


def test_prompts():
    assert ft_prompt(" hello ") == "### Message\nhello\n### Record\n"
    assert "cancel_order" in SYSTEM_PROMPT and "SUBSCRIPTION" in SYSTEM_PROMPT
    turns = chat_messages("hi", [{"message": "m", "target": "t"}])
    assert [t["role"] for t in turns] == ["system", "user", "assistant", "user"]


def test_pick_examples_mixes_intents_and_order_numbers(examples):
    picked = pick_examples(examples, k=4)
    assert len(picked) == 4
    assert len({p["intent"] for p in picked}) == 4
    assert sum(1 for p in picked if p["order_number"]) == 1


class FakeOpenAI:
    """Stands in for the OpenAI client so the hosted path is tested with no key and no cost."""

    def __init__(self, content: str):
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._content = content

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))],
            usage=SimpleNamespace(prompt_tokens=300, completion_tokens=20),
        )


def test_hosted_provider_sends_capped_request_and_counts_tokens():
    client = FakeOpenAI('{"category": "ORDER", "intent": "cancel_order", "order_number": null}')
    provider = OpenAIProvider("some-model", client=client)
    out = provider.complete(["cancel my order", "cancel it"])
    assert len(out) == 2 and len(client.calls) == 2
    assert client.calls[0]["max_tokens"] == 60 and client.calls[0]["temperature"] == 0
    assert client.calls[0]["messages"][0]["role"] == "system"
    assert (provider.input_tokens, provider.output_tokens) == (600, 40)
