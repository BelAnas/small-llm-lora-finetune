from fastapi.testclient import TestClient

from triage.api import create_app
from triage.providers import MockProvider
from triage.service import route


class Fixed:
    """A provider that always writes the same text, to drive the API into each branch."""

    name = "fixed"

    def __init__(self, text: str):
        self.text = text

    def complete(self, messages):
        return [self.text for _ in messages]


def test_health():
    client = TestClient(create_app(MockProvider()))
    assert client.get("/health").json() == {"status": "ok", "provider": "mock"}


def test_route_returns_a_checked_record():
    client = TestClient(create_app(MockProvider()))
    body = client.post("/route", json={"message": "Please cancel order ORD-12345"}).json()
    assert body["intent"] == "cancel_order" and body["category"] == "ORDER"
    assert body["order_number"] == "ORD-12345"
    assert body["needs_review"] is False and body["problems"] == []


def test_invented_order_number_goes_to_review():
    provider = Fixed('{"category": "ORDER", "intent": "cancel_order", "order_number": "ORD-99999"}')
    body = TestClient(create_app(provider)).post("/route", json={"message": "cancel my order"}).json()
    assert body["needs_review"] is True
    assert body["problems"] == ["order number does not appear in the message"]
    assert body["order_number"] == "ORD-99999"  # shown to the reviewer, not acted on


def test_unusable_output_goes_to_review():
    body = TestClient(create_app(Fixed("I am just a model"))).post("/route", json={"message": "hi"}).json()
    assert body["needs_review"] is True and body["intent"] is None


def test_empty_message_is_rejected():
    client = TestClient(create_app(MockProvider()))
    assert client.post("/route", json={"message": ""}).status_code == 422


def test_overlong_message_is_rejected():
    client = TestClient(create_app(MockProvider()))
    assert client.post("/route", json={"message": "x" * 2001}).status_code == 422


def test_service_reports_latency_and_provider():
    routed = route(MockProvider(), "I forgot my password")
    assert routed.provider == "mock" and routed.latency_ms >= 0
    assert routed.intent == "recover_password"
