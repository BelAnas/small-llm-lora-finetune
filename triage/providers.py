"""The systems that can route a message. Each takes messages and returns raw text per message.

They all return text in the same format, so the checks in schema.py and the scorer in
metrics.py treat a rule, a small local model and a hosted model in exactly the same way.
"""
from __future__ import annotations

import random
import re
from typing import Protocol

from .baseline import regex_order_number
from .config import Settings
from .labels import INTENT_TO_CATEGORY
from .prompts import chat_messages, ft_prompt
from .schema import target_json


class Provider(Protocol):
    name: str

    def complete(self, messages: list[str]) -> list[str]: ...


# Ordered keyword rules: the first one that matches wins, so specific rules come first.
_RULES: list[tuple[str, str]] = [
    (r"refund.*(status|where|track|still waiting)|track.*refund", "track_refund"),
    (r"refund polic|money back guarantee|conditions.*refund", "check_refund_policy"),
    (r"refund|reimburs|money back", "get_refund"),
    (r"password|pass ?word|locked out", "recover_password"),
    (r"human|real person|live agent|speak (to|with) (a|an|some)", "contact_human_agent"),
    (r"customer (service|support|care)|support team", "contact_customer_service"),
    (r"newsletter|mailing list", "newsletter_subscription"),
    (r"cancel\w* (fee|charge|penalt)|penalt", "check_cancellation_fee"),
    (r"invoice|bill\b", "get_invoice"),
    (r"(change|update|correct|wrong).*address", "change_shipping_address"),
    (r"address", "set_up_shipping_address"),
    (r"cancel", "cancel_order"),
    (r"where is|track|status of", "track_order"),
    (r"(add|remove|swap|change).*(order|item)", "change_order"),
    (r"payment (method|option)|ways of paying|accept", "check_payment_methods"),
    (r"payment|charged|card", "payment_issue"),
    (r"complain|unhappy|worst", "complaint"),
    (r"feedback|review", "review"),
    (r"(how long|when).*(deliver|arriv|ship)", "delivery_period"),
    (r"deliver|shipping", "delivery_options"),
    (r"(delete|close|erase).*(account|profile)", "delete_account"),
    (r"(sign.?up|regist).*(error|problem|fail)", "registration_problems"),
    (r"sign.?up|register|create.*(account|profile)", "create_account"),
    (r"(switch|upgrade|move).*(account|plan)", "switch_account"),
    (r"account|profile", "edit_account"),
    (r"buy|order|purchase", "place_order"),
]


class MockProvider:
    """Keyword rules, no model. It exists so the tests and the API run with nothing downloaded.

    It is not a baseline anyone should ship: its accuracy is reported only to make clear
    that it is a stand-in.
    """

    name = "mock"

    def complete(self, messages: list[str]) -> list[str]:
        out = []
        for message in messages:
            text = message.lower()
            intent = next((i for pattern, i in _RULES if re.search(pattern, text)), "contact_customer_service")
            out.append(target_json(INTENT_TO_CATEGORY[intent], intent, regex_order_number(message)))
        return out


class LocalProvider:
    """A model on this machine: the fine-tuned one, or the base model driven by a prompt."""

    def __init__(
        self,
        model,
        tokenizer,
        mode: str = "lora",
        examples: list[dict] | None = None,
        max_new_tokens: int = 48,
        batch_size: int = 16,
    ):
        self.model, self.tokenizer, self.mode = model, tokenizer, mode
        self.examples = examples or []
        self.max_new_tokens, self.batch_size = max_new_tokens, batch_size
        self.name = {"lora": "lora", "zero": "base-prompt", "few": "base-examples"}[mode]

    def prompt(self, message: str) -> str:
        from .model import render_chat

        if self.mode == "lora":
            return ft_prompt(message)
        return render_chat(self.tokenizer, chat_messages(message, self.examples if self.mode == "few" else None))

    def complete(self, messages: list[str]) -> list[str]:
        from .model import generate

        prompts = [self.prompt(m) for m in messages]
        return generate(self.model, self.tokenizer, prompts, self.max_new_tokens, self.batch_size)


class OpenAIProvider:
    """A hosted model with the same instructions, for a cost and accuracy comparison."""

    name = "hosted"

    def __init__(self, model: str, client=None, max_tokens: int = 60):
        if client is None:
            from openai import OpenAI

            client = OpenAI()  # reads OPENAI_API_KEY from the environment
        self.client, self.model, self.max_tokens = client, model, max_tokens
        self.input_tokens = self.output_tokens = 0

    def complete(self, messages: list[str]) -> list[str]:
        out = []
        for message in messages:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=chat_messages(message),
                max_tokens=self.max_tokens,
                temperature=0,
                response_format={"type": "json_object"},
            )
            usage = getattr(response, "usage", None)
            if usage is not None:
                self.input_tokens += usage.prompt_tokens
                self.output_tokens += usage.completion_tokens
            out.append(response.choices[0].message.content or "")
        return out


def pick_examples(train: list[dict], k: int = 8, seed: int = 7) -> list[dict]:
    """Worked examples for the prompt: different intents, a quarter of them with an order number."""
    rng = random.Random(seed)
    pool = train[:]
    rng.shuffle(pool)
    want_order = max(1, k // 4)
    picked: list[dict] = []
    seen: set[str] = set()
    for ex in pool:
        need_order = sum(1 for p in picked if p["order_number"]) < want_order
        if ex["intent"] in seen or (need_order and not ex["order_number"]):
            continue
        if not need_order and ex["order_number"]:
            continue
        picked.append(ex)
        seen.add(ex["intent"])
        if len(picked) == k:
            break
    return picked


def build_provider(settings: Settings) -> Provider:
    """The provider named in the settings. Used by the API and the command line."""
    if settings.provider == "mock":
        return MockProvider()
    if settings.provider == "openai":
        return OpenAIProvider(settings.openai_model)
    if settings.provider == "local":
        from .model import has_adapter, load_model, load_tokenizer, set_threads

        if not has_adapter(settings.adapter_dir):
            raise RuntimeError(
                f"No LoRA add-on in {settings.adapter_dir}. Run `python -m triage.cli train` first, "
                "or set TRIAGE_PROVIDER=mock."
            )
        set_threads(settings.threads)
        tokenizer = load_tokenizer(settings.base_model)
        model = load_model(settings.base_model, settings.adapter_dir, merge=True)
        return LocalProvider(model, tokenizer, "lora", max_new_tokens=settings.max_new_tokens)
    raise ValueError(f"Unknown TRIAGE_PROVIDER '{settings.provider}' (use local, mock or openai)")
