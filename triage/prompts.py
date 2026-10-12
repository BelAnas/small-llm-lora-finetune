"""The prompts. The fine-tuned model gets a short one; the untouched model needs the full rules."""
from __future__ import annotations

from .labels import CATEGORIES, INTENT_TO_CATEGORY

# After fine-tuning the model has learned the task, so the prompt only marks where the
# message ends and the record begins. A short prompt is also faster on every request.
FT_TEMPLATE = "### Message\n{message}\n### Record\n"


def ft_prompt(message: str) -> str:
    return FT_TEMPLATE.format(message=message.strip())


def _label_block() -> str:
    lines = []
    for category in CATEGORIES:
        intents = [i for i, c in INTENT_TO_CATEGORY.items() if c == category]
        lines.append(f"{category}: {', '.join(intents)}")
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You route customer support messages. Read the message and answer with one JSON object "
    'and nothing else, in this form: {"category": "...", "intent": "...", "order_number": ...}\n'
    "Pick the category and one of its intents from this list (CATEGORY: intents):\n"
    f"{_label_block()}\n"
    "order_number is the order number written in the message, copied exactly, or null if the "
    "message has none."
)


def chat_messages(message: str, examples: list[dict] | None = None) -> list[dict]:
    """Chat turns for a model that was not fine-tuned. `examples` adds worked examples."""
    turns = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples or []:
        turns.append({"role": "user", "content": ex["message"]})
        turns.append({"role": "assistant", "content": ex["target"]})
    turns.append({"role": "user", "content": message.strip()})
    return turns
