"""Loading the model and generating text on a CPU."""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_threads(threads: int) -> None:
    if threads > 0:
        torch.set_num_threads(threads)


def load_tokenizer(name_or_path: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(str(name_or_path))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding keeps the end of every prompt next to the first generated token,
    # which batched generation with a decoder-only model requires.
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(base_model: str | Path, adapter_dir: str | Path | None = None, merge: bool = True):
    """Load the base model, and the LoRA add-on if a folder is given.

    With merge=True the add-on is folded into the base weights, so serving costs exactly
    the same as the untouched model: no extra layers at inference time.
    """
    model = AutoModelForCausalLM.from_pretrained(str(base_model), dtype=torch.float32)
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
        if merge:
            model = model.merge_and_unload()
    model.eval()
    return model


def has_adapter(adapter_dir: str | Path) -> bool:
    return (Path(adapter_dir) / "adapter_config.json").exists()


@torch.inference_mode()
def generate(
    model, tokenizer, prompts: list[str], max_new_tokens: int = 48, batch_size: int = 16
) -> list[str]:
    """Greedy generation in batches. Greedy, because routing must give the same answer twice."""
    outputs: list[str] = []
    # Sorting by length puts similar prompts in one batch, so little time is spent on padding.
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    results: dict[int, str] = {}
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = tokenizer(
            [prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False
        )
        generated = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = generated[:, batch["input_ids"].shape[1] :]
        for i, text in zip(idx, tokenizer.batch_decode(new_tokens, skip_special_tokens=True)):
            results[i] = text.strip()
    outputs = [results[i] for i in range(len(prompts))]
    return outputs


def render_chat(tokenizer, turns: list[dict]) -> str:
    """Turn chat messages into the prompt text the base model was trained to expect."""
    return tokenizer.apply_chat_template(turns, tokenize=False, add_generation_prompt=True)
