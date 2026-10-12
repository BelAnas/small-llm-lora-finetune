"""LoRA fine-tuning on a CPU.

The base model stays frozen. LoRA adds two small matrices next to chosen weight matrices and
trains only those, which is why this fits on a laptop and the result is a file of a few megabytes.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from .prompts import ft_prompt


@dataclass
class TrainConfig:
    rank: int = 16  # size of the add-on; higher learns more and costs more
    alpha: int = 32  # scaling of the add-on, kept at 2 x rank by convention
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    epochs: float = 2.0
    batch_size: int = 16
    learning_rate: float = 5e-4
    warmup_steps: int = 30
    max_length: int = 160
    eval_every: int = 100
    seed: int = 7


def encode_example(tokenizer, message: str, target: str, max_length: int) -> dict[str, list[int]]:
    """Token ids for prompt + answer, with the prompt masked out of the loss.

    The model should learn to write the record, not to write customer messages, so only the
    answer tokens (and the end-of-text token that teaches it to stop) carry a label.
    """
    answer = tokenizer(target, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    prompt = tokenizer(ft_prompt(message), add_special_tokens=False)["input_ids"]
    # A very long message is cut at the front of the prompt budget, never in the answer.
    prompt = prompt[: max(1, max_length - len(answer))]
    return {"input_ids": prompt + answer, "labels": [-100] * len(prompt) + answer}


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad a batch. Padding is ignored by both attention and the loss."""
    width = max(len(item["input_ids"]) for item in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, item in enumerate(batch):
        n = len(item["input_ids"])
        ids[row, :n] = torch.tensor(item["input_ids"])
        labels[row, :n] = torch.tensor(item["labels"])
        mask[row, :n] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def add_lora(model, cfg: TrainConfig):
    lora = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


def count_parameters(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate_loss(model, encoded: list[dict], pad_id: int, batch_size: int) -> float:
    model.eval()
    total, tokens = 0.0, 0
    for start in range(0, len(encoded), batch_size):
        batch = collate(encoded[start : start + batch_size], pad_id)
        n = int((batch["labels"] != -100).sum())
        total += model(**batch).loss.item() * n
        tokens += n
    model.train()
    return total / max(tokens, 1)


def train(
    model,
    tokenizer,
    train_rows: list[dict],
    val_rows: list[dict],
    out_dir: Path,
    cfg: TrainConfig | None = None,
    log=print,
) -> dict:
    """Train the LoRA add-on, save it to out_dir, and return a summary with the loss log."""
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    model = add_lora(model, cfg)
    trainable, total = count_parameters(model)
    pad_id = tokenizer.pad_token_id
    enc_train = [encode_example(tokenizer, r["message"], r["target"], cfg.max_length) for r in train_rows]
    enc_val = [encode_example(tokenizer, r["message"], r["target"], cfg.max_length) for r in val_rows]

    steps_per_epoch = math.ceil(len(enc_train) / cfg.batch_size)
    total_steps = max(1, int(steps_per_epoch * cfg.epochs))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.learning_rate)

    def lr_at(step: int) -> float:
        # Short warm-up, then a cosine decay to zero: the usual schedule for fine-tuning.
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    history: list[dict] = [
        {"step": 0, "val_loss": round(evaluate_loss(model, enc_val, pad_id, cfg.batch_size), 4)}
    ]
    log(f"trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")
    log(f"steps: {total_steps}, validation loss before training: {history[0]['val_loss']}")

    model.train()
    started = time.time()
    step, running, order = 0, [], []
    while step < total_steps:
        if not order:
            order = list(range(len(enc_train)))
            rng.shuffle(order)
        picked, order = order[: cfg.batch_size], order[cfg.batch_size :]
        batch = collate([enc_train[i] for i in picked], pad_id)
        loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        running.append(loss.item())
        step += 1
        if step % cfg.eval_every == 0 or step == total_steps:
            entry = {
                "step": step,
                "train_loss": round(sum(running) / len(running), 4),
                "val_loss": round(evaluate_loss(model, enc_val, pad_id, cfg.batch_size), 4),
                "minutes": round((time.time() - started) / 60, 1),
            }
            running = []
            history.append(entry)
            log(json.dumps(entry))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    summary = {
        "config": {**asdict(cfg), "target_modules": list(cfg.target_modules)},
        "train_examples": len(enc_train),
        "val_examples": len(enc_val),
        "steps": total_steps,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_pct": round(100 * trainable / total, 2),
        "training_minutes": round((time.time() - started) / 60, 1),
        "cpu_threads": torch.get_num_threads(),
        "history": history,
    }
    (out_dir / "training_log.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
