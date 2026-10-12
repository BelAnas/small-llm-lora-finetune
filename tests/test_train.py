"""The training code, run for real on a tiny random model so it needs no download."""
import json

import torch

from triage.model import generate, has_adapter, load_model
from triage.prompts import ft_prompt
from triage.train import TrainConfig, add_lora, collate, count_parameters, encode_example, train

CFG = TrainConfig(rank=4, alpha=8, epochs=30, batch_size=6, learning_rate=1e-2, warmup_steps=2, eval_every=10)


def test_only_the_answer_carries_a_label(tiny_tokenizer, examples):
    ex = examples[0]
    enc = encode_example(tiny_tokenizer, ex["message"], ex["target"], 160)
    n_prompt = len(tiny_tokenizer(ft_prompt(ex["message"]), add_special_tokens=False)["input_ids"])
    assert enc["labels"][:n_prompt] == [-100] * n_prompt
    assert enc["labels"][n_prompt:] == enc["input_ids"][n_prompt:]
    assert enc["input_ids"][-1] == tiny_tokenizer.eos_token_id  # the model must learn to stop


def test_long_message_is_cut_but_the_answer_is_kept(tiny_tokenizer, examples):
    ex = examples[0]
    enc = encode_example(tiny_tokenizer, ex["message"] * 50, ex["target"], 60)
    answer = tiny_tokenizer(ex["target"], add_special_tokens=False)["input_ids"]
    assert len(enc["input_ids"]) == 60
    assert enc["input_ids"][-len(answer) - 1 : -1] == answer


def test_collate_pads_and_masks(tiny_tokenizer, examples):
    encoded = [encode_example(tiny_tokenizer, e["message"], e["target"], 160) for e in examples[:3]]
    batch = collate(encoded, tiny_tokenizer.pad_token_id)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    for row, enc in enumerate(encoded):
        n = len(enc["input_ids"])
        assert batch["attention_mask"][row].sum() == n
        assert (batch["labels"][row, n:] == -100).all()


def test_lora_trains_only_a_small_part(tiny_model):
    before = sum(p.numel() for p in tiny_model.parameters())
    peft_model = add_lora(tiny_model, CFG)
    trainable, total = count_parameters(peft_model)
    assert 0 < trainable < 0.2 * total
    assert total == before + trainable
    names = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)


def test_training_lowers_the_loss_and_leaves_the_base_untouched(tiny_model, tiny_tokenizer, examples, tmp_path):
    frozen_before = tiny_model.model.embed_tokens.weight.detach().clone()
    summary = train(tiny_model, tiny_tokenizer, examples, examples, tmp_path / "adapter", CFG, log=lambda _: None)
    history = summary["history"]
    assert history[-1]["val_loss"] < history[0]["val_loss"] - 0.1
    assert torch.equal(frozen_before, tiny_model.model.embed_tokens.weight)
    assert has_adapter(tmp_path / "adapter")
    log = json.loads((tmp_path / "adapter" / "training_log.json").read_text())
    assert log["steps"] == 30 and log["trainable_pct"] < 20


def test_saved_adapter_reloads_and_merging_changes_nothing(tiny_model, tiny_tokenizer, examples, tmp_path):
    base_dir = tmp_path / "base"
    tiny_model.save_pretrained(base_dir)
    train(tiny_model, tiny_tokenizer, examples, examples, tmp_path / "adapter", CFG, log=lambda _: None)

    prompts = [ft_prompt(e["message"]) for e in examples]
    unmerged = generate(load_model(base_dir, tmp_path / "adapter", merge=False), tiny_tokenizer, prompts, 24, 4)
    merged = generate(load_model(base_dir, tmp_path / "adapter", merge=True), tiny_tokenizer, prompts, 24, 4)
    base = generate(load_model(base_dir), tiny_tokenizer, prompts, 24, 4)
    assert merged == unmerged  # folding the add-on into the weights does not change the answers
    assert merged != base  # and the add-on does change what the base model writes


def test_batched_generation_matches_one_at_a_time(tiny_model, tiny_tokenizer, examples):
    tiny_model.eval()
    prompts = [ft_prompt(e["message"]) for e in examples]
    together = generate(tiny_model, tiny_tokenizer, prompts, 12, batch_size=6)
    alone = [generate(tiny_model, tiny_tokenizer, [p], 12, batch_size=1)[0] for p in prompts]
    assert together == alone
