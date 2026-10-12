"""Shared fixtures. Nothing here touches the network: the model used in tests is built in memory."""
from __future__ import annotations

import pytest

from triage.data import make_example

MESSAGES = [
    ("I want to cancel order ORD-11111 please", "cancel_order", "ORD-11111"),
    ("where is my order #22222", "track_order", "#22222"),
    ("I forgot my password", "recover_password", None),
    ("how do I get a refund", "get_refund", None),
    ("please send me my invoice", "get_invoice", None),
    ("I need to talk to a human agent", "contact_human_agent", None),
]


@pytest.fixture(scope="session")
def examples() -> list[dict]:
    return [make_example(m, i, o) for m, i, o in MESSAGES]


@pytest.fixture(scope="session")
def tiny_tokenizer(examples):
    """A byte-pair tokenizer trained in memory on the fixture text. No download."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = [e["message"] for e in examples] + [e["target"] for e in examples] + ["### Message\n### Record\n"]
    trainer = trainers.BpeTrainer(
        vocab_size=400, special_tokens=["<unk>", "<pad>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    fast.padding_side = "left"
    return fast


@pytest.fixture()
def tiny_model(tiny_tokenizer):
    """A two-layer model with random weights and the same architecture family as SmolLM2."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(tiny_tokenizer), hidden_size=48, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
        pad_token_id=tiny_tokenizer.pad_token_id, eos_token_id=tiny_tokenizer.eos_token_id,
        bos_token_id=tiny_tokenizer.eos_token_id,
    )
    return LlamaForCausalLM(config)
