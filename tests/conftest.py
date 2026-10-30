"""Shared fixtures.

Unit tests use a tiny randomly initialised Switch Transformer so they run on
CPU in seconds and never download the full model. Tests marked
``integration`` use the real google/switch-base-8 and are skipped unless it is
already in the local Hugging Face cache.
"""

from __future__ import annotations

import os

import pytest
import torch
from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

REAL_MODEL_ID = "google/switch-base-8"


def tiny_switch_config(**overrides) -> SwitchTransformersConfig:
    params = dict(
        vocab_size=96,
        d_model=16,
        d_kv=4,
        d_ff=32,
        num_heads=2,
        num_layers=4,
        num_decoder_layers=4,
        num_experts=4,
        expert_capacity=64,
        encoder_sparse_step=2,
        decoder_sparse_step=2,
        dropout_rate=0.0,
        router_jitter_noise=0.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    params.update(overrides)
    return SwitchTransformersConfig(**params)


def make_tiny_model(seed: int = 0, **overrides) -> SwitchTransformersForConditionalGeneration:
    torch.manual_seed(seed)
    model = SwitchTransformersForConditionalGeneration(tiny_switch_config(**overrides))
    model.eval()
    return model


@pytest.fixture
def tiny_model():
    return make_tiny_model()


@pytest.fixture
def tiny_batch():
    """Right-padded batch: second sequence is shorter in encoder and decoder."""
    g = torch.Generator().manual_seed(1)
    input_ids = torch.randint(2, 96, (2, 10), generator=g)
    attention_mask = torch.ones_like(input_ids)
    input_ids[1, 7:] = 0
    attention_mask[1, 7:] = 0
    labels = torch.randint(2, 96, (2, 6), generator=g)
    labels[1, 4:] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def real_model_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(REAL_MODEL_ID, "config.json")
        return isinstance(path, str)
    except Exception:
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: uses the real google/switch-base-8 model")


requires_real_model = pytest.mark.skipif(
    not real_model_cached(), reason=f"{REAL_MODEL_ID} not found in the local Hugging Face cache"
)


@pytest.fixture
def tiny_tokenizer():
    """Word-level tokenizer built locally (no download), compatible with save/load_pretrained."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2}
    vocab.update({f"w{i}": i for i in range(3, 96)})
    tok = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="</s>", unk_token="<unk>")


def make_tiny_olmoe(seed: int = 0, sharpen_router: float = 40.0, **overrides):
    """Tiny random OLMoE (8 experts, top-2). The router is sharpened so gates are distinct."""
    from transformers import OlmoeConfig, OlmoeForCausalLM

    params = dict(
        vocab_size=96,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=False,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=None,
        tie_word_embeddings=False,
    )
    params.update(overrides)
    torch.manual_seed(seed)
    model = OlmoeForCausalLM(OlmoeConfig(**params)).eval()
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.gate.weight.mul_(sharpen_router)
    return model


@pytest.fixture
def tiny_olmoe():
    return make_tiny_olmoe()
