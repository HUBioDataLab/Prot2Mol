from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class LoadedEncoder:
    name_or_path: str
    tokenizer: Any
    model: torch.nn.Module
    hidden_size: int


def infer_hidden_size(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError(f"Model {type(model)} does not expose a config for hidden-size inference")

    for key in ("hidden_size", "d_model", "embed_dim", "n_embd", "dim"):
        value = getattr(config, key, None)
        if isinstance(value, int) and value > 0:
            return value

    raise ValueError(f"Could not infer hidden size from model config type {type(config)}")


def extract_last_hidden_state(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, (tuple, list)) and outputs:
        first = outputs[0]
        if isinstance(first, torch.Tensor):
            return first
    raise ValueError(f"Unsupported encoder output type: {type(outputs)}")


def encode_tokens(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return extract_last_hidden_state(outputs)


def batch_encode_texts(
    tokenizer: Any,
    texts: Sequence[str],
    max_length: int,
    device: Optional[torch.device] = None,
    add_special_tokens: bool = True,
    padding: str = "max_length",
    truncation: bool = True,
) -> Dict[str, torch.Tensor]:
    encode_kwargs = {
        "add_special_tokens": add_special_tokens,
        "padding": padding,
        "truncation": truncation,
        "max_length": max_length,
        "return_tensors": "pt",
    }
    text_list = list(texts)

    if callable(tokenizer):
        encoded = tokenizer(text_list, **encode_kwargs)
    else:
        encoded = tokenizer.batch_encode_plus(text_list, **encode_kwargs)

    if device is not None:
        encoded = {key: value.to(device) for key, value in encoded.items()}
    return encoded


def load_tokenizer(
    name_or_path: str,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
):
    resolved_kwargs = {
        "clean_up_tokenization_spaces": False,
        **(tokenizer_kwargs or {}),
    }
    return AutoTokenizer.from_pretrained(name_or_path, **resolved_kwargs)


def load_encoder_bundle(
    name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> LoadedEncoder:
    tokenizer = load_tokenizer(tokenizer_name_or_path or name_or_path, tokenizer_kwargs=tokenizer_kwargs)
    resolved_model_kwargs = {
        "add_pooling_layer": False,
        **(model_kwargs or {}),
    }
    model = AutoModel.from_pretrained(name_or_path, **resolved_model_kwargs)
    hidden_size = infer_hidden_size(model)
    return LoadedEncoder(
        name_or_path=name_or_path,
        tokenizer=tokenizer,
        model=model,
        hidden_size=hidden_size,
    )
