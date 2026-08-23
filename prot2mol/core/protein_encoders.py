"""Protein encoder loading and sequence tokenization contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    T5EncoderModel,
    T5Tokenizer,
)

from ..io.hf_utils import resolve_model_path


_NON_STANDARD_AMINO_ACIDS = re.compile(r"[UZOB]")


@dataclass(frozen=True)
class ProteinEncoderSpec:
    model_id: str
    family: str


PROTEIN_ENCODERS = {
    "esm2": ProteinEncoderSpec(
        model_id="facebook/esm2_t33_650M_UR50D",
        family="esm",
    ),
    "prot_t5": ProteinEncoderSpec(
        model_id="Rostlab/prot_t5_xl_uniref50",
        family="t5",
    ),
}


class ProteinEncoder(nn.Module):
    """Thin registered wrapper around a Hugging Face protein encoder."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @property
    def hidden_size(self) -> int:
        config = self.model.config
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(config, "d_model", None)
        if hidden_size is None:
            raise ValueError(f"Cannot infer hidden size from {type(config).__name__}")
        return int(hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            return hidden_states[-1]
        raise ValueError(f"Unsupported encoder output type: {type(outputs).__name__}")


def resolve_protein_model_id(model_name: str, model_id: Optional[str] = None) -> str:
    if model_name not in PROTEIN_ENCODERS:
        choices = ", ".join(sorted(PROTEIN_ENCODERS))
        raise ValueError(f"Unsupported protein encoder '{model_name}'. Choose one of: {choices}")
    return model_id or PROTEIN_ENCODERS[model_name].model_id


def get_protein_encoder(
    model_name: str,
    model_id: Optional[str] = None,
    active: bool = True,
    *,
    pretrained: bool = True,
    revision: str | None = None,
    models_base: str | None = None,
    local_files_only: bool = False,
) -> ProteinEncoder:
    """Build the selected encoder, optionally without downloading base weights.

    Full Prot2Mol checkpoints already contain the protein encoder. In that case
    ``pretrained=False`` reconstructs only the pinned architecture before the
    caller loads the checkpoint state, avoiding a redundant 650M-weight load.
    """

    resolved_id = resolve_protein_model_id(model_name, model_id)
    model_path = resolve_model_path(
        resolved_id,
        models_base=models_base,
        revision=revision,
    )
    common_kwargs = {}
    if revision is not None:
        common_kwargs["revision"] = revision
    if models_base is not None:
        common_kwargs["cache_dir"] = models_base
    if local_files_only:
        common_kwargs["local_files_only"] = True
    if PROTEIN_ENCODERS[model_name].family == "t5":
        if pretrained:
            model = T5EncoderModel.from_pretrained(model_path, **common_kwargs)
        else:
            config = AutoConfig.from_pretrained(model_path, **common_kwargs)
            model = T5EncoderModel(config)
    else:
        if pretrained:
            model = AutoModel.from_pretrained(model_path, **common_kwargs)
        else:
            config = AutoConfig.from_pretrained(model_path, **common_kwargs)
            model = AutoModel.from_config(config)
    encoder = ProteinEncoder(model)
    for parameter in encoder.parameters():
        parameter.requires_grad = bool(active)
    encoder.train(bool(active))
    return encoder


def get_protein_tokenizer(
    model_name: str,
    model_id: Optional[str] = None,
    *,
    revision: str | None = None,
    models_base: str | None = None,
    local_files_only: bool = False,
):
    """Lazily load only the tokenizer matching the selected encoder."""

    resolved_id = resolve_protein_model_id(model_name, model_id)
    model_path = resolve_model_path(
        resolved_id,
        models_base=models_base,
        revision=revision,
    )
    common_kwargs = {}
    if revision is not None:
        common_kwargs["revision"] = revision
    if models_base is not None:
        common_kwargs["cache_dir"] = models_base
    if local_files_only:
        common_kwargs["local_files_only"] = True
    if PROTEIN_ENCODERS[model_name].family == "t5":
        tokenizer = T5Tokenizer.from_pretrained(
            model_path,
            do_lower_case=False,
            legacy=True,
            clean_up_tokenization_spaces=True,
            **common_kwargs,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, **common_kwargs)
    tokenizer.padding_side = "right"
    return tokenizer


def format_protein_sequence(sequence: str, model_name: str) -> str:
    """Normalize plain FASTA for the selected sequence encoder."""

    sequence = str(sequence).strip().upper()
    cleaned = _NON_STANDARD_AMINO_ACIDS.sub("X", sequence)
    if model_name == "prot_t5":
        return " ".join(cleaned)
    return cleaned


def format_protein_sequences(sequences: Iterable[str], model_name: str) -> List[str]:
    return [format_protein_sequence(sequence, model_name) for sequence in sequences]
