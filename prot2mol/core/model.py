"""Protein-conditioned MolGen model."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

from ..io.hf_utils import resolve_model_path
from .protein_encoders import get_protein_encoder


class Prot2MolModel(nn.Module):
    """Generate molecular SELFIES from full protein token representations.

    A sequence encoder produces per-residue states.  A learned projection and
    normalization bridge those states into the pretrained MolGen BART decoder,
    whose native cross-attention conditions every generated token on the protein.
    MolGen's unused molecular encoder layers are removed after loading.
    """

    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._config = dict(config)
        self._trainable_encoder = bool(config.get("train_encoder_model", False))
        self._trainable_projection = bool(config.get("train_projection_model", True))
        self._trainable_decoder = bool(config.get("train_decoder_model", True))

        self.protein_encoder = get_protein_encoder(
            model_name=config["prot_emb_model"],
            model_id=config.get("protein_model_id"),
            active=self._trainable_encoder,
        )

        decoder_model_id = str(config.get("decoder_model_id", "zjunlp/MolGen-large"))
        self.molecule_decoder = AutoModelForSeq2SeqLM.from_pretrained(
            resolve_model_path(decoder_model_id)
        )
        if not hasattr(self.molecule_decoder, "model") or not hasattr(
            self.molecule_decoder.model, "encoder"
        ):
            raise TypeError(f"{decoder_model_id} is not a BART-style encoder-decoder model")

        # The external protein representation is always supplied as
        # encoder_outputs. Retaining MolGen's 12-layer molecular encoder would
        # waste roughly half the checkpoint without participating in training.
        if hasattr(self.molecule_decoder.model.encoder, "layers"):
            self.molecule_decoder.model.encoder.layers = nn.ModuleList()

        tokenizer = config["mol_tokenizer"]
        tokenizer_size = len(tokenizer)
        if tokenizer_size != self.molecule_decoder.config.vocab_size:
            raise ValueError(
                f"Tokenizer vocabulary ({tokenizer_size}) does not match the pretrained "
                f"decoder vocabulary ({self.molecule_decoder.config.vocab_size}). "
                "Use the tokenizer from the same MolGen checkpoint."
            )
        self._validate_tokenizer_contract(tokenizer)

        decoder_dim = int(self.molecule_decoder.config.d_model)
        dropout = float(config.get("conditioning_dropout", 0.1))
        self.conditioning_projection = nn.Sequential(
            nn.Linear(self.protein_encoder.hidden_size, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.Dropout(dropout),
        )

        self._config.update(
            {
                "decoder_model_id": decoder_model_id,
                "protein_model_id": config.get("protein_model_id"),
                "decoder_hidden_size": decoder_dim,
                "protein_hidden_size": self.protein_encoder.hidden_size,
                "vocab_size": tokenizer_size,
            }
        )
        self.update_trainable_components(
            trainable_encoder=self._trainable_encoder,
            trainable_projection=self._trainable_projection,
            trainable_decoder=self._trainable_decoder,
        )
        self.logger.info("Prot2Mol parameters: %s", self.parameter_counts())

    @property
    def config(self):
        return self.molecule_decoder.config

    def _validate_tokenizer_contract(self, tokenizer) -> None:
        special_ids = {
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        missing = [name for name, value in special_ids.items() if value is None]
        if missing:
            raise ValueError(f"Molecule tokenizer is missing required tokens: {missing}")
        if len(set(special_ids.values())) != len(special_ids):
            raise ValueError("Molecule tokenizer pad, BOS, and EOS ids must be distinct")
        decoder_special_ids = {
            name: getattr(self.molecule_decoder.config, name)
            for name in special_ids
        }
        mismatched = {
            name: (special_ids[name], decoder_special_ids[name])
            for name in special_ids
            if special_ids[name] != decoder_special_ids[name]
        }
        if mismatched:
            raise ValueError(
                "Molecule tokenizer special-token ids do not match the pretrained "
                f"decoder config: {mismatched}"
            )
        if self.molecule_decoder.config.decoder_start_token_id is None:
            raise ValueError("Pretrained molecule decoder lacks decoder_start_token_id")

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad

    def update_trainable_components(
        self,
        trainable_encoder: bool,
        trainable_projection: bool = True,
        trainable_decoder: bool = True,
    ) -> None:
        self._trainable_encoder = bool(trainable_encoder)
        self._trainable_projection = bool(trainable_projection)
        self._trainable_decoder = bool(trainable_decoder)
        self._set_requires_grad(self.protein_encoder, self._trainable_encoder)
        self._set_requires_grad(self.conditioning_projection, self._trainable_projection)
        self._set_requires_grad(self.molecule_decoder, self._trainable_decoder)
        self._freeze_unused_molecule_encoder_parameters()
        self.train(self.training)

    def _freeze_unused_molecule_encoder_parameters(self) -> None:
        """Keep DDP from waiting for gradients on bypassed BART encoder weights."""

        shared_parameter_ids = {
            id(parameter) for parameter in self.molecule_decoder.model.shared.parameters()
        }
        for parameter in self.molecule_decoder.model.encoder.parameters():
            if id(parameter) not in shared_parameter_ids:
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.protein_encoder.train(mode and self._trainable_encoder)
        self.conditioning_projection.train(mode and self._trainable_projection)
        self.molecule_decoder.train(mode and self._trainable_decoder)
        return self

    def encode_protein(
        self,
        prot_input_ids: torch.Tensor,
        prot_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        gradients_enabled = self._trainable_encoder and torch.is_grad_enabled()
        with torch.set_grad_enabled(gradients_enabled):
            hidden = self.protein_encoder(prot_input_ids, prot_attention_mask)
        return self.conditioning_projection(hidden)

    def _encoder_outputs(self, protein_embeddings: torch.Tensor) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=protein_embeddings)

    def forward(
        self,
        prot_input_ids: torch.Tensor,
        prot_attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        protein_embeddings = self.encode_protein(prot_input_ids, prot_attention_mask)

        # BART performs its canonical shift-right operation from labels.
        # Passing the unshifted molecule tokens as decoder_input_ids here would
        # leak each target token into its own prediction.
        outputs = self.molecule_decoder(
            input_ids=None,
            attention_mask=prot_attention_mask,
            encoder_outputs=self._encoder_outputs(protein_embeddings),
            decoder_input_ids=None,
            decoder_attention_mask=None,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        result = {"logits": outputs.logits}
        if outputs.loss is not None:
            result["lm_loss"] = outputs.loss
            result["loss"] = outputs.loss
        return result

    def generate_from_protein_embeddings(
        self,
        protein_embeddings: torch.Tensor,
        prot_attention_mask: torch.Tensor,
        **generation_kwargs,
    ) -> torch.Tensor:
        generation_kwargs.setdefault("pad_token_id", self.config.pad_token_id)
        generation_kwargs.setdefault("bos_token_id", self.config.bos_token_id)
        generation_kwargs.setdefault("eos_token_id", self.config.eos_token_id)
        generation_kwargs.setdefault(
            "decoder_start_token_id", self.config.decoder_start_token_id
        )
        return self.molecule_decoder.generate(
            encoder_outputs=self._encoder_outputs(protein_embeddings),
            attention_mask=prot_attention_mask,
            **generation_kwargs,
        )

    def generate(
        self,
        prot_input_ids: torch.Tensor,
        prot_attention_mask: torch.Tensor,
        protein_embeddings: torch.Tensor | None = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        if protein_embeddings is None:
            protein_embeddings = self.encode_protein(prot_input_ids, prot_attention_mask)
        return self.generate_from_protein_embeddings(
            protein_embeddings,
            prot_attention_mask,
            **generation_kwargs,
        )

    def get_encoder_hidden_states(self, prot_input_ids, prot_attention_mask):
        return self.encode_protein(prot_input_ids, prot_attention_mask)

    def get_encoder(self):
        return self.protein_encoder

    def get_decoder(self):
        return self.molecule_decoder

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def num_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def parameter_counts(self) -> dict[str, int]:
        return {
            "protein_encoder": sum(p.numel() for p in self.protein_encoder.parameters()),
            "conditioning_projection": sum(
                p.numel() for p in self.conditioning_projection.parameters()
            ),
            "molecule_decoder": sum(p.numel() for p in self.molecule_decoder.parameters()),
            "molecule_decoder_trainable": sum(
                p.numel() for p in self.molecule_decoder.parameters() if p.requires_grad
            ),
            "total": self.num_parameters(),
            "trainable": self.num_trainable_parameters(),
        }


def create_prot2mol_model(config: Mapping[str, Any]) -> Prot2MolModel:
    return Prot2MolModel(config)
