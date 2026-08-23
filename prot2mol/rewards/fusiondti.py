"""Frozen FusionDTI binary-activity scorer for GRPO ablations.

This is a checkpoint-compatible reimplementation of the CAN inference path
published in the FusionDTI Hugging Face Space.  It intentionally preserves the
published ablation contract: SaProt masked-LM logits (446 channels), SELFormer
hidden states (768 channels), and a binary sigmoid output.

Upstream source: https://github.com/ZhaohanM/FusionDTI (MIT Space artifacts).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn


FUSIONDTI_SPACE_ID = "Zhaohan-Meng/FusionDTI"
FUSIONDTI_SPACE_REVISION = "5bc15620b46d2870d0208a084f7e1b2c4bb98a6a"
FUSIONDTI_PROTEIN_MODEL_ID = "westlake-repl/SaProt_650M_AF2"
FUSIONDTI_PROTEIN_MODEL_REVISION = "d9b9ad00ef61c0990e611b2b43f2231c7de24b38"
FUSIONDTI_MOLECULE_MODEL_ID = "HUBioDataLab/SELFormer"
FUSIONDTI_MOLECULE_MODEL_REVISION = "177d98b158e999a6cb7fc9743dbfe1e8a17c57e5"


@dataclass(frozen=True)
class _CheckpointSpec:
    dataset: str
    filename: str
    sha256: str


_CHECKPOINTS = {
    "bindingdb": _CheckpointSpec(
        dataset="BindingDB",
        filename="save_model_ckp/BindingDB_CAN/best_model.ckpt",
        sha256="7984a2e11cc3b577720755d24e188763112008f9bd8f9342808aa12ad80a1153",
    ),
    "biosnap": _CheckpointSpec(
        dataset="Biosnap",
        filename="save_model_ckp/Biosnap_CAN/best_model.ckpt",
        sha256="351bec96ac3e43f7e98b7c62a03fe178a6e2556e2422b1bf948e07204ae2825e",
    ),
    "human": _CheckpointSpec(
        dataset="Human",
        filename="save_model_ckp/Human_CAN/best_model.ckpt",
        sha256="4a007ccbb7e6d101db080a2e1d2c7f966f03e5d947f960074c96a2648800c53f",
    ),
}


@dataclass(frozen=True)
class FusionDTIArtifactPaths:
    """Pinned local files required by the published FusionDTI scorer."""

    dataset: str
    checkpoint: Path
    selfies_vocab: Path
    special_tokens_map: Path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_fusiondti_artifacts(
    dataset: str = "BindingDB",
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> FusionDTIArtifactPaths:
    """Resolve a pinned FusionDTI checkpoint and its exact SELFIES vocabulary."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - declared project dependency
        raise ImportError("huggingface_hub is required to load FusionDTI") from error

    key = dataset.strip().lower()
    if key not in _CHECKPOINTS:
        supported = ", ".join(spec.dataset for spec in _CHECKPOINTS.values())
        raise ValueError(f"unsupported FusionDTI dataset {dataset!r}; choose {supported}")
    spec = _CHECKPOINTS[key]
    download_kwargs = {
        "repo_id": FUSIONDTI_SPACE_ID,
        "repo_type": "space",
        "revision": FUSIONDTI_SPACE_REVISION,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "local_files_only": local_files_only,
    }
    checkpoint = Path(hf_hub_download(filename=spec.filename, **download_kwargs))
    selfies_vocab = Path(
        hf_hub_download(filename="tokenizer/vocab.json", **download_kwargs)
    )
    special_tokens_map = Path(
        hf_hub_download(
            filename="tokenizer/special_tokens_map.json",
            **download_kwargs,
        )
    )
    actual_sha256 = _file_sha256(checkpoint)
    if actual_sha256 != spec.sha256:
        raise RuntimeError(
            f"FusionDTI {spec.dataset} checkpoint checksum mismatch: "
            f"expected {spec.sha256}, got {actual_sha256}"
        )
    return FusionDTIArtifactPaths(
        dataset=spec.dataset,
        checkpoint=checkpoint,
        selfies_vocab=selfies_vocab,
        special_tokens_map=special_tokens_map,
    )


class FusionDTISelfiesTokenizer:
    """Tokenizer matching FusionDTI Space ``DrugTokenizer`` semantics."""

    _SELFIES_TOKEN = re.compile(r"\[([^\[\]]+)\]")
    _REQUIRED_SPECIAL_TOKENS = (
        "cls_token",
        "sep_token",
        "unk_token",
        "pad_token",
    )

    def __init__(
        self,
        vocab: Mapping[str, int],
        special_tokens: Mapping[str, str],
    ):
        self.vocab = {str(token): int(index) for token, index in vocab.items()}
        self.special_tokens = {
            str(name): str(token) for name, token in special_tokens.items()
        }
        missing = [
            name
            for name in self._REQUIRED_SPECIAL_TOKENS
            if name not in self.special_tokens
            or self.special_tokens[name] not in self.vocab
        ]
        if missing:
            raise ValueError(
                "FusionDTI tokenizer is missing required special tokens: "
                + ", ".join(missing)
            )
        self.cls_token_id = self.vocab[self.special_tokens["cls_token"]]
        self.sep_token_id = self.vocab[self.special_tokens["sep_token"]]
        self.unk_token_id = self.vocab[self.special_tokens["unk_token"]]
        self.pad_token_id = self.vocab[self.special_tokens["pad_token"]]

    @classmethod
    def from_files(
        cls,
        vocab_path: str | Path,
        special_tokens_path: str | Path,
    ) -> "FusionDTISelfiesTokenizer":
        with Path(vocab_path).open(encoding="utf-8") as stream:
            vocab = json.load(stream)
        with Path(special_tokens_path).open(encoding="utf-8") as stream:
            raw_special_tokens = json.load(stream)
        special_tokens = {
            name: value["content"] if isinstance(value, Mapping) else value
            for name, value in raw_special_tokens.items()
        }
        return cls(vocab=vocab, special_tokens=special_tokens)

    def encode(self, sequence: str) -> tuple[list[int], list[int]]:
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("FusionDTI molecule inputs must be non-empty SELFIES strings")
        tokens = self._SELFIES_TOKEN.findall(sequence)
        if not tokens:
            raise ValueError(f"FusionDTI expected SELFIES, got {sequence!r}")
        input_ids = [self.cls_token_id]
        input_ids.extend(self.vocab.get(token, self.unk_token_id) for token in tokens)
        input_ids.append(self.sep_token_id)
        return input_ids, [1] * len(input_ids)

    def batch_encode(
        self,
        sequences: Sequence[str],
        *,
        max_length: int,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        if max_length < 2:
            raise ValueError("max_length must leave room for FusionDTI special tokens")
        if not sequences:
            raise ValueError("FusionDTI requires at least one molecule")
        rows: list[list[int]] = []
        masks: list[list[int]] = []
        for sequence in sequences:
            input_ids, attention_mask = self.encode(sequence)
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            padding = max_length - len(input_ids)
            rows.append(input_ids + [self.pad_token_id] * padding)
            masks.append(attention_mask + [0] * padding)
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(
                masks,
                dtype=torch.long,
                device=device,
            ),
        }


class FusionDTICrossAttention(nn.Module):
    """Published four-way CAN fusion block."""

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        group_size: int = 1,
        aggregation: str = "mean_all_tok",
    ):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("FusionDTI hidden_dim must be divisible by num_heads")
        if group_size < 1:
            raise ValueError("FusionDTI group_size must be positive")
        if aggregation not in {"cls", "mean", "mean_all_tok"}:
            raise ValueError(f"unsupported FusionDTI aggregation {aggregation!r}")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads
        self.group_size = group_size
        self.aggregation = aggregation

        self.query_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_d = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def _group(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = embeddings.shape
        if sequence_length % self.group_size:
            raise ValueError(
                "FusionDTI sequence length must be divisible by group_size"
            )
        groups = sequence_length // self.group_size
        grouped_embeddings = embeddings.reshape(
            batch_size,
            groups,
            self.group_size,
            hidden_dim,
        ).mean(dim=2)
        grouped_mask = mask.bool().reshape(
            batch_size,
            groups,
            self.group_size,
        ).any(dim=2)
        return grouped_embeddings, grouped_mask

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*value.shape[:-1], self.num_heads, self.head_size)

    @staticmethod
    def _attention(
        logits: torch.Tensor,
        row_mask: torch.Tensor,
        column_mask: torch.Tensor,
    ) -> torch.Tensor:
        pair_mask = row_mask[:, :, None, None] & column_mask[:, None, :, None]
        masked_logits = torch.where(pair_mask, logits, logits - 1.0e6)
        attention = torch.softmax(masked_logits, dim=2)
        return torch.where(
            row_mask[:, :, None, None],
            attention,
            torch.zeros_like(attention),
        )

    @staticmethod
    def _masked_mean(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(dtype=embeddings.dtype).unsqueeze(-1)
        return (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        protein: torch.Tensor,
        molecule: torch.Tensor,
        protein_mask: torch.Tensor,
        molecule_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        protein, protein_mask = self._group(protein, protein_mask)
        molecule, molecule_mask = self._group(molecule, molecule_mask)

        query_protein = self._heads(self.query_p(protein))
        key_protein = self._heads(self.key_p(protein))
        value_protein = self._heads(self.value_p(protein))
        query_molecule = self._heads(self.query_d(molecule))
        key_molecule = self._heads(self.key_d(molecule))
        value_molecule = self._heads(self.value_d(molecule))

        logits_pp = torch.einsum("blhd,bkhd->blkh", query_protein, key_protein)
        logits_pd = torch.einsum("blhd,bkhd->blkh", query_protein, key_molecule)
        logits_dp = torch.einsum("blhd,bkhd->blkh", query_molecule, key_protein)
        logits_dd = torch.einsum("blhd,bkhd->blkh", query_molecule, key_molecule)

        attention_pp = self._attention(logits_pp, protein_mask, protein_mask)
        attention_pd = self._attention(logits_pd, protein_mask, molecule_mask)
        attention_dp = self._attention(logits_dp, molecule_mask, protein_mask)
        attention_dd = self._attention(logits_dd, molecule_mask, molecule_mask)

        protein_embedding = (
            torch.einsum("blkh,bkhd->blhd", attention_pp, value_protein).flatten(-2)
            + torch.einsum(
                "blkh,bkhd->blhd", attention_pd, value_molecule
            ).flatten(-2)
        ) / 2
        molecule_embedding = (
            torch.einsum("blkh,bkhd->blhd", attention_dp, value_protein).flatten(-2)
            + torch.einsum(
                "blkh,bkhd->blhd", attention_dd, value_molecule
            ).flatten(-2)
        ) / 2

        if self.aggregation == "cls":
            pooled_protein = protein_embedding[:, 0]
            pooled_molecule = molecule_embedding[:, 0]
        elif self.aggregation == "mean_all_tok":
            # This intentionally includes zeroed padding positions, matching the
            # released FusionDTI checkpoint's inference implementation.
            pooled_protein = protein_embedding.mean(dim=1)
            pooled_molecule = molecule_embedding.mean(dim=1)
        else:
            pooled_protein = self._masked_mean(protein_embedding, protein_mask)
            pooled_molecule = self._masked_mean(molecule_embedding, molecule_mask)

        joint_embedding = torch.cat([pooled_protein, pooled_molecule], dim=1)
        return joint_embedding, attention_pd.mean(dim=-1)


class FusionDTIMLP(nn.Module):
    """Checkpoint-named sigmoid classifier used by FusionDTI CAN."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim // 2)
        self.bn2 = nn.BatchNorm1d(input_dim // 2)
        self.fc3 = nn.Linear(input_dim // 2, input_dim // 4)
        self.bn3 = nn.BatchNorm1d(input_dim // 4)
        self.output = nn.Linear(input_dim // 4, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.bn1(torch.relu(self.fc1(value)))
        value = self.bn2(torch.relu(self.fc2(value)))
        value = self.bn3(torch.relu(self.fc3(value)))
        return torch.sigmoid(self.output(value))


class FusionDTIActivityHead(nn.Module):
    """FusionDTI CAN head with names/shapes compatible with released checkpoints."""

    def __init__(
        self,
        protein_dim: int = 446,
        molecule_dim: int = 768,
        *,
        hidden_dim: int = 512,
        num_heads: int = 8,
        group_size: int = 1,
        aggregation: str = "mean_all_tok",
    ):
        super().__init__()
        self.drug_reg = nn.Linear(molecule_dim, hidden_dim)
        self.prot_reg = nn.Linear(protein_dim, hidden_dim)
        self.can_layer = FusionDTICrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            group_size=group_size,
            aggregation=aggregation,
        )
        self.mlp_classifier = FusionDTIMLP(input_dim=hidden_dim * 2)

    def forward(
        self,
        protein_embeddings: torch.Tensor,
        molecule_embeddings: torch.Tensor,
        protein_mask: torch.Tensor,
        molecule_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        protein_embeddings = self.prot_reg(protein_embeddings)
        molecule_embeddings = self.drug_reg(molecule_embeddings)
        joint, protein_molecule_attention = self.can_layer(
            protein_embeddings,
            molecule_embeddings,
            protein_mask,
            molecule_mask,
        )
        return self.mlp_classifier(joint), protein_molecule_attention


def load_fusiondti_head(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> FusionDTIActivityHead:
    """Strictly load a published FusionDTI CAN checkpoint."""

    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:  # pragma: no cover - older PyTorch compatibility
        state_dict = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(state_dict, Mapping) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise TypeError("FusionDTI checkpoint must contain a state dict")
    head = FusionDTIActivityHead()
    head.load_state_dict(state_dict, strict=True)
    head.requires_grad_(False)
    head.eval()
    return head


class FusionDTIActivityScorer(nn.Module):
    """Frozen structure-aware-protein/SELFIES binary probability scorer."""

    protein_representation = "structure_aware"
    molecule_representation = "selfies"

    def __init__(
        self,
        *,
        protein_encoder: nn.Module,
        molecule_encoder: nn.Module,
        activity_head: nn.Module,
        protein_tokenizer,
        molecule_tokenizer: FusionDTISelfiesTokenizer,
        max_length: int = 512,
        batch_size: int = 8,
        protein_cache_size: int = 64,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        if max_length < 2:
            raise ValueError("FusionDTI max_length must be at least 2")
        if batch_size < 1:
            raise ValueError("FusionDTI batch_size must be positive")
        if protein_cache_size < 1:
            raise ValueError("FusionDTI protein_cache_size must be positive")
        if protein_cache_size < batch_size:
            raise ValueError(
                "FusionDTI protein_cache_size must be at least batch_size"
            )
        self.protein_encoder = protein_encoder
        self.molecule_encoder = molecule_encoder
        self.activity_head = activity_head
        self.protein_tokenizer = protein_tokenizer
        self.molecule_tokenizer = molecule_tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.protein_cache_size = protein_cache_size
        self._protein_cache: OrderedDict[
            str,
            tuple[torch.Tensor, torch.Tensor],
        ] = OrderedDict()
        target_device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.to(target_device)
        self.requires_grad_(False)
        self.eval()

    @property
    def device(self) -> torch.device:
        return next(self.activity_head.parameters()).device

    def train(self, mode: bool = True) -> "FusionDTIActivityScorer":
        # A reward scorer is part of the environment, never the optimized policy.
        super().train(False)
        return self

    def _apply(self, function):
        result = super()._apply(function)
        if hasattr(self, "_protein_cache"):
            self._protein_cache.clear()
        return result

    def _protein_inputs(self, sequences: Sequence[str]) -> dict[str, torch.Tensor]:
        if any(not isinstance(sequence, str) or not sequence for sequence in sequences):
            raise ValueError(
                "FusionDTI protein inputs must be non-empty structure-aware strings"
            )
        invalid = [
            sequence
            for sequence in sequences
            if len(sequence) % 2
            or not all(character.isupper() for character in sequence[0::2])
            or not all(
                character.islower() or character == "#"
                for character in sequence[1::2]
            )
        ]
        if invalid:
            raise ValueError(
                "FusionDTI protein inputs must be residue/3Di paired strings"
            )
        too_long = [
            len(sequence) // 2
            for sequence in sequences
            if len(sequence) // 2 > self.max_length - 2
        ]
        if too_long:
            raise ValueError(
                "FusionDTI protein sequence exceeds its no-truncation residue limit "
                f"of {self.max_length - 2}; observed {max(too_long)}"
            )
        encoded = self.protein_tokenizer(
            list(sequences),
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=False,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }

    def _protein_features(
        self,
        sequences: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unique_sequences = list(dict.fromkeys(sequences))
        missing = [
            sequence
            for sequence in unique_sequences
            if sequence not in self._protein_cache
        ]
        if missing:
            protein_inputs = self._protein_inputs(missing)
            protein_output = self.protein_encoder(
                **protein_inputs,
                return_dict=True,
            )
            embeddings = self._encoder_output(protein_output, "logits")
            masks = protein_inputs["attention_mask"]
            for index, sequence in enumerate(missing):
                self._protein_cache[sequence] = (
                    embeddings[index].detach(),
                    masks[index].detach(),
                )
                self._protein_cache.move_to_end(sequence)
                while len(self._protein_cache) > self.protein_cache_size:
                    self._protein_cache.popitem(last=False)
        features = []
        masks = []
        for sequence in sequences:
            embedding, mask = self._protein_cache[sequence]
            self._protein_cache.move_to_end(sequence)
            features.append(embedding)
            masks.append(mask)
        return torch.stack(features), torch.stack(masks)

    @staticmethod
    def _encoder_output(output, name: str) -> torch.Tensor:
        if hasattr(output, name):
            return getattr(output, name)
        if isinstance(output, Mapping) and name in output:
            return output[name]
        raise TypeError(f"FusionDTI encoder output has no {name!r} tensor")

    def _score_batch(
        self,
        protein_sequences: Sequence[str],
        molecule_selfies: Sequence[str],
    ) -> torch.Tensor:
        protein_embeddings, protein_mask = self._protein_features(protein_sequences)
        molecule_inputs = self.molecule_tokenizer.batch_encode(
            molecule_selfies,
            max_length=self.max_length,
            device=self.device,
        )
        molecule_output = self.molecule_encoder(
            **molecule_inputs,
            return_dict=True,
        )
        molecule_embeddings = self._encoder_output(
            molecule_output,
            "last_hidden_state",
        )
        head_dtype = next(self.activity_head.parameters()).dtype
        scores, _ = self.activity_head(
            protein_embeddings.to(dtype=head_dtype),
            molecule_embeddings.to(dtype=head_dtype),
            protein_mask.bool(),
            molecule_inputs["attention_mask"].bool(),
        )
        return scores.reshape(-1)

    def forward(
        self,
        protein_sequences: Sequence[str],
        molecule_sequences: Sequence[str],
    ) -> torch.Tensor:
        if len(protein_sequences) != len(molecule_sequences):
            raise ValueError("FusionDTI protein and molecule batches must align")
        if not protein_sequences:
            return torch.empty(0, dtype=torch.float32)
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(protein_sequences), self.batch_size):
                stop = start + self.batch_size
                chunks.append(
                    self._score_batch(
                        protein_sequences[start:stop],
                        molecule_sequences[start:stop],
                    )
                )
        scores = torch.cat(chunks).detach().float().cpu()
        if not torch.isfinite(scores).all():
            raise RuntimeError("FusionDTI returned non-finite activity probabilities")
        if scores.lt(0.0).any() or scores.gt(1.0).any():
            raise RuntimeError("FusionDTI returned values outside [0, 1]")
        return scores

    @classmethod
    def from_pretrained(
        cls,
        dataset: str = "BindingDB",
        *,
        device: str | torch.device | None = None,
        batch_size: int = 8,
        max_length: int = 512,
        protein_cache_size: int = 64,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> "FusionDTIActivityScorer":
        """Load the exact published encoders, vocabulary, and activity head."""

        try:
            from transformers import AutoModel, EsmForMaskedLM, EsmTokenizer
        except ImportError as error:  # pragma: no cover - declared dependency
            raise ImportError("transformers is required to load FusionDTI") from error

        artifacts = download_fusiondti_artifacts(
            dataset,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        common_model_kwargs = {
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "local_files_only": local_files_only,
        }
        protein_tokenizer = EsmTokenizer.from_pretrained(
            FUSIONDTI_PROTEIN_MODEL_ID,
            revision=FUSIONDTI_PROTEIN_MODEL_REVISION,
            **common_model_kwargs,
        )
        protein_encoder = EsmForMaskedLM.from_pretrained(
            FUSIONDTI_PROTEIN_MODEL_ID,
            revision=FUSIONDTI_PROTEIN_MODEL_REVISION,
            **common_model_kwargs,
        )
        molecule_encoder = AutoModel.from_pretrained(
            FUSIONDTI_MOLECULE_MODEL_ID,
            revision=FUSIONDTI_MOLECULE_MODEL_REVISION,
            **common_model_kwargs,
        )
        molecule_tokenizer = FusionDTISelfiesTokenizer.from_files(
            artifacts.selfies_vocab,
            artifacts.special_tokens_map,
        )
        activity_head = load_fusiondti_head(artifacts.checkpoint)
        return cls(
            protein_encoder=protein_encoder,
            molecule_encoder=molecule_encoder,
            activity_head=activity_head,
            protein_tokenizer=protein_tokenizer,
            molecule_tokenizer=molecule_tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            protein_cache_size=protein_cache_size,
            device=device,
        )
