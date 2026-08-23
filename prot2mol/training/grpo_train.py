#!/usr/bin/env python3
"""Full single-GPU GRPO post-training over unique proteins."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import random
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import wandb

from ..chem.utils import (
    canonic_smiles,
    molecular_property_rows,
    molecular_property_summary,
)
from ..core.protein_encoders import get_protein_tokenizer
from ..data.pipeline import tokenize_protein_sequences_for_inference
from ..io.config import parse_args_with_config
from ..io.hf_utils import (
    load_molgen_tokenizer,
    load_prot2mol_inference_model,
    save_model_config,
)
from ..rewards import (
    FusionDTIActivityScorer,
    TargetActivePropertyStats,
    TargetPropertyShapedActivityScorer,
    internal_diversity_factors,
    scaffold_diversity_summary,
)
from .grpo import GRPOConfig, GRPOTrainer, selfies_to_smiles, valid_selfies


LOGGER = logging.getLogger(__name__)
GIB = 1024**3
STRUCTURE_AWARE_COLUMN_CANDIDATES = (
    "structure_aware_sequence",
    "saprot_sequence",
    "foldseek_sequence",
)


@dataclass(frozen=True)
class ProteinRecord:
    protein_id: str
    protein_sequence: str
    reward_protein_sequence: str


@dataclass(frozen=True)
class EvaluationTarget:
    protein: ProteinRecord
    active_reference_smiles: tuple[str, ...]


def _schema_names(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def _first_available(names: set[str], candidates: Sequence[str]) -> str | None:
    return next((candidate for candidate in candidates if candidate in names), None)


def _iter_parquet_rows(
    path: Path,
    columns: Sequence[str],
    *,
    batch_size: int = 65_536,
) -> Iterable[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(columns)):
        values = batch.to_pydict()
        for index in range(batch.num_rows):
            yield {column: values[column][index] for column in columns}


def _load_structure_mapping(
    path: Path,
    *,
    key_column: str,
    sequence_column: str | None,
) -> tuple[dict[str, str], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Structure-aware protein mapping not found: {path}")
    names = _schema_names(path)
    resolved_sequence_column = sequence_column or _first_available(
        names,
        STRUCTURE_AWARE_COLUMN_CANDIDATES,
    )
    if resolved_sequence_column is None:
        raise ValueError(
            f"{path} has no structure-aware sequence column; expected one of "
            f"{STRUCTURE_AWARE_COLUMN_CANDIDATES}"
        )
    if key_column not in names:
        raise ValueError(
            f"Structure-aware mapping lacks join column {key_column!r}; "
            f"available columns: {sorted(names)}"
        )
    mapping: dict[str, str] = {}
    for row in _iter_parquet_rows(path, (key_column, resolved_sequence_column)):
        key = str(row[key_column] or "").strip()
        value = str(row[resolved_sequence_column] or "").strip()
        if not key or not value:
            continue
        previous = mapping.setdefault(key, value)
        if previous != value:
            raise ValueError(
                f"Structure-aware mapping key {key!r} maps to multiple sequences"
            )
    if not mapping:
        raise ValueError(f"Structure-aware mapping is empty: {path}")
    return mapping, resolved_sequence_column


def _looks_like_structure_aware_sequence(value: str) -> bool:
    """Recognize SaProt's residue/3Di paired-token string representation."""

    if not value or any(character.isspace() for character in value) or len(value) % 2:
        return False
    amino_acids = value[0::2]
    structure_tokens = value[1::2]
    return all(character.isupper() for character in amino_acids) and all(
        character.islower() or character == "#" for character in structure_tokens
    )


def load_unique_training_proteins(
    train_parquet_path: str | Path,
    *,
    protein_id_column: str = "protein_accession",
    protein_sequence_column: str = "protein_sequence",
    structure_aware_path: str | Path | None = None,
    structure_aware_key_column: str = "protein_accession",
    structure_aware_sequence_column: str | None = None,
    required_protein_ids: Sequence[str] | None = None,
) -> list[ProteinRecord]:
    """Load one stable record per unique amino-acid sequence from the train split."""

    train_path = Path(train_parquet_path).expanduser().resolve()
    if not train_path.is_file():
        raise FileNotFoundError(f"Random-split training Parquet not found: {train_path}")
    names = _schema_names(train_path)
    required = {protein_id_column, protein_sequence_column}
    missing = sorted(required.difference(names))
    if missing:
        raise ValueError(f"Training Parquet lacks required columns: {missing}")

    embedded_structure_column = structure_aware_sequence_column
    structure_mapping: dict[str, str] | None = None
    if structure_aware_path is not None:
        structure_mapping, embedded_structure_column = _load_structure_mapping(
            Path(structure_aware_path).expanduser().resolve(),
            key_column=structure_aware_key_column,
            sequence_column=structure_aware_sequence_column,
        )
    elif embedded_structure_column is None:
        embedded_structure_column = _first_available(
            names,
            STRUCTURE_AWARE_COLUMN_CANDIDATES,
        )
    if structure_mapping is None and embedded_structure_column not in names:
        raise ValueError(
            "FusionDTI requires a separate SaProt/Foldseek structure-aware sequence. "
            f"The random-split train file contains only plain amino-acid sequences. "
            "Provide --structure_aware_path with a Parquet mapping containing "
            f"{structure_aware_key_column!r} and one of "
            f"{STRUCTURE_AWARE_COLUMN_CANDIDATES}."
        )

    columns = [protein_id_column, protein_sequence_column]
    lookup_column = structure_aware_key_column
    required_ids = (
        {str(value).strip() for value in required_protein_ids}
        if required_protein_ids is not None
        else None
    )
    if required_ids is not None and not required_ids:
        raise ValueError("required_protein_ids cannot be empty")
    if structure_mapping is not None and lookup_column not in names:
        raise ValueError(
            f"Training Parquet lacks structure mapping join column {lookup_column!r}"
        )
    if structure_mapping is not None and lookup_column not in columns:
        columns.append(lookup_column)
    if structure_mapping is None and embedded_structure_column not in columns:
        columns.append(str(embedded_structure_column))

    by_sequence: dict[str, ProteinRecord] = {}
    missing_mapping_sequences: dict[str, str] = {}
    for row in _iter_parquet_rows(train_path, columns):
        protein_id = str(row[protein_id_column] or "").strip()
        sequence = str(row[protein_sequence_column] or "").strip().upper()
        if not protein_id or not sequence:
            continue
        if required_ids is not None and protein_id not in required_ids:
            continue
        if structure_mapping is not None:
            mapping_key = str(row[lookup_column] or "").strip()
            reward_sequence = structure_mapping.get(mapping_key, "")
            if not reward_sequence:
                if sequence not in by_sequence:
                    missing_mapping_sequences.setdefault(sequence, mapping_key)
                continue
        else:
            reward_sequence = str(row[str(embedded_structure_column)] or "").strip()
            if not reward_sequence:
                if sequence not in by_sequence:
                    missing_mapping_sequences.setdefault(sequence, protein_id)
                continue
        if not _looks_like_structure_aware_sequence(reward_sequence):
            raise ValueError(
                f"Protein {protein_id!r} does not contain a SaProt residue/3Di "
                "paired sequence; refusing to pass a plain sequence to FusionDTI"
            )
        if reward_sequence[0::2] != sequence:
            raise ValueError(
                f"Protein {protein_id!r} structure-aware amino-acid track does not "
                "exactly match its generator protein sequence"
            )
        record = ProteinRecord(protein_id, sequence, reward_sequence)
        previous = by_sequence.setdefault(sequence, record)
        missing_mapping_sequences.pop(sequence, None)
        if previous.reward_protein_sequence != reward_sequence:
            raise ValueError(
                "One amino-acid sequence maps to multiple structure-aware sequences: "
                f"{previous.protein_id!r}, {protein_id!r}"
            )
    if missing_mapping_sequences:
        examples = sorted(missing_mapping_sequences.values())[:5]
        raise ValueError(
            f"Structure-aware mapping is incomplete for {len(missing_mapping_sequences)} "
            f"training protein keys; examples: {examples}"
        )
    proteins = sorted(by_sequence.values(), key=lambda row: (row.protein_id, row.protein_sequence))
    if required_ids is not None:
        found_ids = {protein.protein_id for protein in proteins}
        missing_ids = sorted(required_ids.difference(found_ids))
        if missing_ids:
            raise ValueError(
                f"Required protein IDs were not found with structure mappings: {missing_ids[:5]}"
            )
    if not proteins:
        raise ValueError("No usable unique proteins were found in the training split")
    return proteins


def load_active_references(
    validation_parquet_path: str | Path,
    *,
    protein_sequence_column: str = "protein_sequence",
    smiles_column: str = "smiles",
    label_column: str = "binary_label",
) -> dict[str, tuple[str, ...]]:
    """Collect unique validation actives per protein for target-conditional FCD."""

    path = Path(validation_parquet_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Random-split validation Parquet not found: {path}")
    names = _schema_names(path)
    columns = (protein_sequence_column, smiles_column, label_column)
    missing = sorted(set(columns).difference(names))
    if missing:
        raise ValueError(f"Validation Parquet lacks required columns: {missing}")
    references: dict[str, set[str]] = {}
    for row in _iter_parquet_rows(path, columns):
        if int(row[label_column] or 0) != 1:
            continue
        sequence = str(row[protein_sequence_column] or "").strip().upper()
        smiles = canonic_smiles(row[smiles_column])
        if sequence and smiles:
            references.setdefault(sequence, set()).add(smiles)
    return {
        sequence: tuple(sorted(smiles))
        for sequence, smiles in references.items()
        if smiles
    }


def load_training_active_property_stats(
    train_parquet_path: str | Path,
    protein_sequences: Sequence[str],
    *,
    protein_sequence_column: str = "protein_sequence",
    smiles_column: str = "smiles",
    label_column: str = "binary_label",
) -> dict[str, TargetActivePropertyStats]:
    """Calculate target-property distributions from unique training actives."""

    path = Path(train_parquet_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Random-split training Parquet not found: {path}")
    names = _schema_names(path)
    columns = (protein_sequence_column, smiles_column, label_column)
    missing = sorted(set(columns).difference(names))
    if missing:
        raise ValueError(f"Training Parquet lacks required columns: {missing}")
    requested = {str(sequence).strip().upper() for sequence in protein_sequences}
    if not requested:
        raise ValueError("Property shaping requires at least one target protein")
    active_smiles: dict[str, set[str]] = {sequence: set() for sequence in requested}
    for row in _iter_parquet_rows(path, columns):
        if int(row[label_column] or 0) != 1:
            continue
        sequence = str(row[protein_sequence_column] or "").strip().upper()
        if sequence not in active_smiles:
            continue
        smiles = canonic_smiles(row[smiles_column])
        if smiles:
            active_smiles[sequence].add(smiles)

    stats: dict[str, TargetActivePropertyStats] = {}
    insufficient = []
    for sequence in sorted(requested):
        smiles = sorted(active_smiles[sequence])
        if len(smiles) < 2:
            insufficient.append((sequence, len(smiles)))
            continue
        rows = molecular_property_rows(smiles)
        logp = np.asarray([row["logp"] for row in rows], dtype=np.float64)
        sas = np.asarray([row["sas"] for row in rows], dtype=np.float64)
        stats[sequence] = TargetActivePropertyStats(
            active_count=len(smiles),
            logp_mean=float(logp.mean()),
            logp_std=float(logp.std(ddof=0)),
            sas_mean=float(sas.mean()),
            sas_std=float(sas.std(ddof=0)),
        )
    if insufficient:
        examples = [(sequence[:12], count) for sequence, count in insufficient[:5]]
        raise ValueError(
            "Property shaping requires at least two unique training actives per "
            f"protein; insufficient examples: {examples}"
        )
    return stats


def select_evaluation_targets(
    proteins: Sequence[ProteinRecord],
    active_references: Mapping[str, Sequence[str]],
    *,
    count: int,
    seed: int,
    min_reference_actives: int,
    requested_protein_ids: Sequence[str] = (),
    max_protein_length: int | None = None,
) -> list[EvaluationTarget]:
    by_id = {protein.protein_id: protein for protein in proteins}
    eligible = [
        protein
        for protein in proteins
        if len(active_references.get(protein.protein_sequence, ()))
        >= min_reference_actives
        and (
            max_protein_length is None
            or len(protein.protein_sequence) <= max_protein_length
        )
    ]
    requested = [str(value) for value in requested_protein_ids if str(value)]
    ordered = sorted(
        eligible,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row.protein_id}:{row.protein_sequence}".encode()
        ).hexdigest(),
    )
    if requested:
        missing = [value for value in requested if value not in by_id]
        if missing:
            raise ValueError(f"Requested evaluation protein IDs not found: {missing}")
        selected = [by_id[value] for value in requested]
        insufficient = [
            protein.protein_id
            for protein in selected
            if len(active_references.get(protein.protein_sequence, ()))
            < min_reference_actives
        ]
        if insufficient:
            raise ValueError(
                "Requested evaluation proteins lack enough validation actives for FCD: "
                f"{insufficient}"
            )
        too_long = [
            protein.protein_id
            for protein in selected
            if max_protein_length is not None
            and len(protein.protein_sequence) > max_protein_length
        ]
        if too_long:
            raise ValueError(
                "Requested evaluation proteins exceed the allowed protein context: "
                f"{too_long}"
            )
        selected_ids = {protein.protein_id for protein in selected}
        selected.extend(
            protein
            for protein in ordered
            if protein.protein_id not in selected_ids
        )
        selected = selected[: max(count, len(requested))]
    else:
        selected = ordered[:count]
    return [
        EvaluationTarget(
            protein=protein,
            active_reference_smiles=tuple(active_references[protein.protein_sequence]),
        )
        for protein in selected
    ]


def _build_reference_policy(policy: torch.nn.Module) -> torch.nn.Module:
    """Copy only the decoder while sharing the already-frozen conditioner."""

    reference = copy.copy(policy)
    reference._modules = policy._modules.copy()
    reference._parameters = policy._parameters.copy()
    reference._buffers = policy._buffers.copy()
    reference.molecule_decoder = copy.deepcopy(policy.molecule_decoder)
    reference.protein_encoder = policy.protein_encoder
    reference.conditioning_projection = policy.conditioning_projection
    reference.requires_grad_(False)
    reference.eval()
    return reference


def _trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _load_trainable_state_dict(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    parameters = dict(model.named_parameters())
    expected = {name for name, value in parameters.items() if value.requires_grad}
    actual = set(state)
    if expected != actual:
        raise ValueError(
            "GRPO checkpoint trainable-parameter contract differs: "
            f"missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}"
        )
    with torch.no_grad():
        for name in sorted(expected):
            target = parameters[name]
            value = state[name]
            if target.shape != value.shape:
                raise ValueError(
                    f"GRPO checkpoint shape mismatch for {name}: "
                    f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            target.copy_(value.to(device=target.device, dtype=target.dtype))


def _cosine_warmup_lambda(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return max(float(step + 1) / float(warmup_steps), 1.0e-8)
    denominator = max(1, total_steps - warmup_steps)
    progress = min(max((step - warmup_steps) / denominator, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _evaluation_property_summary(
    smiles: Sequence[str],
) -> dict[str, float | None]:
    """Keep unavailable endpoint chemistry out of cross-protein macro means."""

    properties = molecular_property_summary(smiles)
    if properties["count"] == 0:
        return {
            key: (value if key == "count" else None)
            for key, value in properties.items()
        }
    return properties


def _frechet_distance(
    mean_a: np.ndarray,
    covariance_a: np.ndarray,
    mean_b: np.ndarray,
    covariance_b: np.ndarray,
    *,
    epsilon: float = 1.0e-6,
) -> float:
    """Compute the standard FCD Gaussian distance across SciPy versions."""

    from scipy import linalg

    mean_a = np.atleast_1d(mean_a)
    mean_b = np.atleast_1d(mean_b)
    covariance_a = np.atleast_2d(covariance_a)
    covariance_b = np.atleast_2d(covariance_b)
    if mean_a.shape != mean_b.shape or covariance_a.shape != covariance_b.shape:
        raise ValueError("FCD activation statistics have incompatible shapes")
    difference = mean_a - mean_b
    covariance_mean = linalg.sqrtm(covariance_a.dot(covariance_b))
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(covariance_a.shape[0]) * epsilon
        covariance_mean = linalg.sqrtm(
            (covariance_a + offset).dot(covariance_b + offset)
        )
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(np.diagonal(covariance_mean).imag, 0.0, atol=1.0e-3):
            raise ValueError(
                "FCD covariance square root has a non-negligible imaginary component"
            )
        covariance_mean = covariance_mean.real
    return float(
        difference.dot(difference)
        + np.trace(covariance_a)
        + np.trace(covariance_b)
        - 2.0 * np.trace(covariance_mean)
    )


class _ChemNetFCD:
    """Use fcd-torch ChemNet activations with version-compatible distance math."""

    def __init__(self, implementation):
        self.implementation = implementation

    def __call__(self, *, ref: Sequence[str], gen: Sequence[str]) -> float:
        reference = self.implementation.precalc(list(ref))
        generated = self.implementation.precalc(list(gen))
        return _frechet_distance(
            reference["mu"],
            reference["sigma"],
            generated["mu"],
            generated["sigma"],
        )


class GRPOTrainingRun:
    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        if self.output_dir.exists() and not config.resume_from_checkpoint:
            material_outputs = [
                path
                for path in self.output_dir.iterdir()
                if path.name in {"evaluation", "final", "training_summary.json"}
                or path.name.startswith("checkpoint-")
            ]
            if material_outputs:
                names = sorted(path.name for path in material_outputs)
                raise FileExistsError(
                    f"Refusing to overwrite an existing GRPO run in {self.output_dir}: "
                    f"{names}"
                )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.policy: torch.nn.Module | None = None
        self.reference_policy: torch.nn.Module | None = None
        self.reward_scorer: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.trainer: GRPOTrainer | None = None
        self.proteins: list[ProteinRecord] = []
        self.source_unique_protein_count = 0
        self.evaluation_targets: list[EvaluationTarget] = []
        self.property_stats_by_sequence: dict[
            str,
            TargetActivePropertyStats,
        ] = {}
        self.property_stats_by_reward_sequence: dict[
            str,
            TargetActivePropertyStats,
        ] = {}
        self.molecule_tokenizer = None
        self.protein_tokenizer = None
        self._fcd = None
        self.start_epoch = 0
        self.start_index = 0
        self.proteins_seen = 0
        self.wandb_run = None
        self._saved_checkpoints: set[Path] = set()

    def _load_data(self) -> None:
        required_protein_ids = (
            self.config.eval_protein_ids
            if self.config.train_on_evaluation_panel_only
            and self.config.eval_protein_ids
            else None
        )
        source_proteins = load_unique_training_proteins(
            self.config.train_parquet_path,
            protein_id_column=self.config.protein_id_column,
            protein_sequence_column=self.config.protein_sequence_column,
            structure_aware_path=self.config.structure_aware_path,
            structure_aware_key_column=self.config.structure_aware_key_column,
            structure_aware_sequence_column=self.config.structure_aware_sequence_column,
            required_protein_ids=required_protein_ids,
        )
        references = load_active_references(
            self.config.validation_parquet_path,
            protein_sequence_column=self.config.protein_sequence_column,
            smiles_column=self.config.smiles_column,
            label_column=self.config.label_column,
        )
        self.source_unique_protein_count = len(source_proteins)
        self.evaluation_targets = select_evaluation_targets(
            source_proteins,
            references,
            count=self.config.eval_proteins,
            seed=self.config.seed,
            min_reference_actives=self.config.min_fcd_reference_actives,
            requested_protein_ids=self.config.eval_protein_ids,
            max_protein_length=min(
                self.config.prot_max_length,
                self.config.reward_protein_max_residues,
            ),
        )
        if self.config.eval_proteins and not self.evaluation_targets:
            raise ValueError("No proteins are eligible for target-conditional evaluation")
        if self.config.train_on_evaluation_panel_only:
            self.proteins = [target.protein for target in self.evaluation_targets]
        else:
            self.proteins = source_proteins
        if self.config.property_reward_shaping:
            self.property_stats_by_sequence = load_training_active_property_stats(
                self.config.train_parquet_path,
                [protein.protein_sequence for protein in self.proteins],
                protein_sequence_column=self.config.protein_sequence_column,
                smiles_column=self.config.smiles_column,
                label_column=self.config.label_column,
            )
            self.property_stats_by_reward_sequence = {
                protein.reward_protein_sequence: self.property_stats_by_sequence[
                    protein.protein_sequence
                ]
                for protein in self.proteins
            }
        self._save_cohort_manifest()
        LOGGER.info(
            "Selected %d training proteins from %d unique source proteins; "
            "%d evaluation targets",
            len(self.proteins),
            self.source_unique_protein_count,
            len(self.evaluation_targets),
        )

    def _save_cohort_manifest(self) -> None:
        rows = []
        for index, target in enumerate(self.evaluation_targets):
            row = {
                "cohort_index": index,
                "protein_id": target.protein.protein_id,
                "protein_sequence": target.protein.protein_sequence,
                "structure_aware_sequence": target.protein.reward_protein_sequence,
                "protein_length": len(target.protein.protein_sequence),
                "validation_active_reference_count": len(
                    target.active_reference_smiles
                ),
            }
            stats = self.property_stats_by_sequence.get(
                target.protein.protein_sequence
            )
            if stats is not None:
                row.update(
                    stats.as_dict(
                        allowed_sigma=self.config.property_allowed_sigma
                    )
                )
            rows.append(row)
        pq.write_table(pa.Table.from_pylist(rows), self.output_dir / "cohort.parquet")
        manifest = {
            "selection_seed": self.config.seed,
            "source_unique_proteins": self.source_unique_protein_count,
            "cohort_size": len(rows),
            "required_protein_ids": list(self.config.eval_protein_ids),
            "protein_ids": [row["protein_id"] for row in rows],
            "training_is_cohort_only": self.config.train_on_evaluation_panel_only,
            "generator_max_protein_length": self.config.prot_max_length,
            "reward_max_protein_residues": self.config.reward_protein_max_residues,
            "minimum_unique_active_references": self.config.min_fcd_reference_actives,
            "property_reward_shaping": self.config.property_reward_shaping,
            "property_statistics_source": "unique canonical training-split actives",
            "property_allowed_sigma": self.config.property_allowed_sigma,
            "property_penalty_strength": self.config.property_penalty_strength,
            "property_reward_formula": (
                "activity_probability * exp(-strength * logp_excess_z^2) * "
                "exp(-strength * sas_excess_z^2)"
            ),
            "diversity_reward_shaping": self.config.diversity_reward_shaping,
            "diversity_scope": (
                "valid EOS molecules within each fixed GRPO protein group"
            ),
            "diversity_reward_weight": self.config.diversity_reward_weight,
            "diversity_mean_similarity_weight": (
                self.config.diversity_mean_similarity_weight
            ),
            "diversity_nearest_similarity_weight": (
                1.0 - self.config.diversity_mean_similarity_weight
            ),
            "diversity_duplicate_penalty_factor": (
                self.config.diversity_duplicate_penalty_factor
            ),
            "diversity_morgan_radius": self.config.diversity_morgan_radius,
            "diversity_morgan_bits": self.config.diversity_morgan_bits,
            "final_reward_formula": (
                "property_shaped_reward * ((1 - diversity_reward_weight) + "
                "diversity_reward_weight * (1 - weighted_mean_and_nearest_"
                "group_tanimoto)); canonical duplicates use the configured "
                "duplicate penalty factor"
            ),
        }
        (self.output_dir / "cohort.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _load_models(self) -> None:
        self.molecule_tokenizer = load_molgen_tokenizer(
            models_base=self.config.models_base,
            padding_side="right",
            model_id=self.config.decoder_model_id,
            revision=self.config.decoder_model_revision,
            local_files_only=self.config.local_files_only,
        )
        self.protein_tokenizer = get_protein_tokenizer(
            self.config.prot_emb_model,
            model_id=self.config.protein_model_id,
            revision=self.config.protein_model_revision,
            models_base=self.config.models_base,
            local_files_only=self.config.local_files_only,
        )
        self.policy = load_prot2mol_inference_model(
            model_path=self.config.generator_checkpoint,
            device=self.device,
            mol_tokenizer=self.molecule_tokenizer,
            prot_emb_model=self.config.prot_emb_model,
            protein_model_id=self.config.protein_model_id,
            protein_model_revision=self.config.protein_model_revision,
            decoder_type=self.config.decoder_type,
            decoder_model_id=self.config.decoder_model_id,
            decoder_model_revision=self.config.decoder_model_revision,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            n_emb=self.config.n_emb,
            conditioning_dropout=self.config.conditioning_dropout,
            models_base=self.config.models_base,
            local_files_only=self.config.local_files_only,
            max_mol_len=self.config.max_mol_len,
            prot_max_length=self.config.prot_max_length,
            strict=True,
            logger=LOGGER,
        )
        position_limit = int(
            getattr(self.policy.config, "n_positions", self.config.max_mol_len)
        )
        if self.config.max_mol_len > position_limit:
            raise ValueError(
                f"Requested molecule length {self.config.max_mol_len} exceeds the "
                f"checkpoint position limit {position_limit}"
            )
        self.reference_policy = _build_reference_policy(self.policy)
        self.policy.update_trainable_components(
            trainable_encoder=False,
            trainable_projection=False,
            trainable_decoder=True,
        )
        activity_scorer = FusionDTIActivityScorer.from_pretrained(
            dataset=self.config.fusiondti_dataset,
            device=self.device,
            batch_size=self.config.reward_batch_size,
            max_length=self.config.reward_max_length,
            cache_dir=self.config.models_base,
            local_files_only=self.config.local_files_only,
            protein_cache_size=self.config.reward_protein_cache_size,
        )
        self.reward_scorer = (
            TargetPropertyShapedActivityScorer(
                activity_scorer,
                self.property_stats_by_reward_sequence,
                allowed_sigma=self.config.property_allowed_sigma,
                penalty_strength=self.config.property_penalty_strength,
            )
            if self.config.property_reward_shaping
            else activity_scorer
        )
        trainable_parameters = [
            parameter for parameter in self.policy.parameters() if parameter.requires_grad
        ]
        optimizer_kwargs: dict[str, Any] = {
            "lr": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
        }
        if self.device.type == "cuda":
            optimizer_kwargs["fused"] = True
        self.optimizer = torch.optim.AdamW(trainable_parameters, **optimizer_kwargs)
        requested_rollout_steps = self.config.epochs * len(self.proteins)
        total_rollout_steps = (
            min(requested_rollout_steps, self.config.max_steps)
            if self.config.max_steps is not None
            else requested_rollout_steps
        )
        total_optimizer_steps = total_rollout_steps * self.config.num_iterations
        warmup_steps = int(total_optimizer_steps * self.config.warmup_ratio)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _cosine_warmup_lambda(
                step,
                warmup_steps=warmup_steps,
                total_steps=total_optimizer_steps,
            ),
        )
        self.trainer = GRPOTrainer(
            policy=self.policy,
            reference_policy=self.reference_policy,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            tokenizer=self.molecule_tokenizer,
            reward_function=self.reward_scorer,
            config=GRPOConfig(
                group_size=self.config.group_size,
                clip_epsilon=self.config.clip_epsilon,
                kl_beta=self.config.kl_beta,
                advantage_epsilon=self.config.advantage_epsilon,
                max_length=self.config.max_mol_len,
                min_length=self.config.min_mol_len,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                num_iterations=self.config.num_iterations,
                loss_type=self.config.loss_type,
                max_grad_norm=self.config.max_grad_norm,
                require_eos_for_reward=True,
                diversity_reward_shaping=self.config.diversity_reward_shaping,
                diversity_reward_weight=self.config.diversity_reward_weight,
                diversity_mean_similarity_weight=(
                    self.config.diversity_mean_similarity_weight
                ),
                diversity_duplicate_penalty_factor=(
                    self.config.diversity_duplicate_penalty_factor
                ),
                diversity_morgan_radius=self.config.diversity_morgan_radius,
                diversity_morgan_bits=self.config.diversity_morgan_bits,
                precision=self.config.precision,
            ),
        )
        counts = self.policy.parameter_counts()
        LOGGER.info("Policy parameters: %s", counts)
        LOGGER.info("Only the molecule decoder is trainable")

    def _property_reward_contract(self) -> dict[str, Any]:
        rows = [
            {
                "protein_id": protein.protein_id,
                **self.property_stats_by_sequence[protein.protein_sequence].as_dict(
                    allowed_sigma=self.config.property_allowed_sigma
                ),
            }
            for protein in self.proteins
            if protein.protein_sequence in self.property_stats_by_sequence
        ]
        digest = hashlib.sha256(
            json.dumps(rows, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {
            "enabled": self.config.property_reward_shaping,
            "allowed_sigma": self.config.property_allowed_sigma,
            "penalty_strength": self.config.property_penalty_strength,
            "training_active_statistics_sha256": digest,
            "diversity_enabled": self.config.diversity_reward_shaping,
            "diversity_reward_weight": self.config.diversity_reward_weight,
            "diversity_mean_similarity_weight": (
                self.config.diversity_mean_similarity_weight
            ),
            "diversity_duplicate_penalty_factor": (
                self.config.diversity_duplicate_penalty_factor
            ),
            "diversity_morgan_radius": self.config.diversity_morgan_radius,
            "diversity_morgan_bits": self.config.diversity_morgan_bits,
        }

    def _checkpoint_payload(self, *, epoch: int, next_index: int) -> dict[str, Any]:
        assert self.policy is not None
        assert self.optimizer is not None
        assert self.scheduler is not None
        assert self.trainer is not None
        payload: dict[str, Any] = {
            "format_version": 1,
            "policy_trainable_state": _trainable_state_dict(self.policy),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "global_step": self.trainer.global_step,
            "optimization_step": self.trainer.optimization_step,
            "epoch": epoch,
            "next_index": next_index,
            "proteins_seen": self.proteins_seen,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "generator_checkpoint": str(Path(self.config.generator_checkpoint).resolve()),
            "train_parquet_path": str(Path(self.config.train_parquet_path).resolve()),
            "cohort_protein_ids": [protein.protein_id for protein in self.proteins],
            "property_reward_contract": self._property_reward_contract(),
            "wandb_run_id": self.wandb_run.id if self.wandb_run is not None else None,
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return payload

    def save_checkpoint(self, *, epoch: int, next_index: int) -> Path:
        assert self.trainer is not None
        destination = self.output_dir / f"checkpoint-{self.trainer.global_step}"
        if destination in self._saved_checkpoints:
            if not (destination / "trainer_state.pt").is_file():
                raise RuntimeError(
                    f"Current-run GRPO checkpoint became incomplete: {destination}"
                )
            return destination
        if destination.exists():
            raise FileExistsError(
                f"Refusing to reuse a pre-existing GRPO checkpoint: {destination}"
            )
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}-",
                dir=self.output_dir,
            )
        )
        payload = self._checkpoint_payload(epoch=epoch, next_index=next_index)
        try:
            torch.save(payload, temporary / "trainer_state.pt")
            metadata = {
                key: payload[key]
                for key in (
                    "format_version",
                    "global_step",
                    "optimization_step",
                    "epoch",
                    "next_index",
                    "proteins_seen",
                    "generator_checkpoint",
                    "train_parquet_path",
                    "cohort_protein_ids",
                    "property_reward_contract",
                    "wandb_run_id",
                )
            }
            (temporary / "trainer_state.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        self._saved_checkpoints.add(destination)
        LOGGER.info("Saved GRPO checkpoint to %s", destination)
        return destination

    def _load_resume_checkpoint(self) -> str | None:
        if not self.config.resume_from_checkpoint:
            return None
        assert self.policy is not None
        assert self.optimizer is not None
        assert self.scheduler is not None
        assert self.trainer is not None
        path = Path(self.config.resume_from_checkpoint).expanduser().resolve()
        state_path = path / "trainer_state.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"Missing GRPO resume state: {state_path}")
        payload = torch.load(state_path, map_location=self.device, weights_only=False)
        expected_base = str(Path(self.config.generator_checkpoint).resolve())
        expected_data = str(Path(self.config.train_parquet_path).resolve())
        if payload.get("generator_checkpoint") != expected_base:
            raise ValueError("Resume checkpoint was created from a different generator")
        if payload.get("train_parquet_path") != expected_data:
            raise ValueError("Resume checkpoint was created from a different training split")
        expected_cohort = [protein.protein_id for protein in self.proteins]
        if payload.get("cohort_protein_ids") != expected_cohort:
            raise ValueError("Resume checkpoint was created for a different protein cohort")
        if payload.get("property_reward_contract") != self._property_reward_contract():
            raise ValueError(
                "Resume checkpoint was created with a different property reward contract"
            )
        _load_trainable_state_dict(self.policy, payload["policy_trainable_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.scheduler.load_state_dict(payload["scheduler_state"])
        self.trainer.global_step = int(payload["global_step"])
        self.trainer.optimization_step = int(
            payload.get(
                "optimization_step",
                self.trainer.global_step * self.config.num_iterations,
            )
        )
        self.start_epoch = int(payload["epoch"])
        self.start_index = int(payload["next_index"])
        self.proteins_seen = int(payload.get("proteins_seen", self.trainer.global_step))
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state_all" in payload:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        LOGGER.info("Resumed GRPO from %s at step %d", path, self.trainer.global_step)
        return payload.get("wandb_run_id")

    def _wandb_config(self) -> dict[str, Any]:
        values = vars(self.config).copy()
        for key, value in list(values.items()):
            if isinstance(value, Path):
                values[key] = str(value)
        values.update(
            {
                "unique_training_proteins": len(self.proteins),
                "source_unique_proteins": self.source_unique_protein_count,
                "evaluation_proteins": len(self.evaluation_targets),
                "reward_requires_structure_aware_sequence": True,
                "reward_requires_eos": True,
            }
        )
        if self.policy is not None:
            values.update(
                {
                    f"parameters/{key}": value
                    for key, value in self.policy.parameter_counts().items()
                }
            )
        return values

    def _init_wandb(self, resume_id: str | None) -> None:
        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=self.config.wandb_run_name,
            id=resume_id,
            resume="must" if resume_id else None,
            mode=self.config.wandb_mode,
            config=self._wandb_config(),
            dir=str(self.output_dir),
        )

    def _protein_order(self, epoch: int) -> list[ProteinRecord]:
        order = list(self.proteins)
        random.Random(self.config.seed + epoch).shuffle(order)
        return order

    def _tokenize(self, protein: ProteinRecord) -> tuple[torch.Tensor, torch.Tensor]:
        return tokenize_protein_sequences_for_inference(
            [protein.protein_sequence],
            prot_tokenizer=self.protein_tokenizer,
            prot_emb_model=self.config.prot_emb_model,
            prot_max_length=self.config.prot_max_length,
            device=self.device,
        )

    def _fcd_metric(self):
        if self._fcd is None:
            try:
                from fcd_torch import FCD
            except ImportError as error:
                raise ImportError(
                    "Periodic FCD requires fcd-torch; install requirements.txt"
                ) from error
            # fcd-torch 1.0.7 still calls the NumPy 1.x ``row_stack`` alias,
            # which was removed in NumPy 2. Keep the pinned implementation
            # usable without changing its ChemNet weights or FCD calculation.
            if not hasattr(np, "row_stack"):
                np.row_stack = np.vstack  # type: ignore[attr-defined]
            self._fcd = _ChemNetFCD(
                FCD(
                    device=str(self.device),
                    n_jobs=self.config.fcd_num_workers,
                    batch_size=self.config.fcd_batch_size,
                    model_path=self.config.fcd_model_path,
                    canonize=True,
                )
            )
        return self._fcd

    def _generate_evaluation_group(
        self,
        target: EvaluationTarget,
        *,
        seed: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        assert self.policy is not None
        assert self.reward_scorer is not None
        protein_ids, protein_mask = self._tokenize(target.protein)
        cuda_devices = (
            [self.device.index or 0] if self.device.type == "cuda" else []
        )
        was_training = self.policy.training
        self.policy.eval()
        with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            autocast = (
                torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
                if self.config.precision == "bf16"
                else torch.autocast(device_type=self.device.type, enabled=False)
            )
            with autocast:
                embeddings = self.policy.encode_protein(protein_ids, protein_mask)
                generated_ids = self.policy.generate_from_protein_embeddings(
                    embeddings.repeat(self.config.eval_samples_per_protein, 1, 1),
                    protein_mask.repeat(self.config.eval_samples_per_protein, 1),
                    max_length=self.config.max_mol_len,
                    min_length=self.config.min_mol_len,
                    do_sample=True,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
        if was_training:
            self.policy.train()
        selfies = [
            str(value).replace(" ", "")
            for value in self.molecule_tokenizer.batch_decode(
                generated_ids.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ]
        smiles = [selfies_to_smiles(value) for value in selfies]
        eos_id = int(self.policy.config.eos_token_id)
        terminated = generated_ids[:, 1:].eq(eos_id).any(dim=1).cpu().tolist()
        chemically_valid = [
            bool(valid_selfies(value) and smiles[index])
            for index, value in enumerate(selfies)
        ]
        valid = [
            bool(done and chemical)
            for done, chemical in zip(terminated, chemically_valid)
        ]
        valid_indices = [index for index, value in enumerate(valid) if value]
        valid_smiles = [smiles[index] for index in valid_indices]
        rewards = np.zeros(len(selfies), dtype=np.float32)
        diagnostic_names = (
            "activity_probability",
            "logp_penalty_factor",
            "sas_penalty_factor",
            "property_penalty_factor",
            "logp_excess_z",
            "sas_excess_z",
            "logp_violation",
            "sas_violation",
            "property_shaped_reward",
        )
        diagnostics = {
            name: np.zeros(len(selfies), dtype=np.float32)
            for name in diagnostic_names
        }
        if valid_indices:
            reward_values = self.reward_scorer(
                [target.protein.reward_protein_sequence] * len(valid_indices),
                [selfies[index] for index in valid_indices],
            ).numpy()
            rewards[valid_indices] = reward_values
            diagnostics_provider = getattr(
                self.reward_scorer,
                "last_diagnostics",
                None,
            )
            if callable(diagnostics_provider):
                valid_diagnostics = diagnostics_provider()
                for name in diagnostic_names:
                    values = np.asarray(valid_diagnostics[name], dtype=np.float32)
                    if len(values) != len(valid_indices):
                        raise ValueError(
                            f"Evaluation reward diagnostic {name!r} is misaligned"
                        )
                    diagnostics[name][valid_indices] = values
            else:
                diagnostics["activity_probability"][valid_indices] = reward_values
                diagnostics["logp_penalty_factor"][valid_indices] = 1.0
                diagnostics["sas_penalty_factor"][valid_indices] = 1.0
                diagnostics["property_penalty_factor"][valid_indices] = 1.0
                diagnostics["property_shaped_reward"][valid_indices] = reward_values
        property_rewards = rewards.copy()
        diversity = internal_diversity_factors(
            smiles,
            valid,
            group_size=self.config.group_size,
            reward_weight=self.config.diversity_reward_weight,
            mean_similarity_weight=self.config.diversity_mean_similarity_weight,
            duplicate_penalty_factor=(
                self.config.diversity_duplicate_penalty_factor
            ),
            morgan_radius=self.config.diversity_morgan_radius,
            morgan_bits=self.config.diversity_morgan_bits,
        )
        diagnostics.update(diversity.as_dict())
        global_diversity = internal_diversity_factors(
            smiles,
            valid,
            group_size=len(smiles),
            reward_weight=self.config.diversity_reward_weight,
            mean_similarity_weight=self.config.diversity_mean_similarity_weight,
            duplicate_penalty_factor=(
                self.config.diversity_duplicate_penalty_factor
            ),
            morgan_radius=self.config.diversity_morgan_radius,
            morgan_bits=self.config.diversity_morgan_bits,
        ).as_dict()
        local_comparable = diagnostics["diversity_comparable"].astype(bool)
        global_comparable = global_diversity["diversity_comparable"].astype(bool)
        scaffold_metrics = scaffold_diversity_summary(smiles, valid)
        if self.config.diversity_reward_shaping:
            rewards *= diversity.penalty_factor
        diagnostics["final_reward"] = rewards.copy()
        properties = _evaluation_property_summary(valid_smiles)
        aligned_properties = molecular_property_rows(
            [smiles[index] if valid[index] else "" for index in range(len(smiles))]
        )
        fcd_value: float | None = None
        if len(valid_smiles) >= 2 and len(target.active_reference_smiles) >= 2:
            candidate = float(
                self._fcd_metric()(
                    ref=list(target.active_reference_smiles),
                    gen=valid_smiles,
                )
            )
            if np.isfinite(candidate):
                fcd_value = candidate
        summary = {
            "protein_id": target.protein.protein_id,
            "reference_actives": len(target.active_reference_smiles),
            "generated": len(selfies),
            "valid_fraction": float(np.mean(valid)),
            "terminated_fraction": float(np.mean(terminated)),
            "valid_unique_fraction": (
                len(set(valid_smiles)) / len(valid_smiles) if valid_smiles else 0.0
            ),
            "reward_mean": float(rewards.mean()),
            "valid_reward_mean": (
                float(rewards[valid_indices].mean()) if valid_indices else 0.0
            ),
            "property_shaped_reward_mean": float(property_rewards.mean()),
            "valid_property_shaped_reward_mean": (
                float(property_rewards[valid_indices].mean())
                if valid_indices
                else 0.0
            ),
            "activity_probability_mean": float(
                diagnostics["activity_probability"].mean()
            ),
            "valid_activity_probability_mean": (
                float(
                    diagnostics["activity_probability"][valid_indices].mean()
                )
                if valid_indices
                else 0.0
            ),
            "property_penalty_factor_mean": (
                float(
                    diagnostics["property_penalty_factor"][valid_indices].mean()
                )
                if valid_indices
                else 0.0
            ),
            "logp_penalty_factor_mean": (
                float(diagnostics["logp_penalty_factor"][valid_indices].mean())
                if valid_indices
                else 0.0
            ),
            "sas_penalty_factor_mean": (
                float(diagnostics["sas_penalty_factor"][valid_indices].mean())
                if valid_indices
                else 0.0
            ),
            "logp_violation_fraction": (
                float(diagnostics["logp_violation"][valid_indices].mean())
                if valid_indices
                else 0.0
            ),
            "sas_violation_fraction": (
                float(diagnostics["sas_violation"][valid_indices].mean())
                if valid_indices
                else 0.0
            ),
            "diversity_penalty_factor_mean": (
                float(
                    diagnostics["diversity_penalty_factor"][valid_indices].mean()
                )
                if valid_indices
                else 0.0
            ),
            "internal_diversity_mean": (
                1.0
                - float(
                    diagnostics["mean_tanimoto_similarity"][local_comparable].mean()
                )
                if local_comparable.any()
                else 0.0
            ),
            "mean_tanimoto_similarity": (
                float(
                    diagnostics["mean_tanimoto_similarity"][local_comparable].mean()
                )
                if local_comparable.any()
                else 0.0
            ),
            "max_tanimoto_similarity": (
                float(
                    diagnostics["max_tanimoto_similarity"][local_comparable].max()
                )
                if local_comparable.any()
                else 0.0
            ),
            "nearest_neighbor_tanimoto_similarity_mean": (
                float(
                    diagnostics["max_tanimoto_similarity"]
                    [local_comparable]
                    .mean()
                )
                if local_comparable.any()
                else 0.0
            ),
            "combined_tanimoto_similarity_mean": (
                float(
                    diagnostics["combined_tanimoto_similarity"]
                    [local_comparable]
                    .mean()
                )
                if local_comparable.any()
                else 0.0
            ),
            "diversity_score_mean": (
                float(
                    diagnostics["diversity_score"][local_comparable].mean()
                )
                if local_comparable.any()
                else 0.0
            ),
            "exact_duplicate_fraction": (
                float(diagnostics["exact_duplicate"][valid_indices].mean())
                if valid_indices
                else 0.0
            ),
            "diversity_comparable_fraction": (
                float(diagnostics["diversity_comparable"][valid_indices].mean())
                if valid_indices
                else 0.0
            ),
            "global_internal_diversity_mean": (
                1.0
                - float(
                    global_diversity["mean_tanimoto_similarity"]
                    [global_comparable]
                    .mean()
                )
                if global_comparable.any()
                else 0.0
            ),
            "global_mean_tanimoto_similarity": (
                float(
                    global_diversity["mean_tanimoto_similarity"]
                    [global_comparable]
                    .mean()
                )
                if global_comparable.any()
                else 0.0
            ),
            "global_max_tanimoto_similarity": (
                float(
                    global_diversity["max_tanimoto_similarity"]
                    [global_comparable]
                    .max()
                )
                if global_comparable.any()
                else 0.0
            ),
            "global_nearest_neighbor_tanimoto_similarity_mean": (
                float(
                    global_diversity["max_tanimoto_similarity"]
                    [global_comparable]
                    .mean()
                )
                if global_comparable.any()
                else 0.0
            ),
            "global_combined_tanimoto_similarity_mean": (
                float(
                    global_diversity["combined_tanimoto_similarity"]
                    [global_comparable]
                    .mean()
                )
                if global_comparable.any()
                else 0.0
            ),
            "global_diversity_score_mean": (
                float(
                    global_diversity["diversity_score"][global_comparable].mean()
                )
                if global_comparable.any()
                else 0.0
            ),
            "global_exact_duplicate_fraction": (
                float(global_diversity["exact_duplicate"][valid_indices].mean())
                if valid_indices
                else 0.0
            ),
            **scaffold_metrics,
            "fcd": fcd_value,
            **properties,
        }
        molecule_rows = [
            {
                "protein_id": target.protein.protein_id,
                "sample_index": index,
                "generated_selfies": selfies[index],
                "generated_smiles": smiles[index],
                "terminated": bool(terminated[index]),
                "chemically_valid": chemically_valid[index],
                "reward_eligible": valid[index],
                "activity_reward": float(rewards[index]),
                "activity_probability": float(
                    diagnostics["activity_probability"][index]
                ),
                "property_shaped_reward": float(property_rewards[index]),
                "final_reward": float(rewards[index]),
                "logp_penalty_factor": float(
                    diagnostics["logp_penalty_factor"][index]
                ),
                "sas_penalty_factor": float(
                    diagnostics["sas_penalty_factor"][index]
                ),
                "property_penalty_factor": float(
                    diagnostics["property_penalty_factor"][index]
                ),
                "logp_excess_z": float(diagnostics["logp_excess_z"][index]),
                "sas_excess_z": float(diagnostics["sas_excess_z"][index]),
                "logp_violation": bool(diagnostics["logp_violation"][index]),
                "sas_violation": bool(diagnostics["sas_violation"][index]),
                "diversity_penalty_factor": float(
                    diagnostics["diversity_penalty_factor"][index]
                ),
                "mean_tanimoto_similarity": float(
                    diagnostics["mean_tanimoto_similarity"][index]
                ),
                "max_tanimoto_similarity": float(
                    diagnostics["max_tanimoto_similarity"][index]
                ),
                "combined_tanimoto_similarity": float(
                    diagnostics["combined_tanimoto_similarity"][index]
                ),
                "diversity_score": float(diagnostics["diversity_score"][index]),
                "global_mean_tanimoto_similarity": float(
                    global_diversity["mean_tanimoto_similarity"][index]
                ),
                "global_max_tanimoto_similarity": float(
                    global_diversity["max_tanimoto_similarity"][index]
                ),
                "exact_duplicate": bool(diagnostics["exact_duplicate"][index]),
                "global_exact_duplicate": bool(
                    global_diversity["exact_duplicate"][index]
                ),
                "diversity_comparable": bool(
                    diagnostics["diversity_comparable"][index]
                ),
                **aligned_properties[index],
            }
            for index in range(len(selfies))
        ]
        return summary, molecule_rows

    @staticmethod
    def _macro(rows: Sequence[Mapping[str, Any]], key: str) -> float:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return float(np.mean(values)) if values else 0.0

    def _save_evaluation_snapshot(
        self,
        *,
        snapshot_name: str,
        global_step: int,
        metrics: Mapping[str, float],
        per_protein_rows: Sequence[Mapping[str, Any]],
        molecule_rows: Sequence[Mapping[str, Any]],
    ) -> Path:
        snapshot_dir = self.output_dir / "evaluation" / snapshot_name
        if snapshot_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite evaluation snapshot: {snapshot_dir}"
            )
        snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{snapshot_name}-",
                dir=snapshot_dir.parent,
            )
        )
        metadata = {
            "snapshot": snapshot_name,
            "global_step": global_step,
            "sampling_seed": self.config.eval_seed,
            "samples_per_protein": self.config.eval_samples_per_protein,
            "diversity_group_size": self.config.group_size,
            "diversity_group_count_per_protein": (
                self.config.eval_samples_per_protein // self.config.group_size
            ),
            "diversity_reward_shaping": self.config.diversity_reward_shaping,
            "diversity_reward_weight": self.config.diversity_reward_weight,
            "diversity_mean_similarity_weight": (
                self.config.diversity_mean_similarity_weight
            ),
            "diversity_duplicate_penalty_factor": (
                self.config.diversity_duplicate_penalty_factor
            ),
            "diversity_morgan_radius": self.config.diversity_morgan_radius,
            "diversity_morgan_bits": self.config.diversity_morgan_bits,
            "protein_ids": [row["protein_id"] for row in per_protein_rows],
            "generator_checkpoint": str(
                Path(self.config.generator_checkpoint).expanduser().resolve()
            ),
        }
        try:
            pq.write_table(
                pa.Table.from_pylist(list(per_protein_rows)),
                temporary / "per_protein_metrics.parquet",
            )
            pq.write_table(
                pa.Table.from_pylist(list(molecule_rows)),
                temporary / "generated_molecules.parquet",
            )
            (temporary / "metrics.json").write_text(
                json.dumps(dict(metrics), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            (temporary / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(snapshot_dir)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        LOGGER.info("Saved %s evaluation snapshot to %s", snapshot_name, snapshot_dir)
        return snapshot_dir

    def evaluate(
        self,
        *,
        global_step: int,
        snapshot_name: str | None = None,
    ) -> dict[str, float]:
        if not self.evaluation_targets:
            return {}
        started = time.perf_counter()
        results = [
            self._generate_evaluation_group(
                target,
                seed=self.config.eval_seed + index,
            )
            for index, target in enumerate(self.evaluation_targets)
        ]
        rows = [summary for summary, _ in results]
        molecule_rows = [
            {"global_step": global_step, "snapshot": snapshot_name, **row}
            for _, generated_rows in results
            for row in generated_rows
        ]
        metrics = {
            "eval/protein_count": float(len(rows)),
            "eval/generated_count": float(sum(row["generated"] for row in rows)),
            "eval/valid_fraction_macro": self._macro(rows, "valid_fraction"),
            "eval/terminated_fraction_macro": self._macro(rows, "terminated_fraction"),
            "eval/valid_unique_fraction_macro": self._macro(
                rows, "valid_unique_fraction"
            ),
            "eval/reward_mean_macro": self._macro(rows, "reward_mean"),
            "eval/valid_reward_mean_macro": self._macro(rows, "valid_reward_mean"),
            "eval/property_shaped_reward_mean_macro": self._macro(
                rows,
                "property_shaped_reward_mean",
            ),
            "eval/valid_property_shaped_reward_mean_macro": self._macro(
                rows,
                "valid_property_shaped_reward_mean",
            ),
            "eval/activity_probability_mean_macro": self._macro(
                rows,
                "activity_probability_mean",
            ),
            "eval/valid_activity_probability_mean_macro": self._macro(
                rows,
                "valid_activity_probability_mean",
            ),
            "eval/property_penalty_factor_mean_macro": self._macro(
                rows,
                "property_penalty_factor_mean",
            ),
            "eval/logp_penalty_factor_mean_macro": self._macro(
                rows,
                "logp_penalty_factor_mean",
            ),
            "eval/sas_penalty_factor_mean_macro": self._macro(
                rows,
                "sas_penalty_factor_mean",
            ),
            "eval/logp_violation_fraction_macro": self._macro(
                rows,
                "logp_violation_fraction",
            ),
            "eval/sas_violation_fraction_macro": self._macro(
                rows,
                "sas_violation_fraction",
            ),
            "eval/diversity_penalty_factor_mean_macro": self._macro(
                rows,
                "diversity_penalty_factor_mean",
            ),
            "eval/internal_diversity_mean_macro": self._macro(
                rows,
                "internal_diversity_mean",
            ),
            "eval/mean_tanimoto_similarity_macro": self._macro(
                rows,
                "mean_tanimoto_similarity",
            ),
            "eval/max_tanimoto_similarity_macro": self._macro(
                rows,
                "max_tanimoto_similarity",
            ),
            "eval/nearest_neighbor_tanimoto_similarity_mean_macro": self._macro(
                rows,
                "nearest_neighbor_tanimoto_similarity_mean",
            ),
            "eval/combined_tanimoto_similarity_mean_macro": self._macro(
                rows,
                "combined_tanimoto_similarity_mean",
            ),
            "eval/diversity_score_mean_macro": self._macro(
                rows,
                "diversity_score_mean",
            ),
            "eval/exact_duplicate_fraction_macro": self._macro(
                rows,
                "exact_duplicate_fraction",
            ),
            "eval/diversity_comparable_fraction_macro": self._macro(
                rows,
                "diversity_comparable_fraction",
            ),
            "eval/global_internal_diversity_mean_macro": self._macro(
                rows,
                "global_internal_diversity_mean",
            ),
            "eval/global_mean_tanimoto_similarity_macro": self._macro(
                rows,
                "global_mean_tanimoto_similarity",
            ),
            "eval/global_max_tanimoto_similarity_macro": self._macro(
                rows,
                "global_max_tanimoto_similarity",
            ),
            "eval/global_nearest_neighbor_tanimoto_similarity_mean_macro": (
                self._macro(
                    rows,
                    "global_nearest_neighbor_tanimoto_similarity_mean",
                )
            ),
            "eval/global_combined_tanimoto_similarity_mean_macro": self._macro(
                rows,
                "global_combined_tanimoto_similarity_mean",
            ),
            "eval/global_diversity_score_mean_macro": self._macro(
                rows,
                "global_diversity_score_mean",
            ),
            "eval/global_exact_duplicate_fraction_macro": self._macro(
                rows,
                "global_exact_duplicate_fraction",
            ),
            "eval/scaffold_available_fraction_macro": self._macro(
                rows,
                "scaffold_available_fraction",
            ),
            "eval/scaffold_unique_fraction_macro": self._macro(
                rows,
                "scaffold_unique_fraction",
            ),
            "eval/qed_mean_macro": self._macro(rows, "qed_mean"),
            "eval/sas_mean_macro": self._macro(rows, "sas_mean"),
            "eval/logp_mean_macro": self._macro(rows, "logp_mean"),
            "eval/property_available_fraction": float(
                sum(row["count"] > 0 for row in rows) / len(rows)
            ),
            "eval/fcd_macro": self._macro(rows, "fcd"),
            "eval/fcd_available_fraction": float(
                sum(row["fcd"] is not None for row in rows) / len(rows)
            ),
            "eval/runtime_seconds": time.perf_counter() - started,
            "trainer/global_step": float(global_step),
        }
        wandb_target_ids = set(self.config.wandb_target_metric_ids)
        for row in rows:
            if row["protein_id"] not in wandb_target_ids:
                continue
            prefix = f"eval/targets/{row['protein_id']}"
            for key in (
                "valid_fraction",
                "terminated_fraction",
                "valid_unique_fraction",
                "reward_mean",
                "valid_reward_mean",
                "property_shaped_reward_mean",
                "valid_property_shaped_reward_mean",
                "activity_probability_mean",
                "valid_activity_probability_mean",
                "property_penalty_factor_mean",
                "logp_penalty_factor_mean",
                "sas_penalty_factor_mean",
                "logp_violation_fraction",
                "sas_violation_fraction",
                "diversity_penalty_factor_mean",
                "internal_diversity_mean",
                "mean_tanimoto_similarity",
                "max_tanimoto_similarity",
                "nearest_neighbor_tanimoto_similarity_mean",
                "combined_tanimoto_similarity_mean",
                "diversity_score_mean",
                "exact_duplicate_fraction",
                "diversity_comparable_fraction",
                "global_internal_diversity_mean",
                "global_mean_tanimoto_similarity",
                "global_max_tanimoto_similarity",
                "global_nearest_neighbor_tanimoto_similarity_mean",
                "global_combined_tanimoto_similarity_mean",
                "global_diversity_score_mean",
                "global_exact_duplicate_fraction",
                "scaffold_available_fraction",
                "scaffold_unique_fraction",
                "qed_mean",
                "sas_mean",
                "logp_mean",
                "fcd",
            ):
                if row[key] is not None:
                    metrics[f"{prefix}/{key}"] = float(row[key])
        if snapshot_name is not None:
            self._save_evaluation_snapshot(
                snapshot_name=snapshot_name,
                global_step=global_step,
                metrics=metrics,
                per_protein_rows=rows,
                molecule_rows=molecule_rows,
            )
        if self.wandb_run is not None:
            table = wandb.Table(
                columns=list(rows[0]),
                data=[[row[column] for column in rows[0]] for row in rows],
            )
            self.wandb_run.log(
                {**metrics, "eval/per_protein": table},
                step=global_step,
            )
        LOGGER.info("Evaluation at step %d: %s", global_step, metrics)
        return metrics

    def _runtime_metrics(self, step_seconds: float) -> dict[str, float]:
        values = {
            "runtime/step_seconds": step_seconds,
            "runtime/process_rss_gib": 0.0,
        }
        try:
            import psutil

            values["runtime/process_rss_gib"] = psutil.Process().memory_info().rss / GIB
        except ImportError:
            pass
        if self.device.type == "cuda":
            values.update(
                {
                    "runtime/gpu_memory_allocated_gib": torch.cuda.memory_allocated(
                        self.device
                    )
                    / GIB,
                    "runtime/gpu_memory_reserved_gib": torch.cuda.memory_reserved(
                        self.device
                    )
                    / GIB,
                    "runtime/gpu_peak_memory_allocated_gib": torch.cuda.max_memory_allocated(
                        self.device
                    )
                    / GIB,
                    "runtime/gpu_peak_memory_reserved_gib": torch.cuda.max_memory_reserved(
                        self.device
                    )
                    / GIB,
                }
            )
        return values

    def _save_final_model(self) -> Path:
        assert self.policy is not None
        final_dir = self.output_dir / "final"
        if final_dir.exists():
            raise FileExistsError(f"Refusing to overwrite final model: {final_dir}")
        temporary = Path(
            tempfile.mkdtemp(prefix=".final-", dir=self.output_dir)
        )
        model_config = dict(getattr(self.policy, "_config", {}))
        model_config.update(
            {
                "max_mol_len": self.config.max_mol_len,
                "prot_max_length": self.config.prot_max_length,
                "train_encoder_model": False,
                "train_projection_model": False,
                "train_decoder_model": False,
            }
        )
        try:
            torch.save(self.policy.state_dict(), temporary / "pytorch_model.bin")
            save_model_config(str(temporary), model_config, logger=LOGGER)
            (temporary / "grpo_config.json").write_text(
                json.dumps(
                    self._wandb_config(),
                    indent=2,
                    sort_keys=True,
                    default=str,
                ),
                encoding="utf-8",
            )
            temporary.replace(final_dir)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return final_dir

    def run(self) -> dict[str, Any]:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")
            torch.cuda.manual_seed_all(self.config.seed)
            torch.cuda.reset_peak_memory_stats(self.device)

        self._load_data()
        self._load_models()
        resume_id = self._load_resume_checkpoint()
        self._init_wandb(resume_id)
        assert self.trainer is not None
        assert self.scheduler is not None
        assert self.optimizer is not None
        last_epoch = self.start_epoch
        last_next_index = self.start_index
        try:
            if self.config.eval_at_start and self.trainer.global_step == 0:
                self.evaluate(global_step=0, snapshot_name="start")
            stop = False
            for epoch in range(self.start_epoch, self.config.epochs):
                order = self._protein_order(epoch)
                index_start = self.start_index if epoch == self.start_epoch else 0
                for index in range(index_start, len(order)):
                    if (
                        self.config.max_steps is not None
                        and self.trainer.global_step >= self.config.max_steps
                    ):
                        stop = True
                        break
                    protein = order[index]
                    started = time.perf_counter()
                    protein_ids, protein_mask = self._tokenize(protein)
                    output = self.trainer.step(
                        protein_input_ids=protein_ids,
                        protein_attention_mask=protein_mask,
                        protein_sequences=[protein.protein_sequence],
                        reward_protein_sequences=[protein.reward_protein_sequence],
                    )
                    self.proteins_seen += 1
                    global_step = self.trainer.global_step
                    last_epoch = epoch
                    last_next_index = index + 1
                    if last_next_index == len(order):
                        last_epoch = epoch + 1
                        last_next_index = 0
                    metrics = {
                        **output.metrics,
                        **self._runtime_metrics(time.perf_counter() - started),
                        "train/learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                        "train/epoch": float(epoch + (index + 1) / len(order)),
                        "train/proteins_seen": float(self.proteins_seen),
                        "train/unique_proteins": float(len(self.proteins)),
                        "trainer/global_step": float(global_step),
                        "trainer/optimization_step": float(
                            self.trainer.optimization_step
                        ),
                    }
                    if global_step % self.config.logging_steps == 0:
                        self.wandb_run.log(metrics, step=global_step)
                        LOGGER.info(
                            "step=%d epoch=%.4f protein=%s reward=%.4f valid=%.3f eos=%.3f",
                            global_step,
                            metrics["train/epoch"],
                            protein.protein_id,
                            metrics["grpo/reward_mean"],
                            metrics["grpo/valid_fraction"],
                            metrics["grpo/terminated_fraction"],
                        )
                    if self.config.eval_steps and global_step % self.config.eval_steps == 0:
                        self.evaluate(global_step=global_step)
                    if self.config.save_steps and global_step % self.config.save_steps == 0:
                        self.save_checkpoint(epoch=last_epoch, next_index=last_next_index)
                self.start_index = 0
                if stop:
                    break
            if self.evaluation_targets:
                self.evaluate(
                    global_step=self.trainer.global_step,
                    snapshot_name="end",
                )
            checkpoint = self.save_checkpoint(
                epoch=last_epoch,
                next_index=last_next_index,
            )
            final_dir = self._save_final_model()
            summary = {
                "global_step": self.trainer.global_step,
                "proteins_seen": self.proteins_seen,
                "unique_training_proteins": len(self.proteins),
                "cohort_protein_ids": [
                    protein.protein_id for protein in self.proteins
                ],
                "checkpoint": str(checkpoint),
                "final_model": str(final_dir),
                "start_snapshot": str(self.output_dir / "evaluation" / "start"),
                "end_snapshot": str(self.output_dir / "evaluation" / "end"),
            }
            (self.output_dir / "training_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return summary
        finally:
            if self.wandb_run is not None:
                self.wandb_run.finish()


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = parser.add_argument_group("Random-assay data")
    data.add_argument("--train_parquet_path", required=True)
    data.add_argument("--validation_parquet_path", required=True)
    data.add_argument("--protein_id_column", default="protein_accession")
    data.add_argument("--protein_sequence_column", default="protein_sequence")
    data.add_argument("--smiles_column", default="smiles")
    data.add_argument("--label_column", default="binary_label")
    data.add_argument("--structure_aware_path", default=None)
    data.add_argument("--structure_aware_key_column", default="protein_accession")
    data.add_argument("--structure_aware_sequence_column", default=None)

    model = parser.add_argument_group("Generator")
    model.add_argument("--generator_checkpoint", required=True)
    model.add_argument("--prot_emb_model", choices=["esm2", "prot_t5"], default="esm2")
    model.add_argument("--protein_model_id", default="facebook/esm2_t33_650M_UR50D")
    model.add_argument("--protein_model_revision", default=None)
    model.add_argument("--decoder_type", choices=["gpt2", "molgen"], default="gpt2")
    model.add_argument("--decoder_model_id", default="zjunlp/MolGen-large")
    model.add_argument("--decoder_model_revision", default=None)
    model.add_argument("--n_layer", type=int, default=12)
    model.add_argument("--n_head", type=int, default=16)
    model.add_argument("--n_emb", type=int, default=1280)
    model.add_argument("--conditioning_dropout", type=float, default=0.1)
    model.add_argument("--models_base", default=None)
    model.add_argument("--max_mol_len", type=int, default=200)
    model.add_argument("--min_mol_len", type=int, default=3)
    model.add_argument("--prot_max_length", type=int, default=1000)

    reward = parser.add_argument_group("Frozen FusionDTI reward")
    reward.add_argument(
        "--fusiondti_dataset",
        choices=["BindingDB", "Biosnap", "Human"],
        default="BindingDB",
    )
    reward.add_argument("--reward_batch_size", type=int, default=8)
    reward.add_argument("--reward_max_length", type=int, default=512)
    reward.add_argument("--reward_protein_max_residues", type=int, default=510)
    reward.add_argument("--reward_protein_cache_size", type=int, default=64)
    reward.add_argument(
        "--property_reward_shaping",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    reward.add_argument("--property_allowed_sigma", type=float, default=2.0)
    reward.add_argument("--property_penalty_strength", type=float, default=0.5)
    reward.add_argument(
        "--diversity_reward_shaping",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    reward.add_argument(
        "--diversity_reward_weight",
        type=float,
        default=0.5,
    )
    reward.add_argument(
        "--diversity_mean_similarity_weight",
        type=float,
        default=0.5,
    )
    reward.add_argument(
        "--diversity_duplicate_penalty_factor",
        type=float,
        default=0.0,
    )
    reward.add_argument("--diversity_morgan_radius", type=int, default=2)
    reward.add_argument("--diversity_morgan_bits", type=int, default=2048)
    reward.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    training = parser.add_argument_group("GRPO")
    training.add_argument("--epochs", type=int, default=1)
    training.add_argument("--max_steps", type=int, default=None)
    training.add_argument("--group_size", type=int, default=8)
    training.add_argument("--learning_rate", type=float, default=1.0e-6)
    training.add_argument("--weight_decay", type=float, default=0.01)
    training.add_argument("--warmup_ratio", type=float, default=0.03)
    training.add_argument("--clip_epsilon", type=float, default=0.2)
    training.add_argument("--kl_beta", type=float, default=0.01)
    training.add_argument("--advantage_epsilon", type=float, default=1.0e-6)
    training.add_argument("--num_iterations", type=int, default=2)
    training.add_argument("--loss_type", choices=["grpo", "bnpo"], default="grpo")
    training.add_argument("--max_grad_norm", type=float, default=1.0)
    training.add_argument("--temperature", type=float, default=1.0)
    training.add_argument("--top_p", type=float, default=0.9)
    training.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--device", default="cuda")
    training.add_argument(
        "--train_on_evaluation_panel_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict GRPO updates to the fixed evaluation cohort",
    )

    evaluation = parser.add_argument_group("Periodic evaluation")
    evaluation.add_argument("--eval_steps", type=int, default=500)
    evaluation.add_argument("--eval_at_start", action=argparse.BooleanOptionalAction, default=True)
    evaluation.add_argument("--eval_proteins", type=int, default=4)
    evaluation.add_argument("--eval_protein_ids", nargs="*", default=[])
    evaluation.add_argument(
        "--wandb_target_metric_ids",
        nargs="*",
        default=["P31749", "P24941"],
        help="Panel proteins that receive individual W&B scalar series",
    )
    evaluation.add_argument("--eval_samples_per_protein", type=int, default=64)
    evaluation.add_argument("--eval_seed", type=int, default=17)
    evaluation.add_argument("--min_fcd_reference_actives", type=int, default=201)
    evaluation.add_argument("--fcd_batch_size", type=int, default=256)
    evaluation.add_argument("--fcd_num_workers", type=int, default=1)
    evaluation.add_argument("--fcd_model_path", default=None)

    output = parser.add_argument_group("Logging and checkpoints")
    output.add_argument("--output_dir", required=True)
    output.add_argument("--resume_from_checkpoint", default=None)
    output.add_argument("--logging_steps", type=int, default=1)
    output.add_argument("--save_steps", type=int, default=5)
    output.add_argument("--wandb_project", default="prot2mol-grpo")
    output.add_argument("--wandb_entity", default=None)
    output.add_argument("--wandb_run_name", default=None)
    output.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    output.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    config = parse_args_with_config(parser, section="grpo", argv=argv)
    positive = {
        "epochs": config.epochs,
        "group_size": config.group_size,
        "learning_rate": config.learning_rate,
        "max_grad_norm": config.max_grad_norm,
        "logging_steps": config.logging_steps,
        "reward_batch_size": config.reward_batch_size,
        "reward_protein_max_residues": config.reward_protein_max_residues,
        "reward_protein_cache_size": config.reward_protein_cache_size,
        "property_allowed_sigma": config.property_allowed_sigma,
        "property_penalty_strength": config.property_penalty_strength,
        "diversity_morgan_radius": config.diversity_morgan_radius,
        "diversity_morgan_bits": config.diversity_morgan_bits,
        "eval_samples_per_protein": config.eval_samples_per_protein,
    }
    invalid = {key: value for key, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"GRPO values must be positive: {invalid}")
    if config.group_size < 2:
        raise ValueError("group_size must be at least 2")
    bounded_diversity = {
        "diversity_reward_weight": config.diversity_reward_weight,
        "diversity_mean_similarity_weight": (
            config.diversity_mean_similarity_weight
        ),
        "diversity_duplicate_penalty_factor": (
            config.diversity_duplicate_penalty_factor
        ),
    }
    invalid_diversity = {
        key: value
        for key, value in bounded_diversity.items()
        if not 0.0 <= value <= 1.0
    }
    if invalid_diversity:
        raise ValueError(
            f"GRPO diversity weights must be in [0, 1]: {invalid_diversity}"
        )
    if config.max_steps is not None and config.max_steps < 1:
        raise ValueError("max_steps must be positive when provided")
    if not 0.0 <= config.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if config.reward_protein_max_residues > config.reward_max_length - 2:
        raise ValueError(
            "reward_protein_max_residues must leave room for FusionDTI's two "
            "protein special tokens"
        )
    if config.max_mol_len != 200:
        LOGGER.warning(
            "This checkpoint was trained with a 200-token molecule context; got %d",
            config.max_mol_len,
        )
    if config.eval_proteins < 0 or config.min_fcd_reference_actives < 2:
        raise ValueError("Evaluation counts are invalid")
    if config.eval_samples_per_protein % config.group_size:
        raise ValueError(
            "eval_samples_per_protein must contain complete GRPO diversity groups"
        )
    return config


def main() -> None:
    config = parse_arguments()
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    summary = GRPOTrainingRun(config).run()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
