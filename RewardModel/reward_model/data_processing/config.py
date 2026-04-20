from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple


DEFAULT_ACTIVITY_TYPES: Tuple[str, ...] = (
    "IC50",
    "XC50",
    "EC50",
    "AC50",
    "Ki",
    "Kd",
    "Potency",
    "ED50",
)

DEFAULT_CONFIDENCE_SCORES: Tuple[int, ...] = (8, 9)
DEFAULT_SPLIT_RATIOS: Tuple[float, float, float] = (0.8, 0.1, 0.1)


@dataclass(eq=True)
class ChemblPreprocessConfig:
    sqlite_path: Optional[str] = None
    output_dir: Optional[str] = None
    source_url: str = "ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest"
    download_url: Optional[str] = None
    chembl_release: Optional[str] = None
    protein_model_name_or_path: str = "facebook/esm2_t12_35M_UR50D"
    molecule_model_name_or_path: str = "HUBioDataLab/SELFormer"
    protein_tokenizer_name_or_path: Optional[str] = None
    molecule_tokenizer_name_or_path: Optional[str] = None
    protein_max_length: int = 1024
    molecule_max_length: int = 512
    tokenization_batch_size: int = 256
    activity_threshold: float = 6.0
    assay_type: str = "B"
    target_type: str = "SINGLE PROTEIN"
    organism: str = "Homo sapiens"
    confidence_scores: Tuple[int, ...] = field(default_factory=lambda: DEFAULT_CONFIDENCE_SCORES)
    standard_types: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_ACTIVITY_TYPES)
    exclude_variants: bool = True
    min_group_size: int = 2
    min_assays_per_protein_for_holdout: int = 3
    split_ratios: Tuple[float, float, float] = field(default_factory=lambda: DEFAULT_SPLIT_RATIOS)
    split_seed: int = 42
    overwrite: bool = False
    write_parquet: bool = True
    raw_dir_name: str = "raw"
    curated_dir_name: str = "curated"
    tokenized_dir_name: str = "tokenized"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.protein_max_length <= 0:
            raise ValueError("protein_max_length must be > 0")
        if self.molecule_max_length <= 0:
            raise ValueError("molecule_max_length must be > 0")
        if self.tokenization_batch_size <= 0:
            raise ValueError("tokenization_batch_size must be > 0")
        if self.activity_threshold <= 0:
            raise ValueError("activity_threshold must be > 0")
        if not self.assay_type:
            raise ValueError("assay_type must be provided")
        if not self.target_type:
            raise ValueError("target_type must be provided")
        if not self.organism:
            raise ValueError("organism must be provided")
        if not self.confidence_scores:
            raise ValueError("confidence_scores must not be empty")
        if not self.standard_types:
            raise ValueError("standard_types must not be empty")
        if self.min_group_size < 2:
            raise ValueError("min_group_size must be >= 2")
        if self.min_assays_per_protein_for_holdout < 2:
            raise ValueError("min_assays_per_protein_for_holdout must be >= 2")
        if len(self.split_ratios) != 3:
            raise ValueError("split_ratios must contain train/valid/test values")
        ratio_sum = sum(self.split_ratios)
        if ratio_sum <= 0:
            raise ValueError("split_ratios must sum to a positive value")
        if any(value < 0.0 for value in self.split_ratios):
            raise ValueError("split_ratios must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChemblPreprocessConfig":
        return cls(**dict(payload))

    def save_json(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
        return path

    @classmethod
    def load_json(cls, path: str) -> "ChemblPreprocessConfig":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Config JSON at {path} must contain a mapping")
        return cls.from_dict(payload)
