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


@dataclass(eq=True)
class ChemblPreprocessConfig:
    sqlite_path: Optional[str] = None
    output_dir: Optional[str] = None
    source_url: str = "ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest"
    download_url: Optional[str] = None
    chembl_release: Optional[str] = None
    activity_threshold: float = 6.0
    assay_type: str = "B"
    target_type: str = "SINGLE PROTEIN"
    organism: str = "Homo sapiens"
    confidence_scores: Tuple[int, ...] = field(default_factory=lambda: DEFAULT_CONFIDENCE_SCORES)
    standard_types: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_ACTIVITY_TYPES)
    exclude_variants: bool = True
    min_group_size: int = 2
    overwrite: bool = False
    write_parquet: bool = True
    raw_dir_name: str = "raw"
    curated_dir_name: str = "curated"

    def __post_init__(self) -> None:
        if not isinstance(self.confidence_scores, tuple):
            self.confidence_scores = tuple(self.confidence_scores)
        if not isinstance(self.standard_types, tuple):
            self.standard_types = tuple(self.standard_types)
        self.validate()

    def validate(self) -> None:
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
