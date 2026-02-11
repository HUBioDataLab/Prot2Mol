import json
import os
from dataclasses import dataclass

import pandas as pd

from ..data.pipeline import get_processed_data_path


@dataclass
class NormalizationStats:
    pchembl_mean: float
    pchembl_std: float
    pchembl_threshold: float
    pchembl_huber_delta_norm: float


class NormalizationService:
    """Resolve pChEMBL normalization constants used by training and evaluation."""

    def __init__(
        self,
        selfies_path: str,
        train_pchembl_head: bool,
        pchembl_huber_delta_raw: float,
        logger=None,
        default_threshold: float = 6.0,
    ):
        self.selfies_path = selfies_path
        self.train_pchembl_head = train_pchembl_head
        self.pchembl_huber_delta_raw = pchembl_huber_delta_raw
        self.default_threshold = default_threshold
        self.logger = logger

    def load_or_compute(self) -> NormalizationStats:
        """Load stats from preprocessing cache, or compute from CSV as fallback."""
        if not self.train_pchembl_head:
            if self.logger is not None:
                self.logger.info("pChEMBL head training is disabled. Using neutral normalization constants.")
            return NormalizationStats(
                pchembl_mean=0.0,
                pchembl_std=1.0,
                pchembl_threshold=self.default_threshold,
                pchembl_huber_delta_norm=self.pchembl_huber_delta_raw,
            )

        cached_stats = self._load_cached_stats()
        if cached_stats is not None:
            pchembl_mean, pchembl_std, pchembl_threshold = cached_stats
        else:
            pchembl_mean, pchembl_std, pchembl_threshold = self._compute_stats_from_csv()

        if pchembl_std > 0:
            delta_norm = self.pchembl_huber_delta_raw / pchembl_std
        else:
            delta_norm = self.pchembl_huber_delta_raw

        if self.logger is not None:
            self.logger.info(
                "Huber delta: raw=%.3f, normalized=%.3f",
                self.pchembl_huber_delta_raw,
                delta_norm,
            )

        return NormalizationStats(
            pchembl_mean=pchembl_mean,
            pchembl_std=pchembl_std,
            pchembl_threshold=pchembl_threshold,
            pchembl_huber_delta_norm=delta_norm,
        )

    def _load_cached_stats(self):
        cache_dir = os.environ.get("DATASETS_CACHE_DIR", "/gpfs/projects/etur29/atabey/datasets")
        processed_data_path = get_processed_data_path(self.selfies_path, cache_dir=cache_dir)
        stats_path = os.path.join(processed_data_path, "pchembl_stats.json")
        if not os.path.exists(stats_path):
            return None

        try:
            with open(stats_path, "r", encoding="utf-8") as handle:
                stats = json.load(handle)
            pchembl_mean = float(stats["pchembl_mean"])
            pchembl_std = float(stats["pchembl_std"])
            pchembl_threshold = float(stats.get("pchembl_threshold", self.default_threshold))
            if self.logger is not None:
                self.logger.info(
                    "Loaded pChEMBL stats from cache: mean=%.3f, std=%.3f, threshold=%.3f",
                    pchembl_mean,
                    pchembl_std,
                    pchembl_threshold,
                )
            return pchembl_mean, pchembl_std, pchembl_threshold
        except Exception as exc:
            if self.logger is not None:
                self.logger.warning("Failed to load cached pChEMBL stats: %s", exc)
            return None

    def _compute_stats_from_csv(self):
        if self.logger is not None:
            self.logger.info("Preparing normalization constants from dataset (fallback)...")

        df = pd.read_csv(self.selfies_path)
        pchembl_values = df["pchembl_value_Median"].dropna()
        pchembl_mean = float(pchembl_values.mean())
        pchembl_std = float(pchembl_values.std(ddof=0))
        pchembl_threshold = float(self.default_threshold)

        if self.logger is not None:
            self.logger.info(
                "pChEMBL normalization range: mean=%.3f, std=%.3f",
                pchembl_mean,
                pchembl_std,
            )
            self.logger.info("pChEMBL positive threshold set to: >=%.3f", pchembl_threshold)

        return pchembl_mean, pchembl_std, pchembl_threshold
