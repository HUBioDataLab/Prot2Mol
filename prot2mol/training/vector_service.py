import os
from typing import Optional

import numpy as np
import pandas as pd
import selfies as sf

from ..chem.utils_fps import generate_morgan_fingerprints_parallel


class VectorService:
    """Load or build train fingerprints used by generation metrics."""

    def __init__(
        self,
        selfies_path: str,
        logger=None,
        data_dir: Optional[str] = None,
        max_molecules: int = 100000,
        chunk_size: int = 10000,
    ):
        self.selfies_path = selfies_path
        self.logger = logger
        self.max_molecules = max_molecules
        self.chunk_size = chunk_size

        if data_dir is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(repo_root, "data")
        self.data_dir = data_dir
        self.train_vecs_path = os.path.join(self.data_dir, "train_vecs.npy")

    def load_or_generate(self):
        """Return training vectors, generating them on-demand if missing."""
        if os.path.exists(self.train_vecs_path):
            if self.logger is not None:
                self.logger.info("Loading existing training vectors from %s", self.train_vecs_path)
            vectors = np.load(self.train_vecs_path)
            if self.logger is not None:
                self.logger.info("Loaded training vectors with shape: %s", vectors.shape)
            return vectors

        if self.logger is not None:
            self.logger.warning("Training vectors file not found at %s", self.train_vecs_path)
            self.logger.info("Attempting to generate training vectors automatically...")

        try:
            vectors = self._generate_training_vectors()
            os.makedirs(self.data_dir, exist_ok=True)
            np.save(self.train_vecs_path, vectors)
            if self.logger is not None:
                self.logger.info(
                    "Successfully generated and loaded training vectors from %s",
                    self.train_vecs_path,
                )
            return vectors
        except Exception as exc:
            if self.logger is not None:
                self.logger.warning("Failed to auto-generate training vectors: %s", exc)
                self.logger.warning(
                    "You can manually generate train_vecs.npy using: python generate_train_vecs.py"
                )
            return None

    def _generate_training_vectors(self):
        if self.logger is not None:
            self.logger.info("Generating training vectors from dataset...")
            self.logger.info("Loading dataset from: %s", self.selfies_path)

        smiles_list = []
        total_rows = 0

        for chunk_df in pd.read_csv(self.selfies_path, chunksize=self.chunk_size):
            total_rows += len(chunk_df)
            selfies_chunk = chunk_df["Compound_SELFIES"].tolist()

            for selfies_str in selfies_chunk:
                try:
                    if selfies_str and isinstance(selfies_str, str):
                        smiles = sf.decoder(selfies_str)
                        if smiles and smiles.strip():
                            smiles_list.append(smiles)
                except Exception:
                    continue

            if self.logger is not None:
                self.logger.info(
                    "Processed %s rows, %s valid SMILES so far...",
                    total_rows,
                    len(smiles_list),
                )

            if len(smiles_list) >= self.max_molecules:
                if self.logger is not None:
                    self.logger.info(
                        "Limiting to %s molecules for training vectors",
                        len(smiles_list),
                    )
                break

        if not smiles_list:
            raise ValueError("No valid SMILES found in the dataset")

        if self.logger is not None:
            self.logger.info(
                "Generating Morgan fingerprints for %s molecules...",
                len(smiles_list),
            )

        train_vecs = generate_morgan_fingerprints_parallel(
            smiles=smiles_list,
            radius=2,
            nBits=1024,
            n_jobs=None,
        )

        if self.logger is not None:
            self.logger.info("Generated training vectors with shape: %s", train_vecs.shape)

        return train_vecs
