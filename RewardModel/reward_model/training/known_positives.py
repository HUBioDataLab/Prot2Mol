"""Training-only positives used to mask known false contrastive negatives.

The index is external to the scorer: inference does not consult measurement
labels. Training and loss evaluation must explicitly receive the same index.
Its canonical digest is recorded with the run so it can be reconstructed from
the preserved training partition without consulting held-out labels.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math

import numpy as np
import torch


def identity_mapping(values):
    return {identity: index for index, identity in enumerate(sorted(set(map(str, values))))}


def positive_mask(positives_by_target, query_targets, candidate_molecules):
    candidates = list(candidate_molecules)
    return torch.tensor([[candidate in positives_by_target.get(target, ()) for candidate in candidates]
                         for target in query_targets], dtype=torch.bool)


class KnownPositiveIndex:
    def __init__(self, positives_by_target, *, active_threshold, source_rows):
        if not math.isfinite(float(active_threshold)):
            raise ValueError('Active threshold must be finite')
        self.positives_by_target = {str(target): frozenset(map(str, molecules))
                                   for target, molecules in positives_by_target.items()}
        self.active_threshold = float(active_threshold)
        digest = hashlib.sha256()
        for target in sorted(self.positives_by_target):
            for molecule in sorted(self.positives_by_target[target]):
                digest.update((json.dumps([target, molecule], separators=(',', ':')) + '\n').encode())
        self.provenance = dict(
            policy='Any curated training observation strictly above the active threshold after float32 conversion',
            active_threshold=self.active_threshold, source_rows=int(source_rows),
            positive_pairs=sum(map(len, self.positives_by_target.values())),
            canonical_pairs_sha256=digest.hexdigest(), heldout_labels_used=False)

    @classmethod
    def from_training_dataset(cls, training_examples, *, active_threshold):
        required = {'target_chembl_id', 'compound_id', 'pchembl_value'}
        if not required <= set(training_examples.column_names):
            raise ValueError('Known-positive index requires training identities and continuous labels')
        positives = defaultdict(set)
        metadata = training_examples.select_columns(sorted(required))
        for start in range(0, len(metadata), 65536):
            rows = metadata[start:start+65536]
            for target, molecule, value in zip(rows['target_chembl_id'], rows['compound_id'], rows['pchembl_value']):
                if target is None or molecule is None or not math.isfinite(float(value)):
                    raise ValueError('Invalid training identity/label in known-positive index')
                if float(np.float32(value)) > active_threshold:
                    positives[str(target)].add(str(molecule))
        return cls(positives, active_threshold=active_threshold, source_rows=len(training_examples))

    def mask(self, query_targets, candidate_molecules):
        return positive_mask(self.positives_by_target, map(str, query_targets), map(str, candidate_molecules))

    def encoded_for(self, target_values, molecule_values):
        """Translate training positives into this evaluation partition's ID map.

        This reads identities only from the evaluation data. An empty mapping
        is a valid result for molecule-disjoint evaluation partitions.
        """
        targets, molecules = identity_mapping(target_values), identity_mapping(molecule_values)
        return {targets[target]: frozenset(molecules[molecule] for molecule in positive_molecules if molecule in molecules)
                for target, positive_molecules in self.positives_by_target.items() if target in targets}
