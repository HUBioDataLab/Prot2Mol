"""Verified whole-unit training subsets without copying token payloads.

Only the small binary-label column is replaced. Arrow token buffers remain
shared with the original cache; a row-index view restricts all training access.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .prediction_exports import file_sha256


def subset_pair_labels(frame, indices, threshold=6.):
    """Compute medians only after selection, using original source precision."""
    selected = frame.iloc[np.asarray(indices)].copy()
    if selected.empty or not np.isfinite(selected.pchembl_value).all():
        raise ValueError('Training subset needs finite, nonempty observations')
    median = selected.groupby(['target_chembl_id', 'compound_id'], sort=False).pchembl_value.transform('median')
    return (median >= threshold).to_numpy(dtype=np.uint8)


def check_whole_units(unit_ids, indices):
    units, indices = np.asarray(unit_ids), np.asarray(indices)
    if indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices):
        raise ValueError('Subset indices must be a nonempty integer vector')
    if indices[0] < 0 or indices[-1] >= len(units) or np.any(np.diff(indices.astype(np.int64)) <= 0):
        raise ValueError('Subset indices must be sorted, distinct and in bounds')
    # np.isin uses a nested scan for object/string IDs. A hash lookup keeps
    # this linear in rows even with hundreds of thousands of assay units.
    expected = np.flatnonzero(pd.Index(units).isin(pd.unique(units[indices])))
    if not np.array_equal(expected, indices):
        raise ValueError('Training subset fragments an indivisible unit')


def indexed_label_view(dataset, indices, labels, fingerprint):
    """Replace one Arrow column before indexing; never flatten token indices."""
    import pyarrow as pa
    indices, labels = np.asarray(indices), np.asarray(labels)
    if getattr(dataset, '_indices', None) is not None:
        raise ValueError('Apply the training view to its original unindexed cache')
    if labels.shape != indices.shape or not np.isin(labels, [0,1]).all():
        raise ValueError('Subset labels must be aligned binary values')
    if indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices) or indices.min() < 0 or indices.max() >= len(dataset) or len(np.unique(indices)) != len(indices):
        raise ValueError('Invalid training view indices')
    # Values outside the selected view are deliberately unused. This avoids
    # importing full-training pair medians into a smaller subset's labels.
    full_labels = np.zeros(len(dataset), dtype=np.uint8)
    full_labels[indices] = labels
    field = dataset.data.schema.field('binary_label')
    # Dataset.add_column only flattens when an index view already exists.
    # Replace the small column on the original cache first. This also works
    # with multi-shard ConcatenationTable, whose set_column is not implemented.
    relabelled = dataset.remove_columns('binary_label').add_column(
        'binary_label', pa.array(full_labels, type=field.type),
        new_fingerprint=hashlib.sha256((fingerprint+':labels').encode()).hexdigest())
    relabelled = relabelled.select_columns(dataset.column_names)
    return relabelled.select(indices.tolist(), new_fingerprint=fingerprint)


def load_training_view(dataset, data_config):
    path = Path(data_config.train_view_manifest)
    if file_sha256(path) != data_config.train_view_sha256:
        raise ValueError('Training view manifest changed after configuration')
    record = json.loads(path.read_text())
    if record['format_version'] != 1 or record['label_policy'] != 'within_selected_training_pair_median':
        raise ValueError('Unsupported training view contract')
    if data_config.enforce_binary_label_threshold:
        raise ValueError('Pair-median training views require harmonized binary-label semantics')
    source = record['source']
    if Path(source['parquet_path']).resolve() != Path(data_config.train_parquet_path).resolve():
        raise ValueError('Training view belongs to another source partition')
    if Path(source['cache_dir']).resolve() != (Path(data_config.tokenized_dataset_dir)/'train_examples').resolve():
        raise ValueError('Training view belongs to another token cache')
    if file_sha256(Path(source['parquet_path'])) != source['parquet_sha256']:
        raise ValueError('Training source changed after subset preparation')
    if dataset._fingerprint != source['fingerprint'] or len(dataset) != source['rows']:
        raise ValueError('Training dataset identity or cardinality changed')
    for name, digest in source['cache_files_sha256'].items():
        if Path(name).name != name or file_sha256(Path(source['cache_dir'])/name) != digest:
            raise ValueError('Training cache bytes changed after subset preparation')
    def artifact(name):
        item = record[name]
        target = Path(item['path'])
        if file_sha256(target) != item['sha256']:
            raise ValueError(f'Training subset artifact changed: {name}')
        return target
    indices = np.load(artifact('indices'), allow_pickle=False)
    labels = np.load(artifact('labels'), allow_pickle=False)
    metadata = pd.read_parquet(artifact('source_metadata'))
    if len(metadata) != len(dataset):
        raise ValueError('Training metadata does not cover the cache')
    for name in ('group_id', 'target_chembl_id', 'compound_id'):
        if not np.array_equal(metadata[name].astype(str), np.asarray(dataset[name], dtype=str)):
            raise ValueError('Training metadata identities are not in cache order')
    if not np.array_equal(metadata.pchembl_value.to_numpy(dtype=np.float32), np.asarray(dataset['pchembl_value'], dtype=np.float32)):
        raise ValueError('Training metadata continuous labels changed')
    check_whole_units(metadata.unit.to_numpy(), indices)
    if not np.array_equal(labels, subset_pair_labels(metadata, indices, record['activity_threshold'])):
        raise ValueError('Binary labels were not derived solely from selected training observations')
    view = indexed_label_view(dataset, indices, labels, data_config.train_view_sha256)
    if len(view) != record['statistics']['rows']:
        raise ValueError('Training view row count differs from frozen statistics')
    provenance = dict(manifest_path=str(path.resolve()), manifest_sha256=data_config.train_view_sha256,
        view_fingerprint=view._fingerprint, source_fingerprint=dataset._fingerprint,
        label_policy=record['label_policy'], unit_definition=record['unit_definition'],
        statistics=record['statistics'], token_payloads_copied=False)
    return view, provenance
