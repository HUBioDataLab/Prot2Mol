"""Compact, self-identifying predictions from the existing evaluation pass.

No extra inference or embeddings: this preserves epoch trajectories when large
optimizer checkpoints are eventually removed. The manifest is committed last;
an incomplete pair of files is never evidence of a completed export.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def export_evaluation_predictions(*, output_dir, phase, metric_prefix, step,
                                  epoch, dataset, group_names, tensors, metadata):
    for token in (phase, metric_prefix):
        if not re.fullmatch(r"[a-zA-Z0-9_]+", token):
            raise ValueError("Prediction export phase/prefix must be simple identifiers")
    if step < 0:
        raise ValueError("Prediction export step must be nonnegative")
    arrays = {}
    for name, tensor in tensors.items():
        if tensor is None:
            continue
        tensor = tensor.detach().cpu()
        array = (tensor.float() if tensor.is_floating_point() else tensor).numpy().reshape(-1)
        if len(array) != len(dataset) or not np.isfinite(array).all():
            raise ValueError(f"Incomplete or nonfinite prediction column: {name}")
        arrays[name] = array
    if not np.array_equal(arrays["row_index"], np.arange(len(dataset))):
        raise ValueError("Prediction export requires every source row exactly once in order")
    for name in ("group_id", "target_chembl_id", "compound_id"):
        arrays[name] = np.asarray([str(value) for value in dataset[name]])
    group_indices = arrays["evaluation_group_index"]
    if group_indices.dtype.kind not in "iu" or np.any(group_indices < 0) or np.any(group_indices >= len(group_names)):
        raise ValueError("Invalid evaluation group indices")
    if not np.array_equal(np.asarray(group_names, dtype=str)[group_indices], arrays["group_id"]):
        raise ValueError("Prediction group identities differ from source rows")
    for name in ("pchembl_value", "binary_label"):
        if not np.array_equal(arrays[name], np.asarray(dataset[name], dtype=np.float32)):
            raise ValueError(f"Prediction labels differ from source rows: {name}")

    directory = Path(output_dir) / "prediction_exports" / phase
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{metric_prefix}_step{step:08d}"
    target = directory / f"{stem}.npz"
    manifest_path = directory / f"{stem}.json"
    manifest = _json_value({
        "format_version": 1, "phase": phase, "metric_prefix": metric_prefix,
        "step": int(step), "epoch": epoch, "rows": len(dataset),
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "columns": sorted(arrays), "predictions_file": target.name,
        **metadata,
    })
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        digest = previous.pop("predictions_sha256")
        if previous != manifest or not target.exists() or file_sha256(target) != digest:
            raise RuntimeError("Existing prediction export has different provenance or is corrupt")
        with np.load(target, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(not np.array_equal(saved[key], value) for key, value in arrays.items()):
                raise RuntimeError("Refusing to replace an existing prediction trajectory")
        return manifest_path
    temporary = target.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(target)
    manifest["predictions_sha256"] = file_sha256(target)
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary_manifest.replace(manifest_path)
    return manifest_path
