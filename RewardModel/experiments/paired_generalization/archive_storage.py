"""Verified, reversible relocation to the storage explicitly authorized by the user."""
from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil

ROOT = Path("/home/atabey/Prot2Mol")
ARCHIVE = Path("/media/16TB/archive/atabey/Prot2Mol/paired-generalization-storage")
MANIFEST = ROOT / ".cache/paired_generalization_archive_manifest.jsonl"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def record(value):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(value), flush=True)


def archive_file(source):
    if source.is_symlink():
        return
    destination = ARCHIVE / source.relative_to(ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    if destination.exists():
        source_hash = digest(source)
    else:
        temporary = destination.with_name(destination.name + ".copying")
        checksum = hashlib.sha256()
        with source.open("rb") as incoming, temporary.open("xb") as outgoing:
            for block in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
                checksum.update(block)
                outgoing.write(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        shutil.copystat(source, temporary)
        temporary.replace(destination)
        source_hash = checksum.hexdigest()
    after = source.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError(f"Source changed during copy: {source}")
    if destination.stat().st_size != before.st_size or digest(destination) != source_hash:
        raise RuntimeError(f"Archive verification failed: {source}")
    link = source.with_name(source.name + ".archive-link")
    link.symlink_to(destination)
    link.replace(source)
    record({"source": str(source), "archive": str(destination), "bytes": before.st_size, "sha256": source_hash, "original_path_is_symlink": True})


def archive_tree(source):
    if source.is_symlink():
        if source.resolve() != ARCHIVE / source.relative_to(ROOT):
            raise RuntimeError(f"Unexpected existing symlink: {source}")
        return
    destination = ARCHIVE / source.relative_to(ROOT)
    if destination.exists():
        raise RuntimeError(f"Archive destination already exists: {destination}")
    if any(p.is_symlink() for p in source.rglob("*")):
        raise RuntimeError(f"Tree contains symlinks; needs individual review: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    inventory = []
    for item in sorted(source.rglob("*")):
        if item.is_file():
            target = destination / item.relative_to(source)
            checksum = digest(item)
            if item.stat().st_size != target.stat().st_size or digest(target) != checksum:
                raise RuntimeError(f"Tree verification failed: {item}")
            inventory.append({"path": str(item.relative_to(source)), "bytes": item.stat().st_size, "sha256": checksum})
    backup = source.with_name(source.name + ".archive-source-backup")
    if backup.exists():
        raise RuntimeError(f"Backup already exists: {backup}")
    source.rename(backup)
    try:
        source.symlink_to(destination, target_is_directory=True)
    except BaseException:
        backup.rename(source)
        raise
    shutil.rmtree(backup)
    record({"source": str(source), "archive": str(destination), "files": inventory, "original_path_is_symlink": True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-storage-only", action="store_true")
    options = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("Run only from the authorized Atum project")
    work = ROOT / "RewardModel/outputs/paired_generalization"
    for name in ("prepare", "chembl_download", "baseline_driver"):
        marker = work / f"{name}_launch.json"
        if marker.exists():
            pid = json.loads(marker.read_text())["pid"]
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f"Task process {pid} is still alive; stop it before moving runtime directories")
    if not options.active_storage_only:
        # These are inactive historical GRPO states, not current training outputs.
        for source in sorted((ROOT / "outputs/prot2mol-grpo").rglob("trainer_state.pt")):
            if not source.is_symlink() and source.stat().st_size > 1024**3:
                archive_file(source)
    # Both jobs have exited. Their original project paths remain usable.
    archive_tree(ROOT / "dataset/raw/chembl_37")
    if options.active_storage_only:
        source = work / "baseline_scaffold_seed42_15ep"
        destination = ARCHIVE / source.relative_to(ROOT)
        if source.is_symlink():
            if source.resolve() != destination:
                raise RuntimeError("Unexpected baseline storage symlink")
        elif source.exists():
            raise RuntimeError("Baseline output already exists; refusing to relocate active data")
        else:
            destination.mkdir(parents=True, exist_ok=True)
            source.symlink_to(destination, target_is_directory=True)
            record({"source": str(source), "archive": str(destination), "new_empty_output_directory": True})
    else:
        archive_tree(work)


if __name__ == "__main__":
    main()
