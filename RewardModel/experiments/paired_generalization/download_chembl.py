"""Download and verify the official ChEMBL 37 SQLite release inside the project."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[3]
BASE = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/"
NAME = "chembl_37_sqlite.tar.gz"
EXPECTED_SHA256 = "33c203740555f96067710cdfc1c3c55d890660e5908ec5cbf5817492c290d281"


def main():
    destination = ROOT / "dataset/raw/chembl_37"
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / NAME
    database = destination / "chembl_37.db"
    manifest = destination / "download_verified.json"
    if manifest.exists() and database.is_file():
        print(manifest.read_text(), flush=True)
        return
    with urlopen(BASE + "checksums.txt", timeout=60) as response:
        checksums = response.read().decode()
    official = next(line.split()[0] for line in checksums.splitlines() if line.endswith(NAME))
    if official != EXPECTED_SHA256:
        raise RuntimeError("The official checksum changed; refusing an unreviewed release")
    (destination / "checksums.txt").write_text(checksums)
    subprocess.run([
        "curl", "--fail", "--location", "--retry", "8", "--retry-delay", "10",
        "--connect-timeout", "30", "--continue-at", "-", "--output", str(archive), BASE + NAME,
    ], check=True)
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("Archive checksum mismatch; preserved for inspection, not extracted")
    # Extract only the database bytes, never arbitrary archive paths or links.
    temporary = database.with_suffix(".db.partial")
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            if member.isfile() and Path(member.name).name == database.name:
                with tar.extractfile(member) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                if temporary.stat().st_size != member.size:
                    raise RuntimeError("Extracted database size mismatch")
                temporary.replace(database)
                break
        else:
            raise RuntimeError("Expected database absent from archive")
    manifest.write_text(json.dumps({
        "release": 37, "url": BASE + NAME, "sha256": digest.hexdigest(),
        "archive_bytes": archive.stat().st_size, "database": str(database),
        "database_bytes": database.stat().st_size,
    }, indent=2) + "\n")
    print(manifest.read_text(), flush=True)


if __name__ == "__main__":
    main()
