#!/usr/bin/env python3
"""Build binary protein-ligand hit-prediction sources.

This intentionally keeps the first PubChem pass source-level:

* no RDKit standardization in this script
* no local PAINS filtering in this script
* exact text-level deduplication only

The default PubChem path uses the curated MF-PCBA binding dataset from
Leash-Biosciences/mf-pcba-bind, which is derived from the original MF-PCBA
PubChem retrieval scripts and already includes binding-assay and PAINS curation.
"""

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

HF_MF_PCBA_BIND_DATASET = "Leash-Biosciences/mf-pcba-bind"
HF_MF_PCBA_BIND_API = (
    "https://huggingface.co/api/datasets/Leash-Biosciences/mf-pcba-bind"
)
HF_MF_PCBA_BIND_PARQUET_API = (
    "https://datasets-server.huggingface.co/parquet?"
    "dataset=Leash-Biosciences/mf-pcba-bind"
)
HF_MF_PCBA_BIND_SIZE_API = (
    "https://datasets-server.huggingface.co/size?"
    "dataset=Leash-Biosciences/mf-pcba-bind"
)
HF_MF_PCBA_BIND_METADATA_URL = (
    "https://huggingface.co/datasets/Leash-Biosciences/mf-pcba-bind/"
    "raw/main/MF-PCBA-Assay-Metadata.csv"
)
UNIPROT_ACCESSIONS_FASTA_API = "https://rest.uniprot.org/uniprotkb/accessions"

BINARY_OUTPUT_COLUMNS = [
    "source",
    "supervision",
    "task_type",
    "dataset",
    "source_split",
    "assay_type",
    "screening_stage",
    "assay_group_id",
    "assay_id",
    "target_id",
    "protein_name",
    "protein_category",
    "protein_accession",
    "protein_sequence",
    "compound_id",
    "smiles",
    "binary_label",
    "activity_outcome",
    "dr_value",
    "sd_value",
    "sd_z_score",
    "xc50_uM",
    "log_xc50",
    "assay_num_hits",
    "assay_num_negatives",
    "assay_total_samples",
    "assay_hit_rate_percent",
    "label_rule",
    "upstream_pains_filtered",
]


@dataclass(frozen=True)
class PubChemMfPcbaConfig:
    min_assay_size: int = 100
    max_hit_rate_percent: float = 10.0
    cap_per_assay: int = 50_000
    cap_mode: str = "keep-positives"
    random_seed: int = 42
    include_source_splits: tuple[str, ...] = ("validation", "test")


@dataclass(frozen=True)
class AffinityThresholdConfig:
    pchembl_threshold: float = 6.0
    include_censored_safe_negatives: bool = True
    chunksize: int = 500_000


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def normalize_string(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def normalize_sequence(value) -> str:
    return "".join(normalize_string(value).split()).upper()


def short_hash(value: str, length: int = 16) -> str:
    return hashlib.sha1(normalize_string(value).encode("utf-8")).hexdigest()[:length]


def normalize_single_accession(value) -> str:
    accession = normalize_string(value)
    if ";" in accession or "," in accession:
        return ""
    return accession


def stable_int(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def request_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "Prot2Mol dataset builder"})
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def request_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Prot2Mol dataset builder"})
    with urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8")


def download_url(url: str, output_path: Path, chunk_size: int = 8 * 1024 * 1024) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        log(f"download skip: {output_path} already exists")
        return output_path

    log(f"download start: {url}")
    request = Request(url, headers={"User-Agent": "Prot2Mol dataset builder"})
    with urlopen(request, timeout=120) as response, output_path.open("wb") as handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
    log(f"download done: {output_path} ({output_path.stat().st_size:,} bytes)")
    return output_path


def parse_fasta_sequences(text: str) -> dict[str, str]:
    sequences: dict[str, str] = {}
    accession = None
    parts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if accession and parts:
                sequences[accession] = normalize_sequence("".join(parts))
            header = line[1:].split()[0]
            tokens = header.split("|")
            accession = tokens[1] if len(tokens) >= 2 else header
            parts = []
        else:
            parts.append(line)
    if accession and parts:
        sequences[accession] = normalize_sequence("".join(parts))
    return sequences


def load_sequence_cache(cache_path: Path | None) -> dict[str, str]:
    if cache_path is None or not cache_path.exists():
        return {}
    data = json.loads(cache_path.read_text())
    return {normalize_string(k): normalize_sequence(v) for k, v in data.items() if normalize_sequence(v)}


def save_sequence_cache(cache_path: Path | None, sequences: dict[str, str]) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(dict(sorted(sequences.items())), indent=2))


def fetch_uniprot_sequences(
    accessions: list[str],
    cache_path: Path | None,
    *,
    allow_fetch: bool = True,
    batch_size: int = 100,
) -> dict[str, str]:
    accessions = sorted({normalize_string(accession) for accession in accessions if normalize_string(accession)})
    sequences = load_sequence_cache(cache_path)
    missing = [accession for accession in accessions if accession not in sequences]
    if not missing or not allow_fetch:
        return {accession: sequences[accession] for accession in accessions if accession in sequences}

    log(f"UniProt sequence fetch: {len(missing):,} missing accessions")
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        url = f"{UNIPROT_ACCESSIONS_FASTA_API}?{urlencode({'accessions': ','.join(batch), 'format': 'fasta'})}"
        try:
            fetched = parse_fasta_sequences(request_text(url))
        except Exception as error:
            log(f"UniProt batch failed ({len(batch):,} accessions): {error}")
            if "nodename nor servname" in str(error) or "Name or service not known" in str(error):
                raise
            fetched = {}
            if len(batch) > 1:
                unresolved = []
                for accession in batch:
                    single_url = f"{UNIPROT_ACCESSIONS_FASTA_API}?{urlencode({'accessions': accession, 'format': 'fasta'})}"
                    try:
                        fetched.update(parse_fasta_sequences(request_text(single_url)))
                    except Exception as single_error:
                        unresolved.append((accession, str(single_error)))
                if unresolved:
                    examples = ", ".join(accession for accession, _ in unresolved[:5])
                    log(f"UniProt unresolved in fallback: {len(unresolved):,} accessions ({examples})")
        sequences.update(fetched)
        log(
            "UniProt batch "
            f"{start // batch_size + 1:,}/{math.ceil(len(missing) / batch_size):,}: "
            f"{len(fetched):,}/{len(batch):,} resolved"
        )
    save_sequence_cache(cache_path, sequences)
    return {accession: sequences[accession] for accession in accessions if accession in sequences}


def fill_assay_counts(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    df = frame.copy()
    counts = (
        df.groupby("assay_id")["binary_label"]
        .agg(assay_total_samples="size", assay_num_hits="sum")
        .reset_index()
    )
    counts["assay_num_negatives"] = counts["assay_total_samples"] - counts["assay_num_hits"]
    counts["assay_hit_rate_percent"] = (
        counts["assay_num_hits"] / counts["assay_total_samples"] * 100.0
    )
    count_cols = [
        "assay_num_hits",
        "assay_num_negatives",
        "assay_total_samples",
        "assay_hit_rate_percent",
    ]
    df = df.drop(columns=[column for column in count_cols if column in df.columns], errors="ignore")
    return df.merge(counts, on="assay_id", how="left")


def ensure_binary_schema(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    source = df["source"].iloc[0] if "source" in df.columns and not df.empty else ""
    defaults = {
        "supervision": "binary",
        "task_type": "hit_prediction",
        "assay_type": "",
        "screening_stage": "",
        "assay_group_id": "",
        "protein_category": "",
        "activity_outcome": "",
        "dr_value": pd.NA,
        "sd_value": pd.NA,
        "sd_z_score": pd.NA,
        "xc50_uM": pd.NA,
        "log_xc50": pd.NA,
        "assay_num_hits": pd.NA,
        "assay_num_negatives": pd.NA,
        "assay_total_samples": pd.NA,
        "assay_hit_rate_percent": pd.NA,
        "label_rule": "",
        "upstream_pains_filtered": False,
    }
    for column in BINARY_OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = defaults.get(column, "")

    if source == "PubChem_HTS":
        df.loc[df["assay_type"].eq(""), "assay_type"] = "hts_binding"
        df.loc[df["screening_stage"].eq(""), "screening_stage"] = "mf_pcba_primary_confirmatory"
    df["assay_group_id"] = df["assay_group_id"].map(normalize_string)
    missing_group = df["assay_group_id"].eq("")
    if missing_group.any():
        df.loc[missing_group, "assay_group_id"] = (
            df.loc[missing_group, "source"].map(normalize_string)
            + ":"
            + df.loc[missing_group, "assay_id"].map(normalize_string)
            + ":"
            + df.loc[missing_group, "target_id"].map(normalize_string)
        )
    return df[BINARY_OUTPUT_COLUMNS]


def download_mf_pcba_bind(raw_dir: Path) -> tuple[Path, list[Path], dict]:
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = request_json(HF_MF_PCBA_BIND_API)
    parquet_manifest = request_json(HF_MF_PCBA_BIND_PARQUET_API)
    size_info = request_json(HF_MF_PCBA_BIND_SIZE_API)

    (raw_dir / "hf_dataset_info.json").write_text(json.dumps(dataset_info, indent=2))
    (raw_dir / "parquet_manifest.json").write_text(json.dumps(parquet_manifest, indent=2))
    (raw_dir / "size.json").write_text(json.dumps(size_info, indent=2))

    metadata_path = download_url(
        HF_MF_PCBA_BIND_METADATA_URL,
        raw_dir / "MF-PCBA-Assay-Metadata.csv",
    )

    parquet_paths = []
    for item in parquet_manifest["parquet_files"]:
        split = item["split"]
        filename = item["filename"]
        output_path = raw_dir / "parquet" / split / filename
        download_url(item["url"], output_path)
        parquet_paths.append(output_path)

    return metadata_path, parquet_paths, size_info


def load_assay_metadata(metadata_path: Path, config: PubChemMfPcbaConfig) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    metadata = metadata.rename(
        columns={
            "bind/phenotypic": "assay_kind",
            "Hit_Rate_%": "assay_hit_rate_percent",
            "Num_Hits": "assay_num_hits",
            "Num_Negatives": "assay_num_negatives",
            "Total_Samples": "assay_total_samples",
        }
    )
    required = {
        "AID",
        "assay_kind",
        "protein_name",
        "protein_category",
        "protein_accession",
        "amino_acid_sequence",
        "assay_num_hits",
        "assay_num_negatives",
        "assay_total_samples",
        "assay_hit_rate_percent",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Missing MF-PCBA metadata columns: {missing}")

    metadata["assay_id"] = metadata["AID"].map(normalize_string)
    metadata["assay_kind"] = metadata["assay_kind"].map(lambda value: normalize_string(value).lower())
    metadata["protein_sequence_meta"] = metadata["amino_acid_sequence"].map(normalize_sequence)
    metadata["assay_total_samples"] = pd.to_numeric(metadata["assay_total_samples"], errors="coerce")
    metadata["assay_hit_rate_percent"] = pd.to_numeric(
        metadata["assay_hit_rate_percent"],
        errors="coerce",
    )

    keep = metadata["assay_kind"].eq("bind")
    keep &= metadata["assay_total_samples"] >= config.min_assay_size
    keep &= metadata["assay_hit_rate_percent"] < config.max_hit_rate_percent
    keep &= metadata["protein_sequence_meta"].ne("")
    filtered = metadata.loc[keep].copy()
    log(
        "metadata filters: "
        f"{len(metadata):,} assays -> {len(filtered):,} binding assays "
        f"(min_assay_size={config.min_assay_size}, "
        f"max_hit_rate_percent={config.max_hit_rate_percent})"
    )
    return filtered[
        [
            "assay_id",
            "protein_name",
            "protein_category",
            "protein_accession",
            "protein_sequence_meta",
            "assay_num_hits",
            "assay_num_negatives",
            "assay_total_samples",
            "assay_hit_rate_percent",
        ]
    ]


def normalize_mf_pcba_frame(
    frame: pd.DataFrame,
    source_split: str,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "CID",
        "smiles",
        "binds",
        "Activity",
        "AID",
        "amino_acid_sequence",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing MF-PCBA parquet columns: {missing}")

    df = frame.copy()
    df["assay_id"] = df["AID"].map(normalize_string)
    df = df.merge(metadata, on="assay_id", how="inner", suffixes=("", "_meta"))
    df["binary_label"] = pd.to_numeric(df["binds"], errors="coerce")
    df = df.loc[df["binary_label"].isin([0, 1])].copy()
    df["binary_label"] = df["binary_label"].astype(int)
    df["smiles"] = df["smiles"].map(normalize_string)
    df["protein_sequence"] = df["amino_acid_sequence"].map(normalize_sequence)
    missing_sequence = df["protein_sequence"].eq("")
    df.loc[missing_sequence, "protein_sequence"] = df.loc[missing_sequence, "protein_sequence_meta"]
    df = df.loc[df["smiles"].ne("") & df["protein_sequence"].ne("")].copy()

    protein_accession = df["protein_accession"].map(normalize_string)
    protein_name = df["protein_name"].map(normalize_string)
    df["target_id"] = protein_accession.where(protein_accession.ne(""), protein_name)

    out = pd.DataFrame(
        {
            "source": "PubChem_HTS",
            "supervision": "binary",
            "task_type": "hit_prediction",
            "dataset": HF_MF_PCBA_BIND_DATASET,
            "source_split": source_split,
            "assay_type": "hts_binding",
            "screening_stage": "mf_pcba_primary_confirmatory",
            "assay_group_id": "PubChem_HTS:" + df["assay_id"].map(normalize_string) + ":" + df["target_id"].map(normalize_string),
            "assay_id": df["assay_id"],
            "target_id": df["target_id"],
            "protein_name": protein_name,
            "protein_category": df["protein_category"].map(normalize_string),
            "protein_accession": protein_accession,
            "protein_sequence": df["protein_sequence"],
            "compound_id": "CID:" + df["CID"].astype(str),
            "smiles": df["smiles"],
            "binary_label": df["binary_label"],
            "activity_outcome": df["Activity"].map(normalize_string),
            "dr_value": pd.to_numeric(df.get("DR"), errors="coerce"),
            "sd_value": pd.to_numeric(df.get("SD"), errors="coerce"),
            "sd_z_score": pd.to_numeric(df.get("SD Z-score"), errors="coerce"),
            "xc50_uM": pd.to_numeric(df.get("XC50"), errors="coerce"),
            "log_xc50": pd.to_numeric(df.get("Log XC50"), errors="coerce"),
            "assay_num_hits": pd.to_numeric(df["assay_num_hits"], errors="coerce"),
            "assay_num_negatives": pd.to_numeric(df["assay_num_negatives"], errors="coerce"),
            "assay_total_samples": pd.to_numeric(df["assay_total_samples"], errors="coerce"),
            "assay_hit_rate_percent": pd.to_numeric(df["assay_hit_rate_percent"], errors="coerce"),
            "label_rule": (
                "binders=DR_confirmatory_active; "
                "non_binders=SD_primary_inactive; "
                "metadata_filter=binding_assay_and_hit_rate"
            ),
            "upstream_pains_filtered": True,
        }
    )
    return out[BINARY_OUTPUT_COLUMNS]


def deduplicate_binary_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(frame)
    subset = [
        "source",
        "dataset",
        "assay_id",
        "protein_sequence",
        "smiles",
        "binary_label",
    ]
    deduped = frame.drop_duplicates(subset=subset, keep="first").copy()
    return deduped, before - len(deduped)


def cap_assays(frame: pd.DataFrame, config: PubChemMfPcbaConfig) -> tuple[pd.DataFrame, int]:
    if config.cap_per_assay <= 0:
        return frame.copy(), 0

    capped_groups = []
    dropped = 0
    for assay_id, group in frame.groupby("assay_id", sort=True):
        if len(group) <= config.cap_per_assay:
            capped_groups.append(group)
            continue

        seed = stable_int(assay_id, config.random_seed)
        if config.cap_mode == "random":
            sampled = group.sample(n=config.cap_per_assay, random_state=seed)
        elif config.cap_mode == "keep-positives":
            positives = group.loc[group["binary_label"] == 1]
            negatives = group.loc[group["binary_label"] == 0]
            if len(positives) >= config.cap_per_assay:
                sampled = positives.sample(n=config.cap_per_assay, random_state=seed)
            else:
                remaining = config.cap_per_assay - len(positives)
                sampled_negatives = negatives.sample(
                    n=min(remaining, len(negatives)),
                    random_state=seed,
                )
                sampled = pd.concat([positives, sampled_negatives], ignore_index=False)
        else:
            raise ValueError(f"Unsupported cap_mode: {config.cap_mode}")

        dropped += len(group) - len(sampled)
        capped_groups.append(sampled)

    if not capped_groups:
        return frame.iloc[0:0].copy(), dropped
    capped = pd.concat(capped_groups, ignore_index=True)
    capped = capped.sample(frac=1.0, random_state=config.random_seed).reset_index(drop=True)
    return capped[BINARY_OUTPUT_COLUMNS], dropped


def summarize_binary(frame: pd.DataFrame) -> dict:
    assay_counts = frame.groupby("assay_id").size() if not frame.empty else pd.Series(dtype=int)
    source_counts = frame["source"].value_counts().to_dict() if not frame.empty else {}
    label_counts = frame["binary_label"].value_counts().sort_index().to_dict() if not frame.empty else {}
    return {
        "rows": int(len(frame)),
        "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "assays": int(frame["assay_id"].nunique()) if not frame.empty else 0,
        "proteins": int(frame["protein_sequence"].nunique()) if not frame.empty else 0,
        "smiles": int(frame["smiles"].nunique()) if not frame.empty else 0,
        "compounds": int(frame["compound_id"].nunique()) if not frame.empty else 0,
        "min_rows_per_assay": int(assay_counts.min()) if len(assay_counts) else 0,
        "max_rows_per_assay": int(assay_counts.max()) if len(assay_counts) else 0,
    }


def build_pubchem_mf_pcba(
    raw_dir: Path,
    output_dir: Path,
    config: PubChemMfPcbaConfig,
    skip_download: bool = False,
    metadata_path: Path | None = None,
    parquet_paths: list[Path] | None = None,
    limit_rows: int | None = None,
) -> dict:
    started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    size_info = {}
    if skip_download:
        metadata_path = metadata_path or raw_dir / "MF-PCBA-Assay-Metadata.csv"
        if parquet_paths is None:
            parquet_paths = sorted((raw_dir / "parquet").glob("*/*.parquet"))
    else:
        metadata_path, parquet_paths, size_info = download_mf_pcba_bind(raw_dir)

    if metadata_path is None or not metadata_path.exists():
        raise FileNotFoundError("MF-PCBA metadata CSV not found.")
    if not parquet_paths:
        raise FileNotFoundError("No MF-PCBA parquet files found.")

    metadata = load_assay_metadata(metadata_path, config)
    frames = []
    raw_rows = 0
    for path in parquet_paths:
        source_split = path.parent.name
        if source_split not in config.include_source_splits:
            log(f"skip split: {path}")
            continue
        log(f"read parquet: {path}")
        frame = pd.read_parquet(path)
        if limit_rows is not None:
            frame = frame.head(limit_rows)
        raw_rows += len(frame)
        normalized = normalize_mf_pcba_frame(frame, source_split, metadata)
        log(f"normalized {path.name}: {len(frame):,} raw rows -> {len(normalized):,} kept rows")
        frames.append(normalized)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=BINARY_OUTPUT_COLUMNS)

    after_metadata_filters = len(combined)
    combined, exact_duplicates_removed = deduplicate_binary_rows(combined)
    after_deduplication = len(combined)
    combined, cap_dropped = cap_assays(combined, config)

    output_path = output_dir / "pubchem_mf_pcba_binary.csv"
    parquet_output_path = output_dir / "pubchem_mf_pcba_binary.parquet"
    combined.to_csv(output_path, index=False)
    combined.to_parquet(parquet_output_path, index=False)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "PubChem HTS via MF-PCBA-Bind",
        "config": asdict(config),
        "inputs": {
            "hf_dataset": HF_MF_PCBA_BIND_DATASET,
            "raw_dir": str(raw_dir),
            "metadata_path": str(metadata_path),
            "parquet_paths": [str(path) for path in parquet_paths],
            "hf_size": size_info,
        },
        "counts": {
            "raw_rows": int(raw_rows),
            "after_metadata_filters": int(after_metadata_filters),
            "exact_duplicates_removed": int(exact_duplicates_removed),
            "after_deduplication": int(after_deduplication),
            "assay_cap_dropped": int(cap_dropped),
            "final": summarize_binary(combined),
        },
        "outputs": {
            "pubchem_mf_pcba_binary": str(output_path),
            "pubchem_mf_pcba_binary_parquet": str(parquet_output_path),
        },
        "runtime_seconds": round(perf_counter() - started, 2),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"wrote {output_path} ({len(combined):,} rows)")
    log(f"wrote {parquet_output_path}")
    log(f"wrote {summary_path}")
    return summary


def write_binary_dataset(frame: pd.DataFrame, output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = ensure_binary_schema(frame)
    csv_path = output_dir / f"{stem}.csv"
    parquet_path = output_dir / f"{stem}.parquet"
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)
    return {stem: str(csv_path), f"{stem}_parquet": str(parquet_path)}


def load_cemm_smiles(raw_dir: Path) -> pd.DataFrame:
    mapping_frames = []
    candidates = [
        (raw_dir / "Table-S1.csv", ",", "fragId", "SMILES"),
        (raw_dir / "fid2can_fff_all.tsv", "\t", "fid", "smiles"),
        (raw_dir / "cemm_smiles.csv", ",", "fid", "smiles"),
    ]
    for path, sep, id_column, smiles_column in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path, sep=sep)
        if id_column not in frame.columns or smiles_column not in frame.columns:
            continue
        mapping_frames.append(
            frame[[id_column, smiles_column]]
            .rename(columns={id_column: "fragId", smiles_column: "smiles"})
            .assign(
                fragId=lambda df: df["fragId"].map(normalize_string),
                smiles=lambda df: df["smiles"].map(normalize_string),
            )
        )

    if not mapping_frames:
        raise FileNotFoundError(f"No CeMM fragment SMILES mapping found under {raw_dir}")

    mapping = pd.concat(mapping_frames, ignore_index=True)
    mapping = mapping.loc[mapping["fragId"].ne("") & mapping["smiles"].ne("")]
    mapping = mapping.drop_duplicates(subset=["fragId"], keep="first")
    return mapping


def build_cemm(
    raw_dir: Path,
    output_dir: Path,
    sequence_cache_path: Path | None,
    *,
    allow_sequence_fetch: bool = True,
) -> dict:
    started = perf_counter()
    screen_path = raw_dir / "extracted" / "finalScreen.tsv"
    if not screen_path.exists():
        raise FileNotFoundError(f"CeMM primary screen file not found: {screen_path}")

    log(f"read CeMM primary screen: {screen_path}")
    screen = pd.read_csv(screen_path, sep="\t")
    raw_rows = len(screen)
    required = {
        "accession",
        "geneName",
        "fragId",
        "mdfClass",
        "l2fc",
        "l2fcM",
        "ml10adjP",
        "ml10p",
        "expId",
    }
    missing = sorted(required - set(screen.columns))
    if missing:
        raise ValueError(f"Missing CeMM columns: {missing}")

    screen["mdfClass"] = pd.to_numeric(screen["mdfClass"], errors="coerce")
    screen = screen.loc[screen["mdfClass"].isin([0, 2, 3])].copy()
    after_label_filter = len(screen)
    screen["binary_label"] = screen["mdfClass"].ge(2).astype(int)

    smiles = load_cemm_smiles(raw_dir)
    screen["fragId"] = screen["fragId"].map(normalize_string)
    screen = screen.merge(smiles, on="fragId", how="left")
    missing_smiles = int(screen["smiles"].isna().sum())
    screen = screen.loc[screen["smiles"].notna() & screen["smiles"].ne("")].copy()

    accessions = screen["accession"].map(normalize_string)
    sequences = fetch_uniprot_sequences(
        accessions.tolist(),
        sequence_cache_path,
        allow_fetch=allow_sequence_fetch,
    )
    screen["protein_accession"] = accessions
    screen["protein_sequence"] = screen["protein_accession"].map(sequences).fillna("")
    missing_sequences = int(screen["protein_sequence"].eq("").sum())
    screen = screen.loc[screen["protein_sequence"].ne("")].copy()

    screen["protein_name"] = screen["geneName"].map(normalize_string)
    screen["target_id"] = screen["protein_accession"].where(
        screen["protein_accession"].ne(""),
        screen["protein_name"],
    )
    screen["assay_id"] = (
        "CeMM_primary:"
        + screen["expId"].map(normalize_string)
        + ":"
        + screen["fragId"].map(normalize_string)
    )
    screen["assay_group_id"] = screen["assay_id"]
    screen["activity_outcome"] = "mdfClass=" + screen["mdfClass"].astype("Int64").astype(str)

    out = pd.DataFrame(
        {
            "source": "CeMM",
            "supervision": "binary",
            "task_type": "hit_prediction",
            "dataset": "ligand_discovery_cemm",
            "source_split": "screening",
            "assay_type": "fragment_chemoproteomics",
            "screening_stage": "primary_screen",
            "assay_group_id": screen["assay_group_id"],
            "assay_id": screen["assay_id"],
            "target_id": screen["target_id"],
            "protein_name": screen["protein_name"],
            "protein_category": "",
            "protein_accession": screen["protein_accession"],
            "protein_sequence": screen["protein_sequence"],
            "compound_id": "CeMM:" + screen["fragId"].map(normalize_string),
            "smiles": screen["smiles"].map(normalize_string),
            "binary_label": screen["binary_label"],
            "activity_outcome": screen["activity_outcome"],
            "dr_value": pd.NA,
            "sd_value": pd.NA,
            "sd_z_score": pd.NA,
            "xc50_uM": pd.NA,
            "log_xc50": pd.NA,
            "assay_num_hits": pd.NA,
            "assay_num_negatives": pd.NA,
            "assay_total_samples": pd.NA,
            "assay_hit_rate_percent": pd.NA,
            "label_rule": (
                "primary_screen: positive mdfClass>=2; "
                "negative mdfClass==0; dropped mdfClass==1"
            ),
            "upstream_pains_filtered": False,
        }
    )
    out = fill_assay_counts(ensure_binary_schema(out))
    out, exact_duplicates_removed = deduplicate_binary_rows(out)
    out = fill_assay_counts(ensure_binary_schema(out))
    out = ensure_binary_schema(out)
    outputs = write_binary_dataset(out, output_dir, "cemm_binary")
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "CeMM ligand discovery primary screen",
        "inputs": {
            "raw_dir": str(raw_dir),
            "screen_path": str(screen_path),
            "sequence_cache_path": str(sequence_cache_path) if sequence_cache_path else None,
        },
        "counts": {
            "raw_rows": int(raw_rows),
            "after_label_filter": int(after_label_filter),
            "missing_smiles_rows": missing_smiles,
            "missing_sequence_rows": missing_sequences,
            "exact_duplicates_removed": int(exact_duplicates_removed),
            "final": summarize_binary(out),
        },
        "label_rule": (
            "primary_screen: positive mdfClass>=2; "
            "negative mdfClass==0; dropped mdfClass==1"
        ),
        "outputs": outputs,
        "runtime_seconds": round(perf_counter() - started, 2),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"wrote {outputs['cemm_binary']} ({len(out):,} rows)")
    log(f"wrote {outputs['cemm_binary_parquet']}")
    log(f"wrote {summary_path}")
    return summary


def build_midas(
    raw_dir: Path,
    output_dir: Path,
    sequence_cache_path: Path | None,
    *,
    q_threshold: float = 0.01,
    allow_sequence_fetch: bool = True,
) -> dict:
    started = perf_counter()
    extracted_dir = raw_dir / "extracted"
    metabolites_path = extracted_dir / "science.abm3452_data_s1.txt"
    proteins_path = extracted_dir / "science.abm3452_data_s3.txt"
    measurements_path = extracted_dir / "science.abm3452_data_s4.txt"
    for path in [metabolites_path, proteins_path, measurements_path]:
        if not path.exists():
            raise FileNotFoundError(f"MIDAS file not found: {path}")

    metabolites = pd.read_csv(metabolites_path, sep="\t", encoding="latin-1")
    proteins = pd.read_csv(proteins_path, sep="\t", encoding="latin-1")
    measurements = pd.read_csv(measurements_path, sep="\t", encoding="latin-1")
    raw_rows = len(measurements)

    metabolites = metabolites.rename(
        columns={
            "MIDAS_ID": "metabolite_midas_id",
            "SMILES": "smiles",
            "HMDB_ID": "hmdb_id",
        }
    )
    proteins = proteins.rename(
        columns={
            "MIDAS_ID": "protein_midas_id",
            "Uniprot_entry": "protein_accession",
        }
    )
    df = measurements.merge(
        metabolites[
            [
                "Metabolite",
                "metabolite_midas_id",
                "smiles",
                "hmdb_id",
            ]
        ],
        on="Metabolite",
        how="left",
    )
    df = df.merge(
        proteins[
            [
                "protein_midas_id",
                "protein_accession",
                "Protein_name",
                "Gene_name",
            ]
        ],
        left_on="Protein",
        right_on="protein_midas_id",
        how="left",
    )
    missing_smiles = int(df["smiles"].isna().sum())
    missing_proteins = int(df["protein_accession"].isna().sum())
    df["smiles"] = df["smiles"].map(normalize_string)
    df["raw_protein_accession"] = df["protein_accession"].map(normalize_string)
    multi_accession_mask = (
        df["raw_protein_accession"].str.contains(";", regex=False).fillna(False)
        | df["raw_protein_accession"].str.contains(",", regex=False).fillna(False)
    )
    multi_accession_rows = int(multi_accession_mask.sum())
    df["protein_accession"] = df["raw_protein_accession"].map(normalize_single_accession)
    df = df.loc[df["smiles"].ne("") & df["protein_accession"].ne("")].copy()

    accessions = df["protein_accession"].map(normalize_string)
    sequences = fetch_uniprot_sequences(
        accessions.tolist(),
        sequence_cache_path,
        allow_fetch=allow_sequence_fetch,
    )
    df["protein_accession"] = accessions
    df["protein_sequence"] = df["protein_accession"].map(sequences).fillna("")
    missing_sequences = int(df["protein_sequence"].eq("").sum())
    df = df.loc[df["protein_sequence"].ne("")].copy()

    q_values = pd.to_numeric(df["q_value"], errors="coerce")
    df = df.loc[q_values.notna()].copy()
    q_values = pd.to_numeric(df["q_value"], errors="coerce")
    df["binary_label"] = q_values.lt(q_threshold).astype(int)
    df["protein_name"] = df["Gene_name"].map(normalize_string)
    missing_names = df["protein_name"].eq("")
    df.loc[missing_names, "protein_name"] = df.loc[missing_names, "Protein_name"].map(normalize_string)
    df["target_id"] = df["protein_accession"].where(df["protein_accession"].ne(""), df["protein_name"])
    df["assay_id"] = "MIDAS:" + df["Protein"].map(normalize_string)
    df["activity_outcome"] = (
        "q_value="
        + pd.to_numeric(df["q_value"], errors="coerce").map(lambda value: f"{value:.6g}")
        + ";log2_corrected_fold_change="
        + pd.to_numeric(df["Log2(corrected_fold_change)"], errors="coerce").map(
            lambda value: f"{value:.6g}"
        )
    )
    compound_id = df["metabolite_midas_id"].map(normalize_string)
    missing_compound_id = compound_id.eq("")
    compound_id.loc[missing_compound_id] = df.loc[missing_compound_id, "hmdb_id"].map(normalize_string)
    missing_compound_id = compound_id.eq("")
    compound_id.loc[missing_compound_id] = df.loc[missing_compound_id, "Metabolite"].map(normalize_string)

    out = pd.DataFrame(
        {
            "source": "MIDAS",
            "supervision": "binary",
            "task_type": "hit_prediction",
            "dataset": "Hicks_2023_MIDAS",
            "source_split": "screening",
            "assay_type": "metabolite_binding",
            "screening_stage": "fia_ms_midas",
            "assay_group_id": df["assay_id"],
            "assay_id": df["assay_id"],
            "target_id": df["target_id"],
            "protein_name": df["protein_name"],
            "protein_category": "",
            "protein_accession": df["protein_accession"],
            "protein_sequence": df["protein_sequence"],
            "compound_id": "MIDAS:" + compound_id,
            "smiles": df["smiles"],
            "binary_label": df["binary_label"],
            "activity_outcome": df["activity_outcome"],
            "dr_value": pd.NA,
            "sd_value": pd.NA,
            "sd_z_score": pd.NA,
            "xc50_uM": pd.NA,
            "log_xc50": pd.NA,
            "assay_num_hits": pd.NA,
            "assay_num_negatives": pd.NA,
            "assay_total_samples": pd.NA,
            "assay_hit_rate_percent": pd.NA,
            "label_rule": f"positive q_value<{q_threshold}; negative q_value>={q_threshold}",
            "upstream_pains_filtered": False,
        }
    )
    out = fill_assay_counts(ensure_binary_schema(out))
    out, exact_duplicates_removed = deduplicate_binary_rows(out)
    out = fill_assay_counts(ensure_binary_schema(out))
    out = ensure_binary_schema(out)
    outputs = write_binary_dataset(out, output_dir, "midas_binary")
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "MIDAS protein-metabolite screen",
        "config": {"q_threshold": q_threshold},
        "inputs": {
            "raw_dir": str(raw_dir),
            "metabolites_path": str(metabolites_path),
            "proteins_path": str(proteins_path),
            "measurements_path": str(measurements_path),
            "sequence_cache_path": str(sequence_cache_path) if sequence_cache_path else None,
        },
        "counts": {
            "raw_rows": int(raw_rows),
            "missing_smiles_rows": missing_smiles,
            "missing_protein_rows": missing_proteins,
            "multi_accession_rows": multi_accession_rows,
            "missing_sequence_rows": missing_sequences,
            "exact_duplicates_removed": int(exact_duplicates_removed),
            "final": summarize_binary(out),
        },
        "outputs": outputs,
        "runtime_seconds": round(perf_counter() - started, 2),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"wrote {outputs['midas_binary']} ({len(out):,} rows)")
    log(f"wrote {outputs['midas_binary_parquet']}")
    log(f"wrote {summary_path}")
    return summary


def normalize_affinity_threshold_chunk(
    frame: pd.DataFrame,
    config: AffinityThresholdConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {
        "source",
        "assay_id",
        "target_id",
        "protein_sequence",
        "compound_id",
        "smiles",
        "activity_type",
        "activity_value_uM",
        "activity_qualifier",
        "pchembl_value",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing affinity columns for thresholding: {missing}")

    df = frame.copy()
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df["activity_qualifier"] = df["activity_qualifier"].fillna("=").map(normalize_string)
    df["protein_sequence"] = df["protein_sequence"].map(normalize_sequence)
    df["smiles"] = df["smiles"].map(normalize_string)
    valid = df["pchembl_value"].notna() & df["protein_sequence"].ne("") & df["smiles"].ne("")

    exact = valid & df["activity_qualifier"].eq("=")
    positive = exact & df["pchembl_value"].ge(config.pchembl_threshold)
    negative = exact & df["pchembl_value"].lt(config.pchembl_threshold)
    censored_safe_negative = (
        valid
        & df["activity_qualifier"].eq(">")
        & df["pchembl_value"].lt(config.pchembl_threshold)
    )
    censored_ambiguous = (
        valid
        & df["activity_qualifier"].eq(">")
        & df["pchembl_value"].ge(config.pchembl_threshold)
    )
    if config.include_censored_safe_negatives:
        keep = positive | negative | censored_safe_negative
    else:
        keep = positive | negative

    df = df.loc[keep].copy()
    if df.empty:
        return pd.DataFrame(columns=BINARY_OUTPUT_COLUMNS), {
            "input_rows": int(len(frame)),
            "invalid_rows": int((~valid).sum()),
            "exact_positive_rows": int(positive.sum()),
            "exact_negative_rows": int(negative.sum()),
            "censored_safe_negative_rows": int(censored_safe_negative.sum()),
            "censored_ambiguous_rows": int(censored_ambiguous.sum()),
            "kept_rows": 0,
        }

    df["binary_label"] = df["pchembl_value"].ge(config.pchembl_threshold).astype(int)
    safe_negative_mask = df["activity_qualifier"].eq(">")
    df.loc[safe_negative_mask, "binary_label"] = 0
    source = df["source"].map(normalize_string)
    assay_id = df["assay_id"].map(normalize_string)
    protein_sequence = df["protein_sequence"]
    target_id = df["target_id"].map(normalize_string)
    missing_target = target_id.eq("")
    target_id.loc[missing_target] = "SEQ:" + protein_sequence.loc[missing_target].map(short_hash)
    activity_type = df["activity_type"].map(normalize_string)
    activity_value = pd.to_numeric(df["activity_value_uM"], errors="coerce")
    pchembl = pd.to_numeric(df["pchembl_value"], errors="coerce")
    qualifier = df["activity_qualifier"].map(normalize_string)

    out = pd.DataFrame(
        {
            "source": source,
            "supervision": "binary_from_continuous_affinity",
            "task_type": "hit_prediction",
            "dataset": "chembl_bindingdb_general_filtered",
            "source_split": "affinity_thresholded",
            "assay_type": "continuous_affinity_threshold",
            "screening_stage": "affinity_to_binary",
            "assay_group_id": source + ":" + assay_id + ":" + target_id,
            "assay_id": assay_id,
            "target_id": target_id,
            "protein_name": "",
            "protein_category": "",
            "protein_accession": "",
            "protein_sequence": protein_sequence,
            "compound_id": df["compound_id"].map(normalize_string),
            "smiles": df["smiles"].map(normalize_string),
            "binary_label": df["binary_label"],
            "activity_outcome": (
                activity_type
                + qualifier
                + activity_value.map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
                + "uM;pChEMBL="
                + pchembl.map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
            ),
            "dr_value": pd.NA,
            "sd_value": pd.NA,
            "sd_z_score": pd.NA,
            "xc50_uM": pd.NA,
            "log_xc50": pd.NA,
            "assay_num_hits": pd.NA,
            "assay_num_negatives": pd.NA,
            "assay_total_samples": pd.NA,
            "assay_hit_rate_percent": pd.NA,
            "label_rule": (
                f"exact affinity positive pChEMBL>={config.pchembl_threshold}; "
                f"exact affinity negative pChEMBL<{config.pchembl_threshold}; "
                "censored '>' retained only as safe negative below threshold"
            ),
            "upstream_pains_filtered": False,
        }
    )
    return ensure_binary_schema(out), {
        "input_rows": int(len(frame)),
        "invalid_rows": int((~valid).sum()),
        "exact_positive_rows": int(positive.sum()),
        "exact_negative_rows": int(negative.sum()),
        "censored_safe_negative_rows": int(censored_safe_negative.sum()),
        "censored_ambiguous_rows": int(censored_ambiguous.sum()),
        "kept_rows": int(len(out)),
    }


def add_counts(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    out = dict(left)
    for key, value in right.items():
        out[key] = int(out.get(key, 0) + value)
    return out


def build_affinity_threshold_binary(
    input_path: Path,
    output_dir: Path,
    config: AffinityThresholdConfig,
) -> dict:
    started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "chembl_bindingdb_threshold_binary.csv"
    if output_path.exists():
        output_path.unlink()

    aggregate_counts: dict[str, int] = {}
    first_chunk = True
    for chunk_index, chunk in enumerate(pd.read_csv(input_path, chunksize=config.chunksize), start=1):
        normalized, counts = normalize_affinity_threshold_chunk(chunk, config)
        aggregate_counts = add_counts(aggregate_counts, counts)
        if not normalized.empty:
            normalized.to_csv(output_path, index=False, mode="w" if first_chunk else "a", header=first_chunk)
            first_chunk = False
        log(
            f"affinity threshold chunk {chunk_index}: "
            f"{counts['input_rows']:,} input -> {counts['kept_rows']:,} kept"
        )

    if first_chunk:
        pd.DataFrame(columns=BINARY_OUTPUT_COLUMNS).to_csv(output_path, index=False)

    log(f"read thresholded CSV for final dedupe: {output_path}")
    combined = pd.read_csv(output_path)
    before_dedup = len(combined)
    combined, exact_duplicates_removed = deduplicate_binary_rows(ensure_binary_schema(combined))
    combined = fill_assay_counts(ensure_binary_schema(combined))
    combined = ensure_binary_schema(combined)
    combined.to_csv(output_path, index=False)
    parquet_output_path = output_dir / "chembl_bindingdb_threshold_binary.parquet"
    combined.to_parquet(parquet_output_path, index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "ChEMBL/BindingDB continuous affinity thresholded to binary",
        "config": asdict(config),
        "inputs": {"affinity_csv": str(input_path)},
        "counts": aggregate_counts
        | {
            "before_exact_deduplication": int(before_dedup),
            "exact_duplicates_removed": int(exact_duplicates_removed),
            "final": summarize_binary(combined),
        },
        "outputs": {
            "chembl_bindingdb_threshold_binary": str(output_path),
            "chembl_bindingdb_threshold_binary_parquet": str(parquet_output_path),
        },
        "runtime_seconds": round(perf_counter() - started, 2),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"wrote {output_path} ({len(combined):,} rows)")
    log(f"wrote {parquet_output_path}")
    log(f"wrote {summary_path}")
    return summary


def read_binary_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def merge_binary_sources(input_paths: list[Path], output_dir: Path) -> dict:
    started = perf_counter()
    frames = []
    input_counts = {}
    for path in input_paths:
        log(f"read binary source: {path}")
        frame = ensure_binary_schema(read_binary_dataset(path))
        input_counts[str(path)] = int(len(frame))
        frames.append(frame)

    if frames:
        merged = pd.concat(frames, ignore_index=True)
    else:
        merged = pd.DataFrame(columns=BINARY_OUTPUT_COLUMNS)
    raw_rows = len(merged)

    merged, exact_duplicates_removed = deduplicate_binary_rows(merged)
    after_exact = len(merged)

    pair_cols = ["protein_sequence", "smiles"]
    conflict_keys = (
        merged.groupby(pair_cols)["binary_label"].nunique().reset_index(name="label_count")
    )
    conflict_keys = conflict_keys.loc[conflict_keys["label_count"] > 1, pair_cols]
    if not conflict_keys.empty:
        conflicts = merged.merge(conflict_keys, on=pair_cols, how="inner")
    else:
        conflicts = merged.iloc[0:0].copy()
    conflict_rows = len(conflicts)

    output_dir.mkdir(parents=True, exist_ok=True)
    conflict_path = output_dir / "binary_label_conflicts.csv"
    conflicts.to_csv(conflict_path, index=False)
    if conflict_rows:
        merged = merged.merge(conflict_keys, on=pair_cols, how="left", indicator=True)
        merged = merged.loc[merged["_merge"] == "left_only"].drop(columns=["_merge"])

    source_priority = {
        "ChEMBL": 0,
        "BindingDB": 1,
        "PubChem_HTS": 2,
        "CeMM": 3,
        "MIDAS": 4,
    }
    merged["_source_priority"] = merged["source"].map(source_priority).fillna(99)
    merged["_row_order"] = range(len(merged))
    merged = merged.sort_values(["_source_priority", "_row_order"])
    before_pair_dedupe = len(merged)
    merged = merged.drop_duplicates(
        subset=["protein_sequence", "smiles", "binary_label"],
        keep="first",
    )
    pair_label_duplicates_removed = before_pair_dedupe - len(merged)
    merged = merged.drop(columns=["_source_priority", "_row_order"]).reset_index(drop=True)
    merged = ensure_binary_schema(merged)

    outputs = write_binary_dataset(merged, output_dir, "binary_sources_merged")
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "merged binary hit-prediction sources",
        "inputs": {
            "paths": [str(path) for path in input_paths],
            "input_counts": input_counts,
        },
        "counts": {
            "raw_rows": int(raw_rows),
            "exact_duplicates_removed": int(exact_duplicates_removed),
            "after_exact_deduplication": int(after_exact),
            "conflict_rows_dropped": int(conflict_rows),
            "conflict_pairs": int(len(conflict_keys)),
            "pair_label_duplicates_removed": int(pair_label_duplicates_removed),
            "final": summarize_binary(merged),
        },
        "conflict_policy": "drop protein_sequence+smiles pairs with both binary labels before pair-label dedupe",
        "dedupe_policy": "after conflicts, keep first row per protein_sequence+smiles+binary_label using source priority ChEMBL, BindingDB, PubChem_HTS, CeMM, MIDAS",
        "outputs": outputs | {"binary_label_conflicts": str(conflict_path)},
        "runtime_seconds": round(perf_counter() - started, 2),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"wrote {outputs['binary_sources_merged']} ({len(merged):,} rows)")
    log(f"wrote {outputs['binary_sources_merged_parquet']}")
    log(f"wrote {conflict_path} ({conflict_rows:,} rows)")
    log(f"wrote {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build binary protein-ligand hit-prediction sources."
    )
    parser.add_argument(
        "--source",
        choices=("pubchem_mf_pcba", "cemm", "midas", "affinity_threshold", "merge", "merge_all_binary"),
        default="pubchem_mf_pcba",
        help="Source builder to run.",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Directory for raw source files. Defaults depend on --source.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for normalized binary outputs. Defaults depend on --source.",
    )
    parser.add_argument(
        "--sequence-cache",
        default="dataset/raw/binary_sources/uniprot_sequences.json",
        help="JSON cache for UniProt protein sequences used by CeMM/MIDAS.",
    )
    parser.add_argument(
        "--no-sequence-fetch",
        action="store_true",
        help="Use only the existing sequence cache for CeMM/MIDAS.",
    )
    parser.add_argument("--input", nargs="*", default=None, help="Input files for --source merge.")
    parser.add_argument("--midas-q-threshold", type=float, default=0.01)
    parser.add_argument("--pchembl-threshold", type=float, default=6.0)
    parser.add_argument(
        "--exclude-censored-safe-negatives",
        action="store_true",
        help="For affinity thresholding, drop all censored rows instead of retaining safe negatives.",
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--parquet", nargs="*", default=None, help="Local parquet file(s).")
    parser.add_argument("--min-assay-size", type=int, default=100)
    parser.add_argument("--max-hit-rate-percent", type=float, default=10.0)
    parser.add_argument("--cap-per-assay", type=int, default=50_000)
    parser.add_argument(
        "--cap-mode",
        choices=("keep-positives", "random"),
        default="keep-positives",
        help="How to cap large assays. Use 'random' for a closer paper-style random cap.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--source-splits",
        nargs="+",
        default=["validation", "test"],
        help="Source splits to include from the MF-PCBA-Bind dataset.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Debug option: read only this many rows from each parquet file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source == "cemm":
        build_cemm(
            raw_dir=Path(args.raw_dir or "dataset/raw/binary_sources/cemm"),
            output_dir=Path(args.output_dir or "dataset/processed/binary_affinity/cemm"),
            sequence_cache_path=Path(args.sequence_cache) if args.sequence_cache else None,
            allow_sequence_fetch=not args.no_sequence_fetch,
        )
        return

    if args.source == "midas":
        build_midas(
            raw_dir=Path(args.raw_dir or "dataset/raw/binary_sources/midas"),
            output_dir=Path(args.output_dir or "dataset/processed/binary_affinity/midas"),
            sequence_cache_path=Path(args.sequence_cache) if args.sequence_cache else None,
            q_threshold=args.midas_q_threshold,
            allow_sequence_fetch=not args.no_sequence_fetch,
        )
        return

    if args.source == "merge":
        input_paths = [
            Path(path)
            for path in (
                args.input
                or [
                    "dataset/processed/binary_affinity/pubchem_mf_pcba/pubchem_mf_pcba_binary.parquet",
                    "dataset/processed/binary_affinity/cemm/cemm_binary.parquet",
                    "dataset/processed/binary_affinity/midas/midas_binary.parquet",
                ]
            )
        ]
        merge_binary_sources(
            input_paths=input_paths,
            output_dir=Path(args.output_dir or "dataset/processed/binary_affinity/merged"),
        )
        return

    if args.source == "affinity_threshold":
        build_affinity_threshold_binary(
            input_path=Path(args.input[0] if args.input else "dataset/processed/general_affinity/chembl_bindingdb_general_filtered.csv"),
            output_dir=Path(args.output_dir or "dataset/processed/binary_affinity/chembl_bindingdb_threshold"),
            config=AffinityThresholdConfig(
                pchembl_threshold=args.pchembl_threshold,
                include_censored_safe_negatives=not args.exclude_censored_safe_negatives,
                chunksize=args.chunksize,
            ),
        )
        return

    if args.source == "merge_all_binary":
        input_paths = [
            Path(path)
            for path in (
                args.input
                or [
                    "dataset/processed/binary_affinity/chembl_bindingdb_threshold/chembl_bindingdb_threshold_binary.parquet",
                    "dataset/processed/binary_affinity/pubchem_mf_pcba/pubchem_mf_pcba_binary.parquet",
                    "dataset/processed/binary_affinity/cemm/cemm_binary.parquet",
                    "dataset/processed/binary_affinity/midas/midas_binary.parquet",
                ]
            )
        ]
        merge_binary_sources(
            input_paths=input_paths,
            output_dir=Path(args.output_dir or "dataset/processed/binary_affinity/all_sources_binary"),
        )
        return

    config = PubChemMfPcbaConfig(
        min_assay_size=args.min_assay_size,
        max_hit_rate_percent=args.max_hit_rate_percent,
        cap_per_assay=args.cap_per_assay,
        cap_mode=args.cap_mode,
        random_seed=args.random_seed,
        include_source_splits=tuple(args.source_splits),
    )
    build_pubchem_mf_pcba(
        raw_dir=Path(args.raw_dir or "dataset/raw/binary_sources/pubchem_mf_pcba"),
        output_dir=Path(args.output_dir or "dataset/processed/binary_affinity/pubchem_mf_pcba"),
        config=config,
        skip_download=args.skip_download,
        metadata_path=Path(args.metadata_path) if args.metadata_path else None,
        parquet_paths=[Path(path) for path in args.parquet] if args.parquet else None,
        limit_rows=args.limit_rows,
    )


if __name__ == "__main__":
    main()
