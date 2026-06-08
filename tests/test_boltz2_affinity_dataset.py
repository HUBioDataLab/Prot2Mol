import json
import zipfile

import pandas as pd

from data_processing.build_binary_affinity_sources import (
    AffinityThresholdConfig,
    BINARY_OUTPUT_COLUMNS,
    PubChemMfPcbaConfig,
    build_affinity_threshold_binary,
    build_cemm,
    build_midas,
    build_pubchem_mf_pcba,
    merge_binary_sources,
)
from data_processing.build_boltz2_affinity_from_sources import (
    extract_bindingdb_affinity,
    parse_bindingdb_measurement,
)
from data_processing.prepare_boltz2_affinity_dataset import (
    BOLTZ2_OUTPUT_COLUMNS,
    PROT2MOL_OUTPUT_COLUMNS,
    Boltz2AffinityConfig,
    prepare_dataset,
    prepare_dataset_chunked,
)
from data_processing.build_protein_cluster_splits import SplitConfig, build_splits


def test_prepare_boltz2_affinity_dataset_outputs_spec_and_prot2mol_csv(tmp_path):
    rows = [
        {
            "source": "ChEMBL",
            "assay_id": "CHEMBL_A1",
            "target_id": "CHEMBL_T1",
            "target_chembl_id": "CHEMBL_T1",
            "protein_sequence": "MKTLLV",
            "smiles": smiles,
            "standard_type": "IC50",
            "standard_value": value_nm,
            "standard_units": "nM",
            "standard_relation": "=",
            "confidence_score": 9,
            "target_type": "SINGLE PROTEIN",
            "assay_type": "B",
            "source_unreliable": False,
            "molecule_chembl_id": compound_id,
        }
        for smiles, value_nm, compound_id in [
            ("CCO", 10.0, "CMP1"),
            ("CCN", 100.0, "CMP2"),
            ("CCC", 1000.0, "CMP3"),
            ("CCCl", 10000.0, "CMP4"),
        ]
    ]
    rows.extend(
        [
            {
                "source": "ChEMBL",
                "assay_id": "CHEMBL_A1",
                "target_id": "CHEMBL_T1",
                "protein_sequence": "MKTLLV",
                "smiles": "CCBr",
                "standard_type": "IC50",
                "standard_value": 1000.0,
                "standard_units": "nM",
                "standard_relation": "=",
                "confidence_score": 8,
                "target_type": "SINGLE PROTEIN",
                "assay_type": "B",
            },
            {
                "source": "ChEMBL",
                "assay_id": "CHEMBL_A1",
                "target_id": "CHEMBL_T1",
                "protein_sequence": "MKTLLV",
                "smiles": "CCF",
                "standard_type": "IC50",
                "standard_value": 1000.0,
                "standard_units": "nM",
                "standard_relation": ">",
                "confidence_score": 9,
                "target_type": "SINGLE PROTEIN",
                "assay_type": "B",
            },
        ]
    )
    input_path = tmp_path / "chembl.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)

    summary = prepare_dataset(
        input_paths=[str(input_path)],
        output_dir=str(tmp_path / "out"),
        config=Boltz2AffinityConfig(
            min_assay_size=4,
            min_unique_values=4,
            min_activity_std=0.1,
            min_unique_fraction=0.5,
            val_ratio=0.0,
            test_ratio=0.0,
        ),
    )

    full = pd.read_csv(summary["outputs"]["boltz2_affinity_full"])
    prot2mol = pd.read_csv(summary["outputs"]["prot2mol_training"])
    saved_summary = json.loads((tmp_path / "out" / "summary.json").read_text())

    assert list(full.columns) == BOLTZ2_OUTPUT_COLUMNS
    assert list(prot2mol.columns) == PROT2MOL_OUTPUT_COLUMNS
    assert len(full) == 4
    assert len(prot2mol) == 4
    assert set(full["split"]) == {"train"}
    assert full["is_pains"].eq(False).all()
    assert full["heavy_atom_count"].le(50).all()
    assert prot2mol["pchembl_value_Median"].max() == 8.0
    assert prot2mol["pchembl_value_Median"].min() == 5.0
    assert saved_summary["counts"]["raw_rows"] == 6
    assert saved_summary["counts"]["after_source_filters"] == 5
    assert saved_summary["counts"]["after_activity_filters"] == 4


def test_prepare_boltz2_affinity_dataset_can_keep_censored_in_full_export(tmp_path):
    rows = []
    for idx, (smiles, value_um, relation) in enumerate(
        [
            ("CCO", 0.01, "="),
            ("CCN", 0.1, "="),
            ("CCC", 1.0, "="),
            ("CCCl", 10.0, "="),
            ("CCCC", 20.0, ">"),
        ],
        start=1,
    ):
        rows.append(
            {
                "source": "BindingDB",
                "doi": "10.0000/example",
                "target_id": "BDB_T1",
                "protein_sequence": "AAAA",
                "smiles": smiles,
                "activity_type": "Ki",
                "activity_value_uM": value_um,
                "activity_qualifier": relation,
                "num_protein_chains": 1,
                "compound_id": f"BDB{idx}",
            }
        )
    input_path = tmp_path / "bindingdb.tsv"
    pd.DataFrame(rows).to_csv(input_path, sep="\t", index=False)

    summary = prepare_dataset(
        input_paths=[str(input_path)],
        output_dir=str(tmp_path / "out"),
        config=Boltz2AffinityConfig(
            min_assay_size=4,
            min_unique_values=4,
            min_activity_std=0.1,
            min_unique_fraction=0.5,
            include_censored=True,
            val_ratio=0.0,
            test_ratio=0.0,
        ),
    )

    full = pd.read_csv(summary["outputs"]["boltz2_affinity_full"])
    prot2mol = pd.read_csv(summary["outputs"]["prot2mol_training"])

    assert len(full) == 5
    assert full["is_censored"].sum() == 1
    assert len(prot2mol) == 4


def test_prepare_boltz2_affinity_dataset_deduplicates_bindingdb_after_chembl(tmp_path):
    rows = [
        {
            "source": "ChEMBL",
            "assay_id": "CHEMBL_A1",
            "target_id": "T1",
            "protein_sequence": "MKTLLV",
            "smiles": "CCO",
            "standard_type": "IC50",
            "standard_value": 100.0,
            "standard_units": "nM",
            "standard_relation": "=",
            "confidence_score": 9,
            "target_type": "SINGLE PROTEIN",
            "assay_type": "B",
            "molecule_chembl_id": "CHEMBL_C1",
        },
        {
            "source": "BindingDB",
            "doi": "10.0000/duplicate",
            "target_id": "T1",
            "protein_sequence": "MKTLLV",
            "smiles": "OCC",
            "activity_type": "IC50",
            "activity_value_uM": 0.1,
            "activity_qualifier": "=",
            "num_protein_chains": 1,
            "compound_id": "BDB_DUP",
        },
        {
            "source": "BindingDB",
            "doi": "10.0000/independent",
            "target_id": "T1",
            "protein_sequence": "MKTLLV",
            "smiles": "CCO",
            "activity_type": "IC50",
            "activity_value_uM": 0.2,
            "activity_qualifier": "=",
            "num_protein_chains": 1,
            "compound_id": "BDB_KEEP",
        },
    ]
    input_path = tmp_path / "mixed.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)

    summary = prepare_dataset(
        input_paths=[str(input_path)],
        output_dir=str(tmp_path / "out"),
        config=Boltz2AffinityConfig(
            min_assay_size=1,
            min_unique_values=1,
            min_activity_std=0.0,
            min_unique_fraction=0.0,
            val_ratio=0.0,
            test_ratio=0.0,
        ),
    )

    full = pd.read_csv(summary["outputs"]["boltz2_affinity_full"])
    assert summary["counts"]["bindingdb_chembl_overlap_removed"] == 1
    assert "BDB_DUP" not in set(full["compound_id"])
    assert "BDB_KEEP" in set(full["compound_id"])


def test_parse_bindingdb_measurement_handles_qualifiers():
    assert parse_bindingdb_measurement(">10000") == (">", 10000.0)
    assert parse_bindingdb_measurement("= 3.2") == ("=", 3.2)
    assert parse_bindingdb_measurement("<1,000") == ("<", 1000.0)
    assert parse_bindingdb_measurement("") == (None, None)


def test_extract_bindingdb_affinity_from_zip(tmp_path):
    bindingdb_frame = pd.DataFrame(
        {
            "Ligand SMILES": ["CCO", "CCC", "CCN"],
            "BindingDB Target Chain Sequence": ["MKT", "MKT", "MKT"],
            "Number of Protein Chains in Target (>1 implies a multichain complex)": ["1", "2", "1"],
            "Article DOI": ["10.1/example", "10.1/example", "10.1/example"],
            "BindingDB MonomerID": ["1", "2", "3"],
            "UniProt (SwissProt) Primary ID of Target Chain": ["P1", "P1", "P1"],
            "Ki (nM)": [">100", "20", ""],
            "IC50 (nM)": ["", "30", "40"],
        }
    )
    tsv_path = tmp_path / "BindingDB_All.tsv"
    bindingdb_frame.to_csv(tsv_path, sep="\t", index=False)
    zip_path = tmp_path / "BindingDB_All.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(tsv_path, arcname="BindingDB_All.tsv")

    output_path = tmp_path / "bindingdb_affinity.tsv"
    extract_bindingdb_affinity(zip_path, output_path, chunksize=2)

    out = pd.read_csv(output_path, sep="\t")
    assert len(out) == 2
    assert set(out["activity_type"]) == {"Ki", "IC50"}
    assert out["num_protein_chains"].eq(1).all()
    assert out["activity_value_uM"].tolist() == [0.1, 0.04]


def test_prepare_boltz2_affinity_dataset_chunked_matches_small_output(tmp_path):
    rows = [
        {
            "source": "ChEMBL",
            "assay_id": "CHEMBL_A1",
            "target_id": "CHEMBL_T1",
            "target_chembl_id": "CHEMBL_T1",
            "protein_sequence": "MKTLLV",
            "smiles": smiles,
            "standard_type": "IC50",
            "standard_value": value_nm,
            "standard_units": "nM",
            "standard_relation": "=",
            "confidence_score": 9,
            "target_type": "SINGLE PROTEIN",
            "assay_type": "B",
            "molecule_chembl_id": compound_id,
        }
        for smiles, value_nm, compound_id in [
            ("CCO", 10.0, "CMP1"),
            ("CCN", 100.0, "CMP2"),
            ("CCC", 1000.0, "CMP3"),
            ("CCCl", 10000.0, "CMP4"),
        ]
    ]
    input_path = tmp_path / "chembl.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)

    summary = prepare_dataset_chunked(
        input_paths=[str(input_path)],
        output_dir=str(tmp_path / "out"),
        config=Boltz2AffinityConfig(
            min_assay_size=4,
            min_unique_values=4,
            min_activity_std=0.1,
            min_unique_fraction=0.5,
            val_ratio=0.0,
            test_ratio=0.0,
        ),
        chunk_size=2,
    )

    full = pd.read_csv(summary["outputs"]["boltz2_affinity_full"])
    assert len(full) == 4
    assert summary["chunked"] is True
    assert summary["counts"]["after_assay_filters"] == 4
    assert (tmp_path / "out" / "_staging_molecule_filtered.csv").exists()


def test_build_pubchem_mf_pcba_binary_filters_and_caps(tmp_path):
    metadata = pd.DataFrame(
        {
            "AID": ["AID1", "AID2", "AID3"],
            "bind/phenotypic": ["bind", "phenotypic", "bind"],
            "protein_name": ["P1", "P2", "P3"],
            "protein_category": ["kinase", "cell", "enzyme"],
            "protein_accession": ["NP_1", "NP_2", "NP_3"],
            "amino_acid_sequence": ["MKT", "AAA", "GGG"],
            "Num_Hits": [2, 1, 5],
            "Num_Negatives": [100, 100, 10],
            "Total_Samples": [102, 101, 15],
            "Hit_Rate_%": [1.96, 0.99, 33.3],
        }
    )
    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    rows = [
        {
            "CID": 1,
            "smiles": "CCO",
            "binds": 1,
            "Activity": "Active",
            "DR": 5.0,
            "SD": 10.0,
            "SD Z-score": 2.0,
            "XC50": 10.0,
            "Log XC50": 1.0,
            "protein_name": "P1",
            "protein_category": "kinase",
            "protein_accession": "NP_1",
            "amino_acid_sequence": "MKT",
            "AID": "AID1",
        },
        {
            "CID": 1,
            "smiles": "CCO",
            "binds": 1,
            "Activity": "Active",
            "DR": 5.0,
            "SD": 10.0,
            "SD Z-score": 2.0,
            "XC50": 10.0,
            "Log XC50": 1.0,
            "protein_name": "P1",
            "protein_category": "kinase",
            "protein_accession": "NP_1",
            "amino_acid_sequence": "MKT",
            "AID": "AID1",
        },
        {
            "CID": 2,
            "smiles": "CCC",
            "binds": 0,
            "Activity": "Inactive",
            "DR": None,
            "SD": 0.1,
            "SD Z-score": -1.0,
            "XC50": None,
            "Log XC50": None,
            "protein_name": "P1",
            "protein_category": "kinase",
            "protein_accession": "NP_1",
            "amino_acid_sequence": "MKT",
            "AID": "AID1",
        },
        {
            "CID": 3,
            "smiles": "CCN",
            "binds": 0,
            "Activity": "Inactive",
            "DR": None,
            "SD": 0.2,
            "SD Z-score": -0.5,
            "XC50": None,
            "Log XC50": None,
            "protein_name": "P1",
            "protein_category": "kinase",
            "protein_accession": "NP_1",
            "amino_acid_sequence": "MKT",
            "AID": "AID1",
        },
        {
            "CID": 4,
            "smiles": "CCCl",
            "binds": 0,
            "Activity": "Inactive",
            "DR": None,
            "SD": 0.2,
            "SD Z-score": -0.5,
            "XC50": None,
            "Log XC50": None,
            "protein_name": "P2",
            "protein_category": "cell",
            "protein_accession": "NP_2",
            "amino_acid_sequence": "AAA",
            "AID": "AID2",
        },
        {
            "CID": 5,
            "smiles": "CCBr",
            "binds": 1,
            "Activity": "Active",
            "DR": 4.0,
            "SD": 1.0,
            "SD Z-score": 1.0,
            "XC50": 20.0,
            "Log XC50": 1.3,
            "protein_name": "P3",
            "protein_category": "enzyme",
            "protein_accession": "NP_3",
            "amino_acid_sequence": "GGG",
            "AID": "AID3",
        },
    ]
    parquet_path = tmp_path / "validation" / "0000.parquet"
    parquet_path.parent.mkdir()
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)

    summary = build_pubchem_mf_pcba(
        raw_dir=tmp_path / "raw",
        output_dir=tmp_path / "out",
        config=PubChemMfPcbaConfig(cap_per_assay=2, cap_mode="keep-positives"),
        skip_download=True,
        metadata_path=metadata_path,
        parquet_paths=[parquet_path],
    )

    output = pd.read_csv(summary["outputs"]["pubchem_mf_pcba_binary"])
    assert list(output.columns) == BINARY_OUTPUT_COLUMNS
    assert len(output) == 2
    assert set(output["assay_id"]) == {"AID1"}
    assert output["binary_label"].sum() == 1
    assert summary["counts"]["exact_duplicates_removed"] == 1
    assert summary["counts"]["assay_cap_dropped"] == 1


def test_build_cemm_primary_binary_uses_mdf_classes_and_sequence_cache(tmp_path):
    raw_dir = tmp_path / "cemm"
    extracted_dir = raw_dir / "extracted"
    extracted_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "accession": ["P1", "P1", "P2", "P3"],
            "geneName": ["G1", "G1", "G2", "G3"],
            "protHits": [1, 1, 1, 1],
            "fragId": ["C001", "C002", "C001", "C003"],
            "ligHits": [1, 1, 1, 1],
            "mdfClass": [3, 0, 1, 2],
            "l2fc": [1.0, 0.0, 0.5, 2.0],
            "l2fcM": [1.0, 0.0, 0.5, 2.0],
            "ml10adjP": [3.0, 0.1, 0.2, 4.0],
            "ml10p": [3.0, 0.1, 0.2, 4.0],
            "expId": ["E1", "E1", "E1", "E2"],
            "nUniq": [1, 1, 1, 1],
            "nPep": [1, 1, 1, 1],
            "perCovg": [10, 10, 10, 10],
            "rankRel": [1, 2, 3, 4],
        }
    ).to_csv(extracted_dir / "finalScreen.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "fragId": ["C001", "C002"],
            "SMILES": ["CCO", "CCC"],
        }
    ).to_csv(raw_dir / "Table-S1.csv", index=False)
    sequence_cache = tmp_path / "sequences.json"
    sequence_cache.write_text(json.dumps({"P1": "MKT"}))

    summary = build_cemm(
        raw_dir=raw_dir,
        output_dir=tmp_path / "out",
        sequence_cache_path=sequence_cache,
        allow_sequence_fetch=False,
    )

    output = pd.read_csv(summary["outputs"]["cemm_binary"])
    assert list(output.columns) == BINARY_OUTPUT_COLUMNS
    assert len(output) == 2
    assert set(output["binary_label"]) == {0, 1}
    assert output["assay_type"].eq("fragment_chemoproteomics").all()
    assert summary["counts"]["after_label_filter"] == 3
    assert summary["counts"]["missing_smiles_rows"] == 1


def test_build_midas_binary_uses_q_threshold_and_sequence_cache(tmp_path):
    raw_dir = tmp_path / "midas"
    extracted_dir = raw_dir / "extracted"
    extracted_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "Metabolite": ["Met1", "Met2"],
            "Metabolite_screening_Pool": [1, 1],
            "Screened_concentration_µM": [10.0, 10.0],
            "Molecular_formula": ["C2H6O", "C3H8"],
            "KEGG_ID": ["K1", "K2"],
            "HMDB_ID": ["H1", "H2"],
            "HMDB_taxonomy_sub_class_modified": ["Class", "Class"],
            "SMILES": ["CCO", "CCC"],
            "MIDAS_ID": ["M1", "M2"],
            "Supplier": ["S", "S"],
            "Supplier_product_ID": ["P1", "P2"],
        }
    ).to_csv(extracted_dir / "science.abm3452_data_s1.txt", sep="\t", index=False, encoding="latin-1")
    pd.DataFrame(
        {
            "Protein_name": ["Protein A"],
            "Gene_name": ["GENEA"],
            "Uniprot_entry": ["UP1"],
            "Enzyme_commission_number": ["1.1.1.1"],
            "MIDAS_ID": ["PROTA"],
            "Screened_concentration_(mM)": [1.0],
            "Protein_source": ["lab"],
            "Protein_expression_and_purification": ["method"],
        }
    ).to_csv(extracted_dir / "science.abm3452_data_s3.txt", sep="\t", index=False, encoding="latin-1")
    pd.DataFrame(
        {
            "Metabolite": ["Met1", "Met2"],
            "Protein": ["PROTA", "PROTA"],
            "Log2(corrected_fold_change)": [2.0, 0.1],
            "Log2(fold_change)": [2.1, 0.2],
            "No-signal_model_Log2(corrected_fold_change)_mean": [0.0, 0.0],
            "No-signal_model_Log2(corrected_fold_change)_standard_deviation": [0.1, 0.1],
            "p_value": [0.001, 0.5],
            "q_value": [0.005, 0.5],
        }
    ).to_csv(extracted_dir / "science.abm3452_data_s4.txt", sep="\t", index=False, encoding="latin-1")
    sequence_cache = tmp_path / "sequences.json"
    sequence_cache.write_text(json.dumps({"UP1": "MKT"}))

    summary = build_midas(
        raw_dir=raw_dir,
        output_dir=tmp_path / "out",
        sequence_cache_path=sequence_cache,
        q_threshold=0.01,
        allow_sequence_fetch=False,
    )

    output = pd.read_csv(summary["outputs"]["midas_binary"])
    assert list(output.columns) == BINARY_OUTPUT_COLUMNS
    assert len(output) == 2
    assert output["binary_label"].tolist() == [1, 0]
    assert output["assay_type"].eq("metabolite_binding").all()
    assert summary["counts"]["final"]["label_counts"] == {"0": 1, "1": 1}


def test_merge_binary_sources_drops_label_conflicts_and_pair_duplicates(tmp_path):
    source_a = pd.DataFrame(
        [
            {
                "source": "PubChem_HTS",
                "dataset": "pubchem",
                "source_split": "validation",
                "assay_id": "A1",
                "target_id": "T1",
                "protein_sequence": "MKT",
                "compound_id": "C1",
                "smiles": "CCO",
                "binary_label": 1,
            },
            {
                "source": "PubChem_HTS",
                "dataset": "pubchem",
                "source_split": "validation",
                "assay_id": "A2",
                "target_id": "T1",
                "protein_sequence": "MKT",
                "compound_id": "C2",
                "smiles": "CCC",
                "binary_label": 1,
            },
            {
                "source": "PubChem_HTS",
                "dataset": "pubchem",
                "source_split": "validation",
                "assay_id": "A3",
                "target_id": "T1",
                "protein_sequence": "MKT",
                "compound_id": "C3",
                "smiles": "CCN",
                "binary_label": 0,
            },
        ]
    )
    source_b = pd.DataFrame(
        [
            {
                "source": "CeMM",
                "dataset": "cemm",
                "source_split": "screening",
                "assay_id": "B1",
                "target_id": "T1",
                "protein_sequence": "MKT",
                "compound_id": "C4",
                "smiles": "CCO",
                "binary_label": 1,
            },
            {
                "source": "CeMM",
                "dataset": "cemm",
                "source_split": "screening",
                "assay_id": "B2",
                "target_id": "T1",
                "protein_sequence": "MKT",
                "compound_id": "C5",
                "smiles": "CCC",
                "binary_label": 0,
            },
        ]
    )
    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    source_a.to_csv(path_a, index=False)
    source_b.to_csv(path_b, index=False)

    summary = merge_binary_sources([path_a, path_b], tmp_path / "merged")

    output = pd.read_csv(summary["outputs"]["binary_sources_merged"])
    conflicts = pd.read_csv(summary["outputs"]["binary_label_conflicts"])
    assert len(output) == 2
    assert "CCC" not in set(output["smiles"])
    assert output["smiles"].tolist().count("CCO") == 1
    assert len(conflicts) == 2
    assert summary["counts"]["conflict_pairs"] == 1
    assert summary["counts"]["pair_label_duplicates_removed"] == 1


def test_build_affinity_threshold_binary_keeps_safe_censored_negatives(tmp_path):
    input_path = tmp_path / "affinity.csv"
    pd.DataFrame(
        [
            {
                "source": "ChEMBL",
                "assay_id": "A1",
                "target_id": "T1",
                "protein_sequence": "MKT",
                "compound_id": "C1",
                "smiles": "CCO",
                "activity_type": "IC50",
                "activity_value_uM": 0.1,
                "activity_qualifier": "=",
                "pchembl_value": 7.0,
            },
            {
                "source": "ChEMBL",
                "assay_id": "A1",
                "target_id": "T1",
                "protein_sequence": "MKT",
                "compound_id": "C2",
                "smiles": "CCC",
                "activity_type": "IC50",
                "activity_value_uM": 10.0,
                "activity_qualifier": "=",
                "pchembl_value": 5.0,
            },
            {
                "source": "BindingDB",
                "assay_id": "D1",
                "target_id": "",
                "protein_sequence": "MKT",
                "compound_id": "B1",
                "smiles": "CCN",
                "activity_type": "Ki",
                "activity_value_uM": 20.0,
                "activity_qualifier": ">",
                "pchembl_value": 4.7,
            },
            {
                "source": "BindingDB",
                "assay_id": "D1",
                "target_id": "",
                "protein_sequence": "MKT",
                "compound_id": "B2",
                "smiles": "CCCl",
                "activity_type": "Ki",
                "activity_value_uM": 0.2,
                "activity_qualifier": ">",
                "pchembl_value": 6.7,
            },
        ]
    ).to_csv(input_path, index=False)

    summary = build_affinity_threshold_binary(
        input_path=input_path,
        output_dir=tmp_path / "out",
        config=AffinityThresholdConfig(pchembl_threshold=6.0, chunksize=2),
    )

    output = pd.read_csv(summary["outputs"]["chembl_bindingdb_threshold_binary"])
    assert list(output.columns) == BINARY_OUTPUT_COLUMNS
    assert len(output) == 3
    assert output["binary_label"].tolist() == [1, 0, 0]
    assert output["supervision"].eq("binary_from_continuous_affinity").all()
    assert output["target_id"].str.len().gt(0).all()
    assert summary["counts"]["censored_safe_negative_rows"] == 1
    assert summary["counts"]["censored_ambiguous_rows"] == 1


def test_build_protein_cluster_splits_exact_mode_keeps_cluster_disjoint(tmp_path):
    binary_rows = pd.DataFrame(
        [
            {
                "source": "ChEMBL",
                "assay_group_id": "A1:T1",
                "assay_id": "A1",
                "protein_sequence": "MKT",
                "compound_id": "C1",
                "smiles": "CCO",
                "binary_label": 1,
            },
            {
                "source": "ChEMBL",
                "assay_group_id": "A1:T1",
                "assay_id": "A1",
                "protein_sequence": "MKT",
                "compound_id": "C2",
                "smiles": "CCC",
                "binary_label": 0,
            },
            {
                "source": "CeMM",
                "assay_group_id": "A2:T2",
                "assay_id": "A2",
                "protein_sequence": "AAAA",
                "compound_id": "C3",
                "smiles": "CCN",
                "binary_label": 1,
            },
            {
                "source": "MIDAS",
                "assay_group_id": "A3:T3",
                "assay_id": "A3",
                "protein_sequence": "GGGG",
                "compound_id": "C4",
                "smiles": "CCCl",
                "binary_label": 0,
            },
        ]
    )
    binary_all = tmp_path / "binary_all.parquet"
    binary_screen = tmp_path / "binary_screen.parquet"
    binary_threshold = tmp_path / "binary_threshold.parquet"
    binary_rows.to_parquet(binary_all, index=False)
    binary_rows.iloc[2:].to_parquet(binary_screen, index=False)
    binary_rows.iloc[:2].to_parquet(binary_threshold, index=False)

    ranking = pd.DataFrame(
        [
            {
                "source": "ChEMBL",
                "assay_id": "A1",
                "target_id": "T1",
                "protein_sequence": "MKT",
                "compound_id": "C1",
                "smiles": "CCO",
                "activity_type": "IC50",
                "activity_qualifier": "=",
                "pchembl_value": 7.0,
            },
            {
                "source": "ChEMBL",
                "assay_id": "A1",
                "target_id": "T1",
                "protein_sequence": "MKT",
                "compound_id": "C2",
                "smiles": "CCC",
                "activity_type": "IC50",
                "activity_qualifier": "=",
                "pchembl_value": 5.0,
            },
            {
                "source": "BindingDB",
                "assay_id": "A2",
                "target_id": "T2",
                "protein_sequence": "AAAA",
                "compound_id": "C3",
                "smiles": "CCN",
                "activity_type": "Ki",
                "activity_qualifier": "=",
                "pchembl_value": 6.5,
            },
        ]
    )
    ranking_path = tmp_path / "ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    summary = build_splits(
        binary_all_path=binary_all,
        binary_screen_path=binary_screen,
        binary_threshold_path=binary_threshold,
        ranking_affinity_path=ranking_path,
        output_dir=tmp_path / "splits",
        config=SplitConfig(
            cluster_mode="exact",
            val_ratio=0.34,
            test_ratio=0.34,
            ranking_max_pairs_per_split=10,
            ranking_max_pairs_per_group=10,
        ),
    )

    cluster_split = pd.read_csv(summary["outputs"]["cluster_split"])
    assert cluster_split["protein_cluster_90"].is_unique
    assert summary["leakage_checks"]["train_val_cluster_overlap"] == 0
    assert summary["leakage_checks"]["train_test_cluster_overlap"] == 0
    assert summary["leakage_checks"]["val_test_cluster_overlap"] == 0

    seen_by_split = {}
    for split in ["train", "val", "test"]:
        path = summary["outputs"]["split_datasets"]["binary_all_source"]["outputs"].get(split)
        if path:
            seen_by_split[split] = set(pd.read_parquet(path)["protein_sequence"])
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not (seen_by_split.get(left, set()) & seen_by_split.get(right, set()))

    pair_counts = summary["outputs"]["split_datasets"]["ranking_pairs"]["counts"]
    assert sum(pair_counts.values()) >= 1
