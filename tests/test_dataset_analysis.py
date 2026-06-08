import csv

from data_processing.analyze_dataset import summarize_dataset


def test_summarize_dataset_reports_core_counts(tmp_path):
    csv_path = tmp_path / "toy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Target_FASTA",
                "Target_CHEMBL_ID",
                "Compound_SELFIES",
                "Compound_SMILES",
                "pchembl_value_Median",
                "Compound_CID",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "Target_FASTA": "AAAA",
                    "Target_CHEMBL_ID": "CHEMBL_T1",
                    "Compound_SELFIES": "[C]",
                    "Compound_SMILES": "C",
                    "pchembl_value_Median": "5.0",
                    "Compound_CID": "CID1",
                },
                {
                    "Target_FASTA": "AAAB",
                    "Target_CHEMBL_ID": "CHEMBL_T1",
                    "Compound_SELFIES": "[C][O]",
                    "Compound_SMILES": "CO",
                    "pchembl_value_Median": "7.0",
                    "Compound_CID": "CID1",
                },
                {
                    "Target_FASTA": "AAAA",
                    "Target_CHEMBL_ID": "CHEMBL_T1",
                    "Compound_SELFIES": "[N]",
                    "Compound_SMILES": "N",
                    "pchembl_value_Median": "6.0",
                    "Compound_CID": "CID2",
                },
                {
                    "Target_FASTA": "BBBB",
                    "Target_CHEMBL_ID": "CHEMBL_T2",
                    "Compound_SELFIES": "[C]",
                    "Compound_SMILES": "C",
                    "pchembl_value_Median": "8.0",
                    "Compound_CID": "CID1",
                },
                {
                    "Target_FASTA": "BBBB",
                    "Target_CHEMBL_ID": "CHEMBL_T2",
                    "Compound_SELFIES": "[O]",
                    "Compound_SMILES": "O",
                    "pchembl_value_Median": "6.5",
                    "Compound_CID": "",
                },
            ]
        )

    summary, top_targets, top_compounds = summarize_dataset(csv_path, top_n=2)

    assert summary["row_count"] == 5
    assert summary["column_count"] == 6
    assert summary["unique_counts"]["targets_by_chembl_id"] == 2
    assert summary["unique_counts"]["compounds_by_cid_or_smiles"] == 3
    assert summary["unique_counts"]["protein_compound_pairs"] == 4
    assert summary["consistency_checks"]["targets_with_multiple_fastas"] == 1
    assert summary["consistency_checks"]["compounds_with_multiple_smiles_for_same_key"] == 1
    assert summary["duplicate_pairs"]["count"] == 1
    assert summary["duplicate_pairs"]["rows_in_duplicate_pairs"] == 2
    assert summary["duplicate_pairs"]["pairs_with_label_variation"] == 1
    assert summary["target_row_concentration"]["top_1_share"] == 0.6
    assert top_targets[0]["Target_CHEMBL_ID"] == "CHEMBL_T1"
    assert top_targets[0]["unique_compounds"] == 2
    assert top_compounds[0]["Compound_Key"] == "CID1"
    assert top_compounds[0]["unique_targets"] == 2
