import csv
import json

from data_processing.analyze_prior_tendency import run_prior_tendency_analysis


def test_prior_tendency_analysis_computes_expected_bins_and_scores(tmp_path):
    csv_path = tmp_path / "toy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Compound_SMILES",
                "Target_FASTA",
                "pchembl_value_Median",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"Compound_SMILES": "C", "Target_FASTA": "AAAA", "pchembl_value_Median": "7.0"},
                {"Compound_SMILES": "C", "Target_FASTA": "BBBB", "pchembl_value_Median": "7.1"},
                {"Compound_SMILES": "O", "Target_FASTA": "AAAA", "pchembl_value_Median": "5.0"},
                {"Compound_SMILES": "O", "Target_FASTA": "BBBB", "pchembl_value_Median": "5.1"},
            ]
        )

    analysis = run_prior_tendency_analysis(
        csv_path=csv_path,
        positive_threshold=6.0,
        monte_carlo_iterations=0,
    )

    assert analysis.row_count == 4
    assert analysis.positive_label_rate_g == 0.5

    assert analysis.drug.entity_count == 2
    assert analysis.target.entity_count == 2

    assert analysis.drug.tendency_bin_frequencies["0.0"] == 0.5
    assert analysis.drug.tendency_bin_frequencies["1.0"] == 0.5
    assert analysis.target.tendency_bin_frequencies["0.5"] == 1.0
    assert analysis.drug.overall_prior_tendency_Z == 1.0
    assert analysis.target.overall_prior_tendency_Z == 0.5

    payload = json.loads(json.dumps(analysis, default=lambda obj: obj.__dict__))
    assert payload["drug"]["asymptotic_p_value"] < 0.05
    assert payload["target"]["asymptotic_p_value"] == 1.0
