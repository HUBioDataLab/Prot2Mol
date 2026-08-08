import pandas as pd

from data_processing.build_chembl_generation_dataset import (
    GenerationSplitConfig,
    build_chembl_generation_dataset,
    choose_validation_clusters,
)


def test_validation_cluster_selection_is_deterministic_and_close_to_target():
    counts = pd.Series(
        {f"cluster_{index:03d}": (index % 7) + 1 for index in range(100)}
    )
    first = choose_validation_clusters(counts, 0.05, seed=17, trials=200)
    second = choose_validation_clusters(counts, 0.05, seed=17, trials=200)
    selected_rows = int(counts.loc[list(first)].sum())

    assert first == second
    assert abs(selected_rows / counts.sum() - 0.05) <= 1 / counts.sum()


def test_builder_uses_strict_threshold_and_whole_mmseqs_clusters(tmp_path):
    rows = []
    clusters = []
    for cluster_index in range(20):
        sequence = f"MSEQ{cluster_index:02d}"
        clusters.append(
            {
                "protein_sequence": sequence,
                "protein_cluster_50": f"cluster_{cluster_index:02d}",
            }
        )
        for molecule_index in range(5):
            rows.append(
                {
                    "protein_sequence": sequence,
                    "compound_selfies": "[C]",
                    "smiles": "C",
                    "compound_id": f"C{cluster_index}_{molecule_index}",
                    "assay_id": f"A{cluster_index}_{molecule_index}",
                    "pchembl_value": 6.0 if molecule_index == 0 else 7.0,
                }
            )

    source = tmp_path / "all.parquet"
    cluster_map = tmp_path / "clusters.csv"
    output = tmp_path / "generation"
    pd.DataFrame(rows).to_parquet(source, index=False)
    pd.DataFrame(clusters).to_csv(cluster_map, index=False)

    summary = build_chembl_generation_dataset(
        source,
        cluster_map,
        output,
        GenerationSplitConfig(
            validation_fraction=0.05,
            pchembl_threshold=6.0,
            seed=3,
            assignment_trials=200,
        ),
    )
    train = pd.read_parquet(output / "train.parquet")
    validation = pd.read_parquet(output / "validation.parquet")

    assert len(train) + len(validation) == 20
    assert train["pchembl_value"].gt(6.0).all()
    assert validation["pchembl_value"].gt(6.0).all()
    assert set(train["protein_cluster_50"]).isdisjoint(validation["protein_cluster_50"])
    assert set(train["protein_sequence"]).isdisjoint(validation["protein_sequence"])
    assert not (output / "test.parquet").exists()
    assert summary["validation"]["cluster_overlap"] == 0
    assert summary["validation"]["test_split_written"] is False
    assert summary["counts"]["duplicate_protein_molecule_rows_removed"] == 60
    assert set(pd.concat([train, validation])["source_positive_row_count"]) == {4}
