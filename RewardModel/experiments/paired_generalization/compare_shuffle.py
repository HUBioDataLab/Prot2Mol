"""Paired whole-unit uncertainty for the prespecified protein-shuffle contrast."""
import json

import numpy as np
import pandas as pd

if __package__:
    from .run import WORK, sha256, write_json
    from .analyze_predictions import unit_bootstrap
else:
    from run import WORK, sha256, write_json
    from analyze_predictions import unit_bootstrap


def main():
    results = {}
    for regime in ("A", "B"):
        directories = [WORK / "tier2" / arm / regime for arm in ("kronecker_ridge", "protein_shuffled_kronecker")]
        manifests = [json.loads((p / "complete.json").read_text()) for p in directories]
        if manifests[0]["data_sha256"] != manifests[1]["data_sha256"] or manifests[0]["feature_provenance"] != manifests[1]["feature_provenance"]:
            raise RuntimeError("The paired comparison requires identical source data and feature maps")
        frames = [pd.read_parquet(p / "test/groups.parquet").set_index("group_id").sort_index() for p in directories]
        left, right = frames
        if not left.index.equals(right.index):
            raise RuntimeError("Paired assay identities do not match")
        for column in ("component", "original_ranking_eligible"):
            if not np.array_equal(left[column], right[column]):
                raise RuntimeError("Paired units or eligibility rules do not match")
        eligible = left.original_ranking_eligible
        if left.loc[eligible, "spearman"].isna().any() or right.loc[eligible, "spearman"].isna().any():
            raise RuntimeError("This contrast requires full defined coverage in both models")
        difference = left[["component", "spearman"]].copy()
        difference["spearman"] = left.spearman - right.spearman
        difference.loc[~eligible, "spearman"] = np.nan
        results[regime] = {"contrast": "paired Kronecker minus protein-shuffled Kronecker",
            "test_macro_difference": float(difference.spearman.mean()),
            "eligible_groups": int(eligible.sum()), "paired_unit_bootstrap": unit_bootstrap(difference),
            "sources_sha256": {str(p.relative_to(WORK)): sha256(p / "complete.json") for p in directories},
            "interpretation": "same held-out assays and same resampled whole units for both scores; conditional on one fitted model and one protein permutation per split, not training-seed or permutation uncertainty"}
    write_json(WORK / "tier2/protein_shuffle_comparison.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
