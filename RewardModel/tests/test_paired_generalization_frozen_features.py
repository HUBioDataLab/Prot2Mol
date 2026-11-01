import pandas as pd
import pytest

from experiments.paired_generalization.frozen_features import unique_entities


def test_entity_mapping_preserves_distinct_ids_but_rejects_ambiguous_input():
    frame = pd.DataFrame({"id": ["b", "a", "a", "c"], "sequence": ["ABC", "DEF", "DEF", "ABC"]})
    result = unique_entities(frame, "id", "sequence")
    assert result.to_dict("list") == {"row_index": [0, 1, 2], "entity_id": ["a", "b", "c"], "text": ["DEF", "ABC", "ABC"]}
    with pytest.raises(ValueError, match="multiple"):
        unique_entities(pd.DataFrame({"id": ["a", "a"], "sequence": ["ABC", "DEF"]}), "id", "sequence")
    with pytest.raises(ValueError, match="present"):
        unique_entities(pd.DataFrame({"id": ["a"], "sequence": [None]}), "id", "sequence")
