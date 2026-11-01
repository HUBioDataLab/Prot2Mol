import numpy as np
import pandas as pd

from experiments.paired_generalization.target_coverage import stratify


def test_counts_unique_training_assays_bins_keep_undefined_and_ineligible_groups():
    groups = pd.DataFrame(dict(group_id=['cold__a','one__b','few__c','few__d','many__e','one__f'],
        size=[3,3,3,3,3,1], component=[0,1,2,2,3,4],
        original_ranking_eligible=[True,True,True,True,True,False],
        spearman=[1.,.5,np.nan,-1.,0.,1.]))
    annotated, rows = stratify(groups,pd.Series({'one':1,'few':5,'many':101}))
    assert annotated.support_bin.tolist() == ['0','1','2-5','2-5','>100','1']
    by_bin = {row['support_bin']:row for row in rows}
    assert by_bin['1']['assays'] == 2 and by_bin['1']['eligible_assays'] == 1
    assert by_bin['1']['macro_spearman'] == .5
    assert by_bin['2-5']['defined_fraction'] == .5
    assert by_bin['2-5']['macro_spearman'] == -1.
    assert by_bin['2-5']['unit_bootstrap']['units'] == 1
    assert by_bin['6-20']['macro_spearman'] is None
    assert groups.spearman.iloc[-1] == 1.
