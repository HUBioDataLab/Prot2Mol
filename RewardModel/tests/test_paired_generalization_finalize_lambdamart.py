import copy
import json

import pytest

from experiments.paired_generalization import lambdamart
from experiments.paired_generalization.finalize_lambdamart import verify_fixed_trial, validation_reproduction
from experiments.paired_generalization.run import sha256, write_json


def fixture(tmp_path):
    settings = dict(max_depth=3,reg_lambda=1.,min_child_weight=1.)
    contract = dict(base_parameters={'objective':'rank:ndcg'},evaluate_every=10,max_rounds=500,patience_rounds=100)
    write_json(tmp_path/'fit_contract.json',contract)
    trial = tmp_path/'trial00';trial.mkdir()
    (trial/'model.ubj').write_bytes(b'completed depth3 weights')
    records = [dict(rounds=r,macro_spearman=1/r) for r in range(10,111,10)]
    result = dict(parameters={**contract['base_parameters'],**settings},rounds_trained=110,
                  selected=records[0],model_sha256=sha256(trial/'model.ubj'))
    write_json(trial/'complete.json',result)
    write_json(trial/'validation_history.json',dict(history=records,selected=records[0]))
    decision = dict(selected_trial=0,max_depth=3,selected_setting=settings,remaining_search_cancelled=True,
                    termination_verified=True,completed_depth3={'A':dict(completion_sha256=sha256(trial/'complete.json'),model_sha256=result['model_sha256'])})
    return decision,result


def test_fixed_depth3_uses_completed_validation_selected_tree_count(tmp_path):
    decision,result = fixture(tmp_path)
    assert verify_fixed_trial(tmp_path,decision,'A')[1]==result


@pytest.mark.parametrize('damage',['choice','weights','completion','winner','history','live'])
def test_rejects_changed_choice_checkpoint_or_validation_history(tmp_path,damage):
    decision,_ = fixture(tmp_path)
    if damage=='choice':decision['max_depth']=5
    elif damage=='weights':(tmp_path/'trial00/model.ubj').write_bytes(b'changed')
    elif damage=='completion':(tmp_path/'trial00/complete.json').write_text('{}')
    elif damage=='live':decision['termination_verified']=False
    else:
        path = tmp_path/'trial00/validation_history.json'
        record = json.loads(path.read_text())
        if damage=='winner':record['history'][1]['macro_spearman']=.8
        else:record['history'].pop(2)
        write_json(path,record)
    with pytest.raises(ValueError):verify_fixed_trial(tmp_path,decision,'A')


def test_user_stop_prevents_both_sequential_and_independent_training(tmp_path,monkeypatch):
    monkeypatch.setattr(lambdamart,'WORK',tmp_path)
    write_json(tmp_path/'tier2/lambdamart/user_decision.json',dict(stop=True))
    for precompute in [None,0,3,4]:
        with pytest.raises(RuntimeError,match='User ended'):
            lambdamart.main('A',precompute)


def test_validation_rescoring_reports_roundoff_but_rejects_metric_or_coverage_changes():
    reference = dict(macro_spearman=.05,eligible_groups=4000,defined_groups=4000)
    observed = dict(reference,macro_spearman=.05+1.4e-8)
    result = validation_reproduction(observed,reference)
    assert result['difference']==observed['macro_spearman']-reference['macro_spearman']
    assert result['difference']!=0
    for changed in [dict(reference,macro_spearman=.051),dict(reference,defined_groups=3999)]:
        with pytest.raises(ValueError):validation_reproduction(changed,reference)
