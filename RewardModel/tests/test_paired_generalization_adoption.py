from contextlib import contextmanager, nullcontext
import json

import pytest

from experiments.paired_generalization.adopt_lambdamart import publish, verify_trial
from experiments.paired_generalization.run import sha256, write_json


@pytest.fixture
def completed_trial(tmp_path):
    source, dest = tmp_path/'worker04', tmp_path/'main'
    source.mkdir()
    dest.mkdir()
    contract = dict(base_parameters={'objective': 'rank:ndcg'}, settings=[{}]*5,
                    max_rounds=500, evaluate_every=10, patience_rounds=100,
                    data_sha256={'train': 'frozen'}, feature_provenance={'same': True})
    write_json(source/'fit_contract.json', dict(**contract, precomputed_trial=4))
    write_json(dest/'fit_contract.json', contract)
    trial = source/'trial04'
    trial.mkdir()
    (trial/'model.ubj').write_bytes(b'weights')
    history = [dict(rounds=r, macro_spearman=1/r) for r in range(10, 111, 10)]
    result = dict(parameters=contract['base_parameters'], rounds_trained=110,
                  selected=history[0], model_sha256=sha256(trial/'model.ubj'))
    write_json(trial/'validation_history.json', dict(history=history, selected=history[0]))
    write_json(trial/'complete.json', result)
    write_json(source/'complete.json', dict(**contract, precomputed_trial=4, trials=[dict(trial=4, **result)]))
    return source, dest


def test_publish_retains_source_and_model_provenance(completed_trial):
    source, dest = completed_trial
    publish(source, dest, 4, nullcontext())
    assert (source/'trial04/model.ubj').read_bytes() == (dest/'trial04/model.ubj').read_bytes()
    assert json.loads((dest/'trial04/adoption.json').read_text())['source'] == str(source)


@pytest.mark.parametrize('kind', ['data', 'weights', 'early', 'winner', 'history'])
def test_rejects_invalid_precomputation(completed_trial, kind):
    source, dest = completed_trial
    if kind == 'data':
        path = dest/'fit_contract.json'
        data = json.loads(path.read_text())
        data['data_sha256']['train'] = 'different'
    elif kind == 'weights':
        (source/'trial04/model.ubj').write_bytes(b'wrong')
        with pytest.raises(ValueError):
            verify_trial(source, dest, 4)
        return
    else:
        path = source/'trial04/validation_history.json'
        data = json.loads(path.read_text())
        if kind == 'early':
            data['history'][-1]['macro_spearman'] = .5
            data['selected'] = data['history'][-1]
            result = json.loads((source/'trial04/complete.json').read_text())
            result['selected'] = data['selected']
            write_json(source/'trial04/complete.json', result)
            parent = json.loads((source/'complete.json').read_text())
            parent['trials'] = [dict(trial=4, **result)]
            write_json(source/'complete.json', parent)
        elif kind == 'winner':
            data['history'][1]['macro_spearman'] = .5
        else:
            data['history'].pop(2)
    write_json(path, data)
    with pytest.raises(ValueError):
        verify_trial(source, dest, 4)


def test_refuses_late_target_and_exits_guard(completed_trial):
    source, dest = completed_trial
    exited = []
    @contextmanager
    def race():
        (dest/'trial04').mkdir()
        (dest/'trial04/live').write_text('keep')
        try:
            yield
        finally:
            exited.append(True)
    with pytest.raises(FileExistsError):
        publish(source, dest, 4, race())
    assert exited == [True]
    assert (dest/'trial04/live').read_text() == 'keep'
    assert not list(dest.glob('.adopt_*'))
