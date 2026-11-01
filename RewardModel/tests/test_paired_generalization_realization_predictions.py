import json
import numpy as np
import pytest

from experiments.paired_generalization.realization_protocol import make_views
from experiments.paired_generalization.realization_predictions import choose_epochs, validation_trajectories, validate_export
from experiments.paired_generalization.run import sha256, write_json
from test_paired_generalization_realization_protocol import fixture


def test_each_selector_uses_only_its_validation_roles_and_reports_coverage():
    rows, views, _ = make_views(*fixture())
    good = rows.pchembl_value.to_numpy()
    bad = -good
    scores = [(1, 10, good.copy()), (2, 20, bad.copy())]
    first = validation_trajectories(rows, views, scores)
    assert all(item['selected']['epoch'] == 1 for item in first.values())
    # Change only original test rows in BOTH epochs. Original validation must
    # stay exactly unchanged even though those rows validate other allocations.
    changed = [(epoch, step, values.copy()) for epoch, step, values in scores]
    for _, _, values in changed:
        values[views['original_seed42_test']] *= -1
    second = validation_trajectories(rows, views, changed)
    assert second['original_seed42'] == first['original_seed42']
    assert second['equal_size_seed43'] != first['equal_size_seed43']
    assert second['equal_size_seed44'] != first['equal_size_seed44']
    constant = good.copy()
    constant[rows.component == 0] = 0
    result = validation_trajectories(rows, views, [(1,10,constant)])
    assert result['original_seed42']['selected']['defined_fraction'] == .5


def test_epoch_selection_earliest_tie_and_rejects_undefined_or_missing_scores():
    rows = [dict(epoch=e,step=10*e,macro_spearman=.2,defined_fraction=1.) for e in (3,1,2)]
    assert choose_epochs(rows)['epoch'] == 1
    with pytest.raises(ValueError, match='No defined'):
        choose_epochs([dict(epoch=1,step=10,macro_spearman=None,defined_fraction=0.)])
    data, views, _ = make_views(*fixture())
    with pytest.raises(ValueError, match='Incomplete'):
        validation_trajectories(data, views, [(1,10,np.zeros(len(data)-1))])


def test_corrupt_or_wrong_checkpoint_export_is_rejected(tmp_path):
    path = tmp_path/'predictions.parquet'
    path.write_bytes(b'fixture')
    job = dict(step=10, weights_sha256='checkpoint-one')
    write_json(tmp_path/'complete.json', dict(job=job, predictions_sha256=sha256(path)))
    assert validate_export(tmp_path, job)['job'] == job
    with pytest.raises(ValueError, match='provenance'):
        validate_export(tmp_path, dict(step=10, weights_sha256='checkpoint-two'))
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='payload'):
        validate_export(tmp_path, job)


def test_complete_allocation_selection_verifies_every_epoch_and_payload(tmp_path, monkeypatch):
    from experiments.paired_generalization import realization_predictions as module
    monkeypatch.setattr(module, 'WORK', tmp_path)
    rows, views, counts = make_views(*fixture())
    directory = tmp_path/'controlled/realizations'
    directory.mkdir(parents=True)
    rows.to_parquet(directory/'rows.parquet', index=False)
    np.savez_compressed(directory/'views.npz', **views)
    write_json(tmp_path/'parent_plan.json', {})
    write_json(tmp_path/'assignment_audit.json', {})
    contract = dict(artifacts_sha256={name:sha256(directory/name) for name in ('rows.parquet','views.npz')},
        control_plan_path=str(tmp_path/'parent_plan.json'), control_plan_sha256=sha256(tmp_path/'parent_plan.json'),
        assignment_audit_path=str(tmp_path/'assignment_audit.json'), assignment_audit_sha256=sha256(tmp_path/'assignment_audit.json'),
        realizations=counts, limitations='fixture')
    write_json(directory/'plan.json', contract)
    # select_validation loads the module's explicit contract through this hook.
    original_load = module.load_contract
    monkeypatch.setattr(module, 'load_contract', lambda path=None: original_load(path or directory))
    run = tmp_path/'run'
    write_json(run/'launch_provenance.json', dict(config_sha256='fixture'))
    output = module.destination(run)
    jobs = [dict(epoch=epoch,step=epoch*10,weights_sha256=f'weights-{epoch}') for epoch in range(1,16)]
    frozen = dict(run=str(run),config_sha256='fixture',contract_path=str(directory/'plan.json'),
        contract_sha256=sha256(directory/'plan.json'),jobs=jobs,source_weights=[],
        launch_provenance_sha256=sha256(run/'launch_provenance.json'))
    write_json(output/'jobs.json', frozen)
    for job in jobs:
        for split in ('val','test'):
            frame = rows[rows.original_split == split].reset_index(drop=True).copy()
            frame['row_index'] = np.arange(len(frame))
            frame['ranking_score'] = frame.pchembl_value * (1 if job['epoch'] == 1 else -1)
            location = output/f"epoch{job['epoch']:02d}_{split}"
            location.mkdir(parents=True)
            frame.to_parquet(location/'predictions.parquet', index=False)
            write_json(location/'complete.json', dict(
                job=dict(**job,split=split,jobs_sha256=sha256(output/'jobs.json')),
                rows=len(frame),predictions_sha256=sha256(location/'predictions.parquet')))
    result = module.select_validation(run)
    assert result['selected_steps'] == [10]
    assert not result['test_metrics_computed']
    assert len(result['export_manifests']) == 30
    assert module.verify_validation(run) == result
    payload = output/'epoch15_test/predictions.parquet'
    payload.write_bytes(b'damaged last epoch not selected')
    with pytest.raises(ValueError, match='payload changed'):
        module.verify_validation(run)
    assert json.loads((output/'validation_complete.json').read_text()) == result
