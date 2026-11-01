import copy
import json
from pathlib import Path

import pytest
import yaml

from experiments.paired_generalization import control_plan, learning_curve_driver as driver
from experiments.paired_generalization.run import sha256, write_json
from test_paired_generalization_control_plan import ROOT, noise_fixture


def curve_fixture(tmp_path):
    primary = control_plan.build_plan(ROOT,tmp_path,noise_fixture())
    directory = tmp_path/'learning_curves'
    jobs = []
    for fraction in (.1,.25,.5):
        name = f'unit_curve_fraction{int(fraction*100):03d}'
        for original in primary['tuning_jobs'][:6]:
            job = copy.deepcopy(original)
            job['id'] = control_plan.job_id(name,job['regime'],42,job['learning_rate_multiplier'])
            job.update(arm=name,fraction=fraction)
            job['config']['training']['output_dir'] = str(directory/'runs'/job['id'])
            job['config']['data'].update(train_view_manifest=str(directory/f"{job['regime']}_{fraction}.json"),
                                         train_view_sha256='f'*64)
            jobs.append(job)
    plan = dict(tuning_jobs=jobs)
    write_json(directory/'plan.json',plan)
    for job in jobs:
        control_plan.materialize_job(job,directory,sha256(directory/'plan.json'))
    return directory,plan


def finish_tuning(directory, job):
    metadata = json.loads((directory/'jobs'/f"{job['id']}.json").read_text())
    output = Path(metadata['output_dir'])
    # A prefers .3, B prefers 1. The B winner must be used on both splits.
    score = ({1.:.3,.3:.2,3.:.1} if job['regime']=='B' else {1.:.1,.3:.9,3.:.2})[job['learning_rate_multiplier']]
    write_json(output/'experiment_summary.json',dict(config_sha256=metadata['config_sha256'],test_evaluated=False,test_metrics={}))
    for rank in range(4):
        write_json(output/f'rank_{rank}_complete.json',dict(rank=rank,world_size=4,config_sha256=metadata['config_sha256']))
    write_json(output/'retained/retention.json',dict(config_sha256=metadata['config_sha256'],selections={'best_macro':{'step':10,'metric':score}}))
    snapshot = output/'prediction_exports/trajectory/eval_step00000010.json'
    snapshot.parent.mkdir(parents=True)
    snapshot.with_suffix('.npz').write_bytes(b'fixture payload')
    write_json(snapshot,dict(predictions_sha256=sha256(snapshot.with_suffix('.npz')),
                            metrics=dict(eval_macro_spearman=score,eval_spearman_defined_fraction=1.)))
    write_json(output/'controlled_training_complete.json',dict(config_sha256=metadata['config_sha256'],
        plan_sha256=metadata['plan_sha256'],evidence_sha256={'experiment_summary.json':sha256(output/'experiment_summary.json')}))


def test_curve_selection_keeps_subset_budget_and_uses_only_b_validation(tmp_path):
    directory,plan = curve_fixture(tmp_path)
    for job in plan['tuning_jobs'][:6]:
        finish_tuning(directory,job)
    result = driver.confirm_fraction(.1,directory)
    assert result['selection']['learning_rate_multiplier']==1.
    assert {(m['seed'],m['regime']) for m in result['confirmation_jobs']}=={(43,'A'),(43,'B'),(44,'A'),(44,'B')}
    for metadata in result['confirmation_jobs']:
        config = yaml.safe_load(Path(metadata['config_path']).read_text())
        source = next(j for j in plan['tuning_jobs'][:6] if j['regime']==metadata['regime'] and j['learning_rate_multiplier']==1.)
        expected = copy.deepcopy(source['config'])
        expected['training'].update(seed=metadata['seed'],output_dir=metadata['output_dir'])
        assert config==json.loads(json.dumps(expected))
        assert config['training']['num_train_epochs']==15
        assert config['training']['evaluate_test'] is False
    assert driver.confirm_fraction(.1,directory)==result
    assert driver.next_job(directory,plan)['id']==result['confirmation_jobs'][0]['id']


@pytest.mark.parametrize('damage',['unfinished','missing_rank','test_exposure','prediction_bytes','config'])
def test_curve_selection_rejects_incomplete_or_changed_evidence_before_new_configs(tmp_path,damage):
    directory,plan = curve_fixture(tmp_path)
    for job in plan['tuning_jobs'][:6]:
        finish_tuning(directory,job)
    metadata = json.loads((directory/'jobs'/f"{plan['tuning_jobs'][1]['id']}.json").read_text())
    output = Path(metadata['output_dir'])
    if damage=='unfinished':
        (output/'controlled_training_complete.json').unlink()
    elif damage=='missing_rank':
        (output/'rank_2_complete.json').unlink()
    elif damage=='test_exposure':
        path = output/'experiment_summary.json'
        record = json.loads(path.read_text());record['test_evaluated']=True;write_json(path,record)
        done = output/'controlled_training_complete.json'
        record = json.loads(done.read_text());record['evidence_sha256']['experiment_summary.json']=sha256(path);write_json(done,record)
    elif damage=='prediction_bytes':
        (output/'prediction_exports/trajectory/eval_step00000010.npz').write_bytes(b'changed')
    else:
        Path(metadata['config_path']).write_text('{}')
    with pytest.raises((ValueError,FileNotFoundError)):
        driver.confirm_fraction(.1,directory)
    assert not (directory/'selections/fraction010.json').exists()
    assert not list((directory/'configs').glob('*seed43*'))


def test_curve_queue_does_not_skip_an_incomplete_fit(tmp_path):
    directory,plan = curve_fixture(tmp_path)
    first = driver.next_job(directory,plan)
    assert first['id']==plan['tuning_jobs'][0]['id']
    output = Path(first['output_dir'])
    write_json(output/'experiment_summary.json',dict(apparent_completion=True))
    assert driver.next_job(directory,plan)==first
    with pytest.raises(ValueError,match='smaller fractions'):
        driver.confirm_fraction(1.,directory)


def test_curve_cannot_start_while_primary_driver_owns_global_lock(tmp_path,monkeypatch):
    import fcntl
    monkeypatch.setattr(driver,'WORK',tmp_path)
    monkeypatch.setattr(driver,'ROOT',Path('/home/atabey/Prot2Mol'))
    (tmp_path/'controlled').mkdir()
    def no_preparation():
        raise AssertionError('A competing driver must stop before preparation or launch')
    monkeypatch.setattr(driver,'verify_prepared',no_preparation)
    with (tmp_path/'controlled/driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):driver.run_next()


def test_full_endpoint_requires_all_ten_shared_fits_without_creating_curve_fits(tmp_path,monkeypatch):
    from test_paired_generalization_control_driver import completed_tuning_fixture
    primary = completed_tuning_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(driver,'WORK',tmp_path)
    first = driver.full_endpoint_gate(primary)
    assert not first['ready'] and len(first['missing'])==6
    directory = tmp_path/'controlled'
    for job in primary['tuning_jobs'][:6]:
        metadata = json.loads((directory/'jobs'/f"{job['id']}.json").read_text())
        output = Path(metadata['output_dir'])
        write_json(output/'controlled_training_complete.json',dict(config_sha256=metadata['config_sha256'],
            plan_sha256=metadata['plan_sha256'],evidence_sha256={'experiment_summary.json':sha256(output/'experiment_summary.json')}))
    second = driver.full_endpoint_gate(primary)
    assert not second['ready'] and len(second['missing'])==4 and len(second['evidence'])==6
    for metadata in control_plan.confirm('full')['confirmation_jobs']:
        output = Path(metadata['output_dir'])
        write_json(output/'experiment_summary.json',dict(fixture='completed confirmation'))
        write_json(output/'controlled_training_complete.json',dict(config_sha256=metadata['config_sha256'],
            plan_sha256=metadata['plan_sha256'],evidence_sha256={'experiment_summary.json':sha256(output/'experiment_summary.json')}))
    final = driver.full_endpoint_gate(primary)
    assert final['ready'] and len(final['evidence'])==10
    assert not (tmp_path/'learning_curves/runs').exists()
