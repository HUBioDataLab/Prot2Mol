"""Frozen weight-axis membership and validation-only A/B choices; no training."""
import copy
import json
from pathlib import Path

import pytest

from experiments.paired_generalization import weight_selection as module
from experiments.paired_generalization.run import sha256,write_json
from test_paired_generalization_control_plan import ROOT,noise_fixture


def records_fixture():
    records=[]
    for regime in ('A','B'):
        for arm in module.WEIGHTS:
            best=.3 if regime=='A' else (3. if arm=='contrastive_high' else 1.)
            peak=.8 if (regime,arm)==('A','bce_low') else (.7 if (regime,arm)==('B','contrastive_high') else .2)
            for lr in module.control_plan.LEARNING_RATE_MULTIPLIERS:
                records.append(dict(arm=arm,regime=regime,learning_rate_multiplier=lr,
                    validation_macro_spearman=peak-.01*abs(lr-best),validation_defined_fraction=1.))
    return records


def test_weight_comparison_uses_each_validation_split_and_exposes_confirmation_limits():
    records=records_fixture();result=module.choose(records)
    assert (result['choices']['A']['arm'],result['choices']['A']['learning_rate_multiplier'])==('bce_low',.3)
    assert (result['choices']['B']['arm'],result['choices']['B']['learning_rate_multiplier'])==('contrastive_high',3.)
    assert result['different_weights'] and result['different_learning_rates']
    assert not result['choices']['A']['matches_existing_confirmation_lr']
    assert result['choices']['A']['planned_confirmation_job_ids']==[]
    assert len(result['choices']['B']['planned_confirmation_job_ids'])==4
    assert result['accounting']['unique_tuning_fits']==30 and result['accounting']['tuning_fits_per_split']==15
    assert result['new_fits_created']==0 and not result['test_used_for_selection']
    for record in records:record['test_macro_spearman']=100 if record['arm']=='full' else -100
    assert module.choose(records)==result


def test_family_ties_prefer_reference_and_lr_ties_prefer_lower_multiplier():
    records=records_fixture()
    for record in records:record['validation_macro_spearman']=.4
    result=module.choose(records)
    assert all((r['arm'],r['learning_rate_multiplier'])==('full',.3) for r in result['choices'].values())
    assert all(r['matches_existing_confirmation_lr'] for r in result['choices'].values())


@pytest.mark.parametrize('damage',['missing','duplicate','extra_arm','extra_lr'])
def test_incomplete_or_expanded_candidate_grid_is_rejected(damage):
    records=records_fixture()
    if damage=='missing':records.pop()
    elif damage=='duplicate':records[-1]=records[0]
    elif damage=='extra_arm':records[0]['arm']='unplanned'
    else:records[0]['learning_rate_multiplier']=10.
    with pytest.raises(ValueError,match='grid'):module.choose(records)


def test_nonfinite_candidates_are_excluded_and_serialized_as_null_with_coverage_reported():
    records=records_fixture();records[0]['validation_macro_spearman']=float('nan')
    records[1]['validation_defined_fraction']=float('inf')
    winner=next(
        r for r in records if (r['arm'],r['regime'],r['learning_rate_multiplier'])==('bce_low','A',.3))
    winner['validation_defined_fraction']=.25
    result=module.choose(records)
    assert result['within_setting']['A']['full']['learning_rate_multiplier']==3.
    assert result['choices']['A']['validation_defined_fraction']==.25
    assert result['all_candidates']['A'][0]['validation_macro_spearman'] is None
    json.dumps(result,allow_nan=False)
    for record in records:
        if record['arm']=='full' and record['regime']=='A':record['validation_defined_fraction']=0.
    with pytest.raises(ValueError,match='No finite'):module.choose(records)


@pytest.fixture
def frozen(tmp_path,monkeypatch):
    root=tmp_path/'project';work=root/'outputs';directory=work/'controlled'
    root.mkdir();monkeypatch.setattr(module,'ROOT',root);monkeypatch.setattr(module,'WORK',work)
    monkeypatch.setattr(module,'software',lambda:dict(fixture_source=True))
    primary=module.control_plan.build_plan(ROOT,work,noise_fixture())
    write_json(directory/'plan.json',primary)
    code=root/'training.py';code.write_text('fixture qualified training source')
    sources={'training.py':sha256(code)}
    runtime=dict(world_size=4,sources=sources,signature=module.control_plan.digest_json(dict(sources=sources,plan_sha256=sha256(directory/'plan.json'))))
    candidates={(r['arm'],r['regime'],r['learning_rate_multiplier']):r for r in records_fixture()}
    metadata=[]
    for job in module.family_jobs(primary):
        item=module.control_plan.materialize_job(job,directory,sha256(directory/'plan.json'));metadata.append(item)
        output=Path(item['output_dir']);candidate=candidates[job['arm'],job['regime'],job['learning_rate_multiplier']]
        write_json(output/'experiment_summary.json',dict(config_sha256=item['config_sha256'],test_evaluated=False,test_metrics={}))
        for rank in range(4):write_json(output/f'rank_{rank}_complete.json',dict(rank=rank,world_size=4,config_sha256=item['config_sha256']))
        write_json(output/'retained/retention.json',dict(config_sha256=item['config_sha256'],selections={'best_macro':dict(step=10,metric=candidate['validation_macro_spearman'])}))
        snapshot=output/'prediction_exports/trajectory/eval_step00000010.json';snapshot.parent.mkdir(parents=True)
        snapshot.with_suffix('.npz').write_bytes(b'fixture scalar validation payload')
        write_json(snapshot,dict(predictions_sha256=sha256(snapshot.with_suffix('.npz')),
            metrics=dict(eval_macro_spearman=candidate['validation_macro_spearman'],eval_spearman_defined_fraction=1.)))
        write_json(output/'runtime_qualification.json',runtime)
        write_json(output/'controlled_training_complete.json',dict(config_sha256=item['config_sha256'],plan_sha256=item['plan_sha256'],
            evidence_sha256={'experiment_summary.json':sha256(output/'experiment_summary.json')}))
    return dict(root=root,work=work,directory=directory,primary=primary,metadata=metadata,code=code)


@pytest.mark.parametrize('damage',['membership','weights','readout','epochs','missing_job'])
def test_registered_family_cannot_change_scope_or_training_budget(frozen,damage):
    plan=copy.deepcopy(frozen['primary'])
    if damage=='membership':next(a for a in plan['arms'] if a['name']=='bce_low')['family']='elsewhere'
    elif damage=='weights':plan['tuning_jobs'][0]['config']['model']['ranking_loss_weight']=.5
    elif damage=='readout':plan['tuning_jobs'][0]['config']['model']['ranking_readout']='cosine'
    elif damage=='epochs':plan['tuning_jobs'][0]['config']['training']['num_train_epochs']=12
    else:plan['tuning_jobs'].pop(0)
    with pytest.raises(ValueError):module.family_jobs(plan)


def test_actual_registration_and_completion_chain_reads_validation_only_and_freezes_exact_choice(frozen):
    work=frozen['work'];before={str(p):sha256(p) for p in work.rglob('*') if p.is_file()}
    result=module.build_selection()
    assert len(result['validation_evidence'])==30 and result['choices']['B']['arm']=='contrastive_high'
    assert {str(p):sha256(p) for p in work.rglob('*') if p.is_file()}==before
    for item in frozen['metadata']:write_json(Path(item['output_dir'])/'unrelated_test_results.json',dict(score=999,winner='full'))
    assert module.build_selection()==result
    path=module.prepare();assert module.verify()==result
    changed=module.read(path);changed['choices']['B']['arm']='full';write_json(path,changed)
    with pytest.raises(ValueError,match='Frozen weight comparison'):module.verify()


@pytest.mark.parametrize('damage',['missing_fit','config','validation_payload','runtime','source','exposed_training_summary'])
def test_missing_or_changed_fit_evidence_prevents_weight_choice(frozen,damage):
    item=frozen['metadata'][-1];output=Path(item['output_dir'])
    if damage=='missing_fit':(output/'controlled_training_complete.json').unlink()
    elif damage=='config':Path(item['config_path']).write_text('changed')
    elif damage=='validation_payload':(output/'prediction_exports/trajectory/eval_step00000010.npz').write_bytes(b'changed')
    elif damage=='runtime':
        path=output/'runtime_qualification.json';r=module.read(path);r['world_size']=2;write_json(path,r)
    elif damage=='source':frozen['code'].write_text('changed')
    else:
        path=output/'experiment_summary.json';r=module.read(path);r['test_evaluated']=True;write_json(path,r)
    with pytest.raises((ValueError,RuntimeError,FileNotFoundError)):module.prepare()
    assert not module.destination().exists()


def test_readiness_does_not_create_selection_or_more_fits(frozen):
    output=Path(frozen['metadata'][-1]['output_dir']);(output/'controlled_training_complete.json').unlink()
    before={str(p):sha256(p) for p in frozen['work'].rglob('*') if p.is_file()}
    result=module.preflight()
    assert len(result['verified_training_completions'])==29 and len(result['missing_training_completions'])==1
    assert not result['selection_created'] and result['new_fits_created']==0 and not result['test_metrics_read']
    assert {str(p):sha256(p) for p in frozen['work'].rglob('*') if p.is_file()}==before
