"""Preserve frozen task membership while removing unrequested seeds/rates."""
import copy

import pytest

from experiments.paired_generalization import requested_probe_protocol as module
from experiments.paired_generalization.run import write_json, sha256
from test_paired_generalization_adaptation_protocol import fixture_protocol
from test_paired_generalization_objective_probe_protocol import fixture_inputs


@pytest.fixture
def inputs():
    primary, views, groups = fixture_protocol()
    ap = module.adaptation.build_protocol(primary, views, groups)
    primary, frozen = fixture_inputs()
    op = module.objective.build_protocol(primary, frozen)
    gp = dict(sample_cap=25600, pair_budget=262144, stages=['encoder_pooled','contrastive_projected'], seed=42)
    bases = {r:dict(job=dict(id=f'full__{r}__seed42__lr1p0', arm='full', regime=r, seed=42, learning_rate_multiplier=1.),
        rules={'best_macro':dict(checkpoint='checkpoint-6', epoch=6)},
        checkpoint=dict(id='checkpoint-6', path=f'/fixture/{r}', weights_sha256='fixture')) for r in ('A', 'B')}
    return dict(adaptation=ap, objective=op, geometry=gp), bases


def test_only_single_seed_and_fixed_reference_slots_change(inputs):
    plans, bases = inputs; before = copy.deepcopy(inputs)
    result = module.amend(plans, bases)
    assert inputs == before
    a, o = result['adaptation'], result['objective']
    assert a['seeds'] == o['seeds'] == [42]
    assert a['accounting']['support_fits']*3 == plans['adaptation']['accounting']['support_fits']
    assert a['accounting']['additional_learning_rates'] == 0
    for old, new in zip(plans['adaptation']['cells'], a['cells'], strict=True):
        assert new == dict(old, total_fits=old['total_fits']//3)
    for name in ('unavailable', 'k_values', 'scopes', 'epochs', 'batching', 'optimizer_policy', 'baseline_policy'):
        assert a[name] == plans['adaptation'][name]
    assert o['accounting'] == dict(existing_model_slots=2, diagnostic_pairs=64,
        nominal_steps_per_source_and_scope=1, additional_fits=0, additional_lr_candidates=0)
    for name in ('context_policy', 'step_policy', 'precision', 'parameter_dtype', 'autocast',
                 'allow_tf32', 'dropout', 'scopes', 'source_objectives', 'target_objectives'):
        assert o[name] == plans['objective'][name]
    for slot in o['checkpoint_slots']:
        assert slot['seed'] == 42 and slot['candidate_job_ids'] == [slot['selected_job_id']]
        assert slot['selected_checkpoint'] == bases[slot['regime']]['checkpoint']
    assert result['geometry']['design'] == plans['geometry']
    assert len(result['geometry']['jobs']) == 6
    assert not a['model_fitting'] and not o['executable_gpu_queue']


@pytest.mark.parametrize('damage', ['base_seed', 'base_lr', 'base_arm', 'base_rule', 'old_seeds',
    'fit_count', 'validation_seed', 'slot_count', 'batch_count', 'candidate_rate'])
def test_expanded_or_unverified_probe_scope_is_rejected(inputs, damage):
    plans, bases = inputs
    if damage == 'base_seed': bases['A']['job']['seed'] = 43
    elif damage == 'base_lr': bases['B']['job']['learning_rate_multiplier'] = .3
    elif damage == 'base_arm': bases['A']['job']['arm'] = 'frozen'
    elif damage == 'base_rule': bases['A']['rules']['best_loss'] = bases['A']['rules'].pop('best_macro')
    elif damage == 'old_seeds': plans['adaptation']['seeds'] = [42,43]
    elif damage == 'fit_count': plans['adaptation']['cells'][0]['total_fits'] += 1
    elif damage == 'validation_seed': plans['adaptation']['cells'][0]['selection_seed'] = 43
    elif damage == 'slot_count': plans['objective']['checkpoint_slots'].pop(0)
    elif damage == 'batch_count': plans['objective']['checkpoint_slots'][0]['batch_pairs'] = 64
    else: plans['objective']['candidates']['full__A__seed42__lr1p0']['learning_rate_multiplier'] = 3.
    with pytest.raises(ValueError): module.amend(plans, bases)


def test_snapshot_budget_preserves_all_validation_epochs_and_two_selected_outputs(inputs):
    plans, bases = inputs; result = module.amend(plans, bases)
    cells = result['adaptation']['cells']; sizes = dict(heads=100, heads_cross_attention=200)
    budget = module.snapshot_budget(cells, sizes)
    manual = 0
    for cell, row in zip(cells, budget['cells'], strict=True):
        roles = cell['task_bindings']['B']
        n = len(roles['across'] if cell['kind'] == 'across' else roles['validation'])
        total = sum(cell['fits_per_seed'].values())
        assert row['maximum_snapshots'] == 15*n+2*(total-n)
        manual += (15*n+2*(total-n))*sizes[cell['scope']]
    assert budget['raw_tensor_bytes'] == manual
    assert budget['maximum_snapshots'] > result['adaptation']['accounting']['support_fits']*2


def test_mmap_snapshot_sizing_reads_only_parameters_in_requested_scopes(tmp_path, inputs, monkeypatch):
    import torch
    _, bases = inputs
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    state = {'ranking_head.weight':torch.zeros(2,3), 'fusion.weight':torch.zeros(4,5),
        'protein_encoder.weight':torch.zeros(100,100)}
    for regime in ('A','B'):
        directory = tmp_path / regime; directory.mkdir()
        torch.save(state, directory/'pytorch_model.bin')
        bases[regime]['checkpoint'].update(path=str(directory), weights_sha256=sha256(directory/'pytorch_model.bin'))
    files = {}; sizes = module.checkpoint_snapshot_sizes(bases, files)
    assert sizes == dict(heads=24, heads_cross_attention=104) and len(files) == 2
    (tmp_path/'A/pytorch_model.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='input changed'): module.checkpoint_snapshot_sizes(bases, {})


@pytest.fixture
def tier3(tmp_path, monkeypatch):
    directory = tmp_path/'training'; scope_path = directory/'plan.json'
    jobs = [dict(id=f'job{i}') for i in range(24)]
    references = [dict(id='full_A'),dict(id='full_B')]
    scope = dict(jobs=jobs,references=dict(jobs=references))
    write_json(scope_path,scope)
    evidence = [dict(job=j['id']) for j in jobs]
    write_json(directory/'complete.json',dict(
        phase='requested_fixed_config_training_complete_evaluations_and_probes_pending',
        plan_sha256=sha256(scope_path),training_fits_created=24,parameter_search=False,
        additional_seed_fits=0,evidence=evidence))
    monkeypatch.setattr(module,'verify',lambda:dict(scope_plan_sha256=sha256(scope_path)))
    monkeypatch.setattr(module.evaluation,'scope',lambda:(scope_path,scope))
    monkeypatch.setattr(module.evaluation.control_queue,'require_completed',lambda _:evidence)
    paths=[]
    for j in [*references,*jobs]:
        p=tmp_path/'analyses'/f"{j['id']}.json";write_json(p,dict(job=j));paths.append(p)
    monkeypatch.setattr(module.results,'verify_analysis',module.read)
    return directory,paths


def test_tier3_gate_requires_all_requested_training_and_selected_analyses(tier3):
    directory,paths=tier3
    result=module.require_tier3(paths)
    assert result['requested_fits']==26 and not result['scientific_probes_complete']
    with pytest.raises(ValueError,match='every requested'):module.require_tier3(paths[:-1])
    record=module.read(directory/'complete.json');record['parameter_search']=True
    write_json(directory/'complete.json',record)
    with pytest.raises(ValueError,match='not complete'):module.require_tier3(paths)


def test_tier3_gate_rejects_duplicate_or_replaced_model(tier3):
    _,paths=tier3
    with pytest.raises(ValueError,match='every requested'):module.require_tier3(paths[:-1]+paths[:1])
    write_json(paths[-1],dict(job=dict(id='unrequested_seed43')))
    with pytest.raises(ValueError,match='unrequested'):module.require_tier3(paths)
