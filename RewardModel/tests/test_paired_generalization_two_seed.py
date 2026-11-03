"""User-amended seed scope, real frozen selection, and bounded evaluation handoff."""
import copy
from pathlib import Path

import pytest

from experiments.paired_generalization import two_seed_policy as policy
from experiments.paired_generalization import two_seed_summary as summary
from experiments.paired_generalization import two_seed_full_evaluation_queue as queue
from experiments.paired_generalization.run import sha256, write_json
from test_paired_generalization_full_evaluation_queue import prepared as legacy_prepared
from test_paired_generalization_confirmation_queue import prepared as confirmation_prepared
from test_paired_generalization_confirmation_queue import ready as finish_tuning, complete
from test_paired_generalization_controlled_summary import records_fixture


def two_seed_records():
    records, decision = records_fixture()
    return [r for r in records if r['job']['seed'] != 44], decision


def test_eight_fit_summary_separates_tuning_from_single_confirmation():
    records, decision = two_seed_records()
    assert len(summary.expected_jobs('full', 1.)) == 8
    result = summary.summarize_records(records, decision, 'selection')
    cell = result['selected']['best_macro']['test']['ranking_score']
    values = cell['B']['seed_spread']
    assert set(cell['B']['points']) == {'42', '43'}
    assert values['both_seeds']['original_eligible_macro_spearman']['mean'] == pytest.approx(.205)
    assert values['single_confirmation']['original_eligible_macro_spearman']['mean'] == pytest.approx(.21)
    assert values['single_confirmation']['original_eligible_macro_spearman']['sample_sd'] is None
    assert not values['confirmation_seed_variance_available']
    assert 'all_three' not in values and 'confirmation_only' not in values
    assert len(result['tuning_candidates']) == 3
    assert not result['test_used_for_selection']
    records[0]['checkpoints']['late']['partitions']['test']['metrics']['ranking_score']['original_eligible_macro_spearman'] = .99
    assert summary.summarize_records(records, decision, 'selection')['selected_learning_rate_multiplier'] == 1.


@pytest.mark.parametrize('damage', ['seed44', 'missing', 'duplicate', 'changed_selection', 'other_lr'])
def test_amended_summary_rejects_incomplete_or_wrong_seed_grid(damage):
    records, decision = two_seed_records()
    if damage == 'seed44':
        records.append(next(r for r in records_fixture()[0] if r['job']['seed'] == 44))
    elif damage == 'missing':
        records.pop()
    elif damage == 'duplicate':
        records[-1] = records[0]
    elif damage == 'changed_selection':
        records[-1]['selection_sha256'] = 'other'
    else:
        records[-1]['job']['learning_rate_multiplier'] = .3
    with pytest.raises(ValueError):
        summary.summarize_records(records, decision, 'selection')


def test_single_confirmation_undefined_value_is_not_zero_or_variance():
    result = summary.seed_spread({'42': {'rho': .2}, '43': {'rho': None}})
    assert result['single_confirmation']['rho']['defined'] == 0
    assert result['single_confirmation']['rho']['mean'] is None
    assert result['single_confirmation']['rho']['sample_sd'] is None
    with pytest.raises(ValueError):
        summary.seed_spread({'42': {'rho': .2}, '43': {'rho': .3}, '44': {'rho': .4}})


@pytest.fixture
def prepared(legacy_prepared, monkeypatch):
    value = legacy_prepared
    root, work = value['root'], value['work']
    finish_tuning(value)
    for module in (policy, queue):
        monkeypatch.setattr(module, 'ROOT', root)
        monkeypatch.setattr(module, 'WORK', work)
        source = root / 'RewardModel/experiments/paired_generalization' / Path(module.__file__).name
        source.write_text('fixture amended source')
        monkeypatch.setattr(module, '__file__', str(source))
    monkeypatch.setattr(queue.controlled_summary, 'software', lambda: dict(
        source_sha256={str(value['reporting_source'].relative_to(root)): sha256(value['reporting_source'])}, packages={}))
    selection = policy.control_plan.confirm('full')
    parent = work / 'controlled/full_confirmation_queue'
    write_json(parent / 'selection.json', selection)
    write_json(parent / 'upstream_completion.json', queue.confirmation_queue.require_tuning_completion(queue.read(parent / 'plan.json')))
    confirmations = [j for j in selection['confirmation_jobs'] if j['seed'] == 43]
    for job in confirmations:
        complete(job)
    value['processes'].clear()
    completed = queue.control_queue.require_completed([*value['jobs'], *confirmations])
    intent = dict(user_instruction=policy.USER_INSTRUCTION, seeds=[42, 43], skipped_seed=44,
        stop_processes=[value['evaluation_parent_identity']])
    write_json(policy.destination() / 'cancellation_intent.json', intent)
    done = dict(phase='seed44_cancelled_for_user_two_seed_amendment',
        intent_sha256=sha256(policy.destination() / 'cancellation_intent.json'),
        B44_not_started=True, partial_A44_preserved=True, completed_fits=8,
        completed_evidence=completed)
    path = policy.destination() / 'cancellation_complete.json'
    write_json(path, done)
    monkeypatch.setattr(policy, 'CANCELLATION_SHA256', sha256(path))
    path = queue.prepare()
    queue.validate_plan(queue.read(path))
    monkeypatch.setattr(queue.controlled_summary, 'verify_analysis', lambda path, design: queue.read(path))
    def summarize(arm, paths):
        assert arm == 'full' and len(paths) == 8
        assert all('__seed44__' not in queue.read(p)['job_id'] for p in paths)
        result = work / 'controlled_summaries_two_seed/full/fixture/complete.json'
        write_json(result, dict(arm=arm, fixture=True))
        return result
    monkeypatch.setattr(queue.controlled_summary, 'summarize', summarize)
    value.update(new_target=path.parent, selected=selection, confirmations=confirmations)
    return value


def test_cancelled_three_seed_queue_transitions_to_exactly_eight_evaluations(prepared):
    result = queue.run()
    expected = [j['id'] for j in [*prepared['jobs'], *prepared['confirmations']]]
    assert prepared['evaluated'] == prepared['analyzed'] == expected
    assert len(result['evidence']) == 8 and not prepared['calls']
    assert all('__seed44__' not in identity for identity in prepared['evaluated'])
    assert result['phase'] == 'full_two_seed_evaluation_and_summary_complete_other_work_pending'
    assert not result['study_complete'] and not result['test_used_for_selection']
    with pytest.raises(RuntimeError, match='already invoked'):
        queue.run()


def test_future_confirmation_registration_contains_only_seed43(prepared):
    selected = policy.confirm('full')
    assert [(j['seed'], j['regime']) for j in selected['confirmation_jobs']] == [(43, 'A'), (43, 'B')]
    assert selected['selection'] == prepared['selected']['selection']
    assert policy.verify()['primary_accounting']['total_runs'] == 8 * len(queue.read(prepared['work'] / 'controlled/plan.json')['arms'])
    with pytest.raises(ValueError, match='seed42/43'):
        policy.require_allowed_job(dict(seed=44, phase='confirmation'))


@pytest.mark.parametrize('damage', ['live_retired', 'receipt', 'missing_fit', 'policy', 'source', 'partial_output', 'B44_started'])
def test_unverified_amendment_or_partial_work_never_scores(prepared, damage):
    if damage == 'live_retired':
        identity = prepared['evaluation_parent_identity']
        prepared['processes'][identity['pid']] = identity
    elif damage == 'receipt':
        (policy.destination() / 'cancellation_complete.json').write_text('{}')
    elif damage == 'missing_fit':
        (Path(prepared['confirmations'][0]['output_dir']) / 'controlled_training_complete.json').unlink()
    elif damage == 'policy':
        record = policy.read(policy.destination() / 'policy.json'); record['seeds'].append(44)
        write_json(policy.destination() / 'policy.json', record)
    elif damage == 'source':
        prepared['reporting_source'].write_text('changed')
    elif damage == 'partial_output':
        (prepared['work'] / 'postselection' / prepared['jobs'][0]['id']).mkdir(parents=True)
    else:
        (prepared['work'] / 'controlled/runs/full__B__seed44__lr1p0').mkdir()
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        queue.run()
    assert not prepared['evaluated'] and not prepared['analyzed']


def test_resource_wait_then_eight_jobs_without_training(prepared, monkeypatch):
    prepared['holders'].append(dict(gpu_uuid=policy.control_plan.GPU_UUIDS[0], pid=9))
    waits = []
    def sleep(seconds):
        waits.append(seconds); assert not prepared['evaluated']
        prepared['holders'].clear()
    monkeypatch.setattr(queue.time, 'sleep', sleep)
    queue.run()
    assert waits == [30] and len(prepared['evaluated']) == 8 and not prepared['calls']


def test_failed_started_job_is_not_retried(prepared, monkeypatch):
    def fail(identity, workers):
        prepared['evaluated'].append(identity)
        (prepared['work'] / 'postselection' / identity).mkdir(parents=True)
        raise RuntimeError('worker failed')
    monkeypatch.setattr(queue.postselection, 'run_evaluation', fail)
    with pytest.raises(RuntimeError, match='worker failed'):
        queue.run()
    assert len(prepared['evaluated']) == 1
    with pytest.raises(RuntimeError, match='already invoked'):
        queue.run()


def test_summary_bound_to_two_seed_policy_and_saved_artifacts(tmp_path, monkeypatch):
    import test_paired_generalization_controlled_summary as fixtures
    monkeypatch.setattr(fixtures, 'module', summary)
    monkeypatch.setattr(policy, 'WORK', tmp_path)
    monkeypatch.setattr(policy, 'verify', lambda: {})
    write_json(policy.destination() / 'policy.json', dict(seeds=[42, 43]))
    paths, _ = fixtures.persist_records(tmp_path, monkeypatch)
    result_path = summary.summarize('full', paths)
    result = summary.read(result_path)
    assert 'controlled_summaries_two_seed' in str(result_path)
    assert len(result['composition_pairs']) == 4 and len(result['paired_checkpoint_changes']) == 4
    assert result['source']['seed_policy_sha256'] == sha256(policy.destination() / 'policy.json')
    assert summary.summarize('full', paths) == result_path
    changed = copy.deepcopy(result); changed['payload_sha256'] = 'bad'
    write_json(result_path, changed)
    with pytest.raises(ValueError, match='Completed summary changed'):
        summary.summarize('full', paths)
