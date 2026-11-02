"""Bounded handoff uses real frozen selection/configs and simulated evaluation.

Native inference and metric correctness are qualified in the evaluator/report
tests; these tests exercise parent identity, completion, resource and retry rules.
"""
import os
from pathlib import Path

import pytest

from experiments.paired_generalization import full_evaluation_queue as module
from experiments.paired_generalization.run import sha256,write_json
from test_paired_generalization_confirmation_queue import prepared as confirmation_prepared,ready as finish_tuning,complete as complete_training


@pytest.fixture
def prepared(confirmation_prepared,monkeypatch):
    value=confirmation_prepared;root,work=value['root'],value['work']
    monkeypatch.setattr(module,'ROOT',root);monkeypatch.setattr(module,'WORK',work)
    source=root/'RewardModel/experiments/paired_generalization/full_evaluation_queue.py';source.write_text('fixture evaluator queue')
    monkeypatch.setattr(module,'__file__',str(source))
    reporting=root/'RewardModel/experiments/paired_generalization/reporting_fixture.py';reporting.write_text('fixture report source')
    monkeypatch.setattr(module.controlled_summary,'software',lambda:dict(source_sha256={str(reporting.relative_to(root)):sha256(reporting)},packages={}))
    write_json(work/'tier1/similarity_composition/design/complete.json',dict(fixture=True))
    write_json(work/'tier1/replicates/noise_summary.json',dict(fixture=True))
    parent=work/'controlled/full_confirmation_queue'
    identity=dict(pid=600,start_ticks=2000,uid=os.getuid(),cwd=str(root),argv=['python','confirmation_queue.py','run'])
    value['processes'][600]=identity
    write_json(parent/'invocation.json',dict(pid=600,plan_sha256=sha256(parent/'plan.json')))
    write_json(parent/'launch_process.json',dict(identity=identity,plan_sha256=sha256(parent/'plan.json')))
    path=module.prepare();module.validate_plan(module.read(path))
    evaluated=[];analyzed=[]
    def evaluate(job_id,workers):
        assert workers==4;evaluated.append(job_id)
        result=dict(phase='fixed_weight_evaluation_complete',fixture=True)
        write_json(work/'postselection'/job_id/'complete.json',result)
        return result
    def verify_evaluation(directory):
        result=module.read(Path(directory)/'complete.json')
        assert result['phase']=='fixed_weight_evaluation_complete'
        return {},{}
    def analyze(job_id):
        analyzed.append(job_id);output=work/'postselection_analysis'/job_id/'fixture/complete.json'
        write_json(output,dict(job_id=job_id,fixture=True));return output
    def summarize(arm,paths):
        assert arm=='full' and len(paths)==10
        assert len({module.read(p)['job_id'] for p in paths})==10
        output=work/'controlled_summaries/full/fixture/complete.json'
        write_json(output,dict(arm=arm,fixture=True));return output
    monkeypatch.setattr(module.postselection,'run_evaluation',evaluate)
    monkeypatch.setattr(module.selected_analysis,'verify_completed_evaluations',verify_evaluation)
    monkeypatch.setattr(module.selected_analysis,'analyze_job',analyze)
    monkeypatch.setattr(module.controlled_summary,'verify_analysis',lambda path,design:module.read(path))
    monkeypatch.setattr(module.controlled_summary,'summarize',summarize)
    value.update(evaluation_target=path.parent,evaluation_parent=parent,evaluation_parent_identity=identity,
        reporting_source=reporting,evaluated=evaluated,analyzed=analyzed,evaluation_plan=module.read(path))
    return value


def ready(value):
    finish_tuning(value)
    parent=value['evaluation_parent'];plan=module.read(parent/'plan.json')
    selection=module.controlled_summary.control_plan.confirm('full')
    write_json(parent/'selection.json',selection)
    module.confirmation_queue.verify_confirmation(selection)
    write_json(parent/'upstream_completion.json',module.confirmation_queue.require_tuning_completion(plan))
    for job in selection['confirmation_jobs']:complete_training(job)
    write_json(parent/'complete.json',dict(phase='full_confirmation_training_complete_other_work_pending',
        plan_sha256=sha256(parent/'plan.json'),selection_sha256=sha256(parent/'selection.json'),
        evidence=module.control_queue.require_completed(selection['confirmation_jobs']),study_complete=False))
    return selection


def test_live_parent_is_waited_even_with_completion_then_exact_ten_are_evaluated(prepared,monkeypatch):
    selection=ready(prepared);prepared['processes'][600]=prepared['evaluation_parent_identity']
    waits=[]
    def sleep(seconds):
        waits.append(seconds)
        assert not prepared['evaluated'] and not prepared['calls']
        assert module.read(prepared['evaluation_target']/'status.json')['phase']=='waiting_for_confirmation_coordinator'
        prepared['processes'].clear()
    monkeypatch.setattr(module.time,'sleep',sleep)
    result=module.run()
    expected=[j['id'] for j in [*prepared['jobs'],*selection['confirmation_jobs']]]
    assert prepared['evaluated']==prepared['analyzed']==expected
    assert len(result['evidence'])==10 and waits==[30] and not prepared['calls']
    assert not result['study_complete'] and not result['test_used_for_selection']
    assert result['summary_sha256']==sha256(Path(result['summary']))
    with pytest.raises(RuntimeError,match='already invoked'):module.run()


@pytest.mark.parametrize('damage',['no_parent_receipt','wrong_parent_receipt','missing_fit','selection','source','reused_pid'])
def test_upstream_damage_or_pid_reuse_without_completion_never_scores(prepared,damage):
    selection=ready(prepared);parent=prepared['evaluation_parent']
    if damage in ('no_parent_receipt','reused_pid'):(parent/'complete.json').unlink()
    if damage=='reused_pid':prepared['processes'][600]={**prepared['evaluation_parent_identity'],'start_ticks':3000}
    elif damage=='wrong_parent_receipt':
        record=module.read(parent/'complete.json');record['evidence'].pop();write_json(parent/'complete.json',record)
    elif damage=='missing_fit':(Path(selection['confirmation_jobs'][0]['output_dir'])/'controlled_training_complete.json').unlink()
    elif damage=='selection':write_json(parent/'selection.json',dict(changed=True))
    elif damage=='source':prepared['reporting_source'].write_text('changed')
    with pytest.raises((ValueError,RuntimeError,FileNotFoundError,KeyError)):module.run()
    assert not prepared['evaluated'] and not prepared['analyzed']


@pytest.mark.parametrize('existing',['postselection','postselection_analysis','controlled_summaries'])
def test_prepare_and_start_refuse_existing_scientific_outputs(prepared,existing):
    work=prepared['work'];identity=prepared['jobs'][0]['id']
    path=work/existing/('full' if existing=='controlled_summaries' else identity);path.mkdir(parents=True)
    with pytest.raises(RuntimeError,match='already exists'):module.prepare()
    ready(prepared)
    with pytest.raises(RuntimeError,match='already exists'):module.run()
    assert not prepared['evaluated']


def test_prepare_requires_exact_owned_live_parent(prepared):
    prepared['processes'][600]={**prepared['evaluation_parent_identity'],'uid':os.getuid()+1}
    with pytest.raises(ValueError,match='exact live owned'):module.prepare()


@pytest.mark.parametrize('gate',['gpu','storage','lock','last_gpu_check'])
def test_resource_waits_do_not_create_partial_work_or_duplicate_evaluation(prepared,monkeypatch,gate):
    ready(prepared);evaluate=module.postselection.run_evaluation;once=[True]
    if gate=='gpu':prepared['holders'].append(dict(gpu_uuid=module.controlled_summary.control_plan.GPU_UUIDS[0],pid=9))
    elif gate=='storage':monkeypatch.setattr(module.control_queue,'storage_gate',lambda:dict(ready=False))
    else:
        def blocked(*args,**kwargs):
            if once:
                once.pop()
                if gate=='lock':raise BlockingIOError('held lock')
                return dict(phase='waiting_for_authorized_gpus')
            return evaluate(*args,**kwargs)
        monkeypatch.setattr(module.postselection,'run_evaluation',blocked)
    waits=[]
    def sleep(seconds):
        waits.append(seconds);assert not prepared['evaluated']
        assert not (prepared['work']/'postselection').exists()
        prepared['holders'].clear()
        monkeypatch.setattr(module.control_queue,'storage_gate',lambda:dict(ready=True))
    monkeypatch.setattr(module.time,'sleep',sleep)
    module.run()
    assert waits==[30] and len(prepared['evaluated'])==len(set(prepared['evaluated']))==10


@pytest.mark.parametrize('failure',['worker_error','missing_completion','lock_after_start','gpu_wait_after_start'])
def test_started_or_failed_evaluation_is_never_retried(prepared,monkeypatch,failure):
    ready(prepared)
    def fail(identity,workers):
        prepared['evaluated'].append(identity)
        if failure!='missing_completion':(prepared['work']/'postselection'/identity).mkdir(parents=True)
        if failure=='worker_error':raise RuntimeError('native worker failed')
        if failure=='lock_after_start':raise BlockingIOError('unexpected after start')
        if failure=='gpu_wait_after_start':return dict(phase='waiting_for_authorized_gpus')
        return dict(phase='fixed_weight_evaluation_complete')
    monkeypatch.setattr(module.postselection,'run_evaluation',fail)
    with pytest.raises((RuntimeError,FileNotFoundError)):module.run()
    assert len(prepared['evaluated'])==1 and not prepared['analyzed']
    assert module.read(prepared['evaluation_target']/'status.json')['phase']=='stopped_for_inspection'
    with pytest.raises(RuntimeError,match='already invoked'):module.run()
    assert len(prepared['evaluated'])==1


def test_source_change_after_native_evaluation_stops_before_analysis(prepared,monkeypatch):
    ready(prepared);evaluate=module.postselection.run_evaluation
    def changed(*args,**kwargs):
        result=evaluate(*args,**kwargs);prepared['reporting_source'].write_text('changed');return result
    monkeypatch.setattr(module.postselection,'run_evaluation',changed)
    with pytest.raises(ValueError,match='Frozen evaluation/reporting input changed'):module.run()
    assert len(prepared['evaluated'])==1 and not prepared['analyzed']


def test_changed_earlier_analysis_cannot_be_included_in_final_completion(prepared,monkeypatch):
    ready(prepared);summarize=module.controlled_summary.summarize
    def changed(arm,paths):
        result=summarize(arm,paths);Path(paths[0]).write_text('{}');return result
    monkeypatch.setattr(module.controlled_summary,'summarize',changed)
    with pytest.raises(ValueError,match='reporting evidence changed'):module.run()
    assert len(prepared['evaluated'])==10 and not (prepared['evaluation_target']/'complete.json').exists()
