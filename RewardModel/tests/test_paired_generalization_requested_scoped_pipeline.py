"""Exercise the rebound native reporting loop, with simulated scientific work."""
import copy
from pathlib import Path
import pytest
from experiments.paired_generalization import requested_scoped_pipeline as module
from experiments.paired_generalization.run import sha256, write_json
from test_paired_generalization_requested_reporting_queue import prepared


@pytest.fixture
def rebound(prepared,monkeypatch):
    p=prepared;root=p['root'];work=p['work']
    monkeypatch.setattr(module,'ROOT',root);monkeypatch.setattr(module,'WORK',work)
    monkeypatch.setattr(module.training,'ROOT',root);monkeypatch.setattr(module.training,'WORK',work)
    # configure() changes only this adapter process; restore both modules after the test.
    monkeypatch.setattr(module.reporting,'destination',module.reporting.destination)
    monkeypatch.setattr(module.probes,'destination',module.probes.destination)
    for obj,name in ((module,'requested_scoped_pipeline.py'),(module.training,'requested_scoped_training.py'),
                     (module.training.retention,'requested_retention.py')):
        path=p['source'].with_name(name);path.write_text('replacement source fixture');monkeypatch.setattr(obj,'__file__',str(path))
    parent=module.training.destination();write_json(parent/'plan.json',dict(fixture_amendment=True))
    monkeypatch.setattr(module.training,'validate',lambda plan:plan==dict(fixture_amendment=True) or pytest.fail('amendment changed'))
    owner={**p['identity'],'pid':701,'start_ticks':3001,'argv':module.expected_parent('reporting')}
    p['processes'][701]=owner
    write_json(parent/'invocation.json',dict(pid=701,plan_sha256=sha256(parent/'plan.json')))
    originals={str(path):sha256(path) for path in [p['target']/'plan.json',p['scope_path']]}
    path=module.prepare('reporting',701,3001)
    return dict(**p,new_target=path.parent,originals=originals,new_owner=owner)


def test_rebound_reporting_preserves_old_plan_and_runs_native_loop_once(rebound,monkeypatch):
    p=rebound; waits=[]
    def sleep(seconds):
        waits.append(seconds);assert not p['calls'];p['complete']()
    monkeypatch.setattr(module.reporting.time,'sleep',sleep)
    done=module.run('reporting')
    assert waits==[600] and len(p['analyzed'])==26 and len(p['calls'])==7
    assert done['additional_seed_fits']==done['new_fits']==0
    assert all(sha256(Path(path))==digest for path,digest in p['originals'].items())
    assert not (p['target']/'invocation.json').exists()
    assert (p['new_target']/'invocation.json').exists()
    with pytest.raises(RuntimeError,match='already invoked'):module.run('reporting')


@pytest.mark.parametrize('damage',['owner','argv','parent_source','old_plan','new_seed'])
def test_rebound_reporting_stops_on_changed_binding_or_scope(rebound,damage):
    p=rebound
    if damage=='owner':p['processes'][701]={**p['new_owner'],'start_ticks':9999}
    elif damage=='argv':
        path=p['new_target']/'plan.json';plan=module.read(path);plan['training_controller']['argv']=['legacy'];write_json(path,plan)
    elif damage=='parent_source':Path(module.training.__file__).write_text('changed')
    elif damage=='old_plan':write_json(p['target']/'plan.json',dict(changed=True))
    else:
        path=p['new_target']/'plan.json';plan=module.read(path);plan['seed']=43;write_json(path,plan)
    with pytest.raises((ValueError,RuntimeError)):module.run('reporting')
    assert not p['calls'] and not p['analyzed']


def test_probe_rebinding_pins_real_new_reporter_command_and_paths(rebound,monkeypatch):
    p=rebound; report_path=p['new_target']/'plan.json'
    owner={**p['new_owner'],'pid':702,'start_ticks':3002,'argv':module.expected_parent('probes')}
    p['processes'][702]=owner
    write_json(report_path.parent/'invocation.json',dict(pid=702,plan_sha256=sha256(report_path)))
    report_plan=module.read(report_path)
    old=dict(format_version=1,boot_id='boot-fixture',seed=42,learning_rate_multiplier=1.,
        reporting_plan='original_report_fixture',reporting_plan_sha256='fixture',reporting_controller=p['identity'],
        job_ids=report_plan['job_ids'],probe_plan_sha256=report_plan['probe_plan_sha256'],
        order=list(module.probes.ORDER),wait_seconds=600,gpu_uuids=report_plan['gpu_uuids'],
        files_sha256={},packages={},support_fit_upper_bound=8088,geometry_exports=6,objective_pairs=64,
        additional_primary_fits=0,parameter_search=False,additional_seed_fits=0)
    original=p['work']/'controlled/requested_probe_queue/plan.json';write_json(original,old)
    # Native probe lifecycle/coverage is tested separately; this checks binding,
    # namespace and source integrity while using the real reporting validator.
    observed=[]
    def validate(plan):
        observed.append(plan)
        assert plan['reporting_plan']==str(report_path)
        assert plan['reporting_plan_sha256']==sha256(report_path)
        assert plan['reporting_controller']==owner
        module.reporting.validate(module.read(report_path))
    monkeypatch.setattr(module.probes,'validate',validate)
    before=sha256(original);path=module.prepare('probes',702,3002)
    plan=module.read(path)
    assert observed and sha256(original)==before
    assert plan['order']==list(module.probes.ORDER) and plan['additional_seed_fits']==0
    assert str(report_path.relative_to(p['root'])) in plan['files_sha256']
    assert path.parent==module.probe_destination()
    p['processes'][702]={**owner,'argv':['wrong']}
    with pytest.raises(ValueError,match='exact live owned'):module.prepare('probes',702,3002)
