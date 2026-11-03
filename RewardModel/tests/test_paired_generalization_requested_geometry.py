"""Actual tiny native inference plus isolated scheduler/provenance qualification."""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from experiments.paired_generalization import requested_geometry as module
from experiments.paired_generalization import representation_geometry as definitions
from experiments.paired_generalization.run import sha256, write_json
from test_paired_generalization_requested_evaluation import native


@pytest.fixture
def prepared(native, tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'ROOT', tmp_path); monkeypatch.setattr(module, 'WORK', tmp_path)
    monkeypatch.setattr(module.evaluation.control_queue, 'runtime_guard', lambda: None)
    monkeypatch.setattr(module.evaluation.control_queue, 'boot_id', lambda: 'fixture-boot')
    monkeypatch.setattr(module, 'software', lambda: dict(source_sha256={}, packages={}))
    monkeypatch.setattr(module.protocol, 'destination', lambda: tmp_path / 'protocol')
    monkeypatch.setattr(module.evaluation.requested, 'destination', lambda: tmp_path / 'training')
    bases = {}
    for job in native['jobs']:
        path = module.evaluation.prepare(job['id']); base = module.read(path)
        bases[job['regime']] = {k: base[k] for k in ('job', 'checkpoint', 'rules')}
    memberships = {}
    for regime in ('A', 'B'):
        for split in module.PARTITIONS:
            frame = pd.read_parquet(tmp_path / 'data' / (split + '.parquet'))
            selected, _ = definitions.sampling_design(frame, regime, split, cap=7)
            path = tmp_path / 'tier4/geometry_design' / f'{regime}_{split}.parquet'
            path.parent.mkdir(parents=True, exist_ok=True); selected.to_parquet(path, index=False)
            memberships[path.name] = sha256(path)
    probe = dict(probes=dict(geometry=dict(resolved_bases=bases, design=dict(artifacts_sha256=memberships))))
    write_json(tmp_path / 'protocol/plan.json', probe)
    write_json(tmp_path / 'protocol/complete.json', dict(fixture=True))
    write_json(tmp_path / 'training/complete.json', dict(fixture=True))
    monkeypatch.setattr(module.protocol, 'verify', lambda: module.read(tmp_path / 'protocol/plan.json'))
    analyses = []
    for index in range(26):
        path = tmp_path / 'analyses' / f'{index}.json'; write_json(path, dict(fixture=index)); analyses.append(path)
    gate = dict(requested_fits=26, analysis_sha256={str(p): sha256(p) for p in analyses})
    def require_gate(paths):
        assert list(paths) == analyses
        return gate
    monkeypatch.setattr(module.protocol, 'require_tier3', require_gate)
    from reward_model import model as model_api
    loader = model_api.load_reward_model
    def with_tokenizers(*args, **kwargs):
        model = loader(*args, **kwargs)
        model.protein_tokenizer = model.molecule_tokenizer = SimpleNamespace(pad_token_id=1)
        return model
    monkeypatch.setattr(model_api, 'load_reward_model', with_tokenizers)
    path = module.prepare(analyses); module.verify_plan(path)
    return dict(path=path, plan=module.read(path), analyses=analyses, root=tmp_path, native=native)


def test_native_checkpoint_and_six_cohorts_preserve_weights(prepared, monkeypatch):
    path = prepared['path']; files = {r: Path(v['checkpoint']['path']) / 'pytorch_model.bin'
        for r, v in prepared['plan']['models'].items()}
    before = {r: sha256(p) for r, p in files.items()}
    from reward_model.training.trainer import RewardModelTrainer
    monkeypatch.setattr(RewardModelTrainer, 'train', lambda *_: pytest.fail('Geometry cannot train'))
    monkeypatch.setattr(RewardModelTrainer, 'create_optimizer', lambda *_: pytest.fail('Geometry cannot create optimizer'))
    for i, regime in enumerate(('A', 'B')):
        module.worker(path, regime, i, cpu=True)
        done, results = module.verify_worker(path, regime, cpu=True)
        assert done['new_fits'] == 0 and done['gpu_uuid'] is None
        assert set(results) == {'train', 'val', 'test'}
        for split, result in results.items():
            assert result['cpu_qualification']
            assert len(result['analysis']['statistics']) == 8
            assert result['analysis']['preprojection_alignment'] is None
            manifest = module.read(path.parent / regime / split / 'export.json')
            assert manifest['native_cosine_max_abs_difference'] <= 1e-5
            assert manifest['parameters_unchanged'] and not manifest['optimizer_created']
        # Verified completed data can be read again; no forward runs twice.
        module.worker(path, regime, i, cpu=True)
        with pytest.raises(ValueError, match='execution mode'): module.verify_worker(path, regime)
    assert before == {r: sha256(p) for r, p in files.items()}


@pytest.mark.parametrize('damage', ['checkpoint', 'membership', 'analysis', 'protocol', 'source', 'tokens'])
def test_changed_input_never_reaches_model_forward(prepared, monkeypatch, damage):
    path = prepared['path']; base = prepared['plan']['models']['A']; root = prepared['root']
    if damage == 'checkpoint': (Path(base['checkpoint']['path']) / 'pytorch_model.bin').write_bytes(b'changed')
    elif damage == 'membership': Path(base['partitions']['train']['membership']).write_bytes(b'changed')
    elif damage == 'analysis': prepared['analyses'][0].write_text('{}')
    elif damage == 'protocol': (root / 'protocol/plan.json').write_text('{}')
    elif damage == 'source': Path(base['partitions']['train']['data']['source']).write_bytes(b'changed')
    else:
        arrow = next(Path(base['partitions']['train']['data']['cache']).glob('*.arrow'))
        arrow.write_bytes(b'changed')
    monkeypatch.setattr(module.geometry, 'export_geometry', lambda *_args, **_kw: pytest.fail('Changed data reached inference'))
    with pytest.raises((ValueError, KeyError)): module.worker(path, 'A', 0, cpu=True)
    assert not (path.parent / 'A/complete.json').exists()


def test_missing_tier3_gate_prevents_plan_creation(prepared, monkeypatch):
    monkeypatch.setattr(module.protocol, 'require_tier3', lambda _: (_ for _ in ()).throw(ValueError('Tier3 incomplete')))
    with pytest.raises(ValueError, match='Tier3 incomplete'): module.build(prepared['analyses'])


def test_raw_partition_needs_no_precomputed_group_id(tmp_path):
    path = tmp_path / 'source.parquet'
    frame = pd.DataFrame(dict(assay_id=['a7', 'a2'], target_chembl_id=['p1', 'p9'],
        compound_id=['m2', 'm1'], pchembl_value=[5., 7.]))
    frame.to_parquet(path, index=False)
    actual = module.source_frame(path)
    assert actual.group_id.tolist() == ['p1__a7', 'p9__a2']
    pd.testing.assert_frame_equal(actual.drop(columns='group_id'), frame)


def test_partial_geometry_and_unapproved_device_are_never_retried(prepared, monkeypatch):
    path = prepared['path']; (path.parent / 'A').mkdir()
    with pytest.raises(RuntimeError, match='Partial geometry'): module.worker(path, 'A', 0, cpu=True)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1'); monkeypatch.setenv('ACCELERATE_USE_CPU', 'false')
    with pytest.raises(RuntimeError, match='authorized Atum'): module.worker(path, 'B', 0)


def test_corrupted_export_cannot_be_reported(prepared):
    path = prepared['path']; module.worker(path, 'A', 0, cpu=True)
    (path.parent / 'A/val/molecule_encoder_pooled.npy').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='artifact changed'): module.verify_worker(path, 'A', cpu=True)


@pytest.fixture
def scheduler(prepared, monkeypatch):
    calls = []
    monkeypatch.setattr(module.evaluation.control_queue, 'gpu_processes', lambda: [])
    monkeypatch.setattr(module.evaluation.control_queue, 'storage_gate', lambda: dict(ready=True))
    def verify(path, regime, **_):
        assert (path.parent / regime / 'complete.json').exists()
        return {}, {s: dict(analysis=dict(simulated=True)) for s in module.PARTITIONS}
    monkeypatch.setattr(module, 'verify_worker', verify)
    class Child:
        def __init__(self, args, **kwargs):
            self.args = args; self.kwargs = kwargs; self.pid = 900 + len(calls); self.code = None; self.terminated = False
            calls.append(self)
            import fcntl
            with (prepared['root'] / 'controlled/driver.lock').open('a') as lock:
                with pytest.raises(BlockingIOError): fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def wait(self):
            self.code = 0
            regime = self.args[self.args.index('--regime') + 1]
            write_json(prepared['path'].parent / regime / 'complete.json', dict(simulated=True))
            return 0
        def poll(self): return self.code
        def terminate(self): self.terminated = True
    monkeypatch.setattr(module.subprocess, 'Popen', Child)
    prepared.update(calls=calls, Child=Child)
    return prepared


def test_two_native_workers_share_lock_and_use_only_assigned_devices(scheduler):
    result = module.run(scheduler['path'])
    assert len(scheduler['calls']) == 2 and set(result['results']) == {'A', 'B'}
    assert not result['scientific_study_complete'] and result['new_fits'] == 0
    for i, call in enumerate(scheduler['calls']):
        assert call.kwargs['env']['CUDA_VISIBLE_DEVICES'] == module.evaluation.GPU_UUIDS[i]
        assert call.kwargs['env']['WANDB_MODE'] == 'disabled'
        assert call.args[-1] == str(i)


@pytest.mark.parametrize('gate', ['gpu', 'storage', 'lock'])
def test_unavailable_resources_start_no_workers(scheduler, monkeypatch, gate):
    import fcntl
    handle = (scheduler['root'] / 'controlled/driver.lock').open('a')
    if gate == 'gpu': monkeypatch.setattr(module.evaluation.control_queue, 'gpu_processes', lambda: [dict(pid=99)])
    elif gate == 'storage': monkeypatch.setattr(module.evaluation.control_queue, 'storage_gate', lambda: dict(ready=False))
    else: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try: result = module.run(scheduler['path'])
    finally: handle.close()
    assert result['phase'].startswith('waiting_for_') and not scheduler['calls']


def test_second_launch_failure_cleans_only_its_first_child(scheduler, monkeypatch):
    def launch(*args, **kwargs):
        if scheduler['calls']: raise OSError('launch failure')
        return scheduler['Child'](*args, **kwargs)
    monkeypatch.setattr(module.subprocess, 'Popen', launch)
    with pytest.raises(OSError, match='launch failure'): module.run(scheduler['path'])
    assert len(scheduler['calls']) == 1 and scheduler['calls'][0].terminated
    assert not (scheduler['path'].parent / 'complete.json').exists()


def test_partial_second_regime_starts_neither_worker(scheduler):
    (scheduler['path'].parent / 'B').mkdir()
    with pytest.raises(RuntimeError, match='Partial geometry'): module.run(scheduler['path'])
    assert not scheduler['calls']
