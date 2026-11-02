import json

import pytest

from experiments.paired_generalization import adaptation_query_qualification as qualify


def test_synthetic_runtime_and_independent_reconstruction_detect_changed_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setenv('ACCELERATE_USE_CPU','true')
    monkeypatch.setenv('WANDB_MODE','disabled')
    output = tmp_path/'qualification'
    result = qualify.qualify_cpu(output)
    assert qualify.verify_cpu(output)==result
    assert result['query_exports_verified']==24 and result['assay_pathway_checks']>0
    assert result['scientific_results']==0 and not result['production_gpu_qualified']
    record = json.loads((output/'qualification.json').read_text())
    assert len(record['synthetic_support_fits'])==8
    assert {x['scope'] for x in record['synthetic_support_fits']}=={'heads','heads_cross_attention'}
    with pytest.raises(RuntimeError,match='Existing qualification'):
        qualify.qualify_cpu(output)
    payload = next(output.rglob('*.npz'))
    with payload.open('ab') as handle:handle.write(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):
        qualify.verify_cpu(output)


def test_qualification_refuses_visible_gpus_before_creating_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setenv('ACCELERATE_USE_CPU','true')
    monkeypatch.setenv('WANDB_MODE','disabled')
    with pytest.raises(RuntimeError,match='CPU-only'):
        qualify.qualify_cpu(tmp_path/'refused')
    assert not (tmp_path/'refused').exists()
