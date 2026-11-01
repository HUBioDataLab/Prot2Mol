import json

import pytest
import torch

from experiments.paired_generalization.protect_trajectory import (
    hold_retention, capture_checkpoint, release_retention, verify_complete)
from experiments.paired_generalization.retain_controlled import retain
from experiments.paired_generalization.run import sha256, write_json
from test_paired_generalization_retention import make_run


def setup(tmp_path):
    run,destination = tmp_path/'run',tmp_path/'protected'
    make_run(run)
    original = (run/'controlled_run.json').read_bytes()
    hold_retention(run,destination,'fixture-config',6)
    return run,destination,original


def capture_all(run,destination):
    for checkpoint in sorted(run.glob('checkpoint-*')):
        capture_checkpoint(run,destination,checkpoint,6)


def test_guard_stops_old_cleanup_and_all_epochs_survive_after_release_and_prune(tmp_path):
    run,destination,original = setup(tmp_path)
    with pytest.raises(ValueError,match='not managed'):
        retain(run,prune=True)
    assert len(list(run.glob('checkpoint-*/pytorch_model.bin'))) == 6
    capture_all(run,destination)
    for epoch in range(1,7):
        assert (run/f'checkpoint-{epoch*10}/pytorch_model.bin').stat().st_ino == (destination/f'epoch{epoch:02d}/pytorch_model.bin').stat().st_ino
    result = release_retention(destination)
    assert len(result['epochs']) == 6 and result['retention_hold_released']
    assert (run/'controlled_run.json').read_bytes() == original
    retain(run,prune=True)
    assert len(list(run.glob('checkpoint-*/pytorch_model.bin'))) == 3
    assert verify_complete(destination) == result
    for epoch in range(1,7):
        tensor = torch.load(destination/f'epoch{epoch:02d}/pytorch_model.bin',weights_only=True)['weight']
        assert tensor[0].item() == epoch


def test_guard_and_capture_resume_without_replacing_the_original_marker(tmp_path):
    run,destination,original = setup(tmp_path)
    before = sha256(destination/'guard.json')
    hold_retention(run,destination,'fixture-config',6)
    assert sha256(destination/'guard.json') == before
    first = capture_checkpoint(run,destination,run/'checkpoint-10',6)
    assert capture_checkpoint(run,destination,run/'checkpoint-10',6) == first
    with pytest.raises(FileNotFoundError):
        release_retention(destination)
    assert not json.loads((run/'controlled_run.json').read_text())['allow_checkpoint_retention']
    capture_all(run,destination)
    release_retention(destination)
    hold_retention(run,destination,'fixture-config',6)
    assert (run/'controlled_run.json').read_bytes() == original


def test_an_incomplete_checkpoint_is_not_captured(tmp_path):
    run,destination,_ = setup(tmp_path)
    checkpoint = run/'checkpoint-10'
    state = (checkpoint/'trainer_state.json').read_bytes()
    (checkpoint/'trainer_state.json').write_text('{')
    assert capture_checkpoint(run,destination,checkpoint,6) is None
    assert not (destination/'epoch01').exists()
    (checkpoint/'trainer_state.json').write_bytes(state)
    capture_checkpoint(run,destination,checkpoint,6)


@pytest.mark.parametrize('damage',['weights','marker','epoch'])
def test_corruption_or_unrelated_marker_changes_prevent_release(tmp_path,damage):
    run,destination,_ = setup(tmp_path)
    capture_all(run,destination)
    if damage == 'weights':
        (destination/'epoch02/pytorch_model.bin').write_bytes(b'corrupt')
    elif damage == 'marker':
        marker = json.loads((run/'controlled_run.json').read_text())
        marker['unrelated_change'] = True
        write_json(run/'controlled_run.json',marker)
    else:
        record = json.loads((destination/'epoch02/complete.json').read_text())
        record['epoch'] = 1
        write_json(destination/'epoch02/complete.json',record)
    with pytest.raises(ValueError):
        release_retention(destination)
    assert not (destination/'complete.json').exists()
    assert not json.loads((run/'controlled_run.json').read_text())['allow_checkpoint_retention']
