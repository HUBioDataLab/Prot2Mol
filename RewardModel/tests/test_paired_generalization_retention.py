import json
from pathlib import Path

import pytest
import torch

from experiments.paired_generalization.retain_controlled import average_weights, retain, selection_rules
from experiments.paired_generalization.run import write_json
from reward_model.training.prediction_exports import export_evaluation_predictions
from test_paired_generalization_prediction_snapshots import snapshot_arguments


def records_fixture():
    return [dict(epoch=i, step=i*10, eval_loss=loss, eval_macro_spearman=score)
            for i, (loss, score) in enumerate(zip([1., .8, .6, .9, 1., 1.1], [.5, .4, .3, .2, .6, .4]), 1)]


def test_selection_separates_best_horizon_from_patience_prefix_and_breaks_ties_early():
    rules = selection_rules(records_fixture())
    assert rules['best_macro']['epoch'] == 5
    assert rules['patience3_macro']['epoch'] == 1
    assert rules['patience3_macro']['stop_epoch'] == 4
    assert rules['best_loss']['epoch'] == rules['patience3_loss']['epoch'] == 3
    assert rules['patience3_loss']['stop_epoch'] == 6
    tied = [dict(epoch=i, step=i, eval_loss=1., eval_macro_spearman=.1) for i in range(1, 5)]
    assert selection_rules(tied)['best_macro']['step'] == 1
    assert selection_rules(tied)['best_loss']['step'] == 1


def make_run(run):
    config_hash = 'fixture-config'
    write_json(run/'controlled_run.json', dict(format_version=1, output_dir=str(run),
        config_sha256=config_hash, allow_checkpoint_retention=True))
    write_json(run/'launch_provenance.json', dict(config_sha256=config_hash, world_size=2,
        config=dict(training=dict(num_train_epochs=6, export_evaluation_predictions=True,
                                  fixed_train_eval_assays=2, metric_for_best_model='eval_macro_spearman',
                                  greater_is_better=True))))
    history = records_fixture()
    final = dict(epoch=6, global_step=60, log_history=history, best_model_checkpoint=str(run/'checkpoint-50'))
    write_json(run/'trainer_state.json', final)
    write_json(run/'config.json', {'model': 'fixture'})
    for rank in range(2):
        write_json(run/f'rank_{rank}_complete.json', dict(rank=rank, world_size=2, config_sha256=config_hash))
    for r in history:
        checkpoint = run/f"checkpoint-{r['step']}"
        write_json(checkpoint/'trainer_state.json', dict(epoch=r['epoch'], global_step=r['step']))
        write_json(checkpoint/'config.json', {'model': 'fixture'})
        torch.save({'weight': torch.tensor([float(r['epoch']), 2.]), 'integer': torch.tensor(3)}, checkpoint/'pytorch_model.bin')
        for name in ('optimizer.pt', 'scheduler.pt', 'rng_state_0.pth', 'training_args.bin'):
            (checkpoint/name).write_bytes(b'payload')
        for prefix in ('eval', 'train_eval'):
            args = snapshot_arguments(run)
            args.update(step=r['step'], epoch=r['epoch'], metric_prefix=prefix,
                        metadata={'metrics': {k:v for k,v in r.items() if k.startswith('eval_')}})
            export_evaluation_predictions(**args)
    root_model = torch.load(run/'checkpoint-50/pytorch_model.bin', weights_only=True)
    torch.save(root_model, run/'pytorch_model.bin')
    selected = dict(eval_macro_spearman=.6, eval_loss=1.)
    write_json(run/'experiment_summary.json', dict(config_sha256=config_hash, eval_metrics=selected))
    args = snapshot_arguments(run)
    args.update(phase='selected', step=60, epoch=6, metadata={'metrics': selected})
    export_evaluation_predictions(**args)


def test_retention_preserves_selections_averages_and_trajectories_before_cleanup(tmp_path):
    make_run(tmp_path)
    staged = retain(tmp_path)
    assert not staged['pruned']
    assert len(list(tmp_path.glob('checkpoint-*/pytorch_model.bin'))) == 6
    average = tmp_path/'retained/average_from_loss_minimum/pytorch_model.bin'
    weights = torch.load(average, weights_only=True)
    torch.testing.assert_close(weights['weight'], torch.tensor([4.5, 2.]), rtol=0, atol=0)
    assert weights['integer'].item() == 3
    pruned = retain(tmp_path, prune=True)
    assert pruned['pruned']
    assert {p.parent.name for p in tmp_path.glob('checkpoint-*/pytorch_model.bin')} == {'checkpoint-10', 'checkpoint-30', 'checkpoint-50'}
    assert not list(tmp_path.glob('checkpoint-*/optimizer.pt'))
    assert len(list(tmp_path.glob('checkpoint-*/trainer_state.json'))) == 6
    assert len(list(tmp_path.glob('prediction_exports/trajectory/*.npz'))) == 12
    assert retain(tmp_path, prune=True) == pruned


@pytest.mark.parametrize('damage', ['rank', 'trajectory', 'root_model', 'marker', 'source_changed'])
def test_invalid_evidence_prevents_cleanup(tmp_path, damage):
    make_run(tmp_path)
    if damage == 'rank':
        (tmp_path/'rank_1_complete.json').unlink()
    elif damage == 'trajectory':
        (tmp_path/'prediction_exports/trajectory/eval_step00000020.npz').write_bytes(b'broken')
    elif damage == 'root_model':
        torch.save({'weight': torch.tensor(9)}, tmp_path/'pytorch_model.bin')
    elif damage == 'marker':
        (tmp_path/'controlled_run.json').unlink()
    else:
        retain(tmp_path)
        torch.save({'weight': torch.tensor(9)}, tmp_path/'checkpoint-20/pytorch_model.bin')
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        retain(tmp_path, prune=True)
    assert len(list(tmp_path.glob('checkpoint-*/pytorch_model.bin'))) == 6
    assert len(list(tmp_path.glob('checkpoint-*/optimizer.pt'))) == 6


def test_average_refuses_different_integer_buffers(tmp_path):
    sources = [tmp_path/'a.bin', tmp_path/'b.bin']
    for i, path in enumerate(sources):
        torch.save({'float': torch.tensor([1.]), 'buffer': torch.tensor(i)}, path)
    with pytest.raises(ValueError, match='Nonfloating'):
        average_weights(sources, tmp_path/'average.bin')
    assert not (tmp_path/'average.bin').exists()
