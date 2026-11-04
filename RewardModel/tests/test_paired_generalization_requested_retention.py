"""Real tiny checkpoint tensors and native prediction payload retention."""
from pathlib import Path
import pytest
import torch
from experiments.paired_generalization import requested_retention as module
from experiments.paired_generalization import realization_predictions as allocation
from experiments.paired_generalization.run import write_json
from reward_model.training.prediction_exports import export_evaluation_predictions
from test_paired_generalization_prediction_snapshots import snapshot_arguments
import test_paired_generalization_retention as fixtures


@pytest.fixture
def run(tmp_path,monkeypatch):
    rows=[dict(epoch=e,step=e*10,eval_loss=20-e,eval_macro_spearman=.6-.01*abs(e-5)) for e in range(1,16)]
    monkeypatch.setattr(fixtures,'records_fixture',lambda:rows);fixtures.make_run(tmp_path)
    marker=module.json.loads((tmp_path/'controlled_run.json').read_text())
    marker.update(seed=42,learning_rate_multiplier=1.,regime='B');write_json(tmp_path/'controlled_run.json',marker)
    launch=module.json.loads((tmp_path/'launch_provenance.json').read_text())
    launch['world_size']=4;launch['config']['training']['num_train_epochs']=15
    write_json(tmp_path/'launch_provenance.json',launch)
    for r in range(4):write_json(tmp_path/f'rank_{r}_complete.json',dict(rank=r,world_size=4,config_sha256='fixture-config'))
    write_json(tmp_path/'trainer_state.json',dict(epoch=15,global_step=150,log_history=rows,best_model_checkpoint=str(tmp_path/'checkpoint-50')))
    summary=module.json.loads((tmp_path/'experiment_summary.json').read_text())
    summary.update(test_evaluated=False,test_metrics={});write_json(tmp_path/'experiment_summary.json',summary)
    args=snapshot_arguments(tmp_path);args.update(phase='selected',step=150,epoch=15,metadata={'metrics':summary['eval_metrics']})
    export_evaluation_predictions(**args)
    monkeypatch.setattr(allocation,'verify_validation',lambda *_:pytest.fail('No allocation inference/selection'))
    return tmp_path


def test_B_retains_best_macro_and_final_without_allocations_or_averaging(run):
    result=module.retain(run,prune=True)
    assert result['additional_realizations'] is None and result['averaging'] is None
    assert not result['inference_performed'] and result['pruned']
    assert {p.parent.name for p in run.glob('checkpoint-*/pytorch_model.bin')}=={'checkpoint-50','checkpoint-150'}
    assert len(result['prediction_manifests'])==31 and len(result['source_weights'])==15
    assert not (run/'retained/average_from_loss_minimum').exists()
    assert len(list(run.glob('checkpoint-*/trainer_state.json')))==15
    assert not list(run.glob('checkpoint-*/optimizer.pt'))
    assert module.retain(run,prune=True)==result


@pytest.mark.parametrize('damage',['rank','root','trajectory','test','seed','horizon','symlink'])
def test_invalid_native_evidence_prevents_any_cleanup(run,damage):
    if damage=='rank':(run/'rank_3_complete.json').unlink()
    elif damage=='root':torch.save({'bad':torch.tensor(0)},run/'pytorch_model.bin')
    elif damage=='trajectory':(run/'prediction_exports/trajectory/eval_step00000020.npz').write_bytes(b'bad')
    elif damage=='symlink':
        path=run/'checkpoint-20/optimizer.pt';path.unlink();path.symlink_to(run/'checkpoint-10/optimizer.pt')
    else:
        path=run/({'test':'experiment_summary.json','seed':'controlled_run.json','horizon':'launch_provenance.json'}[damage])
        item=module.json.loads(path.read_text())
        if damage=='test':item['test_evaluated']=True
        elif damage=='seed':item['seed']=43
        else:item['config']['training']['num_train_epochs']=14
        write_json(path,item)
    with pytest.raises((ValueError,RuntimeError,FileNotFoundError)):module.retain(run,prune=True)
    assert len(list(run.glob('checkpoint-*/pytorch_model.bin')))==15
