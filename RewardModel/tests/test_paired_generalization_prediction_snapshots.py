import json

import numpy as np
import pytest
import torch
from datasets import Dataset
from transformers import TrainingArguments

from reward_model.training.data import RewardAssayListCollator, RewardEvaluationDataset
from reward_model.training.prediction_exports import export_evaluation_predictions, file_sha256
from reward_model.training.trainer import RewardModelTrainer
from test_paired_generalization_adaptation import model, batch


def dataset_fixture():
    inputs = batch()
    return Dataset.from_list([
        dict(example_id=i, group_id=f'p{i//3}__a{i//3}', target_chembl_id=f'p{i//3}',
             assay_id=f'a{i//3}', compound_id=f'm{i}', activity_type='IC50',
             pchembl_value=float(inputs['pchembl_values'][i]), binary_label=float(inputs['activity_labels'][i]),
             **{key: inputs[key][i].tolist() for key in inputs if key.endswith(('input_ids', 'attention_mask'))})
        for i in range(9)])


def test_actual_trainer_exports_all_rows_and_preserves_trajectory_after_best_reload(tmp_path):
    fitted = model().eval()
    dataset = dataset_fixture()
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[],
                             per_device_train_batch_size=3, per_device_eval_batch_size=4,
                             remove_unused_columns=False)
    args.reward_export_evaluation_predictions = True
    trainer = RewardModelTrainer(model=fitted, args=args,
        eval_dataset=RewardEvaluationDataset(dataset, protein_shuffle_sensitivity=False),
        data_collator=RewardAssayListCollator())
    trainer.state.global_step, trainer.state.epoch = 50, 1.
    metrics = trainer.evaluate()
    path = tmp_path / 'prediction_exports/trajectory/eval_step00000050.json'
    manifest = json.loads(path.read_text())
    assert manifest['metrics'] == metrics
    assert manifest['rows'] == len(dataset)
    arrays_path = path.with_suffix('.npz')
    assert manifest['predictions_sha256'] == file_sha256(arrays_path)
    with np.load(arrays_path, allow_pickle=False) as saved:
        assert np.array_equal(saved['row_index'], np.arange(9))
        assert saved['compound_id'].tolist() == list(dataset['compound_id'])
        assert set(saved) >= {'ranking_score', 'activity_logits', 'cosine_similarity'}
        # Compare each batch with the same forward calls, including the tail batch.
        expected = []
        for start in range(0, 9, 4):
            rows = [dataset[i] for i in range(start, min(start+4, 9))]
            tokens = trainer.data_collator.collate_example_tokens(rows)
            with torch.no_grad():
                expected.extend(fitted(**tokens).ranking_score.float().tolist())
        np.testing.assert_allclose(saved['ranking_score'], expected, rtol=1e-6, atol=1e-6)
    digest = file_sha256(arrays_path)
    trainer.prediction_export_phase = 'selected'
    with torch.no_grad():
        next(fitted.ranking_head.parameters()).add_(.01)
    trainer.evaluate()
    assert file_sha256(arrays_path) == digest
    assert (tmp_path / 'prediction_exports/selected/eval_step00000050.json').exists()


def snapshot_arguments(tmp_path):
    dataset = dataset_fixture()
    return dict(output_dir=tmp_path, phase='trajectory', metric_prefix='eval', step=5,
                epoch=1., dataset=dataset, group_names=['p0__a0','p1__a1','p2__a2'],
                tensors=dict(row_index=torch.arange(9), evaluation_group_index=torch.arange(9)//3,
                             pchembl_value=torch.tensor(dataset['pchembl_value']),
                             binary_label=torch.tensor(dataset['binary_label']),
                             ranking_score=torch.arange(9).float()),
                metadata={'metrics': {'eval_macro_spearman': float('nan')}})


def test_export_is_idempotent_but_detects_replacement_and_corruption(tmp_path):
    args = snapshot_arguments(tmp_path)
    path = export_evaluation_predictions(**args)
    assert json.loads(path.read_text())['metrics']['eval_macro_spearman'] is None
    assert export_evaluation_predictions(**args) == path
    args['tensors']['ranking_score'][0] += 1
    with pytest.raises(RuntimeError, match='replace'):
        export_evaluation_predictions(**args)
    path.with_suffix('.npz').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='corrupt'):
        export_evaluation_predictions(**args)


@pytest.mark.parametrize('error', ['missing_row', 'group_alignment', 'label_alignment', 'nonfinite'])
def test_export_rejects_incomplete_or_misaligned_predictions(tmp_path, error):
    args = snapshot_arguments(tmp_path)
    if error == 'missing_row':
        args['tensors']['row_index'][-1] = 0
    elif error == 'group_alignment':
        args['tensors']['evaluation_group_index'][0] = 1
    elif error == 'label_alignment':
        args['tensors']['pchembl_value'][0] += 1
    else:
        args['tensors']['ranking_score'][0] = float('nan')
    with pytest.raises(ValueError):
        export_evaluation_predictions(**args)
    assert not list(tmp_path.rglob('*.json'))
