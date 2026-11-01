import pytest
import torch
from datasets import Dataset

from reward_model.model.losses import ligunity_bidirectional_contrastive_loss
from reward_model.training.known_positives import KnownPositiveIndex
from reward_model.training.data import RewardAssayListCollator
from reward_model.training.evaluation import compute_contrastive_evaluation_loss
from test_paired_generalization_adaptation import model, batch, bundles
from reward_model.model import save_reward_model, load_reward_model


def index_fixture():
    return KnownPositiveIndex.from_training_dataset(Dataset.from_list([
        dict(target_chembl_id='p0', compound_id='m3', pchembl_value=8.),
        dict(target_chembl_id='p0', compound_id='m3', pchembl_value=7.),
        dict(target_chembl_id='p1', compound_id='m8', pchembl_value=6.),
        dict(target_chembl_id='p1', compound_id='m2', pchembl_value=4.),
    ]), active_threshold=6.)


def test_index_uses_only_strict_training_positives_and_is_stable_under_reordering():
    index = index_fixture()
    assert index.provenance['positive_pairs'] == 1
    assert index.provenance['heldout_labels_used'] is False
    assert index.positives_by_target == {'p0': frozenset({'m3'})}
    equivalent = KnownPositiveIndex({'p0': ['m3', 'm3']}, active_threshold=6., source_rows=4)
    assert equivalent.provenance == index.provenance
    torch.testing.assert_close(index.mask(['p1', 'p0', 'new'], ['m8', 'm3', 'new']),
                               torch.tensor([[False, False, False], [False, True, False], [False, False, False]]))
    # Validation identities may remap integer IDs; they cannot add positives.
    assert index.encoded_for(['p1', 'p0'], ['m8', 'm3']) == {0: frozenset({0})}
    assert index.encoded_for(['p0'], ['unseen_molecule']) == {0: frozenset()}
    assert index.provenance == equivalent.provenance


def test_index_threshold_uses_the_same_float32_labels_as_training():
    examples = Dataset.from_list([dict(target_chembl_id='p0', compound_id='m0', pchembl_value=6.+1e-9)])
    index = KnownPositiveIndex.from_training_dataset(examples, active_threshold=6.)
    assert index.provenance['positive_pairs'] == 0


def test_known_positive_mask_removes_negative_gradients_without_masking_own_targets():
    scores = torch.zeros((3, 6), dtype=torch.float64, requires_grad=True)
    owners = torch.tensor([0, 0, 1, 1, 2, 2])
    target_ids, molecule_ids = torch.tensor([10, 20, 10]), torch.arange(6)
    values = torch.full((6,), 8.)
    mask = torch.zeros((3, 6), dtype=torch.bool)
    mask[0, 2] = mask[2, 2] = True
    mask[0, 0] = True  # the supervised positive must survive the mask
    reference = ligunity_bidirectional_contrastive_loss(scores, values, owners, target_ids, molecule_ids, strict_active_only=True)
    augmented = ligunity_bidirectional_contrastive_loss(scores, values, owners, target_ids, molecule_ids,
        strict_active_only=True, mask_policy='training_known', known_positive_mask=mask)
    for before, after in zip(reference[1:], augmented[1:]):
        old_gradient, = torch.autograd.grad(before, scores, retain_graph=True)
        new_gradient, = torch.autograd.grad(after, scores, retain_graph=True)
        assert old_gradient[0, 2] > 0 and new_gradient[0, 2] == 0
        assert new_gradient[0, 0] < 0
        assert after < before
    unmasked = ligunity_bidirectional_contrastive_loss(scores, values, owners, target_ids, molecule_ids,
        strict_active_only=True, mask_policy='none')
    gradient, = torch.autograd.grad(unmasked[0], scores)
    assert gradient[0, 4] > 0  # same-target cross-assay entry is now a negative
    assert unmasked[0] > reference[0]


def test_empty_additional_index_exactly_reproduces_reference_and_missing_index_fails():
    torch.manual_seed(5)
    scores = torch.randn(2, 4, requires_grad=True)
    args = (scores, torch.tensor([8., 5., 7., 4.]), torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1]), torch.arange(4))
    reference = ligunity_bidirectional_contrastive_loss(*args, strict_active_only=True)
    actual = ligunity_bidirectional_contrastive_loss(*args, strict_active_only=True,
        mask_policy='training_known', known_positive_mask=torch.zeros_like(scores, dtype=torch.bool))
    for a, b in zip(reference, actual):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match='boolean'):
        ligunity_bidirectional_contrastive_loss(*args, mask_policy='training_known')


def test_collator_aligns_training_index_with_flattened_rows():
    index = index_fixture()
    rows = [dict(target_chembl_id='p0' if i < 3 else 'p1', compound_id=f'm{i}',
                 protein_input_ids=[3, 4], protein_attention_mask=[1, 1],
                 molecule_input_ids=[5, 6], molecule_attention_mask=[1, 1],
                 pchembl_value=8., binary_label=1) for i in range(6)]
    features = [dict(rows=rows[:3], contrastive_group_sizes=[3], contrastive_assay_ids=['p0__a'], ranking_group_flags=[True]),
                dict(rows=rows[3:], contrastive_group_sizes=[3], contrastive_assay_ids=['p1__b'], ranking_group_flags=[True])]
    collated = RewardAssayListCollator(known_positive_index=index)(features)
    mask = collated['contrastive_known_positive_mask']
    assert mask.shape == (6, 6)
    assert mask[:3, 3].all() and mask.sum() == 3


@pytest.mark.parametrize('policy', ['none', 'reference', 'training_known'])
def test_model_and_full_evaluation_use_the_same_mask_policy(policy):
    fitted = model(contrastive_mask_policy=policy).eval()
    data = batch()
    index = index_fixture()
    targets, molecules = [f'p{i//3}' for i in range(9)], [f'm{i}' for i in range(9)]
    if policy == 'training_known':
        data['contrastive_known_positive_mask'] = index.mask(targets, molecules)
    output = fitted(**data)
    actual = compute_contrastive_evaluation_loss(
        normalized_protein_embeddings=output.normalized_protein_embedding,
        normalized_molecule_embeddings=output.normalized_molecule_embedding,
        pchembl_values=data['pchembl_values'], contrastive_group_ids=data['contrastive_group_ids'],
        target_identity_ids=data['contrastive_target_ids'], molecule_identity_ids=data['contrastive_molecule_ids'],
        temperature=1/float(output.score_scale.detach()), active_threshold=6., assay_batch_size=3,
        strict_active_only=True, mask_policy=policy,
        known_positive_pairs=index.encoded_for(targets, molecules) if policy == 'training_known' else None)
    torch.testing.assert_close(actual, output.contrastive_loss, rtol=1e-6, atol=1e-6)
    output.loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in fitted.contrastive_molecule_projection.parameters())


def test_saved_scorer_needs_no_measurement_index_or_labels_for_inference(tmp_path):
    fitted = model(contrastive_mask_policy='training_known').eval()
    scoring_input = {key: value for key, value in batch().items() if key.endswith(('input_ids', 'attention_mask'))}
    expected = fitted(**scoring_input)
    save_reward_model(fitted, str(tmp_path))
    protein, molecule = bundles()
    restored = load_reward_model(str(tmp_path), protein_bundle=protein, molecule_bundle=molecule).eval()
    actual = restored(**scoring_input)
    assert restored.config.contrastive_mask_policy == 'training_known'
    assert actual.loss is None
    torch.testing.assert_close(actual.ranking_score, expected.ranking_score, rtol=0, atol=0)


def test_trainer_evaluation_receives_the_training_collator_index(tmp_path):
    from transformers import TrainingArguments
    from reward_model.training.data import RewardEvaluationDataset
    from reward_model.training.trainer import RewardModelTrainer
    fitted = model(contrastive_mask_policy='training_known').eval()
    inputs = batch()
    rows = []
    for i in range(9):
        rows.append(dict(example_id=i, group_id=f'p{i//3}__a{i//3}', target_chembl_id=f'p{i//3}',
            assay_id=f'a{i//3}', compound_id=f'm{i}', activity_type='IC50',
            pchembl_value=float(inputs['pchembl_values'][i]), binary_label=float(inputs['activity_labels'][i]),
            **{key: inputs[key][i].tolist() for key in inputs if key.endswith(('input_ids', 'attention_mask'))}))
    index = index_fixture()
    eval_data = RewardEvaluationDataset(Dataset.from_list(rows), protein_shuffle_sensitivity=False)
    trainer = RewardModelTrainer(model=fitted,
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[],
                               per_device_train_batch_size=3, per_device_eval_batch_size=3,
                               remove_unused_columns=False),
        eval_dataset=eval_data, data_collator=RewardAssayListCollator(known_positive_index=index))
    metrics = trainer.evaluate()
    inputs['contrastive_known_positive_mask'] = index.mask([r['target_chembl_id'] for r in rows],
                                                          [r['compound_id'] for r in rows])
    expected = fitted(**inputs)
    assert metrics['eval_contrastive_loss'] == pytest.approx(float(expected.contrastive_loss.detach()), abs=2e-6)
    trainer.data_collator.known_positive_index = None
    with pytest.raises(ValueError, match='positive index'):
        trainer.evaluate()
