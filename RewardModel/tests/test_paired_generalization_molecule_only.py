import copy

import pytest
import torch

from experiments.paired_generalization.qualify_controls import make_batch, make_model
from reward_model.model import RewardModelConfig


OVERRIDES = dict(protein_input_mode='constant',protein_constant_token_ids=[0,6,2])


def changed_proteins(batch):
    altered = dict(batch)
    count = len(batch['protein_input_ids'])
    altered['protein_input_ids'] = torch.arange(count*7).reshape(count,7)%30+2
    altered['protein_attention_mask'] = torch.ones((count,7),dtype=torch.long)
    altered['protein_attention_mask'][:,4:] = 0
    return altered


def test_molecule_only_keeps_initial_weights_and_removes_protein_content_and_length():
    full = make_model(None,{}).eval()
    mono = make_model(None,OVERRIDES).eval()
    assert full.num_trainable_parameters() == mono.num_trainable_parameters()
    assert full.state_dict().keys() == mono.state_dict().keys()
    for key,value in full.state_dict().items():
        torch.testing.assert_close(value,mono.state_dict()[key],rtol=0,atol=0)
    batch = make_batch(mono,torch.device('cpu'),False)
    with torch.no_grad():
        a,b = mono(**batch),mono(**changed_proteins(batch))
        original_full,changed_full = full(**batch),full(**changed_proteins(batch))
    for name in ['ranking_score','activity_logits','cosine_similarity','loss']:
        torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=0,atol=0)
    assert not torch.equal(original_full.ranking_score,changed_full.ranking_score)
    assert torch.unique(a.ranking_score).numel() > 1


def test_training_dropout_and_gradients_do_not_depend_on_protein_deduplication_pattern():
    model = make_model(None,OVERRIDES).train()
    batch = make_batch(model,torch.device('cpu'),False)
    snapshots = []
    for inputs in [batch,changed_proteins(batch)]:
        model.zero_grad(set_to_none=True)
        torch.manual_seed(123)
        result = model(**inputs)
        result.loss.backward()
        snapshots.append({name:p.grad.detach().clone() for name,p in model.named_parameters() if p.grad is not None})
    assert snapshots[0].keys() == snapshots[1].keys()
    assert any(name.startswith('protein_encoder.') for name in snapshots[0])
    assert any(name.startswith('molecule_encoder.') for name in snapshots[0])
    for name,gradient in snapshots[0].items():
        torch.testing.assert_close(gradient,snapshots[1][name],rtol=0,atol=0)


def test_serialized_configuration_restores_constant_input_behavior():
    original = make_model(None,OVERRIDES).eval()
    config = RewardModelConfig.from_dict(original.config.to_dict())
    assert config.protein_constant_token_ids == (0,6,2)
    restored = make_model(None,dict(protein_input_mode=config.protein_input_mode,
                                   protein_constant_token_ids=config.protein_constant_token_ids)).eval()
    restored.load_state_dict(copy.deepcopy(original.state_dict()),strict=True)
    batch = make_batch(original,torch.device('cpu'),False)
    with torch.no_grad():
        torch.testing.assert_close(original(**batch).ranking_score,
                                   restored(**changed_proteins(batch)).ranking_score,rtol=0,atol=0)


@pytest.mark.parametrize('changes',[
    dict(protein_input_mode='unknown'),dict(protein_input_mode='constant'),
    dict(protein_input_mode='constant',protein_constant_token_ids=[]),
    dict(protein_input_mode='constant',protein_constant_token_ids=[-1]),
    dict(protein_input_mode='constant',protein_constant_token_ids=[True]),
    dict(protein_constant_token_ids=[0,6,2]),
])
def test_rejects_ambiguous_constant_input_configurations(changes):
    with pytest.raises(ValueError):
        RewardModelConfig(**changes)
