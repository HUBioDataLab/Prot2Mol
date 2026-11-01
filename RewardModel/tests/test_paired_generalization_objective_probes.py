from types import SimpleNamespace

import pytest
import torch

from experiments.paired_generalization import objective_probes as probes
from experiments.paired_generalization.qualify_controls import make_model, make_batch


RATES = dict(protein_encoder=.1, molecule_encoder=.1, interaction=.1)


class Quadratic(torch.nn.Module):
    def __init__(self, *, weights=(1., 1., 1.)):
        super().__init__()
        self.protein_encoder = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        self.ranking_head = torch.nn.Linear(1, 1, bias=True, dtype=torch.float64)
        self.classification_head = torch.nn.Linear(1, 1, bias=True, dtype=torch.float64)
        self.contrastive_protein_projection = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            for p in self.parameters(): p.fill_(1. if p.ndim == 2 else 0.)
        self.config = SimpleNamespace(pair_scoring_mode='mlp_contrastive', ranking_loss_weight=weights[0],
            classification_loss_weight=weights[1], contrastive_loss_weight=weights[2])

    def forward(self, x, targets, return_dict=True):
        value = self.protein_encoder(x)
        predictions = (self.ranking_head(value), self.classification_head(value), self.contrastive_protein_projection(value))
        losses = [.5*((prediction-target)**2).mean() for prediction,target in zip(predictions,targets)]
        return SimpleNamespace(ranking_loss=losses[0], classification_loss=losses[1], contrastive_loss=losses[2])


def batch(targets):
    return dict(x=torch.ones((1,1),dtype=torch.float64), targets=targets)


def test_exact_quadratic_transfer_uses_validation_and_holds_private_heads_fixed():
    model = Quadratic()
    result = probes.measure(model, batch([0.,2.,0.]), batch([2.,0.,2.]), RATES)
    block = result['gradient_statistics']['protein_encoder.other']
    assert block['cosines']['R__B'] == pytest.approx(-1.)
    assert block['cosines']['R__C'] == pytest.approx(1.)
    # R decreases w from1 to.9; this helps the training target0 but hurts
    # validation target2. B increases w to1.1, helping that validation target.
    r = result['lookahead']['R']['scopes']['encoders']
    b = result['lookahead']['B']['scopes']['encoders']
    assert r['transfer']['R']['before'] == .5
    assert r['transfer']['R']['after'] == pytest.approx(.605)
    assert r['transfer']['R']['relative_transfer'] == pytest.approx(-.21)
    assert b['transfer']['R']['relative_transfer'] == pytest.approx(.19)
    assert r['changed_parameters'] == ['protein_encoder.weight']
    assert r['actual_step_norm'] == pytest.approx(.1)
    assert result['lookahead']['R']['scopes']['interaction']['state'] == 'no_effective_parameter_step'
    assert result['parameters_unchanged'] and not result['optimizer_created']


def test_source_weights_scale_step_and_norm_while_zero_target_loss_has_no_ratio():
    model = Quadratic(weights=(.25,1.,0.))
    result = probes.measure(model, batch([0.,2.,0.]), batch([1.,0.,2.]), RATES)
    stats = result['gradient_statistics']['protein_encoder.other']['objectives']
    assert stats['R']['weighted_norm'] == pytest.approx(.25)
    assert stats['R']['raw_norm'] == pytest.approx(1.)
    assert stats['C']['state'] == 'disabled' and stats['C']['weighted_norm'] is None
    record = result['lookahead']['R']['scopes']['shared']
    assert record['actual_step_norm'] == pytest.approx(.025)
    assert record['transfer']['R']['before'] == 0.
    assert record['transfer']['R']['after'] == pytest.approx(.0003125)
    assert record['transfer']['R']['relative_transfer'] is None
    assert result['lookahead']['C']['state'] == 'disabled'


def test_frozen_or_stationary_shared_gradient_is_not_reported_as_zero_conflict():
    model = Quadratic()
    model.protein_encoder.requires_grad_(False)
    result = probes.measure(model, batch([0.,2.,0.]), batch([2.,0.,2.]), RATES)
    stats = result['gradient_statistics']['protein_encoder.other']
    assert stats['objectives']['R']['state'] == 'frozen'
    assert all(v is None for v in stats['cosines'].values())
    assert result['lookahead']['R']['scopes']['shared']['transfer']['R']['relative_transfer'] is None
    model.protein_encoder.requires_grad_(True)
    result = probes.measure(model, batch([1.,1.,1.]), batch([2.,0.,2.]), RATES)
    assert result['gradient_statistics']['protein_encoder.other']['objectives']['R']['state'] == 'zero_gradient'
    assert result['gradient_statistics']['protein_encoder.other']['cosines']['R__B'] is None


def test_real_architecture_measures_shared_parameters_despite_disconnected_embedding_boundary():
    model = make_model(None,{})
    model.eval()
    training = make_batch(model,torch.device('cpu'),False)
    output = model(**training)
    boundary = torch.autograd.grad(output.ranking_loss,
        (output.normalized_protein_embedding,output.normalized_molecule_embedding),allow_unused=True,retain_graph=True)
    assert boundary == (None,None)
    # Existing normalized-boundary logs therefore cannot answer Q33.
    validation = dict(training,pchembl_values=training['pchembl_values'].flip(0),
                      activity_labels=training['activity_labels'].flip(0))
    rates = dict(protein_encoder=1e-5,molecule_encoder=1e-5,interaction=1e-4)
    result = probes.measure(model,training,validation,rates,scopes=('shared',))
    encoder = result['gradient_statistics']['protein_encoder.encoder.layer.0']
    assert encoder['objectives']['R']['state'] == encoder['objectives']['C']['state'] == 'nonzero'
    assert encoder['cosines']['R__C'] is not None
    assert result['gradient_statistics']['fusion']['objectives']['C']['state'] == 'disconnected'
    assert result['gradient_statistics']['fusion']['cosines']['R__C'] is None
    assert result['lookahead']['C']['scopes']['shared']['changed_parameters']
    assert all(name.startswith(('protein_encoder.','molecule_encoder.'))
               for name in result['lookahead']['C']['scopes']['shared']['changed_parameters'])


@pytest.mark.parametrize('fail', [False, True])
def test_loaded_state_existing_gradients_modes_and_rng_are_preserved_even_on_failure(monkeypatch,fail):
    model = Quadratic()
    model.train();model.classification_head.eval()
    for parameter in model.parameters(): parameter.grad = torch.full_like(parameter,7.)
    modes = [m.training for m in model.modules()]
    parameters = {k:v.clone() for k,v in model.state_dict().items()}
    gradients = {k:p.grad.clone() for k,p in model.named_parameters()}
    rng = torch.random.get_rng_state().clone()
    if fail:
        original = model.forward
        def bad(*args,**kwargs):
            torch.rand(3)
            value = original(*args,**kwargs)
            value.ranking_loss = value.ranking_loss*float('nan')
            return value
        monkeypatch.setattr(model,'forward',bad)
        with pytest.raises(ValueError,match='finite scalar'):
            probes.measure(model,batch([0.,2.,0.]),batch([2.,0.,2.]),RATES)
    else:
        probes.measure(model,batch([0.,2.,0.]),batch([2.,0.,2.]),RATES)
    assert modes == [m.training for m in model.modules()]
    assert torch.equal(rng,torch.random.get_rng_state())
    for k,v in model.state_dict().items():torch.testing.assert_close(v,parameters[k],rtol=0,atol=0)
    for k,p in model.named_parameters():torch.testing.assert_close(p.grad,gradients[k],rtol=0,atol=0)


def test_unregistered_parameters_and_invalid_steps_fail_before_probing():
    model = Quadratic()
    model.extra = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError,match='Unregistered parameter'):
        probes.measure(model,batch([0.,2.,0.]),batch([2.,0.,2.]),RATES)
    model = Quadratic()
    with pytest.raises(ValueError,match='positive finite'):
        probes.measure(model,batch([0.,2.,0.]),batch([2.,0.,2.]),dict(RATES,interaction=0.))
