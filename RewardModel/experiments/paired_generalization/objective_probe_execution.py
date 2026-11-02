"""Execute and summarize frozen E56/E57 batch pairs with a loaded reference.

Worker library only: the caller owns Tier 3 completion, selected checkpoint and
full token provenance, production GPU qualification, and the shared GPU lock.
There is no scientific launcher or checkpoint selection in this module.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from reward_model.training.trainer import RewardModelTrainer

if __package__:
    from .run import ROOT, WORK, sha256
    from .control_plan import GPU_UUIDS, digest_json, _write_immutable
    from .checkpoint_evaluation import parameter_digest
    from . import objective_probe_protocol as protocol
    from . import objective_probe_batches as batches
    from . import objective_probes as probes
    from .verify_objective_probe_batches import compare_batches
else:
    from run import ROOT, WORK, sha256
    from control_plan import GPU_UUIDS, digest_json, _write_immutable
    from checkpoint_evaluation import parameter_digest
    import objective_probe_protocol as protocol
    import objective_probe_batches as batches
    import objective_probes as probes
    from verify_objective_probe_batches import compare_batches


def write(path, value):
    _write_immutable(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def worker_sources():
    names = ('objective_probe_execution.py', 'objective_probe_protocol.py', 'objective_probes.py',
             'objective_probe_batches.py', 'verify_objective_probe_batches.py', 'checkpoint_evaluation.py')
    paths = [Path(__file__).with_name(name) for name in names]
    paths.append(ROOT/'RewardModel/reward_model/training/trainer.py')
    return {str(path.relative_to(ROOT)):sha256(path) for path in paths}


@contextmanager
def full_precision():
    previous = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(previous[0])
        torch.backends.cuda.matmul.allow_tf32 = previous[1]
        torch.backends.cudnn.allow_tf32 = previous[2]


def distribution(values):
    defined = [float(v) for v in values if v is not None]
    if any(not math.isfinite(v) for v in defined):
        raise ValueError('Nonfinite diagnostic value must not be silently dropped')
    result = dict(total=len(values), defined=len(defined), undefined=len(values)-len(defined))
    result.update(mean=float(np.mean(defined)) if defined else None,
        quantiles=dict(zip(('min', 'q25', 'median', 'q75', 'max'),
            map(float, np.quantile(defined, [0, .25, .5, .75, 1])))) if defined else None,
        positive=sum(v > 0 for v in defined), negative=sum(v < 0 for v in defined),
        zero=sum(v == 0 for v in defined))
    return result


def summarize(records):
    """Descriptive batch distributions; no independent-batch uncertainty."""
    if not records or [r['pair_index'] for r in records] != list(range(len(records))):
        raise ValueError('Expected every pair once, in the frozen order')
    results = [r['measurement'] for r in records]
    first = results[0]
    for result in results:
        if any(result[key] != first[key] for key in (
            'parameter_sha256', 'parameter_blocks', 'weights', 'learning_rates', 'precision', 'cpu_qualification')):
            raise ValueError('Cannot pool different checkpoints, rate policies or execution modes')
        if not result['parameters_unchanged'] or result['optimizer_created']:
            raise ValueError('Probe changed the fitted model')
        if set(result['gradient_statistics']) != set(first['gradient_statistics']):
            raise ValueError('Parameter block inventory changed')
    gradients = {}
    for block in first['gradient_statistics']:
        rows = [r['gradient_statistics'][block] for r in results]
        gradients[block] = dict(objectives={}, cosines={})
        for objective in probes.OBJECTIVES:
            values = [r['objectives'][objective] for r in rows]
            gradients[block]['objectives'][objective] = dict(
                states=dict(Counter(v['state'] for v in values)),
                raw_norm=distribution([v['raw_norm'] for v in values]),
                weighted_norm=distribution([v['weighted_norm'] for v in values]))
        for pair in rows[0]['cosines']:
            gradients[block]['cosines'][pair] = distribution([v['cosines'][pair] for v in rows])
    lookahead = {}
    for source in probes.OBJECTIVES:
        lookahead[source] = {}
        for scope in protocol.SCOPES:
            rows = [r['lookahead'][source]['scopes'][scope] for r in results]
            lookahead[source][scope] = dict(states=dict(Counter(r['state'] for r in rows)),
                planned_step_norm=distribution([r['planned_step_norm'] for r in rows]),
                actual_step_norm=distribution([r['actual_step_norm'] for r in rows]),
                targets={target: {field: distribution([r['transfer'][target][field] for r in rows])
                    for field in ('before', 'after', 'absolute_loss_change', 'relative_transfer')}
                    for target in probes.OBJECTIVES})
    return dict(pairs=len(records), gradient_statistics=gradients, lookahead=lookahead,
        independent_batch_intervals=False,
        interpretation='Distributions over fixed overlapping diagnostic contexts, conditional on one fitted model. Exact signs and undefined coverage are reported without a numerical tolerance, significance claim or causal generalization conclusion.')


def checked_batch(record, examples, collator, evidence):
    batch = collator(batches.materialize_batch(record, examples))
    if (record['pair_index'] != evidence['pair_index'] or record['rows'] != evidence['rows']
        or record['source_batch_index'] != evidence['source_batch_index']
        or len(record['items']) != evidence['native_items']
        or compare_batches(batch, batch) != evidence['collated_fields']):
        raise ValueError('Reconstructed batch differs from its verified native fields')
    return batch


def execute_pairs(model, examples, contexts, collator, training, evidence, output, source, *, cpu=False):
    """Measure exactly the provided verified pair inventory, preserving the model.

``evidence`` contains each role's production reconstruction records. Scientific
callers must independently establish the upstream gates documented above; this
function does not promote its input receipts into a Tier 3 completion claim.
"""
    output = Path(output)
    if not cpu and (ROOT != Path('/home/atabey/Prot2Mol') or
        os.environ.get('CUDA_VISIBLE_DEVICES') not in GPU_UUIDS or not output.absolute().is_relative_to(WORK)):
        raise RuntimeError('Scientific probes require authorized Atum project and one permitted GPU')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Use one independent model per diagnostic worker')
    if output.exists():
        raise RuntimeError('Existing probe output requires verification or inspection')
    roles = {'train', 'cold_validation'}
    if set(examples) != roles or set(contexts) != roles or set(evidence) != roles:
        raise ValueError('Only the frozen training and cold-validation roles are allowed')
    count = contexts['train']['selected_batches']
    if type(count) is not int or count < 1 or (not cpu and count != batches.BATCH_PAIRS):
        raise ValueError('Scientific workers require all 32 frozen diagnostic pairs')
    for role in roles:
        if (contexts[role]['selected_batches'] != count or len(evidence[role]['batches']) != count or
            [r['pair_index'] for r in contexts[role]['batches']] != list(range(count))):
            raise ValueError('Training/validation context inventories disagree')
    parameters = dict(model.named_parameters())
    device = next(iter(parameters.values())).device
    if device.type != ('cpu' if cpu else 'cuda') or any(p.device != device for p in parameters.values()):
        raise ValueError('Loaded model must already occupy the assigned single device')
    if any(t.is_floating_point() and t.dtype != torch.float32 for t in model.state_dict().values()):
        raise ValueError('Frozen protocol requires float32 model parameters and buffers')
    weights = {key:float(getattr(model.config, field)) for key, (_, field) in probes.OBJECTIVES.items()}
    if weights != dict(R=.25, B=.25, C=1.):
        raise ValueError('This protocol measures only the full reference objectives')
    rates = protocol.validate_parameter_rates(parameters, training)
    before = parameter_digest(model)
    if source.get('parameter_sha256') != before:
        raise ValueError('Loaded model differs from the caller-verified checkpoint')
    records, artifacts, code = [], {}, worker_sources()
    try:
        with full_precision():
            for index in range(count):
                inputs = {role: checked_batch(contexts[role]['batches'][index], examples[role], collator,
                    evidence[role]['batches'][index]) for role in ('train', 'cold_validation')}
                # The native collator also emits evaluation/export metadata.
                # Use the Trainer's stateless adapter without constructing a
                # Trainer; only its actual model inputs belong in the forward.
                inputs = {role:RewardModelTrainer._build_model_inputs(None,
                    {k:v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()})
                    for role,batch in inputs.items()}
                measured = probes.measure(model, inputs['train'], inputs['cold_validation'], rates,
                    precision='float32', scopes=protocol.SCOPES)
                record = dict(pair_index=index, context_sha256={role:digest_json(contexts[role]['batches'][index])
                    for role in ('train', 'cold_validation')}, measurement=measured)
                name = f'pair_{index:03d}.json'
                write(output/name, record); artifacts[name] = sha256(output/name)
                records.append(record)
        if parameter_digest(model) != before:
            raise RuntimeError('Diagnostic execution changed the loaded model')
        if worker_sources() != code:
            raise RuntimeError('Worker source changed during execution')
        result = dict(source=source, cpu_qualification=cpu, parameter_sha256=before,
            context_sha256=digest_json(contexts), reconstruction_sha256=digest_json(evidence),
            worker_sources_sha256=code, learning_rates=rates, artifacts_sha256=artifacts, summary=summarize(records),
            parameters_unchanged=True, optimizer_created=False, new_fits_created=0,
            scope='Fixed-model batch measurements only. Tier3/checkpoint/token/GPU admission belongs to the caller. No experiment completion is inferred from worker completion.')
        write(output/'complete.json', result)
        return result
    finally:
        if parameter_digest(model) != before:
            raise RuntimeError('Diagnostic execution changed the loaded model')


def verify_complete(output, source, contexts, evidence, *, cpu=False):
    output = Path(output); result = json.loads((output/'complete.json').read_text())
    count = contexts['train']['selected_batches']
    expected = [f'pair_{i:03d}.json' for i in range(count)]
    if (result['source'] != source or result['parameter_sha256'] != source['parameter_sha256'] or result['cpu_qualification'] != cpu or
        result['context_sha256'] != digest_json(contexts) or result['reconstruction_sha256'] != digest_json(evidence)
        or result['worker_sources_sha256'] != worker_sources()
        or set(result['artifacts_sha256']) != set(expected) or not result['parameters_unchanged']
        or result['optimizer_created'] or result['new_fits_created']):
        raise ValueError('Probe provenance, pair coverage or execution mode changed')
    records = []
    for name in expected:
        if sha256(output/name) != result['artifacts_sha256'][name]:
            raise ValueError('Saved diagnostic pair changed')
        record = json.loads((output/name).read_text()); index = record['pair_index']
        if (record['context_sha256'] != {role:digest_json(contexts[role]['batches'][index]) for role in ('train','cold_validation')}
            or record['measurement']['parameter_sha256'] != result['parameter_sha256']
            or record['measurement']['learning_rates'] != result['learning_rates']
            or record['measurement']['cpu_qualification'] != cpu or record['measurement']['precision'] != 'float32'):
            raise ValueError('Saved pair refers to different model, rates or contexts')
        records.append(record)
    if summarize(records) != result['summary']:
        raise ValueError('Diagnostic distributions differ from saved pairs')
    return result


def qualification_fixture():
    """Small synthetic native lists, never scientific data or a selected model."""
    from types import SimpleNamespace
    from datasets import Dataset
    import pandas as pd
    from reward_model.training.data import RewardAssayListCollator
    if __package__:
        from .qualify_controls import make_model
    else:
        from qualify_controls import make_model
    model = make_model(None, {})
    config = SimpleNamespace(training=SimpleNamespace(seed=42, per_device_train_batch_size=12,
        length_bucketing=True, length_bucket_size_multiplier=2),
        data=SimpleNamespace(ranking_max_ligands=16, ranking_opportunity_divisor=32,
            ranking_min_pchembl_span=.5, max_classification_only_per_item=16, sample_all_assays=True),
        model=SimpleNamespace(contrastive_loss_weight=1., ranking_balance_assays=False))
    collator = RewardAssayListCollator(protein_pad_token_id=1, molecule_pad_token_id=1)
    examples, contexts, evidence = {}, {}, {}
    for role, offset in (('train', 0), ('cold_validation', 48)):
        rows = [dict(assay_id=f'a{g}', target_chembl_id=f'p{g%5}', compound_id=f'm{g}_{i}',
            group_id=f'p{g%5}__a{g}', pchembl_value=5.+1.5*i, binary_label=float(i > 0),
            protein_input_ids=[2+g%24]*3, protein_attention_mask=[1]*3,
            molecule_input_ids=[2+i]*4, molecule_attention_mask=[1]*4,
            protein_length=3, molecule_length=4) for g in range(offset, offset+48) for i in range(3)]
        examples[role] = Dataset.from_list(rows)
        metadata = batches.aligned_metadata(pd.DataFrame(rows)[batches.RAW_COLUMNS], examples[role],
            'A', pd.Series({f'a{g}':g for g in range(96)}))
        dataset, order = batches.sampler_layout(examples[role], config, role)
        contexts[role], _ = batches.freeze_batches(dataset, order, metadata, count=2)
        checked = []
        for record in contexts[role]['batches']:
            native = collator([dataset[i] for i in order[record['source_batch_index']]])
            checked.append(dict(pair_index=record['pair_index'], rows=record['rows'],
                source_batch_index=record['source_batch_index'], native_items=len(record['items']),
                collated_fields=compare_batches(native, native)))
        evidence[role] = dict(batches=checked)
    training = dict(learning_rate=1e-4, encoder_learning_rate=1e-5, projection_learning_rate=1e-4)
    return model, examples, contexts, collator, training, evidence


def qualify_cpu(output):
    import copy
    from reward_model.model import LoadedEncoder, load_reward_model, save_reward_model
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('ACCELERATE_USE_CPU') != 'true'
        or os.environ.get('WANDB_MODE') != 'disabled'):
        raise RuntimeError('Explicit CPU-only, W&B-disabled qualification required')
    torch.set_num_threads(2)
    output = Path(output)
    if output.exists(): raise RuntimeError('Existing CPU qualification requires inspection')
    original, examples, contexts, collator, training, evidence = qualification_fixture()
    checkpoint = output/'synthetic_checkpoint'
    save_reward_model(original, str(checkpoint))
    bundles = {side+'_bundle': LoadedEncoder('tiny/'+side, None,
        copy.deepcopy(getattr(original, side+'_encoder')), getattr(original, side+'_encoder').config.hidden_size)
        for side in ('protein', 'molecule')}
    model = load_reward_model(str(checkpoint), device=torch.device('cpu'), **bundles)
    before = parameter_digest(original)
    if parameter_digest(model) != before: raise ValueError('Native checkpoint round-trip changed parameters')
    rng = torch.random.get_rng_state().clone()
    modes = [m.training for m in model.modules()]
    source = dict(synthetic=True, parameter_sha256=before,
        checkpoint_weights_sha256=sha256(checkpoint/'pytorch_model.bin'))
    result = execute_pairs(model, examples, contexts, collator, training, evidence, output/'pairs', source, cpu=True)
    if verify_complete(output/'pairs', source, contexts, evidence, cpu=True) != result:
        raise ValueError('Saved CPU result differs from the worker result')
    if not torch.equal(rng, torch.random.get_rng_state()) or modes != [m.training for m in model.modules()]:
        raise ValueError('Worker changed RNG or loaded module modes')
    qualified = dict(cpu_qualification=True, synthetic_pairs=result['summary']['pairs'],
        native_checkpoint_roundtrip=True, native_batch_fingerprints_checked=True,
        result_reloaded_and_summary_recomputed=True, parameters_and_rng_preserved=True,
        complete_sha256=sha256(output/'pairs/complete.json'), new_fits_created=0,
        scientific_probe_results=0, production_gpu_qualified=False)
    write(output/'qualification.json', qualified)
    return qualified


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Synthetic CPU qualification only; no scientific launcher')
    parser.add_argument('command', choices=['qualify-cpu'])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(qualify_cpu(args.output), indent=2), flush=True)
