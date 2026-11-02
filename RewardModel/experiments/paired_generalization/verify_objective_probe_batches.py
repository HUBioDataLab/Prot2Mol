"""Qualify frozen E56/E57 batch reconstruction on actual cached tokens.

Compare every saved context with native dataset items and the reference native
collator. No model is constructed, no loss/gradient is measured, and no test
partition is read. Full token-payload hashing remains a scientific launch gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import pandas as pd
import torch

if __package__:
    from .run import ROOT, WORK, sha256
    from .control_plan import _write_immutable
    from . import objective_probe_batches as design
else:
    from run import ROOT, WORK, sha256
    from control_plan import _write_immutable
    import objective_probe_batches as design


def compare_batches(expected, actual):
    if set(expected) != set(actual):
        raise ValueError('Collated batch fields differ')
    checks = {}
    for name in sorted(expected):
        left, right = expected[name], actual[name]
        if isinstance(left, torch.Tensor):
            if (not isinstance(right, torch.Tensor) or left.dtype != right.dtype or left.shape != right.shape or
                left.device.type != 'cpu' or right.device.type != 'cpu' or not torch.equal(left, right)):
                raise ValueError('Collated tensor differs: '+name)
            checks[name] = dict(dtype=str(left.dtype), shape=list(left.shape),
                sha256=hashlib.sha256(left.contiguous().numpy().tobytes()).hexdigest())
        else:
            if type(left) is not type(right) or left != right:
                raise ValueError('Collated metadata differs: '+name)
            checks[name] = dict(sha256=design.digest_json(left))
    return checks


def verify_context(record, dataset, native_batches, metadata, full_examples, saved_rows, collator):
    count = record['selected_batches']
    expected_numbers = design.batch_numbers(len(native_batches), count)
    if (record['population_batches'] != len(native_batches) or len(record['batches']) != count or
        [r['pair_index'] for r in record['batches']] != list(range(count)) or
        [r['source_batch_index'] for r in record['batches']] != expected_numbers):
        raise ValueError('Frozen context selection differs from its declared batch positions')
    used, results = set(), []
    for batch in record['batches']:
        item_indices = native_batches[batch['source_batch_index']]
        if [item['item_index'] for item in batch['items']] != item_indices:
            raise ValueError('Native item order differs from the frozen context')
        expected = [dataset[index] for index in item_indices]
        for feature in expected:
            # Native cold-view row positions are local to the view. Frozen
            # records intentionally retain positions in the original cache.
            feature['example_indices'] = metadata.iloc[feature['example_indices']].cache_row_index.astype(int).tolist()
        actual = design.materialize_batch(batch, full_examples)
        if expected != actual:
            raise ValueError('Native item tokens, labels, identities or list boundaries differ')
        flat = [index for item in expected for index in item['example_indices']]
        if flat != batch['cache_row_indices'] or len(flat) != batch['rows']:
            raise ValueError('Native flattened row order differs')
        used.update(flat)
        checks = compare_batches(collator(expected), collator(actual))
        results.append(dict(pair_index=batch['pair_index'], source_batch_index=batch['source_batch_index'],
            rows=len(flat), native_items=len(expected), collated_fields=checks))
    columns = [*design.IDENTITY, *design.LABELS, 'cache_row_index', 'source_row_index', 'unit']
    expected_rows = metadata[metadata.cache_row_index.isin(used)][columns].sort_values('cache_row_index').reset_index(drop=True)
    pd.testing.assert_frame_equal(saved_rows, expected_rows, check_exact=True)
    return dict(batches=results, selected_unique_rows=len(used), exact_native_features=True,
                exact_native_collator_tensors=True, source_row_table_verified=True)


def inputs():
    directory = WORK/'tier4/objective_probe_batches'
    record = design.read(directory/'complete.json')
    if (record['payload_sha256'] != design.digest_json({k:v for k,v in record.items() if k != 'payload_sha256'}) or
        {k:record[k] for k in ('source_sha256', 'packages')} != design.software() or
        record['test_data_read'] or record['model_inference'] or record['new_fits_created'] or record['scientific_probe_results']):
        raise ValueError('Frozen batch preparation changed source or scope')
    if set(record['artifacts_sha256']) != {r+'_'+s+suffix for r in ('A', 'B')
        for s in ('train', 'cold_validation') for suffix in ('.json', '_rows.parquet')}:
        raise ValueError('Expected exactly four context manifests and row tables')
    for path, digest in record['input_sha256'].items(): design.checked(path, digest)
    for path, digest in record['source_sha256'].items(): design.checked(ROOT/path, digest)
    for path, digest in record['artifacts_sha256'].items(): design.checked(directory/path, digest)
    return directory, record


def reference_collator(config):
    from reward_model.model.encoders import load_tokenizer
    from reward_model.training.data import RewardAssayListCollator
    if config.model.contrastive_mask_policy == 'training_known' or config.model.ranking_balance_assays:
        raise ValueError('This qualification covers reference collation, not variant-specific mask/weight construction')
    options = dict(local_files_only=True, cache_dir=str(ROOT/'.hf_cache/hub'))
    names = dict(protein=config.model.protein_tokenizer_name_or_path or config.model.protein_model_name_or_path,
                 molecule=config.model.molecule_tokenizer_name_or_path or config.model.molecule_model_name_or_path)
    tokenizers = {side: load_tokenizer(name, tokenizer_kwargs=dict(options,
        **({'trust_remote_code':config.model.molecule_trust_remote_code} if side == 'molecule' else {})))
        for side, name in names.items()}
    pads = {side: int(getattr(tokenizer, 'pad_token_id', 0) or 0) for side, tokenizer in tokenizers.items()}
    collator = RewardAssayListCollator(dynamic_padding=config.training.dynamic_padding,
        protein_pad_token_id=pads['protein'], molecule_pad_token_id=pads['molecule'])
    return collator, dict(names=names, pad_token_ids=pads, dynamic_padding=config.training.dynamic_padding,
        vocab_sha256={side: design.digest_json(t.get_vocab()) for side,t in tokenizers.items()},
        local_files_only=True, model_weights_loaded=False, training_known_mask=False)


def main():
    from datasets import load_from_disk
    from reward_model.training.config import load_reward_training_config
    if (ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or
        os.environ.get('ACCELERATE_USE_CPU', '').lower() != 'true'):
        raise RuntimeError('Use CPU-only reconstruction qualification in authorized Atum Prot2Mol')
    torch.set_num_threads(2)
    directory, frozen = inputs()
    output = directory/'reconstruction_verification.json'
    if output.exists(): raise RuntimeError('Existing reconstruction verification requires inspection')
    verifier_sha256 = sha256(Path(__file__))
    encoder_source = ROOT/'RewardModel/reward_model/model/encoders.py'
    encoder_sha256 = sha256(encoder_source)
    primary = design.read(WORK/'controlled/plan.json')
    components = pd.read_parquet(WORK/'tier1/data/assay_units.parquet').set_index('assay_id').component
    results, collators = {}, {}
    for regime in ('A', 'B'):
        config = load_reward_training_config(str(WORK/'controlled/configs'/f'full__{regime}__seed42__lr1p0.yaml'))
        collator, collators[regime] = reference_collator(config)
        for role, split in (('train', 'train'), ('cold_validation', 'val')):
            cache = Path(primary['bases'][regime]['data']['tokenized_dataset_dir'])/(split+'_examples')
            full_examples = load_from_disk(str(cache))
            metadata = design.aligned_metadata(pd.read_parquet(primary['bases'][regime]['data'][split+'_parquet_path'],
                columns=design.RAW_COLUMNS), full_examples, regime, components)
            examples = full_examples
            if role == 'cold_validation':
                rows = pd.read_parquet(WORK/'tier4/objective_probe_design'/f'validation_rows_{regime}.parquet')
                examples, metadata = design.cold_view(full_examples, metadata, rows)
            dataset, batches = design.sampler_layout(examples, config, role)
            name = regime+'_'+role
            record = design.read(directory/(name+'.json'))
            if record['selected_batches'] != frozen['regimes'][regime]['batch_pairs']:
                raise ValueError('Frozen regime and context batch counts disagree')
            result = verify_context(record, dataset, batches, metadata, full_examples,
                pd.read_parquet(directory/(name+'_rows.parquet')), collator)
            results[name] = result
            print(json.dumps(dict(context=name, batches=len(result['batches']), selected_rows=result['selected_unique_rows'])), flush=True)
            del full_examples, examples, metadata, dataset
    _, checked = inputs()
    if checked != frozen or sha256(Path(__file__)) != verifier_sha256 or sha256(encoder_source) != encoder_sha256:
        raise ValueError('Qualification source or batch evidence changed during execution')
    result = dict(complete_sha256=sha256(directory/'complete.json'), verifier_sha256=verifier_sha256,
        encoder_source_sha256=encoder_sha256, verified_unix=time.time(), contexts=results, collators=collators,
        batches_verified=sum(len(r['batches']) for r in results.values()),
        selected_token_payloads_read=True, full_token_payloads_rehashed=False, test_data_read=False,
        model_weights_loaded=False, model_inference=False, new_fits_created=0, scientific_probe_results=0,
        scope='Exact production native item/row/token/list and collator tensor parity on all saved diagnostic contexts. Metadata/source/output hashes rechecked. This qualifies reconstruction, not gradient numerics, checkpoint selection, full token-payload integrity or scientific effects.')
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args()
    result = main()
    print(json.dumps(dict(batches_verified=result['batches_verified'], model_inference=False)), flush=True)
