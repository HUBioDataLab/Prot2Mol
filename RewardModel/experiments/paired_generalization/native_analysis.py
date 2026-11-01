"""Incremental CPU audits of committed native validation/training snapshots.

No model is loaded and no test prediction is read. Completed snapshots may be
analyzed while their 15-epoch fit is still running. A prefix is never presented
as a completed trajectory or a final checkpoint-selection result.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .control_plan import GPU_UUIDS, digest_json, _write_immutable
    from .checkpoint_evaluation import read_native_predictions
    from .realization_protocol import IDENTITY, LABELS, align_rows
    from .baseline_common import SOURCE_COLUMNS
    from .label_precision import trainer_label_frame, POLICY
    from .analyze_predictions import rescore, fit_temperature, binary_logloss
    from .noise_band_audit import score_bands
    from .scorer_agreement import group_agreement, summarize as summarize_agreement, matched_endpoint_changes
    from .similarity_composition import verify_design, score_similarity, bootstrap_scores
else:
    from run import ROOT, WORK, sha256, write_json
    from control_plan import GPU_UUIDS, digest_json, _write_immutable
    from checkpoint_evaluation import read_native_predictions
    from realization_protocol import IDENTITY, LABELS, align_rows
    from baseline_common import SOURCE_COLUMNS
    from label_precision import trainer_label_frame, POLICY
    from analyze_predictions import rescore, fit_temperature, binary_logloss
    from noise_band_audit import score_bands
    from scorer_agreement import group_agreement, summarize as summarize_agreement, matched_endpoint_changes
    from similarity_composition import verify_design, score_similarity, bootstrap_scores

SCORERS = ('ranking_score','activity_logits','cosine_similarity')


def read(path):
    return json.loads(Path(path).read_text())


def immutable(path, value):
    _write_immutable(path,json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def software():
    from importlib.metadata import version
    names = ['native_analysis.py','metrics.py','checkpoint_evaluation.py','realization_protocol.py',
             'analyze_predictions.py','noise_band_audit.py','scorer_agreement.py','similarity_composition.py',
             'label_precision.py','baseline_common.py','control_plan.py','run.py']
    paths = [Path(__file__).with_name(name) for name in names]
    paths += [ROOT/'RewardModel/reward_model/training'/name for name in ['config.py','data.py','evaluation.py']]
    return dict(source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in paths},
        packages={name:version(name) for name in ['numpy','scipy','pandas','scikit-learn','datasets','torch']})


def native_frame(manifest, expected, fingerprint, prefix):
    frame, record = read_native_predictions(manifest)
    if record['metric_prefix'] != prefix or record['dataset_fingerprint'] != fingerprint:
        raise ValueError('Snapshot partition or dataset fingerprint changed')
    if len(frame) != len(expected) or not np.isfinite(frame[list(SCORERS)+LABELS].to_numpy()).all():
        raise ValueError('Snapshot is incomplete or nonfinite')
    # The actual collator rounds labels to float32. Arithmetic after export uses
    # Python-float/float64 values, as in the native Spearman implementation.
    frame = trainer_label_frame(frame)
    reference = trainer_label_frame(expected)
    pd.testing.assert_frame_equal(frame[IDENTITY+LABELS].reset_index(drop=True),
        reference[IDENTITY+LABELS].reset_index(drop=True),check_dtype=False,check_exact=True)
    frame['assay_id'] = reference.assay_id.to_numpy()
    return frame,record


def check_native_metric(value, expected, label):
    if value is None or expected is None:
        if value is not None or expected is not None:
            raise ValueError(f'Native metric definedness differs: {label}')
    elif not np.isclose(value,expected,atol=1e-14,rtol=0):
        raise ValueError(f'Native metric parity failed: {label}')


def audit_snapshot(manifest, expected, fingerprint, component_map, output, *, prefix,
                   noise_rms, provenance, regime, similarity_design=None):
    if prefix not in ('eval','train_eval') or regime not in ('A','B'):
        raise ValueError('Only validation and fixed training-probe trajectories are accepted')
    frame, record = native_frame(manifest,expected,fingerprint,prefix)
    source = dict(manifest=str(manifest),manifest_sha256=sha256(manifest),
        payload_sha256=record['predictions_sha256'],provenance=provenance)
    if (output/'complete.json').exists():
        cached = read(output/'complete.json')
        if cached['source'] != source:
            raise ValueError('Completed snapshot analysis has different provenance')
        for name,digest in cached['artifacts_sha256'].items():
            if sha256(output/name) != digest:
                raise ValueError('Completed snapshot analysis artifact changed')
        return cached
    if output.exists():
        raise RuntimeError('Partial snapshot analysis requires inspection')
    if not math.isfinite(noise_rms) or noise_rms <= 0:
        raise ValueError('Noise band requires the original positive training-only RMS')
    metrics, tables = rescore(frame,component_map)
    for scorer,table in tables.items():
        eligible = table.original_ranking_eligible
        metrics[scorer].update(eligible_groups=int(eligible.sum()),
            eligible_defined_fraction=float(table.loc[eligible,'spearman'].notna().mean()) if eligible.any() else None,
            matched_eligible_ndcg={str(k):float(table.loc[eligible,f'ndcg_at_{k}_linear'].mean()) if eligible.any() else None for k in (1,3,5)})
        for name in ('unit_bootstrap_all_defined','unit_bootstrap_original_eligible'):
            metrics[scorer][name]['interpretation'] = (
                f'Whole {"assays" if regime == "A" else "scaffold-assay components"}; '
                + ('descriptive resampling of the fixed training probe, not held-out generalization or training-seed uncertainty.'
                   if prefix == 'train_eval' else
                   'percentile interval conditional on these held-out units, not split-realization or training-seed uncertainty.'))
    rank = metrics['ranking_score']
    check_native_metric(rank['original_eligible_macro_spearman'],record['metrics'][f'{prefix}_macro_spearman'],'macro Spearman')
    check_native_metric(rank['eligible_defined_fraction'],record['metrics'][f'{prefix}_spearman_defined_fraction'],'defined fraction')
    if rank['eligible_groups'] != int(record['metrics'][f'{prefix}_spearman_num_groups']):
        raise ValueError('Native eligible cohort differs')
    noise, noise_tables = score_bands(frame,dict(training_replicate_rms=noise_rms,reference_margin=float(np.log10(3.))),SCORERS)
    agreement = group_agreement(frame,component_map)
    calibration = None
    if prefix == 'eval':
        temperature = fit_temperature(frame.binary_label,frame.activity_logits)
        calibration = dict(temperature=temperature,fit_partition='validation only',
            validation_logloss_before=metrics['classification']['logloss'],
            validation_fit_logloss_after=binary_logloss(frame.binary_label,frame.activity_logits,temperature),
            interpretation='In-sample validation calibration; no test transfer evaluated. Positive temperature preserves logit ordering.')
    similarity, standardization = None,None
    if similarity_design is not None:
        rows,groups,cells = similarity_design
        locations = align_rows(rows,frame)
        ordered = frame.iloc[np.argsort(locations)].reset_index(drop=True)
        similarity, scored_groups = score_similarity(rows,groups,ordered)
        if regime == 'A':
            lookup = cells.set_index('cell').A_group_weight
            weights = scored_groups.cell.map(lookup).fillna(0).to_numpy()*scored_groups.eligible.to_numpy()
            standardization = dict(scores=bootstrap_scores(scored_groups,
                dict(raw=scored_groups.eligible.to_numpy(float),common=(weights>0).astype(float),standardized=weights),
                scored_groups.unit_id.unique()),
                target_supported_mass=float(cells.loc[cells.common_support,'target_mass'].sum()),
                interpretation='A validation reweighted to fixed B validation composition. No B model score or test result is used; matched A/B model comparison remains separate.')
    output.mkdir(parents=True)
    for scorer,table in tables.items():
        table.to_parquet(output/f'groups_{scorer}.parquet',index=False)
    for name,table in noise_tables.items():
        table.to_parquet(output/f'noise_{name.replace("/","_")}.parquet',index=False)
    agreement.to_parquet(output/'agreement.parquet',index=False)
    result = dict(source=source,epoch=record['epoch'],step=record['step'],prefix=prefix,
        rows=len(frame),ranking_readout=record['ranking_readout'],label_precision=POLICY,
        scorer_interpretation='ranking_score is the configured readout; in activity-logit or cosine readout runs it repeats that scorer, rather than representing an independently trained ranking head.',
        native_metrics=record['metrics'],contrastive_score_scale=record.get('contrastive_score_scale'),
        metrics=metrics,noise_bands=noise,agreement=summarize_agreement(agreement),calibration=calibration,
        similarity=similarity,composition_standardization=standardization,
        unit_definition='assay' if regime == 'A' else 'scaffold-assay component',
        probe_limit='Training statistics describe only the fixed eligible-assay probe; they are not full-training or held-out generalization estimates.' if prefix == 'train_eval' else None,
        contrastive_limit='Native aggregate contrastive loss and scale are recorded. Directional/fixed-temperature losses and representation geometry require the protected weights and separate embedding inference.',
        artifacts_sha256={p.name:sha256(p) for p in sorted(output.glob('*.parquet'))})
    immutable(output/'complete.json',result)
    return result


def registration(identity):
    directory = WORK/'controlled'
    if Path(identity).name != identity:
        raise ValueError('Expected a primary-grid job ID')
    path = directory/'jobs'/f'{identity}.json'
    metadata = read(path)
    config_path = Path(metadata['config_path'])
    run = Path(metadata['output_dir'])
    plan = read(directory/'plan.json')
    if (metadata['id'] != identity or metadata['plan_sha256'] != sha256(directory/'plan.json') or
        config_path != directory/'configs'/f'{identity}.yaml' or run != directory/'runs'/identity or
        sha256(config_path) != metadata['config_sha256']):
        raise ValueError('Snapshot job/configuration registration changed')
    launch = read(run/'launch_provenance.json')
    if (launch['config_sha256'] != metadata['config_sha256'] or launch['world_size'] != 4 or
        launch['visible_devices'].split(',') != list(GPU_UUIDS) or launch['spec_sha256'] != plan['spec_sha256']):
        raise ValueError('Snapshot launch provenance differs')
    import yaml
    if launch['config'] != yaml.safe_load(config_path.read_text()):
        raise ValueError('Live launch configuration differs from its frozen YAML')
    if launch['config']['data'].get('train_view_manifest'):
        raise ValueError('Primary trajectory analysis cannot silently accept a training subset')
    return metadata,plan,dict(job_metadata_sha256=sha256(path),launch_sha256=sha256(run/'launch_provenance.json'))


def prepare_references(config, plan, regime, destination):
    """Recover exact cache order and probe; verify all metadata against sources.

    Tokens are never consumed here. Frozen cache state/info hashes and every
    used identity/raw label are checked; large token buffers are not rehashed.
    Their training-time provenance remains the original controlled launch.
    """
    from datasets import load_from_disk
    from reward_model.training.data import select_fixed_ranking_evaluation_subset
    sources, datasets = {}, {}
    for split in ('train','val'):
        path = Path(getattr(config.data,f'{split}_parquet_path'))
        digest = sha256(path)
        if digest != plan['source_data_sha256'][regime][split]:
            raise ValueError('Snapshot source partition differs from the frozen plan')
        raw = pd.read_parquet(path,columns=SOURCE_COLUMNS)
        raw['group_id'] = raw.target_chembl_id+'__'+raw.assay_id
        cache = Path(config.data.tokenized_dataset_dir)/f'{split}_examples'
        for name in ('state.json','dataset_info.json'):
            if sha256(cache/name) != plan['tokenized_data_sha256'][regime][split][name]:
                raise ValueError('Token cache registration changed')
        dataset = load_from_disk(str(cache))
        metadata = dataset.select_columns(IDENTITY+LABELS).to_pandas()
        locations = align_rows(raw,metadata)
        for name in LABELS:
            if not np.array_equal(raw[name].to_numpy()[locations],metadata[name].to_numpy()):
                raise ValueError('Raw cache labels differ from the frozen source')
        datasets[split] = dataset
        sources[split] = dict(path=str(path),sha256=digest,cache=str(cache),loaded_fingerprint=dataset._fingerprint,
            cache_metadata_sha256={n:sha256(cache/n) for n in ('state.json','dataset_info.json')})
    probe = select_fixed_ranking_evaluation_subset(datasets['train'],num_assays=config.training.fixed_train_eval_assays,
        seed=config.training.seed,min_pchembl_span=config.data.ranking_min_pchembl_span)
    references = {}
    for name,dataset in [('eval',datasets['val']),('train_eval',probe)]:
        frame = dataset.select_columns(IDENTITY+LABELS).to_pandas()
        frame['assay_id'] = frame.group_id.str.split('__',n=1).str[1]
        if not (frame.target_chembl_id+'__'+frame.assay_id).equals(frame.group_id):
            raise ValueError('Cannot reconstruct assay identity')
        references[name] = (frame,dataset._fingerprint)
    destination.mkdir(parents=True,exist_ok=True)
    for name,(frame,_) in references.items():
        path = destination/f'{name}.parquet'
        if path.exists():
            # Parquet can restore inferred strings as pandas StringDtype rather
            # than object. Require exact values/order, not that storage dtype.
            pd.testing.assert_frame_equal(pd.read_parquet(path),frame,check_exact=True,check_dtype=False)
        else:
            frame.to_parquet(path,index=False)
    record = dict(sources=sources,probe=dict(seed=config.training.seed,requested_assays=config.training.fixed_train_eval_assays,
        actual_assays=int(references['train_eval'][0].group_id.nunique()),rows=len(probe)),
        references={name:dict(fingerprint=fp,sha256=sha256(destination/f'{name}.parquet')) for name,(_,fp) in references.items()},
        token_payloads_used=False,token_payloads_rehashed=False)
    immutable(destination/'complete.json',record)
    return references,record


def main(identity):
    import fcntl
    from reward_model.training.config import load_reward_training_config
    if ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Use CPU-only analysis in the authorized Atum project')
    metadata,plan,registration_source = registration(identity)
    config = load_reward_training_config(metadata['config_path'])
    if config.data.ranking_min_pchembl_span != .5 or config.model.ranking_min_pchembl_span != .5:
        raise ValueError('Native audit implements the registered headline eligibility span')
    regime = metadata['regime']
    code = software()
    output = WORK/'tier1/native_trajectories'/identity/digest_json(code)[:16]
    output.mkdir(parents=True,exist_ok=True)
    with (output/'analysis.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        references,reference_record = prepare_references(config,plan,regime,output/'references')
        units_path = WORK/'tier1/data/assay_units.parquet'
        units = pd.read_parquet(units_path).sort_values('assay_id')
        if units.assay_id.duplicated().any():
            raise ValueError('Assay/unit map is ambiguous')
        component_map = dict(zip(units.assay_id,np.arange(len(units)))) if regime == 'A' else units.set_index('assay_id').component.to_dict()
        design_path = WORK/'tier1/similarity_composition/design'
        verify_design(design_path)
        similarity_design = tuple(pd.read_parquet(design_path/name) for name in
            (f'rows_{regime}_val.parquet',f'groups_{regime}_val.parquet','weights_val.parquet'))
        source = dict(**registration_source,software=code,reference_sha256=sha256(output/'references/complete.json'),
            units_sha256=sha256(units_path),similarity_design_sha256=sha256(design_path/'complete.json'),
            noise_rms=plan['training_noise_rms'][regime],noise_summary_sha256=plan['noise_summary_sha256'])
        immutable(output/'analysis_plan.json',source)
        directory = Path(metadata['output_dir'])/'prediction_exports/trajectory'
        val = {read(p)['step']:p for p in directory.glob('eval_*.json')}
        train = {read(p)['step']:p for p in directory.glob('train_eval_*.json')}
        records = []
        for step in sorted(set(val)&set(train)):
            pair = {}
            for prefix,path in [('eval',val[step]),('train_eval',train[step])]:
                manifest = read(path)
                if (manifest['phase'] != 'trajectory' or manifest['step'] != step or
                    manifest['ranking_readout'] != config.model.ranking_readout):
                    raise ValueError('Unexpected trajectory snapshot')
                pair[prefix] = audit_snapshot(path,*references[prefix],component_map,output/path.stem,
                    prefix=prefix,noise_rms=source['noise_rms'],provenance=source,regime=regime,
                    similarity_design=similarity_design if prefix == 'eval' else None)
                print(json.dumps(dict(snapshot=path.stem,epoch=manifest['epoch'],
                    macro_spearman=pair[prefix]['metrics']['ranking_score']['original_eligible_macro_spearman'])),flush=True)
            if pair['eval']['epoch'] != pair['train_eval']['epoch']:
                raise ValueError('Validation and training-probe epoch differ')
            records.append(dict(step=step,epoch=pair['eval']['epoch'],
                eval=pair['eval'],train_eval=pair['train_eval']))
        epochs = [r['epoch'] for r in records]
        if epochs != list(range(1,len(records)+1)):
            raise ValueError('Completed trajectory snapshots are not a contiguous epoch prefix')
        horizon = config.training.num_train_epochs
        if len(records) > horizon:
            raise ValueError('More snapshots than registered epochs')
        changes = {}
        if len(records) >= 2:
            for prefix in ('eval','train_eval'):
                first,last = records[0][prefix],records[-1][prefix]
                first_table = pd.read_parquet(output/f"{prefix}_step{first['step']:08d}"/'agreement.parquet')
                last_table = pd.read_parquet(output/f"{prefix}_step{last['step']:08d}"/'agreement.parquet')
                changes[prefix] = matched_endpoint_changes(first_table,last_table)
                for value in changes[prefix].values():
                    value['unit_bootstrap']['interpretation'] = (
                        f'Whole {"assays" if regime == "A" else "scaffold-assay components"}, conditional on this observed trajectory prefix; '
                        'training-probe intervals are descriptive, not held-out or seed uncertainty.')
        summary = dict(job=metadata,analysis_plan_sha256=sha256(output/'analysis_plan.json'),
            epochs_analyzed=epochs,horizon=horizon,complete_horizon_observed=len(records)==horizon,
            test_predictions_read=False,probe=reference_record['probe'],trajectory=records,
            observed_prefix_agreement_changes=changes,
            interpretation='Snapshot analysis only. A prefix cannot establish final peak/overfitting behavior, checkpoint selection, scientific run completion, or multi-seed conclusions. No model inference/training/test scoring occurs here.')
        write_json(output/'trajectory.json',summary)
        write_json(output.parent/'latest.json',dict(path=str(output/'trajectory.json'),sha256=sha256(output/'trajectory.json'),
            epochs_analyzed=epochs,phase='native_trajectory_analysis_current'))
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id',required=True)
    args = parser.parse_args()
    result = main(args.job_id)
    print(json.dumps(dict(epochs_analyzed=result['epochs_analyzed'],horizon=result['horizon'],test_predictions_read=False)))
