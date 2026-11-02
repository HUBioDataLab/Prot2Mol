"""CPU diagnostics for completed, validation-selected primary-grid evaluations.

No model inference, optimization or selection occurs here. The source evaluator
must have completed every registered checkpoint and both held-out partitions.
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
    from . import postselection, native_analysis, similarity_composition
    from .run import ROOT, WORK, sha256
    from .control_plan import GPU_UUIDS, digest_json
    from .checkpoint_evaluation import read_native_predictions
    from .realization_protocol import align_rows
    from .label_precision import POLICY
    from .analyze_predictions import rescore, fit_temperature, binary_logloss
    from .noise_band_audit import score_bands
    from .scorer_agreement import group_agreement, summarize
    from .shift_audit import bbse
else:
    import postselection, native_analysis, similarity_composition
    from run import ROOT, WORK, sha256
    from control_plan import GPU_UUIDS, digest_json
    from checkpoint_evaluation import read_native_predictions
    from realization_protocol import align_rows
    from label_precision import POLICY
    from analyze_predictions import rescore, fit_temperature, binary_logloss
    from noise_band_audit import score_bands
    from scorer_agreement import group_agreement, summarize
    from shift_audit import bbse

SCORERS = native_analysis.SCORERS


def read(path):
    return json.loads(Path(path).read_text())


def finite_json(value):
    """Undefined diagnostics remain null, never an invented zero score."""
    if isinstance(value, dict):
        return {key:finite_json(item) for key,item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def immutable(path, value):
    postselection.immutable_json(path, finite_json(value))


def software():
    result = native_analysis.software()
    for name in ('selected_analysis.py', 'postselection.py', 'shift_audit.py'):
        path = Path(__file__).with_name(name)
        result['source_sha256'][str(path.relative_to(ROOT))] = sha256(path)
    return result


def verify_completed_evaluations(directory, *, cpu_qualification=False):
    """Refuse incomplete selection/evaluation before reading test predictions."""
    directory = Path(directory)
    path = directory/'plan.json'
    plan, complete = read(path), read(directory/'complete.json')
    if (complete['phase'] != 'fixed_weight_evaluation_complete' or
        complete['plan_sha256'] != sha256(path)):
        raise ValueError('Native evaluation completion differs from its frozen plan')
    if plan != postselection.build_evaluation_plan(WORK/'controlled', plan['job']['id']):
        raise ValueError('Native evaluation selection or provenance changed')
    expected = {f"{c['id']}/{s}" for c in plan['checkpoints'] for s in ('val','test')}
    if set(complete['evidence_sha256']) != expected:
        raise ValueError('Native evaluation completion omits or adds checkpoint partitions')
    # Verify all receipts before opening any prediction payload.
    for name,digest in complete['evidence_sha256'].items():
        if sha256(directory/name/'complete.json') != digest:
            raise ValueError('Native evaluation receipt changed')
    evaluations = {}
    for checkpoint in plan['checkpoints']:
        evaluations[checkpoint['id']] = {}
        for split in ('val','test'):
            output = directory/checkpoint['id']/split
            receipt = read(output/'complete.json')
            gpu = receipt['source']['gpu_uuid']
            if (cpu_qualification and gpu is not None) or (not cpu_qualification and gpu not in GPU_UUIDS):
                raise ValueError('Unexpected native evaluation device provenance')
            source = dict(plan_sha256=sha256(path),checkpoint=checkpoint,split=split,gpu_uuid=gpu)
            evaluations[checkpoint['id']][split] = postselection.verify_evaluation(
                output, source, cpu_qualification=cpu_qualification)
    return plan, evaluations


def references(plan, design_path):
    """Verify design/cache metadata without consuming token values."""
    from datasets import load_from_disk
    design = similarity_composition.verify_design(design_path)
    regime = plan['job']['regime']
    result = {}
    for split,source in plan['data'].items():
        if design['source_sha256'].get(source['source']) != source['source_sha256']:
            raise ValueError('Scaffold design and evaluated source partition differ')
        cache = Path(source['cache'])
        for name in ('state.json','dataset_info.json'):
            if sha256(cache/name) != source['cache_sha256'][name]:
                raise ValueError('Evaluated cache metadata changed')
        dataset = load_from_disk(str(cache))
        rows = pd.read_parquet(design_path/f'rows_{regime}_{split}.parquet')
        metadata = dataset.select_columns(native_analysis.IDENTITY+native_analysis.LABELS).to_pandas()
        align_rows(rows,metadata)
        result[split] = dict(
            rows=rows,
            groups=pd.read_parquet(design_path/f'groups_{regime}_{split}.parquet'),
            cells=pd.read_parquet(design_path/f'weights_{split}.parquet'),
            # load_from_disk can derive a new fingerprint from saved formatting.
            # Native evaluation records that loaded value, not the state field.
            fingerprint=dataset._fingerprint)
    return result


def unit_map_for(refs):
    units = pd.concat([value['rows'][['assay_id','unit_id']] for value in refs.values()]).drop_duplicates()
    if units.assay_id.duplicated().any():
        raise ValueError('An assay maps to multiple indivisible units')
    if set(refs['val']['rows'].unit_id) & set(refs['test']['rows'].unit_id):
        raise ValueError('Validation and test must have disjoint indivisible units')
    return units.set_index('assay_id').unit_id.to_dict()


def calibration_and_shift(validation, test, units):
    temperature = fit_temperature(validation.binary_label, validation.activity_logits)
    calibration = dict(temperature=temperature,fit_partition='validation only',
        validation_logloss_before=binary_logloss(validation.binary_label,validation.activity_logits),
        validation_fit_logloss_after=binary_logloss(validation.binary_label,validation.activity_logits,temperature),
        test_logloss_before=binary_logloss(test.binary_label,test.activity_logits),
        test_logloss_after=binary_logloss(test.binary_label,test.activity_logits,temperature),
        interpretation='Positive temperature preserves logit ordering. Validation selected the model and fits temperature; test transfer is conditional on that selection, not independent calibration evidence.')
    return dict(calibration=calibration,bbse=bbse(validation,test,units))


def diagnose_checkpoint(evaluations, refs, *, regime, noise_rms, ranking_readout):
    """Compute the MD diagnostics from row-exact saved native predictions."""
    if regime not in ('A','B') or not math.isfinite(noise_rms) or noise_rms <= 0:
        raise ValueError('Expected A/B and a positive training-only replicate RMS')
    units = unit_map_for(refs)
    frames, partitions, tables = {}, {}, {}
    for split in ('val','test'):
        prefix = 'eval' if split == 'val' else 'test'
        manifest = Path(evaluations[split]['prediction_manifest'])
        candidate,record = read_native_predictions(manifest)
        if record['phase'] != 'fixed_checkpoint' or record['ranking_readout'] != ranking_readout:
            raise ValueError('Unexpected prediction phase or configured readout')
        rows,groups,cells = (refs[split][key] for key in ('rows','groups','cells'))
        locations = align_rows(rows,candidate)
        expected = rows.iloc[locations].reset_index(drop=True)
        frame,record = native_analysis.native_frame(manifest,expected,refs[split]['fingerprint'],prefix)
        frames[split] = frame
        metrics,group_tables = rescore(frame,units)
        for scorer,table in group_tables.items():
            eligible = table.original_ranking_eligible
            metrics[scorer].update(eligible_groups=int(eligible.sum()),
                eligible_defined_fraction=float(table.loc[eligible,'spearman'].notna().mean()) if eligible.any() else None,
                matched_eligible_ndcg={str(k):float(table.loc[eligible,f'ndcg_at_{k}_linear'].mean()) if eligible.any() else None for k in (1,3,5)})
            tables[f'{split}/groups_{scorer}'] = table
        headline = metrics['ranking_score']
        native_analysis.check_native_metric(headline['original_eligible_macro_spearman'],
            record['metrics'][prefix+'_macro_spearman'],'macro Spearman')
        native_analysis.check_native_metric(headline['eligible_defined_fraction'],
            record['metrics'][prefix+'_spearman_defined_fraction'],'defined fraction')
        if headline['eligible_groups'] != int(record['metrics'][prefix+'_spearman_num_groups']):
            raise ValueError('Native eligible cohort differs')
        noise,noise_tables = score_bands(frame,dict(training_replicate_rms=noise_rms,
            reference_margin=float(np.log10(3.))),SCORERS)
        tables.update({f'{split}/noise_{key.replace("/","_")}':table for key,table in noise_tables.items()})
        agreement = group_agreement(frame,units)
        tables[f'{split}/agreement'] = agreement
        ordered = frame.iloc[np.argsort(locations)].reset_index(drop=True)
        strata,standardized = {},{}
        for scorer in SCORERS:
            predictions = ordered.assign(ranking_score=ordered[scorer])
            strata[scorer],scored = similarity_composition.score_similarity(rows,groups,predictions)
            tables[f'{split}/composition_{scorer}'] = scored
            if regime == 'A':
                weights = scored.cell.map(cells.set_index('cell').A_group_weight).fillna(0).to_numpy()*scored.eligible.to_numpy()
                standardized[scorer] = similarity_composition.bootstrap_scores(scored,
                    dict(raw=scored.eligible.to_numpy(float),common=(weights>0).astype(float),standardized=weights),scored.unit_id.unique())
        partitions[split] = dict(rows=len(frame),native_metrics=record['metrics'],metrics=metrics,
            noise_bands=noise,agreement=summarize(agreement),similarity=strata,
            A_composition_standardization=standardized or None)
    result = dict(partitions=partitions,**calibration_and_shift(frames['val'],frames['test'],units),
        regime=regime,ranking_readout=ranking_readout,label_precision=POLICY,
        noise_thresholds=dict(training_replicate_rms=noise_rms,reference_margin=float(np.log10(3.))),
        noise_interpretation='Training-only raw-replicate RMS is a sensitivity band, not uncertainty of an aggregated median. Concordance divides by all pairs; ordering accuracy divides by label-comparable pairs.',
        unit_definition='assay' if regime=='A' else 'scaffold-assay component',
        uncertainty='Whole-unit bootstrap conditional on fixed fitted weights and assignments; not training-seed or realization uncertainty. Composition weights are treated as fixed.',
        scorer_interpretation='ranking_score is the configured deployed readout. In activity-logit or cosine runs it repeats that pathway; inactive or detached heads are not independent trained baselines.',
        composition_interpretation=similarity_composition.DESIGN['inference'])
    return finite_json(result),tables


def verify_artifacts(directory, result):
    for name,digest in result['artifacts_sha256'].items():
        if sha256(directory/name) != digest:
            raise ValueError('Completed analysis artifact changed')


def analyze_job(identity):
    import fcntl
    from reward_model.training.config import load_reward_training_config
    if Path(identity).name != identity:
        raise ValueError('Expected a registered job ID')
    directory = WORK/'postselection'/identity
    plan,evaluations = verify_completed_evaluations(directory)
    config = load_reward_training_config(plan['job']['config_path'])
    if config.model.ranking_min_pchembl_span != .5 or config.data.ranking_min_pchembl_span != .5:
        raise ValueError('Analysis requires the registered headline eligibility rule')
    design_path = WORK/'tier1/similarity_composition/design'
    refs = references(plan,design_path)
    primary = read(WORK/'controlled/plan.json')
    noise_path = WORK/'tier1/replicates/noise_summary.json'
    if sha256(noise_path) != primary['noise_summary_sha256']:
        raise ValueError('Training noise source changed')
    noise = read(noise_path)[plan['job']['regime']]['train']['noise_scale']['rms_replicate_difference']
    if noise != primary['training_noise_rms'][plan['job']['regime']]:
        raise ValueError('Training noise threshold changed')
    source = dict(evaluation_plan_sha256=sha256(directory/'plan.json'),
        evaluation_complete_sha256=sha256(directory/'complete.json'),
        design_sha256=sha256(design_path/'complete.json'),noise_summary_sha256=sha256(noise_path),
        software=software(),token_payloads_used=False,token_payloads_rehashed=False)
    output = WORK/'postselection_analysis'/identity/digest_json(source)[:16]
    output.mkdir(parents=True,exist_ok=True)
    with (output/'analysis.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        immutable(output/'analysis_plan.json',source)
        if (output/'complete.json').exists():
            result = read(output/'complete.json')
            if result['source'] != source:
                raise ValueError('Completed analysis source changed')
            verify_artifacts(output,result)
            return output/'complete.json'
        checkpoints = {}
        for checkpoint in plan['checkpoints']:
            destination = output/checkpoint['id']
            if (destination/'complete.json').exists():
                result = read(destination/'complete.json')
                if result['source'] != source or result['checkpoint'] != checkpoint:
                    raise ValueError('Checkpoint analysis source changed')
                verify_artifacts(destination,result)
            else:
                if destination.exists():
                    raise RuntimeError('Partial checkpoint analysis requires inspection')
                statistics,tables = diagnose_checkpoint(evaluations[checkpoint['id']],refs,
                    regime=plan['job']['regime'],noise_rms=noise,ranking_readout=config.model.ranking_readout)
                destination.mkdir()
                for name,table in tables.items():
                    path = destination/(name+'.parquet');path.parent.mkdir(exist_ok=True)
                    table.to_parquet(path,index=False)
                result = dict(source=source,checkpoint=checkpoint,statistics=statistics,
                    artifacts_sha256={str(p.relative_to(destination)):sha256(p) for p in sorted(destination.rglob('*.parquet'))})
                immutable(destination/'complete.json',result)
            checkpoints[checkpoint['id']] = result['statistics']
            print(json.dumps(dict(job=identity,checkpoint=checkpoint['id'],phase='checkpoint_analysis_complete')),flush=True)
        result = dict(source=source,job=plan['job'],rules=plan['rules'],checkpoints=checkpoints,
            seed_role=plan['seed_role'],selected_learning_rate=plan['selected_learning_rate'],
            selection_sha256=plan['selection_sha256'],phase='selected_prediction_analysis_complete',
            artifacts_sha256={str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*'))
                if p.is_file() and p.name not in ('analysis.lock','analysis_plan.json')},
            interpretation='Per-fit diagnostics only. Matched A/B comparisons, seed spread and alternative allocations remain separate; no diagnostic selects a model.')
        immutable(output/'complete.json',result)
        return output/'complete.json'


def compare_analyses(a_path, b_path, design_path):
    a_path,b_path,design_path = Path(a_path),Path(b_path),Path(design_path)
    a,b = read(a_path),read(b_path)
    if any(result['phase'] != 'selected_prediction_analysis_complete' for result in (a,b)):
        raise ValueError('Both selected-model analyses must be complete')
    if a['job']['regime']!='A' or b['job']['regime']!='B':
        raise ValueError('Expected A then B')
    keys = ('arm','seed','phase','learning_rate_multiplier','plan_sha256')
    if (any(a['job'][key]!=b['job'][key] for key in keys) or
        a['selection_sha256'] != b['selection_sha256'] or set(a['rules']) != set(b['rules'])):
        raise ValueError('A/B analyses are not matched registered comparisons')
    verify_artifacts(a_path.parent,a);verify_artifacts(b_path.parent,b)
    similarity_composition.verify_design(design_path)
    if any(result['source']['design_sha256'] != sha256(design_path/'complete.json') for result in (a,b)):
        raise ValueError('Matched analyses use a different scaffold design')
    results = {}
    for rule in a['rules']:
        results[rule] = {}
        for split in ('val','test'):
            cells = pd.read_parquet(design_path/f'weights_{split}.parquet')
            results[rule][split] = {}
            for scorer in SCORERS:
                tables = [pd.read_parquet(path.parent/result['rules'][rule]['checkpoint']/split/f'composition_{scorer}.parquet')
                    for path,result in ((a_path,a),(b_path,b))]
                results[rule][split][scorer] = similarity_composition.compare_composition(*tables,cells)
    return finite_json(dict(comparisons=results,job_A=a['job'],job_B=b['job'],
        analysis_sha256=dict(A=sha256(a_path),B=sha256(b_path)),
        design_sha256=sha256(design_path/'complete.json'),software=software(),
        interpretation='Matched seed/arm/LR; checkpoint rule applied within each split. Descriptive A-minus-B gaps have no cross-split confidence interval. Per-fit unit intervals do not substitute for seed or allocation spread.'))


def main():
    if ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Use CPU-only analysis in the authorized Atum project')
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action',required=True)
    actions.add_parser('analyze').add_argument('--job-id',required=True)
    pair = actions.add_parser('compare')
    pair.add_argument('--a-job-id',required=True);pair.add_argument('--b-job-id',required=True)
    args = parser.parse_args()
    if args.action == 'analyze':
        print(analyze_job(args.job_id))
    else:
        a,b = analyze_job(args.a_job_id),analyze_job(args.b_job_id)
        result = compare_analyses(a,b,WORK/'tier1/similarity_composition/design')
        output = WORK/'postselection_analysis/comparisons'/digest_json(result)[:16]/'complete.json'
        immutable(output,result);print(output)


if __name__ == '__main__':
    main()
