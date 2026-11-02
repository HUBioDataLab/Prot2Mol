"""Q04: full-horizon descriptive comparison of the two completed A LR fits.

Reuse independently audited native metrics; do not create a fit, choose the
study LR, read test predictions, or treat epochs as independent replicates.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256
    from . import native_analysis as native, verify_native_analysis as verifier, control_driver as driver
    from . import reference_composition as reference
else:
    from run import ROOT, WORK, sha256
    import native_analysis as native, verify_native_analysis as verifier, control_driver as driver
    import reference_composition as reference

read, immutable, bind = reference.read, reference.immutable, reference.bind
JOBS = ('full__A__seed42__lr1p0', 'full__A__seed42__lr0p3')
PINS = {
    JOBS[0]: ('7cc18fafad3e454d1a44fd1c63a7f33f8aff2faba8f8a38921c3eb6a4c8924ff',
              '90faa2eab5acd5642247e356fcbec01bea872bac84797825487cf835f8bfbb80'),
    JOBS[1]: ('8ee661e8f322a2d56750c32fa3783a4441765a370be10ac1dab72b7520c56595',
              'a1f0cafdb350d73ff5173233a8ddd9c4ec4d7b0bd860845743662bd97eb7fb80'),
}
RATES = ('learning_rate', 'projection_learning_rate', 'protein_encoder_learning_rate', 'molecule_encoder_learning_rate')


def check_configurations(first, second):
    first, second = copy.deepcopy(first), copy.deepcopy(second)
    changes = {}
    for config in (first, second):
        config['training'].pop('output_dir')
    for name in RATES:
        a, b = first['training'].pop(name), second['training'].pop(name)
        if a <= 0 or abs(b/a-.3) > 1e-12:
            raise ValueError('Expected exactly the registered LR0.3 change')
        changes[name] = [a, b]
    if first != second:
        raise ValueError('Configurations differ beyond output directory and the four learning rates')
    return changes


def load(identity, evidence):
    job, _, registration = native.registration(identity)
    if identity not in JOBS or not driver.completed_job(job):
        raise ValueError('Expected a completed registered A fit')
    directory = WORK/'tier1/native_trajectories'/identity/native.digest_json(native.software())[:16]
    tp, vp = directory/'trajectory_through_epoch15.json', directory/'verification_through_epoch15.json'
    bind(tp, evidence, PINS[identity][0]); bind(vp, evidence, PINS[identity][1])
    trajectory, verified = read(tp), read(vp)
    bind(directory/'analysis_plan.json', evidence, trajectory['analysis_plan_sha256'])
    plan = read(directory/'analysis_plan.json')
    bind(Path(verifier.__file__), evidence, verified['verifier_sha256'])
    if (trajectory['job'] != job or trajectory['epochs_analyzed'] != list(range(1, 16)) or
        trajectory['horizon'] != 15 or not trajectory['complete_horizon_observed'] or trajectory['test_predictions_read'] or
        verified['test_predictions_read'] or not verified['source_and_artifact_hashes_verified'] or
        verified['trajectory_sha256'] != sha256(tp) or verified['analysis_plan_sha256'] != trajectory['analysis_plan_sha256'] or
        verified['verifier_sha256'] != sha256(Path(verifier.__file__)) or plan['software'] != native.software() or
        any(plan[k] != value for k, value in registration.items())):
        raise ValueError('Completed independent native evidence differs')
    checks = {(c['epoch'], c['prefix']): c for c in verified['checks']}
    if len(verified['checks']) != 30 or set(checks) != {(e, p) for e in range(1, 16) for p in ('eval', 'train_eval')}:
        raise ValueError('Expected thirty unique independently verified snapshots')
    run = Path(job['output_dir'])
    bind(WORK/'controlled/plan.json', evidence, job['plan_sha256'])
    bind(WORK/'controlled/jobs'/f'{identity}.json', evidence, registration['job_metadata_sha256'])
    bind(job['config_path'], evidence, job['config_sha256'])
    bind(run/'launch_provenance.json', evidence, registration['launch_sha256'])
    bind(run/'controlled_training_complete.json', evidence)
    for name, digest in read(run/'controlled_training_complete.json')['evidence_sha256'].items():
        bind(run/name, evidence, digest)
    bind(run/'runtime_qualification.json', evidence); qualification = read(run/'runtime_qualification.json')
    for name, digest in {**plan['software']['source_sha256'], **qualification['sources']}.items():
        bind(ROOT/name, evidence, digest)
    bind(directory/'references/complete.json', evidence, plan['reference_sha256']); refs = read(directory/'references/complete.json')
    frames = {}
    for prefix, entry in refs['references'].items():
        bind(directory/'references'/f'{prefix}.parquet', evidence, entry['sha256'])
        frames[prefix] = pd.read_parquet(directory/'references'/f'{prefix}.parquet')
    for record in refs['sources'].values():
        bind(record['path'], evidence, record['sha256'])
        for name, digest in record['cache_metadata_sha256'].items(): bind(Path(record['cache'])/name, evidence, digest)
    if [r['epoch'] for r in trajectory['trajectory']] != list(range(1, 16)):
        raise ValueError('Native trajectory order differs')
    for row in trajectory['trajectory']:
        for prefix in ('eval', 'train_eval'):
            snapshot = row[prefix]; source = snapshot['source']
            manifest_path = run/'prediction_exports/trajectory'/f"{prefix}_step{row['step']:08d}.json"
            if (snapshot['epoch'] != row['epoch'] or snapshot['step'] != row['step'] or snapshot['prefix'] != prefix or
                snapshot['ranking_readout'] != 'ranking_head' or source['provenance'] != plan or source['manifest'] != str(manifest_path)):
                raise ValueError('Native snapshot identity or provenance differs')
            bind(manifest_path, evidence, source['manifest_sha256']); manifest = read(manifest_path)
            if (Path(manifest['predictions_file']).name != manifest['predictions_file'] or
                manifest['predictions_sha256'] != source['payload_sha256'] or manifest['metrics'] != snapshot['native_metrics']):
                raise ValueError('Native payload metadata differs')
            bind(manifest_path.parent/manifest['predictions_file'], evidence, source['payload_sha256'])
            for name, digest in snapshot['artifacts_sha256'].items(): bind(directory/manifest_path.stem/name, evidence, digest)
            for scorer, value in checks[row['epoch'], prefix]['macros'].items():
                native.check_native_metric(snapshot['metrics'][scorer]['original_eligible_macro_spearman'], value, 'verified scorer macro')
    return dict(trajectory=trajectory, qualification=qualification, references=refs, frames=frames,
        config=read(run/'launch_provenance.json')['config'])


def point(snapshot):
    metrics = snapshot['metrics']['ranking_score']; prefix = snapshot['prefix']
    return dict(rows=snapshot['rows'], eligible_groups=metrics['eligible_groups'],
        eligible_defined_fraction=metrics['eligible_defined_fraction'], macro_spearman=metrics['original_eligible_macro_spearman'],
        ranking_ndcg_eligible=metrics['matched_eligible_ndcg'],
        native_losses={name: snapshot['native_metrics'][prefix+'_'+name] for name in ('loss', 'classification_loss', 'ranking_loss', 'contrastive_loss')})


def compare(first, second):
    rows = []
    for a, b in zip(first, second, strict=True):
        if (a['epoch'], a['step']) != (b['epoch'], b['step']):
            raise ValueError('Matched epochs have different optimizer steps')
        row = dict(epoch=a['epoch'], step=a['step'], reference={}, lower_lr={}, difference={})
        for prefix in ('eval', 'train_eval'):
            x, y = point(a[prefix]), point(b[prefix])
            if any(x[k] != y[k] for k in ('rows', 'eligible_groups')) or x['eligible_defined_fraction'] != 1 or y['eligible_defined_fraction'] != 1:
                raise ValueError('Matched headline cohorts or defined coverage differ')
            row['reference'][prefix], row['lower_lr'][prefix] = x, y
            row['difference'][prefix] = dict(macro_spearman=y['macro_spearman']-x['macro_spearman'],
                native_losses={k:y['native_losses'][k]-x['native_losses'][k] for k in x['native_losses']},
                ranking_ndcg_eligible={k:y['ranking_ndcg_eligible'][k]-x['ranking_ndcg_eligible'][k] for k in x['ranking_ndcg_eligible']})
        rows.append(row)
    if [r['epoch'] for r in rows] != list(range(1, 16)):
        raise ValueError('Comparison requires exactly fifteen matched epochs')
    peaks = {name: max(rows, key=lambda r:r[name]['eval']['macro_spearman'])['epoch'] for name in ('reference', 'lower_lr')}
    return dict(rows=rows, descriptive_validation_peak_epochs=peaks,
        lower_lr_validation_below_reference_at_all_epochs=all(r['difference']['eval']['macro_spearman'] < 0 for r in rows))


def runtime_guard():
    if ROOT != Path('/home/atabey/Prot2Mol') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Use CPU-only Atum Prot2Mol')


def run():
    runtime_guard()
    output = WORK/'tier1/native_lr_comparison/full_A_seed42_lr0p3_vs_lr1p0/complete.json'
    if output.exists(): raise RuntimeError('Comparison already exists; inspect without overwriting')
    evidence = {}; first, second = (load(identity, evidence) for identity in JOBS)
    if first['qualification'] != second['qualification'] or first['references']['sources'] != second['references']['sources'] or first['references']['probe'] != second['references']['probe']:
        raise ValueError('Training qualification, source data or fixed probe differs')
    for prefix in ('eval', 'train_eval'):
        pd.testing.assert_frame_equal(first['frames'][prefix], second['frames'][prefix], check_exact=True, check_dtype=False)
    changes = check_configurations(first['config'], second['config'])
    result = dict(reference_job=JOBS[0], candidate_job=JOBS[1], configuration_differences=changes,
        identical_reference_rows_and_labels=True, **compare(first['trajectory']['trajectory'], second['trajectory']['trajectory']),
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha256(Path(__file__)),
                       str(Path(reference.__file__).relative_to(ROOT)):sha256(Path(reference.__file__))},
        input_sha256=evidence, new_fits_created=0, model_inference=False, test_predictions_read=False,
        learning_rate_selected=False, independent_seed_variance_estimated=False,
        interpretation='Descriptive validation and fixed training-probe comparison of two existing A seed42 fits over the same fifteen-epoch budget. Epochs are not independent replicates. No common LR selection, test performance, seed/split variance, causal mechanism or benefit of longer training is established. NDCG/native losses reuse existing audits; their numerical values are not independently reconstructed by the rank verifier.')
    for path, digest in evidence.items(): bind(path, {}, digest)
    immutable(output, result)
    return output


if __name__ == '__main__': print(run(), flush=True)
