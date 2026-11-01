"""Re-score saved outputs using the Trainer's float32 label values, without inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256, write_json
else:
    from run import ROOT, WORK, sha256, write_json

POLICY = 'float32 tensor values promoted to float64, matching Trainer Python-float metric arithmetic'


def trainer_label_frame(frame):
    """Round labels exactly as the collator does; preserve scores and binary labels."""
    values = frame.pchembl_value.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Expected finite pChEMBL labels')
    return frame.assign(pchembl_value=values.astype(np.float32).astype(np.float64))


def read_predictions(source, trainer_labels=False):
    frame = pd.read_parquet(source)
    identity = dict(predictions_sha256=sha256(source),
                    label_precision=POLICY if trainer_labels else 'original exported label precision')
    return (trainer_label_frame(frame) if trainer_labels else frame), identity


def verify():
    from reward_model.training.evaluation import _compute_groupwise_spearman_with_records
    base = WORK/'tier1'
    exports = sorted((base/'predictions').glob('*/complete.json'))
    if len(exports) != 31:
        raise ValueError('Expected all 31 initial baseline exports')
    stages = ['metrics_float32', 'noise_bands_float32', 'scorer_agreement_float32']
    for stage in stages:
        if not (base/stage/'complete.json').exists():
            raise ValueError(f'Analysis stage is not complete: {stage}')
    checks = []
    for marker in exports:
        info = json.loads(marker.read_text())
        name = info['name']
        frame, identity = read_predictions(marker.parent/'predictions.parquet', True)
        if len(frame) != info['rows'] or not np.array_equal(frame.row_index, np.arange(len(frame))):
            raise ValueError('Source export incomplete or misaligned')
        for stage in stages:
            record = json.loads((base/stage/f'{name}.json').read_text())
            if any(record.get(key) != value for key, value in identity.items()):
                raise ValueError(f'Analysis provenance mismatch: {name}/{stage}')
            if record['checkpoint_sha256'] != info['checkpoint_sha256']:
                raise ValueError('Analysis checkpoint mismatch')
        metric_record = json.loads((base/'metrics_float32'/f'{name}.json').read_text())
        for scorer in ['ranking_score', 'activity_logits', 'cosine_similarity']:
            native, native_rows = _compute_groupwise_spearman_with_records(
                group_ids=frame.group_id.tolist(), ranking_scores=frame[scorer].tolist(),
                pchembl_values=frame.pchembl_value.tolist(), min_group_size=3, min_pchembl_span=.5)
            table = pd.read_parquet(base/'metrics_float32'/f'groups_{name}_{scorer}.parquet')
            eligible = table[table.original_ranking_eligible].sort_values('group_id')
            native_table = pd.DataFrame(native_rows).sort_values('group_id')
            if eligible.group_id.tolist() != native_table.group_id.tolist():
                raise ValueError('Audit and native eligible groups differ')
            np.testing.assert_allclose(eligible.spearman, native_table.spearman, atol=1e-14, rtol=0, equal_nan=True)
            reported = metric_record['metrics'][scorer]['original_eligible_macro_spearman']
            np.testing.assert_allclose(reported, native['eval_macro_spearman'], atol=1e-14, rtol=0)
            checks.append(dict(name=name, scorer=scorer, macro_spearman=reported,
                               eligible_groups=len(eligible), defined_groups=int(eligible.spearman.notna().sum())))
        print(f'Native float32-label parity verified: {name}', flush=True)
    native_summary = WORK/'baseline_scaffold_seed42_15ep/experiment_summary.json'
    original = json.loads(native_summary.read_text())['test_metrics']['test_macro_spearman']
    exported = next(row['macro_spearman'] for row in checks if row['name'] == 'best_test' and row['scorer'] == 'ranking_score')
    np.testing.assert_allclose(exported, original, atol=1e-14, rtol=0)
    write_json(base/'label_precision_complete.json', dict(label_precision=POLICY, exports=31,
        native_parity_checks=checks, native_summary_sha256=sha256(native_summary),
        selected_test_macro=exported, selected_test_native_macro=original,
        original_reports='Preserved in metrics/, noise_bands/, scorer_agreement/; they used raw exported labels',
        corrected_reports=stages, checkpoint_selection_changed=False,
        scope='Fixed saved predictions; no training or inference. Agreement with native arithmetic on identical saved scores is distinct from reproducing every historical inference batch.',
        stage_completion_sha256={stage:sha256(base/stage/'complete.json') for stage in stages}))


def main(stage):
    if ROOT != Path('/home/atabey/Prot2Mol'):
        raise RuntimeError('Run production analysis only in authorized Atum project')
    if stage == 'verify':
        verify()
    elif stage == 'metrics':
        if __package__:
            from .analyze_predictions import analyze_all
        else:
            from analyze_predictions import analyze_all
        analyze_all(trainer_labels=True)
    elif stage == 'noise':
        if __package__:
            from .noise_band_audit import main as analyze
        else:
            from noise_band_audit import main as analyze
        analyze(trainer_labels=True)
    else:
        if __package__:
            from .scorer_agreement import main as analyze
        else:
            from scorer_agreement import main as analyze
        analyze(trainer_labels=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['metrics', 'noise', 'agreement', 'verify'], required=True)
    main(parser.parse_args().stage)
