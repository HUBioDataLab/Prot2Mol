"""Evidence table for Q05-Q08/Q10 from the complete initial B trajectory."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .label_precision import POLICY
else:
    from run import ROOT, WORK, sha256, write_json
    from label_precision import POLICY


def main():
    if ROOT != Path('/home/atabey/Prot2Mol'):
        raise RuntimeError('Use the authorized Atum project')
    base = WORK/'tier1'
    sources = {}

    def read(path):
        sources[str(path.relative_to(WORK))] = sha256(path)
        return json.loads(path.read_text())

    parity = read(base/'label_precision_complete.json')
    if parity['label_precision'] != POLICY or len(parity['native_parity_checks']) != 93:
        raise ValueError('All saved-scorer comparisons must pass native-label parity first')
    states = sorted((WORK/'baseline_scaffold_seed42_15ep').glob('checkpoint-*/trainer_state.json'),
                    key=lambda p:int(p.parent.name.split('-')[-1]))
    state = read(states[-1])
    native = [row for row in state['log_history'] if 'eval_macro_spearman' in row]
    if [row['epoch'] for row in native] != list(range(1,16)):
        raise ValueError('Need the entire 15-epoch native history')
    records, pools = [], {'train':set(), 'val':set()}
    for row in native:
        epoch = int(row['epoch'])
        validation = read(base/'metrics_float32'/f'epoch{epoch:02d}_val.json')
        training = read(base/'metrics_float32'/f'epoch{epoch:02d}_train_probe.json')
        groups_path = base/'metrics_float32'/f'groups_epoch{epoch:02d}_val_ranking_score.parquet'
        sources[str(groups_path.relative_to(WORK))] = sha256(groups_path)
        groups = pd.read_parquet(groups_path)
        eligible = groups[groups.original_ranking_eligible]
        if len(eligible) != row['eval_spearman_num_groups']:
            raise ValueError('Top-metric cohort differs from native headline eligibility')
        item = dict(epoch=epoch, step=row['step'], native_validation={k:v for k,v in row.items() if k.startswith('eval_')},
            train_probe_macro=training['metrics']['ranking_score']['original_eligible_macro_spearman'],
            exported_validation_macro=validation['metrics']['ranking_score']['original_eligible_macro_spearman'],
            ranking_ndcg={str(k):validation['metrics']['ranking_score'][f'macro_ndcg_at_{k}_linear'] for k in (1,3,5)},
            ranking_ndcg_eligible={str(k):float(eligible[f'ndcg_at_{k}_linear'].mean()) for k in (1,3,5)},
            ranking_ndcg_eligible_groups=len(eligible),
            calibration=validation['calibration'])
        for role, suffix in [('train','train_probe'),('val','val')]:
            contrastive = read(base/'contrastive'/f'epoch{epoch:02d}_{suffix}.json')
            pools[role].add(contrastive['candidate_pool_sha256'])
            item[f'{role}_contrastive'] = dict(scale=contrastive['score_scale'], losses=contrastive['losses'],
                                              positive_alignment=contrastive['positive_alignment'])
        records.append(item)
    if any(len(value) != 1 for value in pools.values()):
        raise ValueError('Contrastive candidate identities changed across epochs')
    minima = {key:min(records, key=lambda r:r['native_validation'][key])['epoch'] for key in
              ['eval_loss','eval_ranking_loss','eval_classification_loss','eval_contrastive_loss']}
    first_increases = {key:next(([a['epoch'],b['epoch']] for a,b in zip(records,records[1:])
        if b['native_validation'][key] > a['native_validation'][key]),None) for key in minima}
    maxima = dict(macro_spearman=max(records,key=lambda r:r['native_validation']['eval_macro_spearman'])['epoch'],
                 roc_auc=max(records,key=lambda r:r['native_validation']['eval_roc_auc'])['epoch'],
                 **{f'ndcg_at_{k}':max(records,key=lambda r:r['ranking_ndcg'][str(k)])['epoch'] for k in (1,3,5)},
                 **{f'ndcg_at_{k}_same_eligible':max(records,key=lambda r:r['ranking_ndcg_eligible'][str(k)])['epoch'] for k in (1,3,5)})
    selected = records[maxima['macro_spearman']-1]
    last = records[-1]
    write_json(base/'baseline_trajectory_summary_v2.json', dict(format_version=2, questions=['Q05','Q06','Q07','Q08','Q10'],
        scope='Initial B seed42 only; matched A and controlled multi-seed comparisons pending',
        native_loss_minimum_epochs=minima, native_first_adjacent_increases=first_increases,
        validation_metric_maximum_epochs=maxima,
        post_macro_peak_ndcg_changes={str(k):last['ranking_ndcg'][str(k)]-selected['ranking_ndcg'][str(k)] for k in (1,3,5)},
        post_macro_peak_ndcg_same_eligible_changes={str(k):last['ranking_ndcg_eligible'][str(k)]-selected['ranking_ndcg_eligible'][str(k)] for k in (1,3,5)},
        candidate_pools={key:next(iter(value)) for key,value in pools.items()},
        rows=records, sources_sha256=sources, code_sha256=sha256(Path(__file__)),
        interpretation='Descriptive trajectories and earliest extrema of complete saved validation history. No new test selection. NDCG uses nonnegative within-assay linear gains. Report both all nonconstant groups and the same eligible cohort as headline Spearman; differing cohort definitions must not be mistaken for objective conflict. Fixed-scale rescoring does not establish outcomes of fixed-temperature retraining. The fixed training probe is descriptive, not a full training-set evaluation.'))
    print(json.dumps(dict(loss_minima=minima, metric_maxima=maxima, first=records[0], macro_selected=selected, last=last)),flush=True)


if __name__ == '__main__':
    main()
