"""Evaluate the user-selected, already completed depth-3 LambdaMART fits.

This path never fits a tree or selects another configuration. The interrupted
five-setting search remains preserved and explicitly incomplete.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

if __package__:
    from .run import ROOT, WORK, sha256, write_json
    from .lambdamart import VERSION, pair_features
    from .baseline_common import load_partitions, RankingSelection, evaluate_predictions
else:
    from run import ROOT, WORK, sha256, write_json
    from lambdamart import VERSION, pair_features
    from baseline_common import load_partitions, RankingSelection, evaluate_predictions


def verify_fixed_trial(output, decision, regime):
    if (decision['selected_trial']!=0 or decision['max_depth']!=3 or
        decision['selected_setting']!=dict(max_depth=3,reg_lambda=1.,min_child_weight=1.) or
        not decision['remaining_search_cancelled'] or not decision['termination_verified']):
        raise ValueError('Expected the explicit user choice ending the search at depth 3')
    output = Path(output)
    contract = json.loads((output/'fit_contract.json').read_text())
    path = output/'trial00/complete.json'
    result = json.loads(path.read_text())
    pinned = decision['completed_depth3'][regime]
    if sha256(path)!=pinned['completion_sha256'] or sha256(output/'trial00/model.ubj')!=pinned['model_sha256']:
        raise ValueError('Completed depth-3 evidence changed after the user decision')
    if (result['parameters']!={**contract['base_parameters'],**decision['selected_setting']} or
        result['model_sha256']!=pinned['model_sha256']):
        raise ValueError('Completed trial is not the requested fixed configuration')
    history = json.loads((output/'trial00/validation_history.json').read_text())
    records = history['history']
    every,maximum,patience = (contract[k] for k in ('evaluate_every','max_rounds','patience_rounds'))
    rounds = result['rounds_trained']
    if rounds<=0 or rounds>maximum or rounds%every or [r['rounds'] for r in records]!=list(range(every,rounds+1,every)):
        raise ValueError('Incomplete validation history')
    incumbent = None
    for i,row in enumerate(records):
        score = row['macro_spearman']
        if score is not None:
            if not math.isfinite(score) or not -1<=score<=1:raise ValueError('Invalid validation score')
            if incumbent is None or score>incumbent['macro_spearman']:incumbent=row
        if incumbent and row['rounds']-incumbent['rounds']>=patience and i!=len(records)-1:
            raise ValueError('Trial exceeded its registered early-stopping rule')
    if incumbent is None or history['selected']!=incumbent or result['selected']!=incumbent:
        raise ValueError('Selected tree count is not the earliest validation maximum')
    if rounds!=maximum and rounds-incumbent['rounds']<patience:
        raise ValueError('Depth-3 trial did not finish its registered horizon')
    return contract,result


def validation_reproduction(observed,reference):
    # Fit-time validation uses a CUDA QuantileDMatrix; this pass uses chunked
    # CPU DMatrix prediction. Preserve and report small numerical differences.
    tolerance = 1e-6
    difference = observed['macro_spearman']-reference['macro_spearman']
    if not math.isfinite(difference) or abs(difference)>tolerance:
        raise ValueError('Fixed-model validation differs beyond numerical tolerance')
    if any(observed[k]!=reference[k] for k in ('eligible_groups','defined_groups')):
        raise ValueError('Fixed-model validation coverage changed')
    return dict(saved_fit_time_macro=reference['macro_spearman'],cpu_rescored_macro=observed['macro_spearman'],
        difference=difference,absolute_tolerance=tolerance,coverage_unchanged=True,
        interpretation='Same model weights and source feature bytes. Fit-time CUDA quantile-matrix validation versus chunked CPU dense-matrix prediction; report the measured numerical discrepancy, without changing checkpoint selection.')


def main(regime):
    import fcntl
    import xgboost as xgb
    if ROOT!=Path('/home/atabey/Prot2Mol') or xgb.__version__!=VERSION:
        raise RuntimeError('Use the authorized Atum project and pinned XGBoost')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Fixed tree prediction/metrics use CPU only')
    output = WORK/'tier2/lambdamart'/regime
    with (output/'fit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        decision_path = WORK/'tier2/lambdamart/user_decision.json'
        decision = json.loads(decision_path.read_text())
        contract,trial = verify_fixed_trial(output,decision,regime)
        if (output/'complete.json').exists():
            previous = json.loads((output/'complete.json').read_text())
            if previous.get('user_decision_sha256')!=sha256(decision_path):
                raise ValueError('Existing baseline uses another selection policy')
            for name,digest in previous['artifacts_sha256'].items():
                if sha256(output/name)!=digest:
                    raise ValueError('Finalized baseline evidence changed')
            return previous
        frames,hashes = load_partitions(regime)
        if hashes!=contract['data_sha256']:
            raise ValueError('Fixed baseline source partitions changed')
        feature_base = WORK/'tier2/frozen_features'
        manifest = json.loads((feature_base/'complete.json').read_text())
        if manifest!=contract['feature_provenance']:
            raise ValueError('Frozen feature provenance changed')
        maps,entities = {},{}
        for side in ('protein','molecule'):
            values,table = feature_base/f'{side}.npy',feature_base/f'{side}_entities.parquet'
            if sha256(values)!=manifest[side]['features_sha256'] or sha256(table)!=manifest[side]['entities_sha256']:
                raise ValueError('Frozen feature bytes or index changed')
            maps[side] = np.load(values,mmap_mode='r')
            entities[side] = pd.Index(pd.read_parquet(table,columns=['entity_id']).entity_id)
        target = output/'model.ubj'
        if target.exists() and sha256(target)!=trial['model_sha256']:
            raise ValueError('Existing final model differs from the fixed depth-3 choice')
        if not target.exists():shutil.copyfile(output/'trial00/model.ubj',target)
        model = xgb.Booster(model_file=target)
        model.set_param({'device':'cpu','nthread':8})
        if model.num_boosted_rounds()!=trial['selected']['rounds']:
            raise ValueError('Saved model has a different tree count')
        metrics = {}
        reproduction = None
        for part in ('val','test'):
            scores = np.empty(len(frames[part]))
            for start in range(0,len(scores),32768):
                current = frames[part].iloc[start:start+32768]
                batch = xgb.DMatrix(pair_features(current,maps,entities),nthread=8)
                scores[start:start+len(current)] = model.predict(batch)
            if part=='val':
                reproduced = RankingSelection(frames[part])(scores)
                reproduction = validation_reproduction(reproduced,trial['selected'])
            metrics[part] = evaluate_predictions(frames[part].assign(ranking_score=scores,classification_score=scores),regime,output/part)
            print(json.dumps(dict(regime=regime,partition=part,macro_spearman=metrics[part]['original_eligible_macro_spearman'])),flush=True)
        files = ['model.ubj',*[f'{part}/{name}' for part in ('val','test') for name in ('predictions.parquet','groups.parquet','metrics.json')]]
        completed = [dict(trial=int(p.parent.name.replace('trial','')),**json.loads(p.read_text())) for p in sorted(output.glob('trial*/complete.json'))]
        result = dict(**{k:v for k,v in contract.items() if k!='base_parameters'},
            selected=dict(trial=0,**trial),trials=completed,metrics=metrics,
            model_sha256=sha256(target),user_decision_sha256=sha256(decision_path),
            selection=decision['selection_policy'],grid_search_complete=False,
            validation_reproduction=reproduction,
            evaluated_settings=[decision['selected_setting']],no_new_training=True,
            inference_device='CPU',artifacts_sha256={name:sha256(output/name) for name in files},
            scope='Completed fixed depth-3 A/B baseline requested by the user. Tree count selected on each split validation. The larger search was cancelled and its partial artifacts retained; no claim of best-of-grid tuning.',
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        write_json(output/'complete.json',result)
        return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regime',choices=['A','B'],required=True)
    main(parser.parse_args().regime)
