"""Independent primary metric and native-cosine arithmetic verification."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from .study import OUT,read_json,write_json,verify_sample
except ImportError:
    from study import OUT,read_json,write_json,verify_sample


def verify():
    frame=verify_sample();val=frame[frame.partition=='val'].reset_index(drop=True)
    path=OUT/'analysis/pretrained_content_mean_layer12'
    result=read_json(path/'results.json');saved=np.load(path/'predictions.npz')
    checks={}
    for method,record in result['metrics'].items():
        correlations=[]
        for indices in val.groupby('g',sort=True).indices.values():
            labels=val.s.to_numpy()[indices];scores=saved[method][indices]
            if len(indices)<3 or np.ptp(labels)<.5 or not np.isfinite(scores).all() or np.ptp(scores)==0:continue
            correlations.append(float(spearmanr(labels,scores).statistic))
        macro=float(np.mean(correlations)) if correlations else None
        if len(correlations)!=record['defined_groups']:raise ValueError('Coverage disagreement: '+method)
        error=abs(macro-record['macro_spearman']) if macro is not None else 0.
        if error>1e-12:raise ValueError('Independent Spearman disagreement: '+method)
        checks[method]=dict(defined_groups=len(correlations),macro_spearman=macro,absolute_difference=error)
    arrays={};lookup={}
    for side in ('query','candidate'):
        folder=OUT/'embeddings'/(side+'_pretrained')
        ids=read_json(folder/'ids.json');lookup[side]={v:i for i,v in enumerate(ids)}
        arrays[side]=np.load(folder/'content_mean.npy',mmap_mode='r')[12]
    x=np.asarray(arrays['query'][[lookup['query'][q] for q in val.x]],dtype=np.float64)
    y=np.asarray(arrays['candidate'][[lookup['candidate'][q] for q in val.y]],dtype=np.float64)
    direct=np.sum(x*y,axis=1)/np.sqrt(np.sum(x*x,axis=1)*np.sum(y*y,axis=1))
    cosine_error=float(np.max(np.abs(direct-saved['direct_cosine'])))
    if cosine_error>3e-6:raise ValueError('Stored direct cosine differs from raw-vector calculation')
    high_precision=[]
    for indices in val.groupby('g',sort=True).indices.values():
        labels=val.s.to_numpy()[indices]
        if np.ptp(labels)>=.5 and np.ptp(direct[indices])>0:
            high_precision.append(float(spearmanr(labels,direct[indices]).statistic))
    write_json(OUT/'analysis/independent_verification.json',dict(primary_metrics=checks,
        native_cosine_float64_max_abs_difference=cosine_error,
        native_cosine_float64_macro_spearman=float(np.mean(high_precision)),
        note='Float64 arithmetic is a diagnostic recomputation; reported predictions retain the fixed float32 path.'))
    print('Independent primary metric and raw-vector cosine verification passed',flush=True)


if __name__=='__main__':verify()
