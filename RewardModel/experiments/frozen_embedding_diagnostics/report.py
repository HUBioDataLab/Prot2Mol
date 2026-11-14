"""Complete the fixed inferential checks and write an anonymous evidence bundle."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import time
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import rankdata

try:
    from .study import OUT,ROOT,SEED,GPUS,runtime,read_json,write_json,digest,verify_sample
    from .analysis import Evaluation,unit_vectors
except ImportError:
    from study import OUT,ROOT,SEED,GPUS,runtime,read_json,write_json,digest,verify_sample
    from analysis import Evaluation,unit_vectors


def conditional_label_null(evaluation,predictions,draws=999):
    """Exact group membership is retained; one shared shuffle per group/draw.

    This tests frozen readouts against held-out label exchangeability. It is
    separate from the all-partition shuffled-label diagnostic in analysis.py.
    """
    methods=list(predictions)
    normalized=[];denominators=[];observed=[]
    for method in methods:
        raw=predictions[method][evaluation.indices]
        valid=np.isfinite(raw).all(1)&(np.ptp(raw,axis=1)>0)&evaluation.eligible
        ranks=rankdata(raw,axis=1);ranks-=ranks.mean(1,keepdims=True)
        ranks/=np.maximum(np.linalg.norm(ranks,axis=1,keepdims=True),1e-12)
        ranks[~valid]=0
        normalized.append(ranks.reshape(-1));denominators.append(max(1,valid.sum()))
        observed.append(float((ranks*evaluation.label_ranks).sum()/max(1,valid.sum())))
    matrix=np.array(normalized,dtype=np.float64)
    rng=np.random.default_rng(SEED)
    null=[]
    for start in range(0,draws,50):
        order=np.argsort(rng.random((min(50,draws-start),*evaluation.label_ranks.shape)),axis=2)
        shuffled=np.take_along_axis(np.broadcast_to(evaluation.label_ranks,order.shape),order,axis=2)
        null.extend((shuffled.reshape(len(order),-1)@matrix.T/np.array(denominators)).tolist())
    null=np.asarray(null);maxima=null.max(axis=1)
    return {m:dict(observed=observed[i],p_one_sided=float((1+(null[:,i]>=observed[i]-1e-12).sum())/(draws+1)),
        p_max_statistic_family=float((1+(maxima>=observed[i]-1e-12).sum())/(draws+1)),
        null_interval95=np.quantile(null[:,i],[.025,.975]).tolist(),draws=draws,
        family_size=len(methods),interpretation='Conditional validation-label exchangeability for these fixed readouts; dependence outside groups is not modeled by this shuffle') for i,m in enumerate(methods)}


def coordinate_null(frame,evaluation,draws=999):
    """One globally consistent candidate-coordinate permutation per draw."""
    import torch
    qdir=OUT/'embeddings/query_pretrained';ydir=OUT/'embeddings/candidate_pretrained'
    qids=read_json(qdir/'ids.json');yids=read_json(ydir/'ids.json')
    qlookup={v:i for i,v in enumerate(qids)};ylookup={v:i for i,v in enumerate(yids)}
    q=unit_vectors(np.load(qdir/'content_mean.npy',mmap_mode='r')[12])
    y=unit_vectors(np.load(ydir/'content_mean.npy',mmap_mode='r')[12])
    xrows=torch.as_tensor(q[[qlookup[x] for x in frame.x]],device='cuda:0')
    yrows=torch.as_tensor(y[[ylookup[x] for x in frame.y]],device='cuda:0')
    rng=np.random.default_rng(SEED);null=[]
    with torch.inference_mode():
        for start in range(0,draws,16):
            orders=torch.as_tensor(np.array([rng.permutation(768) for _ in range(min(16,draws-start))]),device='cuda:0')
            # Batched reduction avoids modifying either native representation.
            scores=(yrows[:,orders]*xrows[:,None,:]).sum(-1).T.cpu().numpy()
            null.extend([float(np.nanmean(evaluation.group_scores(p))) for p in scores])
    observed=float(np.nanmean(evaluation.group_scores((xrows*yrows).sum(-1).cpu().numpy())))
    return dict(observed=observed,draws=draws,seed=SEED,p_one_sided=(1+sum(x>=observed for x in null))/(draws+1),
        null_mean=float(np.mean(null)),null_interval95=np.quantile(null,[.025,.975]).tolist(),
        interpretation='Native-axis specificity relative to global candidate-coordinate relabeling, not a general test that representations contain no predictive information')


def holm(values):
    ids=sorted(values,key=values.get);adjusted={};previous=0.
    for i,key in enumerate(ids):
        value=min(1.,max(previous,values[key]*(len(ids)-i)))
        adjusted[key]=value;previous=value
    return adjusted


def fmt(value,digits=4):
    return 'undefined' if value is None or (isinstance(value,float) and not np.isfinite(value)) else f'{value:.{digits}f}'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def report():
    runtime()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in GPUS.values():
        raise RuntimeError('The coordinate control requires an authorized GPU')
    frame=verify_sample();source=OUT/'analysis'
    completed=read_json(source/'complete.json')
    if completed['cases']!=52 or completed['fits']!=0:raise ValueError('Incomplete fixed grid')
    val=frame[frame.partition=='val'].reset_index(drop=True);train=frame[frame.partition=='train'].reset_index(drop=True)
    evaluation=Evaluation(val)
    primary=source/'pretrained_content_mean_layer12'
    result=read_json(primary/'results.json')
    predfile=np.load(primary/'predictions.npz')
    predictions={m:predfile[m] for m in result['metrics']}
    destination=OUT/'shareable';destination.mkdir(exist_ok=True)
    if __package__:
        from .verify_outputs import verify
    else:
        from verify_outputs import verify
    verify()
    shutil.copyfile(source/'independent_verification.json',destination/'independent_verification.json')
    if (source/'inferential_checks.json').exists():
        checks=read_json(source/'inferential_checks.json')
    else:
        checks=dict(label_null_primary=conditional_label_null(evaluation,predictions),coordinate_null_primary=coordinate_null(val,evaluation))
        layer_predictions={}
        for case in sorted(source.glob('pretrained_*_layer*')):
            layer_predictions[case.name]=np.load(case/'predictions.npz')['direct_cosine']
        checks['direct_cosine_layer_family']=conditional_label_null(evaluation,layer_predictions)
        write_json(source/'inferential_checks.json',checks)
    all_readouts=pd.read_parquet(source/'all_readouts.parquet')
    all_readouts.to_csv(destination/'all_readouts.csv',index=False,lineterminator='\n')
    results={p.parent.name:read_json(p) for p in sorted(source.glob('*/results.json'))}
    pvalues={case+'__'+axis:r['p_two_sided'] for case,value in results.items() for axis,r in value['second_order'].items() if r['status']=='measured'}
    adjusted=holm(pvalues)
    for case,value in results.items():
        for axis,r in value['second_order'].items():
            if r['status']=='measured':r['p_holm_across_all_cases']=adjusted[case+'__'+axis]
    write_json(destination/'all_measurements.json',results)
    write_json(destination/'inferential_checks.json',checks)
    summary=read_json(OUT/'sample_summary.json');write_json(destination/'sample_summary.json',summary)
    frame.to_parquet(destination/'anonymous_sample.parquet',index=False)
    shutil.copyfile(source/'profile_overlap.json',destination/'profile_overlap.json')
    swap=read_json(source/'class_swap_coverage.json');write_json(destination/'class_swap_coverage.json',swap)
    for name in ('distance_curves.parquet','class_swap_curves.parquet','label_shuffle_curves.parquet','groups.parquet'):
        shutil.copyfile(primary/name,destination/('primary_'+name))
    # Every curve and group statistic is retained; no original identity map or inputs.
    curves=[];prediction_tables=[]
    for case in sorted(source.glob('*_layer*')):
        stored=np.load(case/'predictions.npz')
        predictions_frame=pd.DataFrame({k:stored[k] for k in stored.files})
        predictions_frame['validation_row']=np.arange(len(val))
        predictions_frame['case']=case.name
        prediction_tables.append(predictions_frame)
        for mode,file in [('native','distance_curves.parquet'),('class_swap','class_swap_curves.parquet'),('label_shuffle','label_shuffle_curves.parquet')]:
            part=pd.read_parquet(case/file);part['case']=case.name;part['control']=mode;curves.append(part)
    pd.concat(curves,ignore_index=True).to_parquet(destination/'all_distance_curves.parquet',index=False)
    pd.concat(prediction_tables,ignore_index=True).to_parquet(destination/'all_validation_predictions.parquet',index=False)
    groups=pd.read_parquet(primary/'groups.parquet')
    random=pd.read_parquet(source/'random_content_mean_layer12/groups.parquet')
    pretraining={m:dict(common_groups=int((groups[m].notna()&random[m].notna()).sum()),difference=float((groups[m]-random[m]).mean()) if (groups[m].notna()&random[m].notna()).any() else None,
        interval95=evaluation.interval((groups[m]-random[m]).to_numpy())) for m in predictions}
    write_json(destination/'pretrained_minus_random.json',pretraining)
    support=train.groupby('x').g.nunique()
    strata=[]
    group_distance=pd.Series(predfile['distance_same'],index=val.index).groupby(val.g).mean().reindex(groups.g).to_numpy()
    cut=np.quantile(group_distance,[.25,.5,.75])
    distance_bins=np.searchsorted(cut,group_distance,side='right')
    for label,mask in [(f'training_groups_{n}',groups.x.map(support).to_numpy()==n) for n in sorted(support.unique())]+[(f'neighbour_distance_quartile_{n+1}',distance_bins==n) for n in range(4)]:
        for method in predictions:
            values=groups[method].to_numpy().copy();values[~mask]=np.nan
            strata.append(dict(subset=label,method=method,groups=int(mask.sum()),defined_groups=int(np.isfinite(values).sum()),macro_spearman=float(np.nanmean(values)) if np.isfinite(values).any() else None,interval95=evaluation.interval(values)))
    write_json(destination/'support_and_distance_strata.json',strata)
    extraction={}
    for path in (OUT/'embeddings').glob('*/complete.json'):
        r=read_json(path)
        if r.get('input_contract_version')!=2 or (r['side']=='query' and r['input_audit']['unknown_tokens']!=0):
            raise ValueError('Only input-validated corrected exports may be reported')
        extraction[path.parent.name]={k:r[k] for k in ['shape','parameters_unchanged','optimizer_created','repeat_forward_max_abs_difference','truncated_inputs','input_tokens_quantiles','max_length','input_audit','prefix_tokens','input_contract_version']}
    write_json(destination/'extraction_checks.json',extraction)
    create_plots(source,destination,all_readouts,predfile)
    rows=[]
    random_results=results['random_content_mean_layer12']['metrics']
    for method,m in result['metrics'].items():
        ci=m['interval95'];ci_text='undefined' if ci is None else f'[{fmt(ci[0])}, {fmt(ci[1])}]'
        rows.append([method,fmt(m['macro_spearman']),ci_text,str(m['defined_groups']),fmt(m['gap_filtered_pair_accuracy']),fmt(random_results[method]['macro_spearman'])])
    comparisons=table(['Comparison','Common groups','Difference','95% interval'],[[k,v['common_groups'],fmt(v['difference']),
        'undefined' if v['interval95'] is None else '['+', '.join(fmt(x) for x in v['interval95'])+']'] for k,v in result['comparisons'].items()])
    second_rows=[]
    for axis,r in results['pretrained_content_mean_layer12']['second_order'].items():
        second_rows.append([axis,r['status'],r['eligible_edges'],r.get('objects_in_edges',0),fmt(r.get('correlation')),fmt(r.get('p_holm_across_all_cases'))])
    qtrunc=extraction['query_pretrained']['truncated_inputs'];ytrunc=extraction['candidate_pretrained']['truncated_inputs']
    text=f'''# Frozen representation pilot

This study uses two independent 768-dimensional encoders, X and Y. It makes no parameter updates and fits no optimized head. The fixed cohort contains **20,000 observed pairs**: 10,000 training references and 10,000 validation observations. It contains **697 distinct queries and 18,565 distinct candidates**, with 1,000 groups and exactly 10 distinct candidates per group in each partition. The two partitions share the queries and share no candidate identities, candidate categories or original split units.

The primary specification was fixed before analysis: **pretrained final-layer content-mean pooling**. Every embedding stage/block (0–12), both content-mean and first-token pooling, and one random initialization of each encoder are retained. The resulting 52 representation cases are descriptive comparisons, not a search for a best test layer. All four corrected extraction processes completed with unchanged parameters and identical repeated forwards. No old experiment queue was resumed. An initial query-input serialization error was detected and its exports and derived results were rejected. This report uses only the corrected, input-validated exports; the selected observations did not change.

## Sample and interpretation

All 697 queries with an eligible group in both source partitions were included. Groups were chosen in query-balanced rounds; each selected query has one or two sampled training groups. Within each group, five distinct candidates were sampled uniformly and five through category-balanced sampling. Selection did not inspect scores. Scores were centred by their complete source-group mean within their own partition. The sample has 9,681 distinct training candidates and 8,884 distinct validation candidates. Among the selected validation groups, **955** satisfy the original scalar-span eligibility rule.

This is a deliberately selected, larger-group pilot with a restricted training-reference library. It is not the full benchmark. Prior validation peaks 0.146427/0.184578 and the prior test result 0.176880 are not matched-cohort comparators. Neighbour distances refer to these 10,000 training references, not the entire source training set.

## Primary held-out ranking results

Macro Spearman averages defined within-group correlations among eligible groups. Constant predictions remain undefined; coverage is shown explicitly. Pair accuracy uses only score gaps greater than 0.78 and gives predicted ties half credit. Intervals use 2,000 whole-split-unit bootstrap draws and condition on the fixed cohort, reference library and encoder realization.

{table(['Readout','Macro Spearman','95% interval','Defined groups','Gap-filtered pair accuracy','Random encoders'],rows)}

{comparisons}

The global direction has the highest score among the fixed primary readouts. The own-query direction is worse on mutually defined groups. The same-query neighbour advantage and the nearest-query borrowing advantage both have intervals spanning zero. Direct native cosine also has an interval spanning zero. This pilot therefore detects candidate-only linear structure, while providing no clear evidence of an additional query-specific advantage under these readout rules and this small per-query reference library. It does not establish that such information is absent.

## Measurement 1: agreement versus distance

![Distance agreement](distance_agreement.png)

The curves report the group-balanced product of centred scores and squared score differences, with separate same-group, same-query/different-group and different-query pairs. Pair draws and distance bins are fixed across cases. They are descriptive curves; the differences are not a causal decomposition of candidate and pair information. Group centering retains comparison-set composition effects. Same-candidate observations across different queries are not noise replicates. This pilot does not estimate a noise ceiling.

The complete curve bins, originating-group counts and pair counts for all cases and controls are in `all_distance_curves.parquet`. Validation-to-training nearest-distance distributions are overlaid separately for same-query and all-query reference libraries. Flat curves only indicate failure to detect local association with this distance and cohort.

## Measurement 2: nearest-neighbour ranking

Each prediction uses five available neighbours, or all if fewer, with inverse cosine-distance weights and numerical epsilon 1e-6. Training labels are the only reference outcomes. Duplicate query/candidate references are averaged before retrieval. Other-query reference values exclude the evaluated query and average equally across contributing queries per candidate. A second other-query control matches the size of the evaluated query's reference library. Neighbourhood and weight settings were fixed without tuning.

The same-query minus other-query comparisons above measure differences between these fixed readouts, not an identified amount of interaction information. Different reference libraries still have different distance distributions. Those distributions are preserved in `all_measurements.json`.

## Measurement 3: directions

Directions average higher-minus-lower **unit candidate-vector differences within each training group**, using score gaps above 0.78, then average contributing groups equally per query. Global and borrowed controls exclude the evaluated query. Borrowing uses five nearest available queries and inverse cosine-distance weights; random borrowing uses the same weights and matches sampled training-group-support bins. No vector is estimated from validation labels.

Own-direction success supports accessible query-specific linear preferences. It does not establish that query embeddings can predict those directions. Nearest-query borrowing matching random borrowing would only be a null for this particular query geometry and borrowing rule. `support_and_distance_strata.json` reports every primary readout by sampled training-group count and validation neighbour-distance quartile. With only one or two sampled groups per query, direction estimates can be unstable.

## Measurement 4: second-order agreement

{table(['Profile axis','Status','Eligible edges','Objects in edges','Correlation','Holm-adjusted p'],second_rows)}

Profiles use training observations exclusively, and a pair must share at least five measured entries with nonconstant profiles. Ten eligible edges are required to report a correlation. Missing observations are never filled with zero. Tests permute whole object identities within observed-support bins 999 times, keeping the profile-overlap graph fixed. Reported adjusted probabilities cover every measured layer/pooling/state/axis comparison. This is observed-overlap profile agreement, **not standard CKA**. Sparse overlap restricts the conclusion to its qualifying subset; insufficient overlap is reported as unavailable.

## Direct native cosine and controls

The direct cosine readout compares the native 768-dimensional pooled vectors with no learned or fixed dimensionality projection. Matching width makes the calculation possible but does not align the coordinate systems. Primary direct-cosine performance is **{fmt(result['metrics']['direct_cosine']['macro_spearman'])}**. Its 999 globally consistent coordinate permutations give mean **{fmt(checks['coordinate_null_primary']['null_mean'])}**, interval **{checks['coordinate_null_primary']['null_interval95']}**, and one-sided probability **{fmt(checks['coordinate_null_primary']['p_one_sided'])}**. This comparison tests native-axis specificity relative to that permutation family.

Each of the 52 cases includes within-category candidate-identity swaps and a fixed score shuffle within every training and validation group. Repeated uses of a candidate share the same swapped embedding. Of {swap['candidates']} unique candidates, {swap['movable_candidates']} belong to categories where a swap is possible; {swap['actually_changed']} identities changed in the fixed draw. An unchanged statistic does not prove that the representation contains only category identity. Category centroids themselves are invariant to within-category permutations.

The primary readouts additionally have a 999-draw conditional validation-label null. All 26 pretrained direct-cosine layer/pooling readouts share a maximum-statistic correction. These shuffles retain group membership but do not model all cross-group dependence; bootstrap intervals retain the original split units. Random initialization uses one seed, so the pretrained-minus-random differences do not include variation across random initializations. Full null statistics and comparisons are retained in the accompanying JSON files.

## Layer profiles and limits

![Layer profiles](layer_profiles.png)

Mean pooling excludes padding, special tokens and X's fixed four-token documented input prefix. First-token pooling uses that prefix's first token for X and a start token for Y. The prefix is fixed for every query and contains no observation labels. Stages 0–12 include the embedding stage and all twelve blocks; the final stage uses each encoder's normalized final output. Input token caps were fixed at 1,024 for X and 202 for Y. **{qtrunc} query inputs and {ytrunc} candidate inputs were truncated**; these results therefore concern those representations, including truncation. Unknown-token coverage is recorded in `extraction_checks.json`; all query symbols must be recognized. All extracted pooled vectors were finite and stored in float32 after inference with reduced-precision arithmetic.

No score here establishes that an arbitrary fitted head cannot succeed. No layer, pooling, neighbour setting, random initialization or threshold was selected by validation performance. These are frozen diagnostic readouts on a fixed validation cohort.

## Evidence files

- `all_readouts.csv`: every native ranking readout across all 52 cases.
- `all_measurements.json`: all metrics, paired comparisons, controls, distances and profile tests.
- `all_distance_curves.parquet`: every distance curve and its support counts.
- `anonymous_sample.parquet`, `all_validation_predictions.parquet`: the anonymous fixed observations and every native validation prediction; validation row indices refer to the validation subset in stored order.
- `profile_overlap.json`: anonymous profile objects, observed overlap counts and qualifying profile edges.
- `primary_groups.parquet`: anonymized group-level primary correlations.
- `inferential_checks.json`: conditional label and coordinate-permutation distributions summarized without selecting a winner.
- `pretrained_minus_random.json`: paired primary differences and conditional intervals.
- `support_and_distance_strata.json`: primary results by reference support and distance.
- `sample_summary.json`, `class_swap_coverage.json`, `extraction_checks.json`: sample and execution evidence.
- `independent_verification.json`: independent per-group rank calculations and direct cosine recomputed from raw pooled vectors.

Original identifiers, source input strings, model identifiers, infrastructure details and private mappings are excluded from this bundle. Exact numerical values can still be matched to other copies; no universal anonymity guarantee is implied.
'''
    (destination/'report.md').write_text(text)
    write_json(destination/'verification.json',dict(cases=52,model_updates=0,optimized_fits=0,primary_predeclared=True,rows=20000,
        original_identifiers_included=False,readout_metric_tests_passed=True,extraction_parameters_unchanged=True,
        note='Consistency checks do not constitute external replication or establish population-wide conclusions'))
    # Scan public text and column metadata; private model manifests are excluded.
    forbidden=re.compile(r'prot2mol|prott5|molformer|chembl|smiles|scaffold|protein|molecule|atum|atabey|Rostlab|HUBioDataLab|/home/|/Users/',re.I)
    for path in destination.iterdir():
        if path.suffix in ('.json','.md','.csv'):
            if forbidden.search(path.read_text()):raise ValueError('Source terminology leaked into '+path.name)
        elif path.suffix=='.parquet':
            f=pd.read_parquet(path)
            if forbidden.search(' '.join(f.columns)):raise ValueError('Source schema leaked')
            for col in f.select_dtypes(include=['object','string']).columns:
                if any(forbidden.search(str(v)) for v in f[col].unique()):raise ValueError('Source identifiers leaked')
    checksums={p.name:digest(p) for p in sorted(destination.iterdir()) if p.is_file() and p.name!='checksums.json'}
    write_json(destination/'checksums.json',checksums)
    archive=OUT/'frozen_representation_pilot.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(destination.iterdir()):
            entry=zipfile.ZipInfo('frozen_representation_pilot/'+p.name,date_time=(1980,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(entry,p.read_bytes())
    write_json(OUT/'study_complete.json',dict(shareable_files=len(list(destination.iterdir())),archive_sha256=digest(archive),fits=0,updates=0,completed=time.time()))
    print('Report and bundle complete',flush=True)


def create_plots(source,destination,all_readouts,predictions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    curves=pd.read_parquet(source/'pretrained_content_mean_layer12/distance_curves.parquet')
    names={'same_group':'Same group','same_query_other_group':'Same query, other group','different_query':'Different query'}
    for relation,part in curves.groupby('relation'):
        part=part.sort_values('mean_distance')
        for ax,column in zip(axes[:2],['agreement','mean_squared_difference']):ax.plot(part.mean_distance,part[column],marker='o',markersize=3,label=names[relation])
    axes[0].set(xlabel='Candidate cosine distance',ylabel='Mean centred-score product',title='Agreement (descriptive)')
    axes[1].set(xlabel='Candidate cosine distance',ylabel='Mean squared score difference',title='Disagreement (descriptive)')
    axes[0].legend(fontsize=8)
    axes[2].hist(predictions['distance_same'],bins=30,histtype='step',density=True,label='Same-query references')
    axes[2].hist(predictions['distance_all'],bins=30,histtype='step',density=True,label='All references')
    axes[2].set(xlabel='Nearest training cosine distance',ylabel='Density',title='Validation distance coverage');axes[2].legend(fontsize=8)
    fig.savefig(destination/'distance_agreement.png',dpi=180,metadata={'Software':'plot'});plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,layout='constrained')
    methods=['direct_cosine','nearest_same','direction_own','direction_nearest']
    for ax,method in zip(axes.ravel(),methods):
        for state,style in [('pretrained','-'),('random','--')]:
            for pool,color in [('content_mean','#1769aa'),('first_token','#c25713')]:
                part=all_readouts[(all_readouts.state==state)&(all_readouts.pooling==pool)&(all_readouts.method==method)].sort_values('layer')
                ax.plot(part.layer,part.macro_spearman,linestyle=style,color=color,label=state+' / '+pool)
        ax.axhline(0,color='gray',linewidth=.7);ax.set(title=method.replace('_',' '),xlabel='Stage (0 = embedding; 12 = final)',ylabel='Validation macro Spearman')
    axes[0,0].legend(fontsize=8)
    fig.savefig(destination/'layer_profiles.png',dpi=180,metadata={'Software':'plot'});plt.close(fig)


if __name__=='__main__':report()
