"""Two fixed, training-free follow-ups: query intervention and supported directions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .study import (ROOT, OUT, DATA, PREVIOUS, GPUS, MODELS, SEED, runtime,
                    read_json, write_json, digest, verify_sample, select_candidates,
                    eligible_groups, token_audit, parameter_digest)
from .analysis import Evaluation, unit_vectors

FOLLOW = OUT / "decision_followup"
RUN = ROOT / "RewardModel/outputs/paired_generalization/controlled/runs/full__B__seed42__lr1p0"
CHECKPOINT = RUN / "checkpoint-5262"
PUBLIC = ['partition','x','y','g','category','unit','s','centered','source_row']


def derangement(ids, seed=SEED):
    """One fixed identity-level permutation; repeated groups retain the same donor."""
    ids=np.array(sorted(set(ids)))
    if len(ids)<2:raise ValueError('A query swap needs at least two queries')
    rng=np.random.default_rng(seed)
    for _ in range(10000):
        swapped=rng.permutation(ids)
        if np.all(swapped!=ids):return dict(zip(ids.tolist(),swapped.tolist()))
    raise RuntimeError('Could not construct the fixed derangement')


def choose_supported_queries(train, val, n=50, training_groups=40, validation_groups=5):
    support=pd.concat({p:eligible_groups(f).groupby(level=0).size()
                       for p,f in [('train',train),('val',val)]},axis=1).fillna(0)
    allowed=sorted(support.index[(support.train>=training_groups)&(support.val>=validation_groups)])
    if len(allowed)<n:raise ValueError('Insufficient queries for the fixed design')
    return sorted(np.random.default_rng(SEED).choice(allowed,n,replace=False).tolist()),support


def matched_group_plan(train):
    """Match actual direction-contributing groups; never consult validation labels."""
    lookup={}
    for (q,g),f in train.groupby(['x','g'],sort=True):
        if np.ptp(f.s.to_numpy())>.78:lookup.setdefault(q,[]).append(g)
    plan={}
    for i,q in enumerate(sorted(train.x.unique())):
        own=lookup.get(q,[])
        others=sorted(x for x,v in lookup.items() if x!=q and v)
        if not own or len(others)<len(own):raise ValueError('Matched global reference infeasible')
        rng=np.random.default_rng(SEED+i)
        donors=rng.choice(others,len(own),replace=False).tolist()
        global_groups=[str(rng.choice(lookup[x])) for x in donors]
        plan[q]=dict(own_groups=own,global_groups=global_groups,global_queries=donors,
                     contributing_groups=len(own),sampled_groups=int(train[train.x==q].g.nunique()))
    return plan


def bootstrap_query(values, queries, equal_query=False):
    frame=pd.DataFrame(dict(q=queries,v=values)).dropna()
    if frame.empty:return None
    agg=frame.groupby('q').v.agg(['sum','count','mean'])
    rng=np.random.default_rng(SEED)
    weights=rng.multinomial(len(agg),np.full(len(agg),1/len(agg)),size=2000)
    draws=(weights@agg['mean'].to_numpy()/len(agg) if equal_query else
           weights@agg['sum'].to_numpy()/(weights@agg['count'].to_numpy()))
    return np.quantile(draws,[.025,.975]).tolist()


def comparison(evaluation,a,b):
    result=evaluation.difference(a,b)
    delta=evaluation.group_scores(a)-evaluation.group_scores(b)
    result['query_cluster_interval95']=bootstrap_query(delta,evaluation.query)
    means=pd.DataFrame(dict(q=evaluation.query,v=delta)).groupby('q').v.mean().dropna()
    result['equal_query_difference']=float(means.mean()) if len(means) else None
    result['equal_query_interval95']=bootstrap_query(delta,evaluation.query,True)
    result['defined_queries']=len(means)
    return result


def independent_metric(frame,prediction,expected):
    values=[]
    for _,ix in frame.groupby('g',sort=True).indices.items():
        label=frame.iloc[ix].s.to_numpy();pred=np.asarray(prediction)[ix]
        if np.ptp(label)<.5 or not np.isfinite(pred).all() or np.ptp(pred)==0:continue
        values.append(float(spearmanr(label,pred).statistic))
    if len(values)!=expected['defined_groups']:raise ValueError('Independent coverage mismatch')
    actual=float(np.mean(values)) if values else None
    if actual is not None and abs(actual-expected['macro_spearman'])>1e-12:
        raise ValueError('Independent rank calculation disagrees')
    return dict(defined_groups=len(values),macro_spearman=actual)


def prepare():
    runtime();verify_sample()
    if (FOLLOW/'prepared.json').exists():verify_prepared();return
    if FOLLOW.exists() and any(p.is_file() for p in FOLLOW.rglob('*')):
        raise RuntimeError('Inspect incomplete follow-up before reuse')
    (FOLLOW/'private').mkdir(parents=True,exist_ok=True)
    old_protocol=read_json(OUT/'private/protocol.json')
    for name,expected in old_protocol['source_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Source dataset changed')
    completion=read_json(RUN/'controlled_training_complete.json')
    if completion['selections']['best_macro']['step']!=5262:raise ValueError('Checkpoint selection changed')
    retained=read_json(RUN/'retained/retention.json')
    weight=digest(CHECKPOINT/'pytorch_model.bin')
    expected=next(v['sha256'] for v in retained['source_weights'] if v['path']=='checkpoint-5262/pytorch_model.bin')
    if weight!=expected:raise ValueError('Retained selected weights changed')
    raw=pd.read_parquet(OUT/'private/source_rows.parquet')
    anchor=raw[raw.partition=='val'].reset_index(drop=True).copy()
    ev=Evaluation(anchor);eligible=set(np.array(ev.names)[ev.eligible]);anchor['evaluate']=anchor.g.isin(eligible)
    if len(eligible)!=955 or len(anchor)!=10000:raise ValueError('Original cohort changed')
    mapping=derangement(anchor.x)
    xtext=anchor.drop_duplicates('x').set_index('x').x_text.to_dict()
    if len(set(xtext.values()))!=len(xtext):raise ValueError('Query identity swap may leave input unchanged')
    anchor['swapped_x']=anchor.x.map(mapping)
    anchor['swapped_text']=anchor.swapped_x.map(xtext)
    val_source=pd.read_parquet(DATA/'val.parquet',columns=['compound_id','target_chembl_id','protein_sequence','compound_selfies','pchembl_value'])
    rows=val_source.iloc[anchor.source_row.to_numpy()].reset_index(drop=True)
    if not np.array_equal(rows.compound_id.astype(str),anchor.y_source):raise ValueError('Candidate source order changed')
    if not np.array_equal(rows.protein_sequence.astype(str),anchor.x_text):raise ValueError('Query source order changed')
    if not np.array_equal(rows.pchembl_value.astype('float32'),anchor.s.astype('float32')):raise ValueError('Labels changed')
    anchor['model_candidate_text']=rows.compound_selfies.to_numpy()
    native=np.load(OUT/'analysis/pretrained_content_mean_layer12/predictions.npz')
    anchor['frozen_global']=native['direction_global']
    if abs(ev.summarize(anchor.frozen_global)['macro_spearman']-.09966411332205222)>1e-12:
        raise ValueError('Frozen baseline does not match original pilot')
    anchor.to_parquet(FOLLOW/'private/anchor.parquet',index=False)
    anchor[PUBLIC+['evaluate','swapped_x','frozen_global']].to_parquet(FOLLOW/'anchor.parquet',index=False)
    write_json(FOLLOW/'query_swap.json',mapping)

    classes=pd.read_parquet(PREVIOUS/'molecule_scaffolds.parquet').set_index('compound_id').scaffold
    units=pd.read_parquet(PREVIOUS/'assay_units.parquet').set_index('assay_id').component
    frames={}
    rename=dict(target_chembl_id='x_source',assay_id='g_source',compound_id='y_source',
                protein_sequence='x_text',smiles='y_text',pchembl_value='s')
    for partition in ('train','val'):
        f=pd.read_parquet(DATA/(partition+'.parquet'),columns=list(rename)).rename(columns=rename)
        f['source_row']=np.arange(len(f));f['partition']=partition
        for col in ('x_source','g_source','y_source'):f[col]=f[col].astype(str)
        f['category_source']=f.y_source.map(classes);f['unit_source']=f.g_source.map(units)
        f['s']=f.s.astype('float32').astype('float64')
        f['centered']=f.s-f.groupby(['x_source','g_source']).s.transform('mean')
        if f[['category_source','unit_source']].isna().any().any():raise ValueError('Missing source mapping')
        frames[partition]=f
    selected,support=choose_supported_queries(frames['train'],frames['val'])
    parts=[]
    for offset,partition in enumerate(('train','val')):
        f=frames[partition];eligible=eligible_groups(f);lookup=f.groupby(['x_source','g_source'],sort=False).indices
        rng=np.random.default_rng(SEED+offset);count=40 if partition=='train' else 5
        for q in selected:
            groups=rng.choice(sorted(eligible.xs(q).index),count,replace=False)
            for g in groups:parts.append(select_candidates(f.iloc[lookup[(q,g)]],rng))
    dense=pd.concat(parts,ignore_index=True)
    for source,alias,prefix in [('x_source','x','XH'),('y_source','y','YH'),('category_source','category','CH'),('unit_source','unit','UH')]:
        maps={v:f'{prefix}{i:06d}' for i,v in enumerate(sorted(dense[source].unique(),key=str))}
        dense[alias]=dense[source].map(maps)
    dense['key']=dense.partition+'__'+dense.x_source+'__'+dense.g_source
    dense['g']=dense.key.map({v:f'GH{i:06d}' for i,v in enumerate(sorted(dense.key.unique()))})
    tr,va=(dense[dense.partition==p] for p in ('train','val'))
    for col in ('y_source','category_source','g_source','unit_source'):
        if set(tr[col])&set(va[col]):raise ValueError('Dense sample partition overlap')
    if len(tr)!=20000 or len(va)!=2500 or dense.x.nunique()!=50:raise ValueError('Dense sample size changed')
    if not (dense.groupby('g').y.nunique()==10).all():raise ValueError('Group identity uniqueness changed')
    plan=matched_group_plan(tr)
    if min(p['contributing_groups'] for p in plan.values())<24:
        raise ValueError('A selected query has fewer than two dozen usable groups; do not silently replace it')
    dense.to_parquet(FOLLOW/'private/dense.parquet',index=False)
    dense[PUBLIC].to_parquet(FOLLOW/'dense.parquet',index=False)
    write_json(FOLLOW/'matched_groups.json',plan)
    query_map=dense.drop_duplicates('x_source').set_index('x_source').x
    support.loc[selected].assign(query=query_map).to_parquet(FOLLOW/'private/source_support.parquet')
    summary=dict(queries=50,training_pairs=len(tr),validation_pairs=len(va),training_groups=2000,validation_groups=250,
        unique_candidates=int(dense.y.nunique()),contributing_groups_per_query={str(k):float(np.quantile([p['contributing_groups'] for p in plan.values()],k)) for k in (0,.5,1)},
        validation_eligible_groups=int(Evaluation(va).eligible.sum()))
    write_json(FOLLOW/'sample_summary.json',summary)
    write_json(FOLLOW/'private/protocol.json',dict(
        model_checkpoint=str(CHECKPOINT),weight_sha256=weight,config_sha256=digest(CHECKPOINT/'config.json'),
        source_sha256=old_protocol['source_sha256'],candidate_model=MODELS['candidate'],seed=SEED,
        true_query_model='Previously selected B full reference, epoch 6, base rate, one existing initialization',
        query_swap='One uniformly sampled identity derangement; same donor for every occurrence of a query; candidate rows and labels unchanged',
        anchor='All 955 eligible original pilot groups, ten fixed observations per group; other 45 groups retain missing predictions',
        dense='50 queries sampled uniformly among queries with at least 40 training and 5 validation groups of at least 10 distinct candidates; query/group/candidate selection uses no labels',
        direction='Final-layer content mean of the same fixed candidate encoder. Unit candidate vectors, within-group signed differences for gaps > 0.78, equal group weights',
        matched_global='For each query with m usable groups among its 40 sampled groups, use exactly m usable groups from m distinct other selected queries. Training gap eligibility only, no validation outcomes. One fixed draw',
        precision='Original checkpoint bf16 inference, paired true/swapped padding and batches; frozen candidate extraction bf16 then float32 pooling',
        inference_only=True,training_updates=0,model_search=False,primary_intervals='2000 paired original-split-unit bootstrap draws, query-cluster sensitivity and equal-query averages',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    files=['private/protocol.json','private/anchor.parquet','anchor.parquet','query_swap.json','private/dense.parquet','dense.parquet','matched_groups.json','sample_summary.json']
    write_json(FOLLOW/'prepared.json',dict(files={n:digest(FOLLOW/n) for n in files}))
    print('Fixed follow-up samples prepared',json.dumps(summary),flush=True)


def verify_prepared():
    verify_sample()
    for name,expected in read_json(FOLLOW/'prepared.json')['files'].items():
        if digest(FOLLOW/name)!=expected:raise ValueError('Frozen follow-up changed: '+name)


def guard_gpu(gpu):
    runtime();verify_prepared()
    if gpu not in GPUS or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPUS[gpu]:raise ValueError('Unauthorized GPU')


def anchor_score(condition,gpu):
    guard_gpu(gpu)
    if condition not in ('true','swapped'):raise ValueError('Unknown query condition')
    output=FOLLOW/('anchor_'+condition);output.mkdir(exist_ok=False)
    os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
    os.environ['HF_HUB_CACHE']=str(ROOT/'.cache/huggingface/hub')
    os.environ['HF_HUB_OFFLINE']='1'
    import torch
    from reward_model.model.io import load_reward_model
    from ..paired_generalization.checkpoint_evaluation import parameter_digest as full_digest
    torch.set_num_threads(4);torch.manual_seed(SEED)
    protocol=read_json(FOLLOW/'private/protocol.json')
    if digest(CHECKPOINT/'pytorch_model.bin')!=protocol['weight_sha256']:raise ValueError('Weights changed')
    model=load_reward_model(str(CHECKPOINT),device=torch.device('cpu'),strict=True).eval().requires_grad_(False)
    if model.config.to_dict()!=read_json(CHECKPOINT/'config.json'):raise ValueError('Model configuration changed')
    before=full_digest(model);model.to('cuda:0')
    f=pd.read_parquet(FOLLOW/'private/anchor.parquet')
    prediction=np.full(len(f),np.nan,dtype=np.float32)
    # Identical true/swapped input padding reduces irrelevant numerical batching effects.
    groups=[ix for _,ix in f[f.evaluate].groupby('g',sort=True).groups.items()]
    order=[]
    for ix in groups:order.extend(ix.tolist())
    repeat_error=None;start=time.time()
    for begin in range(0,len(order),40):
        indices=order[begin:begin+40];batch=f.iloc[indices]
        true=model.tokenize_proteins(batch.x_text.tolist(),device=torch.device('cuda:0'))
        swap=model.tokenize_proteins(batch.swapped_text.tolist(),device=torch.device('cuda:0'))
        candidate=model.tokenize_molecules(batch.model_candidate_text.tolist(),device=torch.device('cuda:0'))
        qlength=int(max(true['attention_mask'].sum(1).max(),swap['attention_mask'].sum(1).max()))
        ylength=int(candidate['attention_mask'].sum(1).max())
        query=true if condition=='true' else swap
        inputs=dict(protein_input_ids=query['input_ids'][:,:qlength],protein_attention_mask=query['attention_mask'][:,:qlength],
                    molecule_input_ids=candidate['input_ids'][:,:ylength],molecule_attention_mask=candidate['attention_mask'][:,:ylength])
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            result=model(**inputs)
            scores=result.ranking_score.float().reshape(-1)
            if result.loss is not None:raise ValueError('Inference unexpectedly computed a training loss')
            if repeat_error is None:
                repeat_error=float((scores-model(**inputs).ranking_score.float().reshape(-1)).abs().max())
                if repeat_error>1e-5:raise ValueError('Nondeterministic checkpoint scoring')
        prediction[indices]=scores.cpu().numpy()
        if begin%400==0:
            write_json(output/'progress.json',dict(completed=min(begin+40,len(order)),total=len(order),seconds=time.time()-start))
    if not np.isfinite(prediction[f.evaluate]).all() or np.isfinite(prediction[~f.evaluate]).any():raise ValueError('Wrong scoring coverage')
    after=full_digest(model)
    if before!=after:raise ValueError('Inference changed model parameters or buffers')
    np.save(output/'predictions.npy',prediction)
    write_json(output/'complete.json',dict(rows=len(order),groups=955,condition=condition,parameter_sha256=before,
        parameters_unchanged=True,optimizer_created=False,config_overrides=False,repeat_forward_max_abs_difference=repeat_error,
        prediction_sha256=digest(output/'predictions.npy'),seconds=time.time()-start,weight_sha256=protocol['weight_sha256']))
    print(condition,'query scoring complete',flush=True)


def group_directions(train,vectors,lookup):
    result={}
    for g,ix in train.groupby('g',sort=True).indices.items():
        f=train.iloc[ix];values=f.s.to_numpy();i,j=np.triu_indices(len(f),1)
        gap=values[i]-values[j];keep=np.abs(gap)>.78
        if not keep.any():continue
        v=vectors[[lookup[y] for y in f.y]]
        result[g]=((v[i[keep]]-v[j[keep]])*np.sign(gap[keep])[:,None]).mean(0)
    return result


def dense_predictions(train,val,vectors,lookup,plan):
    d=group_directions(train,vectors,lookup)
    prediction={name:np.full(len(val),np.nan) for name in ('own','global_matched')}
    for q,ix in val.groupby('x',sort=True).indices.items():
        p=plan[q];own=p['own_groups'];global_groups=p['global_groups']
        if len(own)!=len(global_groups):raise ValueError('Unequal reference support')
        v=vectors[[lookup[y] for y in val.iloc[ix].y]]
        prediction['own'][ix]=v@np.mean([d[g] for g in own],axis=0)
        prediction['global_matched'][ix]=v@np.mean([d[g] for g in global_groups],axis=0)
    return prediction


def dense_score(gpu):
    guard_gpu(gpu)
    output=FOLLOW/'dense_vectors';output.mkdir(exist_ok=False)
    import torch
    from transformers import AutoConfig,AutoTokenizer,AutoModel
    from rdkit import Chem,RDLogger
    RDLogger.DisableLog('rdApp.*');torch.set_num_threads(4);torch.manual_seed(SEED)
    spec=MODELS['candidate'];common=dict(revision=spec['revision'],local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(spec['id'],**common)
    config=AutoConfig.from_pretrained(spec['id'],trust_remote_code=True,**common);config.deterministic_eval=True
    model=AutoModel.from_pretrained(spec['id'],config=config,trust_remote_code=True,**common).eval().requires_grad_(False)
    before=parameter_digest(model)
    if before!=read_json(OUT/'embeddings/candidate_pretrained/complete.json')['parameter_sha256']:
        raise ValueError('Frozen candidate weights differ from the initial pilot')
    model.to('cuda:0')
    f=pd.read_parquet(FOLLOW/'private/dense.parquet')
    records=f[['y','y_text']].drop_duplicates().sort_values('y').reset_index(drop=True)
    if records.y.duplicated().any():raise ValueError('Identity has inconsistent candidate text')
    mols=[Chem.MolFromSmiles(t) for t in records.y_text]
    if any(m is None for m in mols):raise ValueError('Invalid candidate input')
    texts=[Chem.MolToSmiles(m,canonical=True,isomericSmiles=False) for m in mols]
    lengths,audit=token_audit(tokenizer,texts)
    if audit['unknown_tokens']:raise ValueError('Unknown candidate tokens')
    values=np.empty((len(records),768),np.float32);order=np.argsort(lengths,kind='stable')
    start=time.time();repeat_error=None
    for begin in range(0,len(order),64):
        ix=order[begin:begin+64]
        batch=tokenizer([texts[i] for i in ix],padding=True,truncation=True,max_length=spec['max_length'],return_special_tokens_mask=True,return_tensors='pt')
        special=batch.pop('special_tokens_mask').to('cuda:0').bool()
        batch={k:v.to('cuda:0') for k,v in batch.items() if k in ('input_ids','attention_mask')}
        content=batch['attention_mask'].bool()&~special
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            hidden=model(**batch).last_hidden_state
            pooled=(hidden.float()*content[...,None]).sum(1)/content.sum(1)[:,None]
            if repeat_error is None:
                repeat_error=float((hidden.float()-model(**batch).last_hidden_state.float()).abs().max())
                if repeat_error>1e-5:raise ValueError('Nondeterministic frozen encoder')
        values[ix]=pooled.cpu().numpy()
        if begin%1280==0:write_json(output/'progress.json',dict(completed=min(begin+64,len(order)),total=len(order),seconds=time.time()-start))
    if not np.isfinite(values).all() or parameter_digest(model)!=before:raise ValueError('Invalid frozen export')
    np.save(output/'content_mean.npy',values);write_json(output/'ids.json',records.y.tolist())
    tr=f[f.partition=='train'].reset_index(drop=True);va=f[f.partition=='val'].reset_index(drop=True)
    plan=read_json(FOLLOW/'matched_groups.json');lookup={v:i for i,v in enumerate(records.y)}
    predictions=dense_predictions(tr,va,unit_vectors(values),lookup,plan)
    np.savez(FOLLOW/'dense_predictions.npz',**predictions)
    evaluation=Evaluation(va);metrics={k:evaluation.summarize(p) for k,p in predictions.items()}
    independent={k:independent_metric(va,p,metrics[k]) for k,p in predictions.items()}
    comparisons=comparison(evaluation,predictions['own'],predictions['global_matched'])
    group_table=pd.DataFrame(dict(g=evaluation.names,x=evaluation.query,unit=evaluation.unit))
    for name,p in predictions.items():group_table[name]=evaluation.group_scores(p)
    group_table.to_parquet(FOLLOW/'dense_groups.parquet',index=False)
    write_json(FOLLOW/'dense_results.json',dict(metrics=metrics,own_minus_global=comparisons,independent_verification=independent,
        input_audit=audit,truncated_inputs=int((lengths>spec['max_length']).sum()),parameters_unchanged=True,
        repeat_forward_max_abs_difference=repeat_error,optimizer_created=False,parameter_sha256=before,
        vector_sha256=digest(output/'content_mean.npy'),prediction_sha256=digest(FOLLOW/'dense_predictions.npz')))
    print('Supported direction comparison complete',flush=True)


def summarize_anchor():
    f=pd.read_parquet(FOLLOW/'anchor.parquet');e=Evaluation(f)
    predictions={}
    for condition in ('true','swapped'):
        folder=FOLLOW/('anchor_'+condition);receipt=read_json(folder/'complete.json')
        if digest(folder/'predictions.npy')!=receipt['prediction_sha256']:raise ValueError('Anchor predictions changed')
        predictions[condition]=np.load(folder/'predictions.npy')
    if read_json(FOLLOW/'anchor_true/complete.json')['parameter_sha256']!=read_json(FOLLOW/'anchor_swapped/complete.json')['parameter_sha256']:
        raise ValueError('True and swapped passes used different models')
    predictions['frozen_global']=f.frozen_global.to_numpy()
    metrics={name:e.summarize(p) for name,p in predictions.items()}
    independent={name:independent_metric(f,p,metrics[name]) for name,p in predictions.items()}
    group_table=pd.DataFrame(dict(g=e.names,x=e.query,unit=e.unit))
    for name,p in predictions.items():group_table[name]=e.group_scores(p)
    true,swap=predictions['true'][e.indices],predictions['swapped'][e.indices]
    joint=np.isfinite(true).all(1)&np.isfinite(swap).all(1)&e.eligible
    rankings=[];changed=[];offset=[];centred=[]
    for a,b in zip(true[joint],swap[joint]):
        ar,br=rankdata(a),rankdata(b)
        changed.append(not np.array_equal(ar,br))
        if np.ptp(a)>0 and np.ptp(b)>0:rankings.append(float(spearmanr(a,b).statistic))
        offset.append(float(np.mean(a-b)));centred.append(float(np.sqrt(np.mean(((a-a.mean())-(b-b.mean()))**2))))
    sensitivity=dict(groups_with_changed_ranking=int(sum(changed)),groups_compared=len(changed),
        macro_true_swapped_rank_correlation=float(np.mean(rankings)),mean_absolute_score_change=float(np.mean(np.abs(true[joint]-swap[joint]))),
        mean_absolute_group_offset=float(np.mean(np.abs(offset))),mean_within_group_centered_rms_change=float(np.mean(centred)))
    group_table.to_parquet(FOLLOW/'anchor_groups.parquet',index=False)
    result=dict(metrics=metrics,true_minus_swapped=comparison(e,predictions['true'],predictions['swapped']),
        true_minus_frozen_global=comparison(e,predictions['true'],predictions['frozen_global']),
        swapped_minus_frozen_global=comparison(e,predictions['swapped'],predictions['frozen_global']),
        prediction_sensitivity=sensitivity,independent_verification=independent)
    write_json(FOLLOW/'anchor_results.json',result)
    for name,p in predictions.items():f[name]=p
    f.to_parquet(FOLLOW/'anchor_predictions.parquet',index=False)
    return result


def report():
    runtime();verify_prepared()
    a=summarize_anchor();d=read_json(FOLLOW/'dense_results.json');summary=read_json(FOLLOW/'sample_summary.json')
    share=FOLLOW/'shareable';share.mkdir(exist_ok=True)
    def fmt(x):return 'undefined' if x is None else f'{x:.4f}'
    def interval(v):return 'undefined' if v is None else '['+', '.join(fmt(x) for x in v)+']'
    def rows(metrics):return '\n'.join(f'| {k} | {fmt(v["macro_spearman"])} | {interval(v["interval95"])} | {v["defined_groups"]} |' for k,v in metrics.items())
    aa=a['true_minus_swapped'];dd=d['own_minus_global'];support=summary['contributing_groups_per_query']
    text=f'''# Query dependence and supported directions

Two fixed follow-up measurements were completed with no training, parameter updates, checkpoint search or additional initialization search. Only validation observations were scored; test data were not read. This report contains neutral identifiers only.

## Existing model on the original pilot

The previously selected B full-reference model (epoch 6) scored the exact 955 eligible original pilot groups, with ten unchanged observations per group. The other 45 original groups retain missing model predictions. A single fixed identity permutation assigns a different query to every original query. Every occurrence of a query uses the same donor; candidates, labels, category separation and group membership remain unchanged. True and swapped passes used the same model state, inference precision, batch membership and padded lengths. There was no model configuration override and no optimizer.

| Readout | Macro Spearman | 95% split-unit interval | Defined groups |
| --- | --- | --- | --- |
{rows(a['metrics'])}

True minus swapped is **{fmt(aa['difference'])}**, with paired split-unit interval **{interval(aa['interval95'])}** and query-cluster sensitivity interval **{interval(aa['query_cluster_interval95'])}**. True minus the frozen global direction is **{fmt(a['true_minus_frozen_global']['difference'])}**, interval **{interval(a['true_minus_frozen_global']['interval95'])}**. Each comparison uses mutually defined groups.

Query swapping changed the candidate ordering in **{a['prediction_sensitivity']['groups_with_changed_ranking']} of {a['prediction_sensitivity']['groups_compared']} groups**. The mean within-group correlation between true-query and swapped-query rankings was **{fmt(a['prediction_sensitivity']['macro_true_swapped_rank_correlation'])}**. Score offsets and within-group score changes are retained separately in `anchor_results.json`. Similar macro performance would not prove that the query input is ignored; a clear drop establishes useful query dependence for this intervention. A drop does not by itself isolate fine-tuning as its cause: the existing model has different encoders and a learned fusion/scoring path from the frozen representation pilot. This cohort is validation-exposed through the historical checkpoint choice.

## Directions with stronger reference support

Fifty queries were selected uniformly from those with at least 40 training groups and five validation groups containing ten distinct candidates. Selection of queries, groups and candidates did not inspect scores. For each query, 40 training groups and five validation groups were sampled. Each group contains ten distinct candidates, half sampled uniformly and half through category-balanced sampling: **20,000 training references and 2,500 validation observations**. Candidate identities, categories and original split units do not cross partitions.

The same frozen candidate encoder and predefined final-layer content-mean pooling were used. No query encoder is needed for this direction comparison. Unit candidate-vector differences were oriented from lower to higher training scores, using gaps greater than 0.78, then averaged equally within and across groups. Of the 40 sampled groups per query, the numbers contributing at least one qualifying pair ranged from **{int(support['0'])} to {int(support['1'])}**, median **{support['0.5']:.1f}**. No query was replaced based on its observed direction or validation result.

For a query with m contributing groups, its global control uses exactly m contributing groups from m distinct other selected queries, one per donor. The reference count is matched exactly. Only training gap eligibility is used when constructing these controls. Their groups and donors were fixed before embedding extraction. Both directions rank the same held-out candidates.

| Readout | Macro Spearman | 95% split-unit interval | Defined groups |
| --- | --- | --- | --- |
{rows(d['metrics'])}

Own minus matched-global is **{fmt(dd['difference'])}**, with paired split-unit interval **{interval(dd['interval95'])}**, query-cluster interval **{interval(dd['query_cluster_interval95'])}**, and equal-query difference **{fmt(dd['equal_query_difference'])}** with interval **{interval(dd['equal_query_interval95'])}**. Results cover **{dd['defined_queries']} queries**. The matched global control has broader query coverage; matching group counts does not match every aspect of candidate coverage or number of qualifying pairs. All per-query reference memberships and group correlations are supplied.

## Verification and interpretation

Macro Spearman averages within-group correlations with score range at least 0.5. Constant predictions are undefined and coverage is explicit. Intervals use 2,000 paired bootstrap draws; split-unit and query-cluster versions are reported separately, not as a joint two-way dependence model. They condition on the selected checkpoint, sampled reference library and single fixed query permutation. No equivalence margin was declared, so a difference spanning zero is not proof of equivalence.

Independent rank calculations reproduced every headline metric. Model state hashes were unchanged by inference; the two query conditions used identical model states. Frozen candidate weights match the earlier pilot exactly. Candidate input audit found {d['input_audit']['unknown_tokens']} unknown tokens; {d['truncated_inputs']} inputs exceeded the predefined token limit. The full sample, predictions, per-group results, matched references and verification receipts are included. Exact numeric values may remain matchable to other copies.
'''
    (share/'report.md').write_text(text)
    for name in ['anchor_results.json','dense_results.json','sample_summary.json','matched_groups.json','query_swap.json','anchor_predictions.parquet','anchor_groups.parquet','dense.parquet','dense_groups.parquet']:
        if name=='dense_results.json':
            value=dict(d);value.pop('parameter_sha256');write_json(share/name,value)
        else:(share/name).write_bytes((FOLLOW/name).read_bytes())
    val=pd.read_parquet(FOLLOW/'dense.parquet');val=val[val.partition=='val'].reset_index(drop=True)
    for k,p in np.load(FOLLOW/'dense_predictions.npz').items():val[k]=p
    val.to_parquet(share/'dense_predictions.parquet',index=False)
    write_json(share/'execution_checks.json',dict(inference_only=True,optimizer_created=False,updates=0,
        anchor={c:{k:v for k,v in read_json(FOLLOW/('anchor_'+c)/'complete.json').items() if k not in ('parameter_sha256','weight_sha256')} for c in ('true','swapped')},
        frozen_candidate_parameters_unchanged=d['parameters_unchanged'],same_candidate_weights_as_original_pilot=True))
    forbidden=re.compile(r'prot2mol|prott5|molformer|chembl|smiles|selfies|scaffold|protein|molecule|atum|atabey|Rostlab|HUBioDataLab|/home/|/Users/',re.I)
    for p in share.iterdir():
        if p.suffix in ('.md','.json'):
            if forbidden.search(p.read_text()):raise ValueError('Non-neutral public text: '+p.name)
        elif p.suffix=='.parquet':
            f=pd.read_parquet(p)
            if forbidden.search(' '.join(f.columns)):raise ValueError('Non-neutral public schema')
            for col in f.select_dtypes(include=['object','string']):
                if any(forbidden.search(str(x)) for x in f[col].unique()):raise ValueError('Non-neutral identity')
    write_json(share/'checksums.json',{p.name:digest(p) for p in sorted(share.iterdir()) if p.name!='checksums.json'})
    archive=FOLLOW/'query_dependence_followup.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(share.iterdir()):
            entry=zipfile.ZipInfo('query_dependence_followup/'+p.name,date_time=(1980,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(entry,p.read_bytes())
    write_json(FOLLOW/'complete.json',dict(archive_sha256=digest(archive),updates=0,completed=time.time()))
    print('Follow-up report and archive complete',flush=True)


def run():
    prepare();(FOLLOW/'logs').mkdir(exist_ok=True)
    commands=[('true',0,['anchor','--condition','true','--gpu','0']),('swapped',2,['anchor','--condition','swapped','--gpu','2']),('dense',5,['dense','--gpu','5'])]
    jobs=[]
    for name,gpu,args in commands:
        log=(FOLLOW/'logs'/f'{name}.log').open('w')
        env={**os.environ,'CUDA_VISIBLE_DEVICES':GPUS[gpu],'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'}
        p=subprocess.Popen([sys.executable,'-m','experiments.frozen_embedding_diagnostics.followup',*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((name,p,log))
    write_json(FOLLOW/'workers.json',[dict(name=n,pid=p.pid) for n,p,_ in jobs])
    exits=[]
    for name,p,log in jobs:exits.append(dict(name=name,exit_code=p.wait()));log.close()
    write_json(FOLLOW/'worker_exit.json',exits)
    if any(e['exit_code'] for e in exits):raise RuntimeError('Follow-up worker failed; inspect logs before reuse')
    report()


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','anchor','dense','report','run']);p.add_argument('--condition');p.add_argument('--gpu',type=int)
    args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='anchor':anchor_score(args.condition,args.gpu)
    elif args.action=='dense':dense_score(args.gpu)
    elif args.action=='report':report()
    else:run()


if __name__=='__main__':main()
