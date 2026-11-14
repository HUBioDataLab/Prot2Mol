"""Frozen readouts and descriptive geometry on a single predeclared cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

try:
    from .study import OUT,SEED,GPUS,digest,read_json,write_json,verify_sample,runtime
except ImportError:
    from study import OUT,SEED,GPUS,digest,read_json,write_json,verify_sample,runtime


def unit_vectors(a):
    a=np.asarray(a,dtype=np.float32)
    return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)


def cosine_matrix(a,b,device=None):
    if device is None:return np.clip(a@b.T,-1.,1.)
    import torch
    result=np.empty((len(a),len(b)),np.float32)
    right=torch.as_tensor(b.T.copy(),device=device)
    for i in range(0,len(a),1024):
        result[i:i+1024]=torch.as_tensor(a[i:i+1024],device=device).matmul(right).cpu().numpy()
    return np.clip(result,-1.,1.)


def weighted_neighbours(similarity,values,k=5):
    """Rows are evaluated candidates; columns are distinct reference candidates."""
    if similarity.shape[1]==0:
        return np.full(len(similarity),np.nan),np.full(len(similarity),np.nan)
    k=min(k,similarity.shape[1])
    # Fixed input ordering and NumPy's deterministic partition give reproducible
    # ties without sorting every full reference library for every query.
    chosen=np.argpartition(-similarity,kth=k-1,axis=1)[:,:k]
    distances=np.maximum(1.-np.take_along_axis(similarity,chosen,axis=1),0.)
    weights=1./np.maximum(distances,1e-6)
    scores=(weights*np.asarray(values)[chosen]).sum(1)/weights.sum(1)
    return scores,distances[:,0]


class Evaluation:
    _bootstrap_cache={}

    def __init__(self,frame):
        self.frame=frame.reset_index(drop=True)
        self.names=list(self.frame.groupby('g',sort=True).indices)
        self.indices=np.array(list(self.frame.groupby('g',sort=True).indices.values()))
        y=self.frame.s.to_numpy()[self.indices]
        self.eligible=np.ptp(y,axis=1)>=.5
        ranks=rankdata(y,axis=1);ranks-=ranks.mean(1,keepdims=True)
        self.label_ranks=ranks/np.maximum(np.linalg.norm(ranks,axis=1,keepdims=True),1e-12)
        self.unit=self.frame.unit.to_numpy()[self.indices[:,0]]
        self.query=self.frame.x.to_numpy()[self.indices[:,0]]
        self.units,self.unit_index=np.unique(self.unit,return_inverse=True)
        key=tuple(self.units)
        if key not in self._bootstrap_cache:
            rng=np.random.default_rng(SEED)
            self._bootstrap_cache[key]=rng.multinomial(len(self.units),np.full(len(self.units),1/len(self.units)),size=2000)
        self.bootstrap=self._bootstrap_cache[key]
        self.left,self.right=np.triu_indices(y.shape[1],1)
        self.label_gap=y[:,self.left]-y[:,self.right]

    def group_scores(self,prediction):
        a=np.asarray(prediction)[self.indices]
        valid=np.isfinite(a).all(1)&(np.ptp(a,axis=1)>0)&self.eligible
        ranks=rankdata(a,axis=1);ranks-=ranks.mean(1,keepdims=True)
        rho=(ranks*self.label_ranks).sum(1)/np.maximum(np.linalg.norm(ranks,axis=1),1e-12)
        rho[~valid]=np.nan
        return rho

    def interval(self,values):
        ok=np.isfinite(values)
        if not ok.any():return None
        sums=np.bincount(self.unit_index[ok],weights=values[ok],minlength=len(self.units))
        count=np.bincount(self.unit_index[ok],minlength=len(self.units))
        denominator=self.bootstrap@count
        draws=(self.bootstrap@sums)[denominator>0]/denominator[denominator>0]
        return np.quantile(draws,[.025,.975]).tolist()

    def summarize(self,prediction):
        rho=self.group_scores(prediction)
        pred=np.asarray(prediction)[self.indices]
        gaps=pred[:,self.left]-pred[:,self.right]
        comparable=(np.abs(self.label_gap)>.78)&np.isfinite(gaps)
        agreement=(np.sign(gaps)==np.sign(self.label_gap)).astype(float)
        agreement[gaps==0]=.5
        return dict(macro_spearman=float(np.nanmean(rho)) if np.isfinite(rho).any() else None,
            eligible_groups=int(self.eligible.sum()),defined_groups=int(np.isfinite(rho).sum()),
            coverage=float(np.isfinite(rho).sum()/max(1,self.eligible.sum())),
            interval95=self.interval(rho),interval_unit_count=len(self.units),
            gap_filtered_pair_accuracy=float(agreement[comparable].mean()) if comparable.any() else None,
            gap_filtered_pairs=int(comparable.sum()))

    def difference(self,a,b):
        aa,bb=self.group_scores(a),self.group_scores(b)
        common=np.isfinite(aa)&np.isfinite(bb)
        delta=aa-bb
        return dict(common_groups=int(common.sum()),difference=float(np.mean(delta[common])) if common.any() else None,
                    interval95=self.interval(delta))


def neighbour_readouts(train,val,y_vectors,y_lookup,device=None):
    # Repeated references first average within a query/candidate; each candidate
    # then has one vote in a query's own neighbourhood.
    refs=train.groupby(['x','y'],sort=True).centered.mean().reset_index()
    training_y=sorted(refs.y.unique())
    y_position={y:i for i,y in enumerate(training_y)}
    ref_positions=np.array([y_position[y] for y in refs.y])
    similarities=cosine_matrix(y_vectors[[y_lookup[y] for y in val.y]],y_vectors[[y_lookup[y] for y in training_y]],device)
    sums=np.bincount(ref_positions,weights=refs.centered,minlength=len(training_y))
    counts=np.bincount(ref_positions,minlength=len(training_y))
    predictions={k:np.full(len(val),np.nan) for k in ('nearest_same','nearest_other','nearest_other_matched')}
    distances={k:np.full(len(val),np.nan) for k in ('same','other','other_matched','all')}
    distances['all']=1-similarities.max(1)
    for q,indices in val.groupby('x',sort=True).indices.items():
        own=refs[refs.x==q]
        pos=np.array([y_position[y] for y in own.y],dtype=int)
        local=similarities[indices]
        predictions['nearest_same'][indices],distances['same'][indices]=weighted_neighbours(local[:,pos],own.centered.to_numpy())
        remaining_count=counts.copy();remaining_sum=sums.copy()
        remaining_count[pos]-=1;remaining_sum[pos]-=own.centered.to_numpy()
        other=np.flatnonzero(remaining_count>0)
        other_values=remaining_sum[other]/remaining_count[other]
        predictions['nearest_other'][indices],distances['other'][indices]=weighted_neighbours(local[:,other],other_values)
        rng=np.random.default_rng(SEED+int(q[1:]))
        selected=np.sort(rng.choice(len(other),size=min(len(pos),len(other)),replace=False))
        predictions['nearest_other_matched'][indices],distances['other_matched'][indices]=weighted_neighbours(local[:,other[selected]],other_values[selected])
    return predictions,distances


def directions(train,y_vectors,y_lookup):
    per_query={}
    contributing={}
    for (q,g),indices in train.groupby(['x','g'],sort=True).indices.items():
        group=train.iloc[indices]
        values=group.s.to_numpy()
        i,j=np.triu_indices(len(group),1)
        gap=values[i]-values[j];keep=np.abs(gap)>.78
        if not keep.any():continue
        vectors=y_vectors[[y_lookup[y] for y in group.y]]
        d=((vectors[i[keep]]-vectors[j[keep]])*np.sign(gap[keep])[:,None]).mean(0)
        per_query.setdefault(q,[]).append(d)
        contributing[q]=contributing.get(q,0)+1
    return {q:np.mean(v,axis=0) for q,v in per_query.items()},contributing


def direction_readouts(train,val,x_vectors,y_vectors,x_lookup,y_lookup):
    d,support=directions(train,y_vectors,y_lookup)
    valid=[q for q in sorted(d) if np.linalg.norm(d[q])>1e-12]
    result={k:np.full(len(val),np.nan) for k in ('direction_own','direction_global','direction_nearest','direction_random')}
    if not valid:return result,{'queries_with_direction':0,'contributing_groups':{}}
    raw=np.array([d[q] for q in valid]);normed=unit_vectors(raw)
    query_vectors=x_vectors[[x_lookup[q] for q in valid]]
    support_all=train.groupby('x').g.nunique().to_dict()
    for q,indices in val.groupby('x',sort=True).indices.items():
        candidates=y_vectors[[y_lookup[y] for y in val.iloc[indices].y]]
        if q in d:result['direction_own'][indices]=candidates@d[q]
        # Leave the evaluated query out of both the pooled and borrowed controls.
        others=np.array([i for i,x in enumerate(valid) if x!=q])
        if not len(others):continue
        result['direction_global'][indices]=candidates@raw[others].mean(0)
        sim=query_vectors[others]@x_vectors[x_lookup[q]]
        chosen=np.argsort(-sim,kind='stable')[:5]
        near=others[chosen]
        w=1/np.maximum(1-sim[chosen],1e-6)
        borrowed=(normed[near]*w[:,None]).sum(0)/w.sum()
        result['direction_nearest'][indices]=candidates@borrowed
        # Match each borrowed query's sampled training-group support bin.
        rng=np.random.default_rng(SEED+int(q[1:]))
        pool=list(others);random=[]
        for n in near:
            support_bin=int(np.floor(np.log2(support_all[valid[n]])))
            choices=[i for i in pool if int(np.floor(np.log2(support_all[valid[i]])))==support_bin]
            picked=int(rng.choice(choices or pool));random.append(picked);pool.remove(picked)
        result['direction_random'][indices]=candidates@((normed[random]*w[:,None]).sum(0)/w.sum())
    return result,dict(queries_with_direction=len(valid),contributing_groups=support)


def profile_edges(frame,axis,min_overlap=5):
    """Fixed observed-overlap edges. Missing entries never become neutral scores."""
    if axis=='query':
        table=frame.groupby(['x','y']).centered.mean().unstack('y')
    else:
        # Group-balanced category preferences, then one value per query/category.
        table=frame.groupby(['g','x','category']).centered.mean().reset_index().groupby(['category','x']).centered.mean().unstack('x')
    matrix=table.to_numpy();present=np.isfinite(matrix)
    overlap=present.astype(np.float32)@present.astype(np.float32).T
    rows,cols=np.where(np.triu(overlap>=min_overlap,1))
    records=[]
    for i,j in zip(rows,cols):
        mask=present[i]&present[j]
        a,b=matrix[i,mask],matrix[j,mask]
        if np.ptp(a)==0 or np.ptp(b)==0:continue
        ra,rb=rankdata(a),rankdata(b)
        rho=float(np.corrcoef(ra,rb)[0,1])
        records.append((int(i),int(j),rho,int(mask.sum())))
    return dict(ids=table.index.tolist(),edges=records,object_support=present.sum(1).tolist(),min_overlap=min_overlap)


def second_order(profile,vectors,permutations=999):
    edges=profile['edges']
    if len(edges)<10:
        return dict(status='insufficient_overlap',eligible_edges=len(edges),objects=len(profile['ids']),minimum_overlap=profile['min_overlap'])
    edges=np.array(edges);i,j=edges[:,:2].astype(int).T
    similarities=np.clip(unit_vectors(vectors)@unit_vectors(vectors).T,-1,1)
    target=rankdata(edges[:,2]);target-=target.mean();target/=np.linalg.norm(target)
    def statistic(order):
        r=rankdata(similarities[order[i],order[j]]);r-=r.mean();den=np.linalg.norm(r)
        return float(r@target/den) if den>0 else 0.
    observed=statistic(np.arange(len(vectors)))
    # Whole-object permutations, stratified by observed support, keep the
    # profile graph and its observation counts fixed.
    bins=np.floor(np.log2(np.maximum(profile['object_support'],1))).astype(int)
    rng=np.random.default_rng(SEED);null=[]
    movable=sum(int((bins==b).sum()) for b in np.unique(bins) if (bins==b).sum()>1)
    for _ in range(permutations):
        order=np.arange(len(vectors))
        for b in np.unique(bins):
            ix=np.flatnonzero(bins==b);order[ix]=rng.permutation(ix)
        null.append(statistic(order))
    return dict(status='measured',correlation=observed,p_two_sided=(1+sum(abs(x)>=abs(observed) for x in null))/(permutations+1),
        eligible_edges=len(edges),objects=len(vectors),objects_in_edges=len(np.unique(np.r_[i,j])),movable_objects=movable,
        overlap_range=[int(edges[:,3].min()),int(edges[:,3].max())],permutations=permutations,null_quantiles=np.quantile(null,[.025,.5,.975]).tolist(),
        interpretation='Observed-overlap profile association, not standard CKA and not a causal information decomposition')


def pair_design(frame):
    rng=np.random.default_rng(SEED)
    groups=list(frame.groupby('g',sort=True).indices.values())
    query_groups={q:list(part.groupby('g',sort=True).groups.values()) for q,part in frame.groupby('x',sort=True)}
    x=frame.x.to_numpy();all_indices=np.arange(len(frame))
    records=[]
    for ix in groups:
        q=x[ix[0]]
        left,right=np.triu_indices(len(ix),1)
        chosen=rng.choice(len(left),size=min(10,len(left)),replace=False)
        for a,b in zip(left[chosen],right[chosen]):records.append((int(ix[a]),int(ix[b]),'same_group'))
        same=np.array([i for group in query_groups[q] for i in group if i not in set(ix)],dtype=int)
        other=all_indices[x!=q]
        for relation,choices in [('same_query_other_group',same),('different_query',other)]:
            if len(choices):
                for a,b in zip(rng.choice(ix,10,replace=True),rng.choice(choices,10,replace=True)):
                    records.append((int(a),int(b),relation))
    return pd.DataFrame(records,columns=['i','j','relation'])


def distance_curves(frame,pairs,y_vectors,y_lookup,centred=None):
    embeddings=y_vectors[[y_lookup[y] for y in frame.y]]
    a,b=pairs.i.to_numpy(),pairs.j.to_numpy()
    distance=np.clip(1-np.einsum('ij,ij->i',embeddings[a],embeddings[b]),0,2)
    z=frame.centered.to_numpy() if centred is None else centred
    values=pairs.copy()
    values['distance']=distance;values['score_product']=z[a]*z[b]
    values['squared_difference']=(z[a]-z[b])**2
    values['origin_group']=frame.g.to_numpy()[a]
    bins=np.array([0,.025,.05,.1,.2,.3,.4,.5,.75,1.,1.5,2.000001])
    values['bin']=np.clip(np.searchsorted(bins,distance,side='right')-1,0,len(bins)-2)
    # One contribution per originating comparison group in each distance bin.
    grouped=values.groupby(['relation','bin','origin_group']).agg(distance=('distance','mean'),score_product=('score_product','mean'),squared_difference=('squared_difference','mean'),pairs=('i','size')).reset_index()
    curves=grouped.groupby(['relation','bin']).agg(mean_distance=('distance','mean'),agreement=('score_product','mean'),mean_squared_difference=('squared_difference','mean'),origin_groups=('origin_group','nunique'),pairs=('pairs','sum')).reset_index()
    curves['distance_low']=bins[curves.bin];curves['distance_high']=bins[curves.bin+1]
    return curves


def permutation_indices(frame,y_ids):
    candidate=frame[['y','category']].drop_duplicates().set_index('y').reindex(y_ids)
    rng=np.random.default_rng(SEED);mapping=np.arange(len(y_ids));movable=0
    for ix in candidate.reset_index().groupby('category',sort=True).indices.values():
        mapping[ix]=rng.permutation(ix)
        if len(ix)>1:movable+=len(ix)
    return mapping,dict(candidates=len(y_ids),movable_candidates=movable,actually_changed=int((mapping!=np.arange(len(mapping))).sum()))


def shuffled_scores(frame):
    rng=np.random.default_rng(SEED)
    copy=frame.copy()
    for ix in copy.groupby('g',sort=True).indices.values():
        copy.loc[ix,'s']=rng.permutation(copy.loc[ix,'s'].to_numpy())
    copy['centered']=copy.s-copy.full_group_mean
    return copy


def profile_vectors(profile,axis,x_vectors,y_vectors,x_lookup,y_lookup,frame):
    if axis=='query':return x_vectors[[x_lookup[x] for x in profile['ids']]]
    groups=frame[['category','y']].drop_duplicates().groupby('category',sort=True)
    centroids={c:y_vectors[[y_lookup[y] for y in part.y]].mean(0) for c,part in groups}
    return np.array([centroids[c] for c in profile['ids']])


def case_analysis(frame,x_vectors,y_vectors,x_lookup,y_lookup,profiles,pairs,device=None,measure_profiles=True):
    train=frame[frame.partition=='train'].reset_index(drop=True)
    val=frame[frame.partition=='val'].reset_index(drop=True)
    evaluation=Evaluation(val)
    predictions,distances=neighbour_readouts(train,val,y_vectors,y_lookup,device)
    direction,support=direction_readouts(train,val,x_vectors,y_vectors,x_lookup,y_lookup)
    predictions.update(direction)
    predictions['direct_cosine']=np.einsum('ij,ij->i',x_vectors[[x_lookup[q] for q in val.x]],y_vectors[[y_lookup[y] for y in val.y]])
    rng=np.random.default_rng(SEED)
    predictions['direct_coordinate_shuffle']=np.einsum('ij,ij->i',x_vectors[[x_lookup[q] for q in val.x]],y_vectors[[y_lookup[y] for y in val.y]][:,rng.permutation(y_vectors.shape[1])])
    result=dict(metrics={name:evaluation.summarize(pred) for name,pred in predictions.items()},
        comparisons={a+'_minus_'+b:evaluation.difference(predictions[a],predictions[b]) for a,b in [
            ('nearest_same','nearest_other'),('nearest_same','nearest_other_matched'),('direction_own','direction_global'),('direction_nearest','direction_random')]},
        direction_support=support,
        nearest_distances={name:{str(q):float(np.nanquantile(d,q)) for q in (0,.1,.25,.5,.75,.9,1)} for name,d in distances.items()},
        second_order={axis:second_order(profiles[axis],profile_vectors(profiles[axis],axis,x_vectors,y_vectors,x_lookup,y_lookup,train)) for axis in ('query','category')} if measure_profiles else {})
    group_table=pd.DataFrame(dict(g=evaluation.names,x=evaluation.query,unit=evaluation.unit,eligible=evaluation.eligible))
    for name,pred in predictions.items():group_table[name]=evaluation.group_scores(pred)
    return result,predictions,group_table,distance_curves(train,pairs,y_vectors,y_lookup),distances


def analyze(gpu=0):
    import os
    runtime()
    if gpu not in GPUS or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPUS[gpu]:raise RuntimeError('Unauthorized analysis GPU')
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    frame=verify_sample()
    destination=OUT/'analysis';destination.mkdir(exist_ok=True)
    train=frame[frame.partition=='train'].reset_index(drop=True)
    profiles={axis:profile_edges(train,axis) for axis in ('query','category')}
    write_json(destination/'profile_overlap.json',profiles)
    pairs=pair_design(train);pairs.to_parquet(destination/'curve_pairs.parquet',index=False)
    models={}
    for state in ('pretrained','random'):
        models[state]={}
        for side in ('query','candidate'):
            directory=OUT/'embeddings'/(side+'_'+state)
            done=read_json(directory/'complete.json')
            if done['sample_sha256']!=digest(OUT/'sample.parquet') or not done['parameters_unchanged']:
                raise ValueError('Extraction is not fixed to the selected cohort')
            for name,sha in done['files'].items():
                if digest(directory/name)!=sha:raise ValueError('Embedding bytes changed')
            ids=read_json(directory/'ids.json')
            models[state][side]=dict(ids=ids,lookup={v:i for i,v in enumerate(ids)},
                arrays={pool:np.load(directory/(pool+'.npy'),mmap_mode='r') for pool in ('content_mean','first_token')})
    swaps,swap_summary=permutation_indices(frame,models['pretrained']['candidate']['ids'])
    write_json(destination/'class_swap_coverage.json',swap_summary)
    rows=[];start=time.time()
    for state in ('pretrained','random'):
        model=models[state]
        for pool in ('content_mean','first_token'):
            for layer in range(13):
                name=f'{state}_{pool}_layer{layer:02d}'
                case=destination/name
                if (case/'complete.json').exists():
                    saved=read_json(case/'results.json')
                else:
                    case.mkdir(exist_ok=True)
                    x=unit_vectors(model['query']['arrays'][pool][layer])
                    y=unit_vectors(model['candidate']['arrays'][pool][layer])
                    saved,predictions,groups,curves,distances=case_analysis(frame,x,y,model['query']['lookup'],model['candidate']['lookup'],profiles,pairs,'cuda:0')
                    # Identity-consistent within-category swaps, on the same cohort.
                    swapped,_,swapgroups,swapcurves,_=case_analysis(frame,x,y[swaps],model['query']['lookup'],model['candidate']['lookup'],profiles,pairs,'cuda:0',measure_profiles=False)
                    saved['class_swap_metrics']=swapped['metrics']
                    saved['class_swap_comparisons']={m:dict(common_groups=int((groups[m].notna()&swapgroups[m].notna()).sum()),
                        difference=float((groups[m]-swapgroups[m]).mean()) if (groups[m].notna()&swapgroups[m].notna()).any() else None,
                        interval95=Evaluation(frame[frame.partition=='val'].reset_index(drop=True)).interval((groups[m]-swapgroups[m]).to_numpy())) for m in predictions}
                    # A single fixed within-group shuffle across both partitions.
                    shuffled=shuffled_scores(frame)
                    null,_,_,nullcurves,_=case_analysis(shuffled,x,y,model['query']['lookup'],model['candidate']['lookup'],profiles,pairs,'cuda:0',measure_profiles=False)
                    saved['fixed_label_shuffle_metrics']=null['metrics']
                    write_json(case/'results.json',saved)
                    groups.to_parquet(case/'groups.parquet',index=False)
                    curves.to_parquet(case/'distance_curves.parquet',index=False)
                    swapcurves.to_parquet(case/'class_swap_curves.parquet',index=False)
                    nullcurves.to_parquet(case/'label_shuffle_curves.parquet',index=False)
                    np.savez_compressed(case/'predictions.npz',**predictions,**{'distance_'+k:v for k,v in distances.items()})
                    write_json(case/'complete.json',dict(files={p.name:digest(p) for p in case.iterdir() if p.is_file()},seconds=time.time()-start))
                for method,metrics in saved['metrics'].items():
                    rows.append(dict(state=state,pooling=pool,layer=layer,method=method,**metrics))
                write_json(destination/'progress.json',dict(completed_cases=len(rows)//9,total_cases=52,latest=name,elapsed_seconds=time.time()-start))
                print(name,'complete',flush=True)
    pd.DataFrame(rows).to_parquet(destination/'all_readouts.parquet',index=False)
    write_json(destination/'complete.json',dict(cases=52,fits=0,updates=0,seconds=time.time()-start,sample_sha256=digest(OUT/'sample.parquet')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,default=0)
    analyze(parser.parse_args().gpu)
