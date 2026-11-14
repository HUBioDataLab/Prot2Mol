"""Prepare the fixed study and export frozen representations. No optimizer exists here."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/frozen_embedding_diagnostics_20261114"
DATA = ROOT / "dataset/processed/chembl_37/paired_generalization/scaffold_seed42_median"
PREVIOUS = ROOT / "RewardModel/outputs/paired_generalization/tier1/data"
GPUS = {
    0: "GPU-970f3ca6-f824-616b-ec10-b972e15edb47",
    2: "GPU-8f691160-19aa-95bb-14b7-82706085df29",
    5: "GPU-bd4fd417-ac81-eb97-55b4-4cd60e19847a",
    6: "GPU-bf841640-8bdb-c7a0-ad1e-9fadf5635d3a",
}
MODELS = {
    "query": {"id": "Rostlab/prot_t5_base_mt_uniref50", "revision": "3fb12c6025327b105f6f602827a5f66259f334f9", "max_length": 1024},
    "candidate": {"id": "ibm-research/MoLFormer-XL-both-10pct", "revision": "361063d0ad524ef77cf39b08469f6be770dc550f", "max_length": 202},
}
SEED = 42


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime(cpu=False):
    if not cpu and ROOT != Path("/home/atabey/Prot2Mol"):
        raise RuntimeError("Scientific extraction is restricted to the authorized project")
    cache = ROOT / ".cache/frozen_embedding_diagnostics/huggingface"
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_MODULES_CACHE"] = str(cache / "modules")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_MODE"] = "disabled"
    return cache


def eligible_groups(frame, size=10):
    counts = frame.groupby(["x_source", "g_source"], sort=True).y_source.nunique()
    return counts[counts >= size]


def balanced_groups(frame, shared, count=1000, seed=SEED):
    """One group for every shared query, then shuffled query-balanced rounds."""
    rng = np.random.default_rng(seed)
    eligible = eligible_groups(frame)
    choices = {}
    for x in sorted(shared):
        groups = list(eligible.xs(x).index)
        choices[x] = list(rng.permutation(groups))
    if sum(map(len, choices.values())) < count or len(shared) > count:
        raise ValueError("The exact requested balanced sample is infeasible")
    selected = []
    while len(selected) < count:
        available = [x for x in sorted(shared) if choices[x]]
        for x in rng.permutation(available):
            selected.append((str(x), str(choices[x].pop())))
            if len(selected) == count:
                break
    return selected


def select_candidates(group, rng, size=10):
    """Half uniform; half category-balanced. Never inspect outcomes."""
    group = group.sort_values(["y_source", "source_row"], kind="stable")
    # Choose one observation per identity using randomness independent of labels.
    indices = [int(rng.choice(ix)) for ix in group.groupby("y_source", sort=True).groups.values()]
    unique = group.loc[indices]
    if len(unique) < size:
        raise ValueError("Insufficient distinct candidates")
    first = rng.choice(unique.index.to_numpy(), size=size // 2, replace=False).tolist()
    remaining = unique.drop(index=first)
    bins = {str(c): list(rng.permutation(part.index)) for c, part in remaining.groupby("category_source", sort=True)}
    chosen = list(first)
    while len(chosen) < size:
        for category in rng.permutation(sorted(k for k, v in bins.items() if v)):
            chosen.append(int(bins[category].pop()))
            if len(chosen) == size:
                break
    return group.loc[chosen].copy()


def prepare():
    runtime()
    if (OUT / "sample_complete.json").exists():
        verify_sample()
        print("Sample already verified", flush=True)
        return
    if OUT.exists() and any(OUT.glob("sample*")):
        raise RuntimeError("Inspect incomplete sampling output before reuse")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "private").mkdir(exist_ok=True)
    sources = [DATA / (s + ".parquet") for s in ("train", "val")]
    sources += [PREVIOUS / "molecule_scaffolds.parquet", PREVIOUS / "assay_units.parquet"]
    classes = pd.read_parquet(sources[2]).set_index("compound_id").scaffold
    units = pd.read_parquet(sources[3]).set_index("assay_id").component
    frames = {}
    rename = dict(target_chembl_id="x_source", assay_id="g_source", compound_id="y_source",
                  protein_sequence="x_text", smiles="y_text", pchembl_value="s")
    for partition, source in zip(("train", "val"), sources[:2]):
        frame = pd.read_parquet(source, columns=list(rename)).rename(columns=rename)
        frame["source_row"] = np.arange(len(frame))
        for name in ("x_source", "g_source", "y_source"):
            frame[name] = frame[name].astype(str)
        frame["category_source"] = frame.y_source.map(classes)
        frame["unit_source"] = frame.g_source.map(units)
        if frame[["category_source", "unit_source"]].isna().any().any():
            raise ValueError("Missing category or split unit")
        frame["partition"] = partition
        frame["s"] = frame.s.astype("float32").astype("float64")
        frame["full_group_mean"] = frame.groupby(["x_source", "g_source"]).s.transform("mean")
        frame["source_group_size"] = frame.groupby(["x_source", "g_source"]).s.transform("size")
        frames[partition] = frame
    eligible = {p: eligible_groups(f) for p, f in frames.items()}
    shared = sorted(set(eligible["train"].index.get_level_values(0)) & set(eligible["val"].index.get_level_values(0)))
    selected = []
    for offset, (partition, frame) in enumerate(frames.items()):
        rng = np.random.default_rng(SEED + offset)
        lookup = frame.groupby(["x_source", "g_source"], sort=False).indices
        for pair in balanced_groups(frame, shared, seed=SEED + offset):
            selected.append(select_candidates(frame.iloc[lookup[pair]], rng))
    raw = pd.concat(selected, ignore_index=True)
    for column, prefix in (("x_source", "X"), ("y_source", "Y"), ("category_source", "C"), ("unit_source", "U")):
        values = sorted(raw[column].unique(), key=str)
        mapping = {v: f"{prefix}{i:06d}" for i, v in enumerate(values)}
        raw[{"x_source":"x", "y_source":"y", "category_source":"category", "unit_source":"unit"}[column]] = raw[column].map(mapping)
    raw["group_key"] = raw.partition + "__" + raw.x_source + "__" + raw.g_source
    raw["g"] = raw.group_key.map({v:f"G{i:06d}" for i,v in enumerate(sorted(raw.group_key.unique()))})
    for source, text in (("x", "x_text"), ("y", "y_text")):
        if (raw.groupby(source)[text].nunique() != 1).any():
            raise ValueError("An identity has inconsistent inputs")
    train, val = (raw[raw.partition == p] for p in ("train", "val"))
    for name in ("y_source", "category_source", "g_source", "unit_source"):
        if set(train[name]) & set(val[name]):
            raise ValueError("Partition overlap: " + name)
    if set(train.x) != set(val.x) or any(len(f) != 10000 for f in (train,val)):
        raise ValueError("Sample size or query coverage changed")
    if not (raw.groupby("g").y.nunique() == 10).all():
        raise ValueError("Every selected group must have 10 distinct candidates")
    raw["centered"] = raw.s - raw.full_group_mean
    raw.to_parquet(OUT / "private/source_rows.parquet", index=False)
    public_columns = ["partition", "x", "y", "g", "category", "unit", "s", "centered", "full_group_mean", "source_group_size"]
    raw[public_columns].to_parquet(OUT / "sample.parquet", index=False)
    for side, identity, text in (("query", "x", "x_text"), ("candidate", "y", "y_text")):
        raw[[identity, text]].drop_duplicates().sort_values(identity).rename(columns={identity:"id",text:"text"}).to_parquet(OUT / "private" / (side + ".parquet"), index=False)
    protocol = dict(version=1, seed=SEED, models=MODELS, fits=0, updates=0,
        sample="1000 groups per partition, 10 distinct candidates each; all shared eligible queries included, then balanced rounds; half uniform and half category-balanced candidate sampling",
        centering="original complete source-group mean, separately in each partition; labels never affect sampling",
        poolings=["content_mean", "first_token"], layers="embedding stage 0 and blocks 1..12; final stage is last_hidden_state after final normalization",
        primary="pretrained final-layer content_mean; all other layers and poolings descriptive, no best-layer selection",
        distance="cosine on L2-normalized vectors", neighbours=5, neighbour_weight="inverse cosine distance with epsilon 1e-6; no tuning",
        direction_gap=0.78, direction="within-group higher-minus-lower unit candidate embeddings; average within group then across groups per query",
        borrowed_queries=5, permutations=999, bootstrap_draws=2000,
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
        software_git=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    write_json(OUT / "private/protocol.json", protocol)
    summary = dict(total_pairs=len(raw), shared_queries=len(shared), eligible_pool={p:dict(groups=len(e),queries=e.index.get_level_values(0).nunique()) for p,e in eligible.items()},
        partitions={p:dict(pairs=len(f),queries=f.x.nunique(),candidates=f.y.nunique(),groups=f.g.nunique(),categories=f.category.nunique(),units=f.unit.nunique(),
            sampled_eligible_groups=int(f.groupby("g").s.agg(lambda s:np.ptp(s.to_numpy())>=.5).sum()),
            groups_per_query_quantiles={str(q):float(f.groupby("x").g.nunique().quantile(q)) for q in (0,.5,.9,1)}) for p,f in (("train",train),("val",val))})
    write_json(OUT / "sample_summary.json", summary)
    files=[OUT/"sample.parquet",OUT/"private/source_rows.parquet",OUT/"private/query.parquet",OUT/"private/candidate.parquet",OUT/"private/protocol.json",OUT/"sample_summary.json"]
    write_json(OUT / "sample_complete.json", {"files":{str(p.relative_to(OUT)):digest(p) for p in files}})
    print(json.dumps(summary), flush=True)


def verify_sample():
    receipt=read_json(OUT / "sample_complete.json")
    for name, expected in receipt["files"].items():
        if digest(OUT/name) != expected:
            raise ValueError("Frozen sample changed: " + name)
    return pd.read_parquet(OUT/"sample.parquet")


def model_download(side):
    runtime()
    from huggingface_hub import snapshot_download
    spec=MODELS[side]
    path=snapshot_download(spec["id"],revision=spec["revision"],allow_patterns=["*.json","*.py","*.safetensors","pytorch_model.bin","spiece.model"])
    write_json(OUT/"private"/(side+"_download.json"),dict(path=path,**spec))
    print(side,"downloaded",flush=True)


def parameter_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()


def extract(side, state, gpu):
    runtime()
    if gpu not in GPUS or os.environ.get("CUDA_VISIBLE_DEVICES") != GPUS[gpu]:
        raise RuntimeError("Only the four explicitly authorized GPUs may be used")
    verify_sample()
    output=OUT/"embeddings"/(side+"_"+state)
    if (output/"complete.json").exists():
        done=read_json(output/"complete.json")
        for name,sha in done["files"].items():
            if digest(output/name)!=sha: raise ValueError("Embedding artifact changed")
        print(side,state,"already complete",flush=True);return
    output.mkdir(parents=True,exist_ok=True)
    import torch
    from transformers import AutoConfig,AutoModel,AutoTokenizer,T5EncoderModel
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    spec=MODELS[side]
    common=dict(revision=spec["revision"],local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(spec["id"],**common)
    config=AutoConfig.from_pretrained(spec["id"],trust_remote_code=side=="candidate",**common)
    if side=="query":
        model=T5EncoderModel.from_pretrained(spec["id"],config=config,**common) if state=="pretrained" else T5EncoderModel(config)
    else:
        config.deterministic_eval=True
        model=AutoModel.from_pretrained(spec["id"],config=config,trust_remote_code=True,**common) if state=="pretrained" else AutoModel.from_config(config,trust_remote_code=True)
    model.eval().requires_grad_(False)
    if int(getattr(config,"d_model",getattr(config,"hidden_size",0)))!=768:
        raise ValueError("Both encoders must have native width 768")
    before=parameter_digest(model)
    model.to("cuda:0")
    records=pd.read_parquet(OUT/"private"/(side+".parquet"))
    texts=records.text.astype(str).tolist()
    if side=="query":
        texts=[" ".join(re.sub(r"[UZOB]","X",t)) for t in texts]
    else:
        from rdkit import Chem,RDLogger
        RDLogger.DisableLog("rdApp.*")
        mols=[Chem.MolFromSmiles(t) for t in texts]
        if any(m is None for m in mols): raise ValueError("Unparseable candidate input")
        texts=[Chem.MolToSmiles(m,canonical=True,isomericSmiles=False) for m in mols]
    lengths=np.array([len(tokenizer(t,add_special_tokens=True,truncation=False)["input_ids"]) for t in texts])
    order=np.argsort(lengths,kind="stable")
    # All tensors are saved as float32 pooled vectors; no full token arrays persist.
    arrays={pool:np.lib.format.open_memmap(output/(pool+".npy"),mode="w+",dtype="float32",shape=(13,len(records),768)) for pool in ("content_mean","first_token")}
    start=time.time(); batch_size=16 if side=="query" else 64
    repeat_error=None
    for start_row in range(0,len(order),batch_size):
        idx=order[start_row:start_row+batch_size]
        batch=tokenizer([texts[i] for i in idx],padding=True,truncation=True,max_length=spec["max_length"],return_special_tokens_mask=True,return_tensors="pt")
        special=batch.pop("special_tokens_mask").to("cuda:0").bool()
        batch={k:v.to("cuda:0") for k,v in batch.items() if k in ("input_ids","attention_mask")}
        content=batch["attention_mask"].bool() & ~special
        if not content.any(dim=1).all(): raise ValueError("An input has no content tokens")
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            result=model(**batch,output_hidden_states=True,return_dict=True)
            hidden=list(result.hidden_states)
            hidden[-1]=result.last_hidden_state
            if len(hidden)!=13: raise ValueError("Expected embedding stage and twelve blocks")
            pooled={"content_mean":torch.stack([(h.float()*content[...,None]).sum(1)/content.sum(1)[:,None] for h in hidden]),
                    "first_token":torch.stack([h[:,0].float() for h in hidden])}
            if repeat_error is None:
                repeat=model(**batch,output_hidden_states=True,return_dict=True)
                repeat_error=float((repeat.last_hidden_state.float()-result.last_hidden_state.float()).abs().max())
                if repeat_error>1e-5: raise RuntimeError("Non-deterministic repeated forward")
        for pool,values in pooled.items():
            data=values.cpu().numpy()
            if not np.isfinite(data).all(): raise ValueError("Nonfinite embeddings")
            arrays[pool][:,idx,:]=data
        if start_row==0 or start_row//batch_size%20==0:
            write_json(output/"progress.json",dict(completed=min(start_row+batch_size,len(order)),total=len(order),elapsed_seconds=time.time()-start))
            print(side,state,min(start_row+batch_size,len(order)),"/",len(order),flush=True)
    for arr in arrays.values():arr.flush()
    after=parameter_digest(model)
    if before!=after: raise RuntimeError("Encoder parameters changed during extraction")
    write_json(output/"ids.json",records.id.tolist())
    import transformers
    receipt=dict(side=side,state=state,shape=[13,len(records),768],seed=SEED,gpu_uuid=GPUS[gpu],models=MODELS[side],
        parameters_unchanged=True,parameter_sha256=before,optimizer_created=False,repeat_forward_max_abs_difference=repeat_error,
        sample_sha256=digest(OUT/"sample.parquet"),truncated_inputs=int((lengths>spec["max_length"]).sum()),
        input_tokens_quantiles={str(q):float(np.quantile(lengths,q)) for q in (0,.5,.9,.99,1)},
        max_length=spec["max_length"],seconds=time.time()-start,torch=torch.__version__,transformers=transformers.__version__,
        files={p.name:digest(p) for p in [output/"ids.json",output/"content_mean.npy",output/"first_token.npy"]})
    write_json(output/"complete.json",receipt)
    print(side,state,"COMPLETE",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["prepare","download","extract"])
    parser.add_argument("--side",choices=list(MODELS))
    parser.add_argument("--state",choices=["pretrained","random"],default="pretrained")
    parser.add_argument("--gpu",type=int)
    args=parser.parse_args()
    if args.action=="prepare":prepare()
    elif args.action=="download":model_download(args.side)
    else:extract(args.side,args.state,args.gpu)


if __name__=="__main__":main()
