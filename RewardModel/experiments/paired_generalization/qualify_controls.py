"""Multi-process runtime qualification; synthetic inputs are not experiment evidence.

Run with torchrun (2 CPU workers or the four authorized GPU UUIDs). Without
--config, instantiate tiny ESM/RoBERTa stacks without downloading anything.
With --config, load the real pretrained encoders and architecture. Exercise
three optimizer steps, two local microbatches per step, and DDP no_sync.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import sys

import torch
from torch.nn.parallel import DistributedDataParallel
from transformers import EsmConfig, EsmModel, RobertaConfig, RobertaModel, set_seed

ROOT = Path(__file__).resolve().parents[3]
ALLOWED_GPUS = {
    "GPU-970f3ca6-f824-616b-ec10-b972e15edb47", "GPU-8f691160-19aa-95bb-14b7-82706085df29",
    "GPU-bd4fd417-ac81-eb97-55b4-4cd60e19847a", "GPU-bf841640-8bdb-c7a0-ad1e-9fadf5635d3a",
}
sys.path.insert(0, str(ROOT / "RewardModel"))
from reward_model.model import LoadedEncoder, RewardModel, RewardModelConfig
from reward_model.training.config import load_reward_training_config


def cases():
    result = []
    for name, r, b, c in (("R", 1, 0, 0), ("B", 0, 1, 0), ("C", 0, 0, 1),
                           ("RB", 1, 1, 0), ("RC", 1, 0, 1), ("BC", 0, 1, 1),
                           ("RBC", 1, 1, 1)):
        result.append((name, dict(ranking_loss_weight=.25*r, classification_loss_weight=.25*b,
                                 contrastive_loss_weight=c,
                                 ranking_readout="ranking_head" if r else "activity_logit" if b else "cosine")))
    result.extend([
        ("frozen", dict(freeze_protein_encoder=True, freeze_molecule_encoder=True)),
        ("partial", dict(protein_unfrozen_last_layers=1, molecule_unfrozen_last_layers=1)),
        ("lora", dict(encoder_lora_rank=2, encoder_lora_alpha=4.)),
        ("fixed_scale", dict(learn_contrastive_scale=False)),
        ("detach", dict(contrastive_detach_encoders=True)),
        ("listwise_normalized", dict(ranking_normalize_per_group=True)),
        ("assay_balanced", dict(ranking_normalize_per_group=True, ranking_balance_assays=True)),
        ("pairwise_top", dict(ranking_objective="pairwise_logistic", ranking_normalize_per_group=True)),
        ("pairwise_uniform", dict(ranking_objective="pairwise_logistic", ranking_normalize_per_group=True,
                                  ranking_position_weighting="uniform")),
        ("pairwise_soft", dict(ranking_objective="pairwise_logistic", ranking_normalize_per_group=True,
                               ranking_position_weighting="uniform", ranking_affinity_margin=0,
                               ranking_pair_target_temperature=.6)),
    ])
    return result


def make_model(config_path, changes):
    set_seed(42)
    if config_path:
        config = load_reward_training_config(str(config_path)).model.to_dict()
        config.update(changes)
        return RewardModel(config)
    protein = EsmModel(EsmConfig(vocab_size=33, hidden_size=12, num_hidden_layers=3,
        num_attention_heads=3, intermediate_size=24, pad_token_id=1, mask_token_id=32,
        hidden_dropout_prob=.1, attention_probs_dropout_prob=0., token_dropout=False), add_pooling_layer=False)
    molecule = RobertaModel(RobertaConfig(vocab_size=33, hidden_size=16, num_hidden_layers=3,
        num_attention_heads=4, intermediate_size=32, pad_token_id=1,
        hidden_dropout_prob=.1, attention_probs_dropout_prob=0.), add_pooling_layer=False)
    config = dict(protein_model_name_or_path="tiny/protein", molecule_model_name_or_path="tiny/molecule",
        fusion_hidden_dim=12, fusion_num_heads=3, fusion_attention_backend="sdpa", dropout=.1,
        pair_scoring_mode="mlp_contrastive", ranking_head_hidden_dims=(8,),
        classification_head_hidden_dims=(8,), ranking_loss_weight=.25,
        classification_loss_weight=.25, contrastive_loss_weight=1.,
        contrastive_active_threshold=6., contrastive_strict_active_only=True)
    config.update(changes)
    return RewardModel(RewardModelConfig(**config),
        protein_bundle=LoadedEncoder("tiny/protein", None, protein, 12),
        molecule_bundle=LoadedEncoder("tiny/molecule", None, molecule, 16))


def make_batch(model, device, real):
    if real:
        proteins = ["MKTAYIAKQRQISFVK"]*3 + ["GAVLIPFWSTCMNQDE"]*3 + ["ACDEFGHIKLMNPQRS"]*3
        molecules = ["[C]" * count + "[O]" for count in range(1, 10)]
        p = model.protein_tokenizer(proteins, padding=True, return_tensors="pt")
        m = model.molecule_tokenizer(molecules, padding=True, return_tensors="pt")
    else:
        pids = torch.tensor([[3, 4, 1]]*3 + [[5, 6, 1]]*3 + [[7, 8, 1]]*3)
        mids = torch.tensor([[i+3, i+4, 1] for i in range(9)])
        p, m = ({"input_ids": ids, "attention_mask": (ids != 1).long()} for ids in (pids, mids))
    owners = torch.arange(3).repeat_interleave(3)
    values = torch.tensor([8., 6.5, 4.]*3)
    batch = dict(protein_input_ids=p["input_ids"], protein_attention_mask=p["attention_mask"],
        molecule_input_ids=m["input_ids"], molecule_attention_mask=m["attention_mask"],
        pchembl_values=values, activity_labels=(values > 6).float(),
        ranking_group_ids=owners, contrastive_group_ids=owners,
        contrastive_target_ids=owners, contrastive_molecule_ids=torch.arange(9))
    batch = {key: value.to(device) for key, value in batch.items()}
    if model.config.ranking_balance_assays:
        batch.update(ranking_group_weights=torch.tensor([.5]*6 + [2.]*3, device=device),
                     ranking_batch_item_count=3)
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 2:
        raise ValueError("This qualification requires at least two distributed workers")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    gpu = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda", local_rank) if gpu else torch.device("cpu")
    if gpu:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if not visible or not set(visible) <= ALLOWED_GPUS or len(visible) != world:
            raise ValueError("GPU qualification requires one authorized physical GPU UUID per worker")
        torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.distributed.init_process_group("nccl" if gpu else "gloo", timeout=timedelta(minutes=5))
    records = []
    for name, changes in cases():
        model = make_model(args.config, changes).to(device).train()
        batch = make_batch(model, device, bool(args.config))
        ddp = DistributedDataParallel(model, device_ids=[local_rank] if gpu else None, find_unused_parameters=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        losses = []
        # Different rank-local dropout streams test actual averaged gradients.
        set_seed(100 + rank)
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                with (ddp.no_sync() if micro == 0 else nullcontext()):
                    with torch.autocast("cuda", dtype=torch.bfloat16) if gpu else nullcontext():
                        output = ddp(**batch)
                    if not torch.isfinite(output.loss):
                        raise RuntimeError(f"Nonfinite loss in {name}")
                    (output.loss / 2).backward()
            optimizer.step()
            losses.append(float(output.loss.detach()))
        # Every rank must own identical fitted parameters after the synchronized
        # updates, even though its examples/dropout gradients can differ.
        checks = torch.stack([p.detach().double().sum() for p in model.parameters() if p.requires_grad])
        reference = checks.clone()
        torch.distributed.broadcast(reference, 0)
        torch.testing.assert_close(checks, reference, rtol=0, atol=1e-8)
        record = dict(case=name, losses=losses, trainable=model.num_trainable_parameters(),
                      readout=model.config.ranking_readout, adaptation=model.encoder_adaptation_report)
        records.append(record)
        if rank == 0:
            print(json.dumps(record), flush=True)
        del output, optimizer, ddp, model, batch, checks, reference
        gc.collect()
        if gpu:
            torch.cuda.empty_cache()
        torch.distributed.barrier()
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(purpose="runtime qualification only", cases=records,
            world_size=world, device=str(device), torch_version=torch.__version__,
            real_config=str(args.config) if args.config else None,
            visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")), indent=2) + "\n")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
