#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
task_root="$(pwd -P)"
[[ "$task_root" == /home/atabey/Prot2Mol ]] || { echo "This launcher is restricted to /home/atabey/Prot2Mol" >&2; exit 1; }
export HF_HOME="$task_root/.hf_cache"
export XDG_CACHE_HOME="$task_root/.cache"
export TORCH_HOME="$task_root/.cache/torch"
export TRITON_CACHE_DIR="$task_root/.cache/triton"
export CUDA_CACHE_PATH="$task_root/.cache/cuda"
export TMPDIR="$task_root/.cache/paired_generalization_tmp"
export WANDB_DIR="$task_root/RewardModel/outputs/paired_generalization"
export WANDB_CACHE_DIR="$task_root/.wandb_cache"
export WANDB_CONFIG_DIR="$task_root/.cache/wandb_config"
export WANDB_DATA_DIR="$task_root/.cache/wandb_data"
export WANDB_PROJECT=Prot2Mol_RewardModel
export WANDB_ENTITY=hubiodatascilab
export WANDB_RUN_GROUP=paired_generalization_15epoch
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$WANDB_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"
runner=RewardModel/experiments/paired_generalization/run.py
case "${1:-}" in
  prepare|make-smoke)
    export CUDA_VISIBLE_DEVICES=""
    exec .venv/bin/python "$runner" "$1"
    ;;
  smoke|train)
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 0,2,5,6 --query-gpu=uuid --format=csv,noheader | paste -sd, -)"
    [[ "$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")" == 4 ]]
    config=RewardModel/configs/reward_train_paired_generalization_baseline.yaml
    if [[ "$1" == smoke ]]; then
      export WANDB_MODE=disabled
      config=RewardModel/outputs/paired_generalization/smoke_config.yaml
    fi
    exec .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 "$runner" train --config "$config"
    ;;
  *) echo "Usage: $0 prepare|make-smoke|smoke|train" >&2; exit 2 ;;
esac
