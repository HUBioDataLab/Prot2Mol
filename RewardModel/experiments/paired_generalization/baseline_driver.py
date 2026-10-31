"""Wait for prepared data, then run exactly one qualified 15-epoch baseline."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import time

from run import ROOT, WORK, write_json


def main():
    if ROOT != Path("/home/atabey/Prot2Mol"):
        raise RuntimeError("This driver is restricted to the authorized Atum folder")
    status = WORK / "baseline_status.json"
    if status.exists():
        raise RuntimeError("Baseline status already exists; inspect it before any relaunch")
    ranks = [json.loads((WORK / "smoke" / f"rank_{i}_complete.json").read_text()) for i in range(4)]
    if any(r["world_size"] != 4 or not math.isfinite(r["eval_macro_spearman"]) for r in ranks):
        raise RuntimeError("The four-GPU runtime check did not pass")
    prepare_pid = json.loads((WORK / "prepare_launch.json").read_text())["pid"]
    write_json(status, {"phase": "waiting_for_tokenization", "driver_pid": os.getpid(), "prepare_pid": prepare_pid})
    while not (WORK / "tokenization_ready.json").exists():
        try:
            os.kill(prepare_pid, 0)
        except ProcessLookupError:
            write_json(status, {"phase": "preparation_failed", "prepare_pid": prepare_pid})
            raise RuntimeError("Preparation exited without a completion manifest")
        time.sleep(15)
    with (WORK / "baseline_train.log").open("x") as log:
        process = subprocess.Popen(
            ["bash", "RewardModel/experiments/paired_generalization/atum.sh", "train"],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        )
        write_json(status, {"phase": "training", "driver_pid": os.getpid(), "training_pid": process.pid, "started_unix": time.time()})
        result = process.wait()
    output = WORK / "baseline_scaffold_seed42_15ep"
    summary_path = output / "experiment_summary.json"
    if result != 0 or not summary_path.is_file():
        write_json(status, {"phase": "training_failed", "returncode": result, "summary_exists": summary_path.exists()})
        raise RuntimeError("Baseline did not finish successfully; inspect baseline_train.log")
    for rank in range(4):
        if not (output / f"rank_{rank}_complete.json").is_file():
            raise RuntimeError(f"Missing successful completion for rank {rank}")
    states = [(p, json.loads(p.read_text())) for p in output.glob("checkpoint-*/trainer_state.json")]
    state_path, latest = max(states, key=lambda item: item[1]["global_step"])
    if latest["epoch"] < 14.999:
        raise RuntimeError(f"Baseline ended before 15 epochs: {latest['epoch']}")
    write_json(status, {
        "phase": "baseline_complete_experiments_pending", "returncode": result,
        "epoch": latest["epoch"], "global_step": latest["global_step"],
        "best_metric": latest["best_metric"],
        "best_model_checkpoint": latest["best_model_checkpoint"],
        "trainer_state": str(state_path), "summary": str(summary_path),
        "finished_unix": time.time(),
    })
    print(status.read_text(), flush=True)


if __name__ == "__main__":
    main()
