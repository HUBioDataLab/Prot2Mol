"""Wait for the current Tier 1 exports, then perform their tested CPU analysis."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

from export_predictions import checkpoint_jobs
from run import ROOT, WORK, write_json


def main():
    if ROOT != Path("/home/atabey/Prot2Mol"):
        raise RuntimeError("This driver runs only in the authorized Atum project")
    base = WORK / "tier1"
    launch = json.loads((base / "launch.json").read_text())
    jobs = checkpoint_jobs()
    while True:
        completed = [job for job in jobs if (base / "predictions" / job["name"] / "complete.json").exists()]
        data_ready = (base / "data/composition.json").exists()
        write_json(base / "status.json", {"phase": "exporting_and_auditing", "exports_complete": len(completed),
                                          "exports_requested": len(jobs), "data_audit_complete": data_ready})
        if len(completed) == len(jobs) and data_ready:
            break
        for worker in launch["jobs"]:
            name = worker["name"]
            if name == "data_audit":
                needed = not data_ready
            else:
                index = int(name.removeprefix("export_worker"))
                needed = any(not (base / "predictions" / job["name"] / "complete.json").exists() for job in jobs[index::4])
            proc = Path(f"/proc/{worker['pid']}/stat")
            alive = proc.exists() and proc.read_text().split(") ", 1)[1].split()[0] != "Z"
            if needed and not alive:
                write_json(base / "status.json", {"phase": "failed", "worker": name, "log": worker["log"]})
                raise RuntimeError(f"{name} exited before completing its work")
        time.sleep(20)
    write_json(base / "status.json", {"phase": "analyzing_predictions"})
    command = [str(ROOT / ".venv/bin/python"), "RewardModel/experiments/paired_generalization/analyze_predictions.py"]
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    write_json(base / "status.json", {"phase": "initial_tier1_batch_complete_more_experiments_pending" if result.returncode == 0 else "analysis_failed",
                                      "returncode": result.returncode})
    if result.returncode:
        raise RuntimeError("Prediction analysis failed")


if __name__ == "__main__":
    main()
