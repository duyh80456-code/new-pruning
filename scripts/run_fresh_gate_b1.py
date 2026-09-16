"""Workers and two-GPU scheduler for the fresh-seed Gate B1 diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import pandas as pd

from rq2_fresh_seed_gate_b1 import CHECKPOINT_EPOCHS, extract_fresh_state


def run_workers(root, dataset_root, gate_a_summary, gpu_ids=(0, 1)):
    root, dataset_root, gate_a_summary = Path(root), Path(dataset_root), Path(gate_a_summary)
    worker_root = root / "state_workers"; worker_root.mkdir(parents=True, exist_ok=True)
    available = list(map(int, gpu_ids)); pending = list(CHECKPOINT_EPOCHS)
    running, timing = {}, []

    def launch(epoch, gpu):
        output = worker_root / f"epoch_{epoch:03d}"
        metadata = output / "metadata.json"
        if metadata.is_file() and json.loads(metadata.read_text()).get("status") == "FRESH_PAIRWISE_STATE_COMPLETE":
            return 0, 0.0
        command = [
            sys.executable, "-m", "scripts.run_fresh_gate_b1", "--worker",
            "--root", str(root), "--dataset-root", str(dataset_root),
            "--gate-a-summary", str(gate_a_summary), "--epoch", str(epoch),
            "--output", str(output),
        ]
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        started = time.perf_counter()
        code = subprocess.run(command, env=env).returncode
        return code, time.perf_counter() - started

    with ThreadPoolExecutor(max_workers=len(available)) as pool:
        while pending or running:
            while pending and available:
                epoch, gpu = pending.pop(0), available.pop(0)
                print(f"[fresh Gate B1] epoch {epoch} starting on physical GPU {gpu}", flush=True)
                running[pool.submit(launch, epoch, gpu)] = (epoch, gpu)
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                epoch, gpu = running.pop(future); code, seconds = future.result(); available.append(gpu)
                timing.append({"epoch": epoch, "physical_gpu": gpu, "minutes": seconds / 60})
                if code:
                    raise RuntimeError(f"Fresh Gate B1 epoch {epoch} worker failed with {code}")
                print(f"[fresh Gate B1] epoch {epoch} finished in {seconds/60:.1f} min", flush=True)
    pd.DataFrame(timing).sort_values("epoch").to_csv(root / "state_extraction_runtime.csv", index=False)
    return [worker_root / f"epoch_{epoch:03d}" for epoch in CHECKPOINT_EPOCHS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--root", required=True); parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--gate-a-summary", required=True); parser.add_argument("--epoch", type=int)
    parser.add_argument("--output"); args = parser.parse_args()
    if not args.worker:
        raise RuntimeError("Use run_workers from the notebook")
    extract_fresh_state(
        args.root, args.epoch, args.dataset_root, args.gate_a_summary, args.output, "cuda:0"
    )


if __name__ == "__main__":
    main()
