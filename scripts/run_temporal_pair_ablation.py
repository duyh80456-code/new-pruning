"""Two-GPU resumable scheduler for temporal SW pair-policy ablations."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import pandas as pd
import yaml

from rq2_e2e_pairwise_pilot import load_config
from rq2_temporal_pair_training import TEMPORAL_METHODS, train_temporal_branch


def run_temporal_branches(
    root, dataset_root, gate_a_summary, methods=TEMPORAL_METHODS, gpu_ids=(0, 1),
):
    root, dataset_root, gate_a_summary = map(Path, (root, dataset_root, gate_a_summary))
    methods = tuple(map(str, methods)); gpu_ids = tuple(map(int, gpu_ids))
    if not methods or len(set(methods)) != len(methods) or not set(methods).issubset(TEMPORAL_METHODS):
        raise ValueError(f"methods must be a unique subset of {TEMPORAL_METHODS}")
    if len(gpu_ids) != 2:
        raise RuntimeError("Temporal ablation scheduler requires T4 x2")

    def launch(method, gpu):
        command = [
            sys.executable, "-m", "scripts.run_temporal_pair_ablation",
            "--worker", "--root", str(root), "--dataset-root", str(dataset_root),
            "--gate-a-summary", str(gate_a_summary), "--method", method,
        ]
        environment = os.environ.copy(); environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        started = time.perf_counter()
        code = subprocess.run(command, env=environment).returncode
        return code, time.perf_counter() - started

    available, pending, active, rows = list(gpu_ids), list(methods), {}, []
    with ThreadPoolExecutor(max_workers=2) as pool:
        while pending or active:
            while pending and available:
                method, gpu = pending.pop(0), available.pop(0)
                print(f"[temporal scheduler] {method} starting on GPU {gpu}", flush=True)
                active[pool.submit(launch, method, gpu)] = (method, gpu)
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                method, gpu = active.pop(future)
                code, seconds = future.result(); available.append(gpu)
                rows.append({"method": method, "physical_gpu": gpu, "minutes": seconds / 60})
                if code:
                    raise RuntimeError(f"Temporal branch {method} failed with code {code}")
                print(f"[temporal scheduler] {method} finished in {seconds/60:.1f} min", flush=True)
    runtime = pd.DataFrame(rows)
    path = root / "temporal_pair_runtime.csv"
    if path.is_file():
        runtime = pd.concat([pd.read_csv(path), runtime], ignore_index=True)
        runtime = runtime.drop_duplicates("method", keep="last")
    runtime.to_csv(path, index=False)
    return runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--gate-a-summary", required=True)
    parser.add_argument("--method", choices=TEMPORAL_METHODS)
    args = parser.parse_args()
    if not args.worker or args.method is None:
        raise RuntimeError("Select --worker and one temporal method")
    config = load_config(Path(args.root) / "resolved_config.yaml", args.dataset_root)
    train_temporal_branch(config, args.root, args.gate_a_summary, args.method)


if __name__ == "__main__":
    main()

