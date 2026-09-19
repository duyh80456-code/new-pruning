"""Two-T4 scheduler for the frozen E10 soft-SW matched-control gate."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from rq2_e2e_pairwise_pilot import load_config
from rq2_fixed_dense_sw_gate import METHODS
from rq2_fixed_dense_sw_training import train_fixed_branch


def run_workers(root, dataset_root, gpu_ids=(0, 1)):
    root, dataset_root = Path(root), Path(dataset_root)
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise ValueError("Exactly two distinct physical GPUs are required")

    def run_one(method, gpu):
        command = [sys.executable, "-m", "scripts.run_fixed_dense_sw_gate",
                   "--worker", "--root", str(root), "--dataset-root", str(dataset_root),
                   "--method", method]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        started = time.perf_counter()
        code = subprocess.run(command, env=env).returncode
        return {"method": method, "physical_gpu": gpu,
                "minutes": (time.perf_counter() - started) / 60, "exit_code": code}

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_one, method, gpu)
                   for method, gpu in zip(METHODS, gpu_ids)]
        rows = [future.result() for future in as_completed(futures)]
    table = pd.DataFrame(rows).sort_values("method")
    table.to_csv(root / "fixed_dense_sw_runtime.csv", index=False)
    failures = table.loc[table.exit_code.ne(0)]
    if not failures.empty:
        raise RuntimeError(f"Fixed-dense-SW workers failed: {failures.to_dict('records')}")
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--method", choices=METHODS)
    args = parser.parse_args()
    if not args.worker or args.method is None:
        raise RuntimeError("Use --worker with one method")
    config = load_config(Path(args.root) / "resolved_config.yaml", args.dataset_root)
    train_fixed_branch(config, args.root, args.method)


if __name__ == "__main__":
    main()
