"""Two-GPU scheduler for the three-way end-to-end pairwise pilot."""

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

from rq2_e2e_pairwise_pilot import (
    DIAGNOSTIC_STATES,
    METHODS,
    extract_diagnostic_state,
    load_config,
    train_branch,
)


def _schedule(jobs, launch, gpu_ids, label):
    available, pending, running, timings = list(map(int, gpu_ids)), list(jobs), {}, []
    if not available:
        raise RuntimeError("At least one GPU is required")
    with ThreadPoolExecutor(max_workers=len(available)) as pool:
        while pending or running:
            while pending and available:
                job, gpu = pending.pop(0), available.pop(0)
                print(f"[{label}] {job} starting on physical GPU {gpu}", flush=True)
                running[pool.submit(launch, job, gpu)] = (job, gpu)
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                job, gpu = running.pop(future)
                code, seconds = future.result(); available.append(gpu)
                timings.append({"job": str(job), "physical_gpu": gpu, "minutes": seconds / 60})
                if code:
                    raise RuntimeError(f"{label} job {job} failed with code {code}")
                print(f"[{label}] {job} finished in {seconds/60:.1f} min", flush=True)
    return pd.DataFrame(timings)


def run_branches(root, dataset_root, gate_a_summary, gpu_ids=(0, 1)):
    root, dataset_root, gate_a_summary = map(Path, (root, dataset_root, gate_a_summary))

    def launch(method, gpu):
        command = [
            sys.executable, "-m", "scripts.run_e2e_pairwise_pilot", "--branch-worker",
            "--root", str(root), "--dataset-root", str(dataset_root),
            "--gate-a-summary", str(gate_a_summary), "--method", method,
        ]
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        started = time.perf_counter()
        return subprocess.run(command, env=env).returncode, time.perf_counter() - started

    timings = _schedule(METHODS, launch, gpu_ids, "e2e branches")
    timings.to_csv(root / "branch_runtime.csv", index=False)
    return timings


def run_diagnostics(root, dataset_root, gate_a_summary, gpu_ids=(0, 1)):
    root, dataset_root, gate_a_summary = map(Path, (root, dataset_root, gate_a_summary))

    def launch(job, gpu):
        method, epoch = job
        output = root / "diagnostics" / f"{method}_E{epoch}"
        metadata = output / "metadata.json"
        if metadata.is_file():
            try:
                if json.loads(metadata.read_text()).get("status") == "E2E_PAIRWISE_DIAGNOSTIC_COMPLETE":
                    return 0, 0.0
            except json.JSONDecodeError:
                pass
        command = [
            sys.executable, "-m", "scripts.run_e2e_pairwise_pilot", "--diagnostic-worker",
            "--root", str(root), "--dataset-root", str(dataset_root),
            "--gate-a-summary", str(gate_a_summary), "--method", method,
            "--epoch", str(epoch), "--output", str(output),
        ]
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        started = time.perf_counter()
        return subprocess.run(command, env=env).returncode, time.perf_counter() - started

    timings = _schedule(DIAGNOSTIC_STATES, launch, gpu_ids, "e2e diagnostics")
    timings.to_csv(root / "diagnostic_runtime.csv", index=False)
    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-worker", action="store_true")
    parser.add_argument("--diagnostic-worker", action="store_true")
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--gate-a-summary", required=True)
    parser.add_argument("--method")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.root)
    config = load_config(root / "resolved_config.yaml", args.dataset_root)
    if args.branch_worker:
        train_branch(config, root, args.gate_a_summary, args.method)
    elif args.diagnostic_worker:
        extract_diagnostic_state(
            root, args.method, args.epoch, args.dataset_root,
            args.gate_a_summary, args.output, "cuda:0",
        )
    else:
        raise RuntimeError("Select one worker mode")


if __name__ == "__main__":
    main()
