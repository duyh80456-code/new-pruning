"""Parallel read-only runner for the Uniform-win mechanism diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from rq2_uniform_win_diagnostics import momentum_probe_state


MOMENTUM_STATES = (("common_warmup", 10),) + tuple(
    (method, epoch)
    for method in ("uniform", "resource", "pure_sw", "resource_geo")
    for epoch in (50, 100)
)


def _complete(path: Path) -> bool:
    metadata = path / "metadata.json"
    required = (
        path / "width_momentum_alignment.csv",
        path / "pair_momentum_alignment.csv",
        path / "policy_momentum_stability.csv",
        path / "gradient_grams.npy",
    )
    if not metadata.is_file() or not all(item.is_file() for item in required):
        return False
    try:
        return json.loads(metadata.read_text()).get("status") == (
            "UNIFORM_WIN_MOMENTUM_DIAGNOSTIC_COMPLETE"
        )
    except json.JSONDecodeError:
        return False


def run_momentum_states(
    root, dataset_root, gate_a_summary, output_dir, gpu_ids=(0, 1),
    states=MOMENTUM_STATES,
):
    root, dataset_root, gate_a_summary, output_dir = map(
        Path, (root, dataset_root, gate_a_summary, output_dir)
    )
    gpu_ids = tuple(map(int, gpu_ids))
    if len(gpu_ids) != 2:
        raise RuntimeError("Momentum diagnostic is optimized for exactly two GPUs")
    states = tuple((str(method), int(epoch)) for method, epoch in states)
    pending = [
        state for state in states
        if not _complete(output_dir / "momentum_states" / f"{state[0]}_E{state[1]}")
    ]
    if not pending:
        return pd.DataFrame(columns=["method", "epoch", "physical_gpu", "seconds"])

    def launch(state, gpu):
        method, epoch = state
        destination = output_dir / "momentum_states" / f"{method}_E{epoch}"
        command = [
            sys.executable, "-m", "scripts.run_uniform_win_diagnostics",
            "--worker", "--root", str(root), "--dataset-root", str(dataset_root),
            "--gate-a-summary", str(gate_a_summary), "--output", str(destination),
            "--method", method, "--epoch", str(epoch),
        ]
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        started = time.perf_counter()
        code = subprocess.run(command, env=env).returncode
        return state, gpu, code, time.perf_counter() - started

    rows = []
    queue = list(pending)
    with ThreadPoolExecutor(max_workers=2) as pool:
        active = {}
        for gpu in gpu_ids:
            if queue:
                state = queue.pop(0)
                active[pool.submit(launch, state, gpu)] = gpu
        while active:
            for future in as_completed(tuple(active)):
                gpu = active.pop(future)
                state, _, code, seconds = future.result()
                method, epoch = state
                print(
                    f"[momentum scheduler] {method} E{epoch} finished on GPU {gpu} "
                    f"in {seconds/60:.1f} min", flush=True,
                )
                if code:
                    raise RuntimeError(f"Momentum worker {method} E{epoch} failed: {code}")
                rows.append({
                    "method": method, "epoch": epoch,
                    "physical_gpu": gpu, "seconds": seconds,
                })
                if queue:
                    next_state = queue.pop(0)
                    active[pool.submit(launch, next_state, gpu)] = gpu
                break
    runtime = pd.DataFrame(rows).sort_values(["epoch", "method"])
    runtime.to_csv(output_dir / "momentum_runtime.csv", index=False)
    return runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--gate-a-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method")
    parser.add_argument("--epoch", type=int)
    args = parser.parse_args()
    if not args.worker or args.method is None or args.epoch is None:
        raise RuntimeError("Use run_momentum_states or select --worker with a state")
    momentum_probe_state(
        args.root, args.dataset_root, args.gate_a_summary,
        args.method, args.epoch, args.output, "cuda:0",
    )


if __name__ == "__main__":
    main()

