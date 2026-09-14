"""Validation-only inspection of completed FinalGeo confirmatory checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
import yaml

from data import build_interim_validation_loaders
from finalgeo_confirmatory import (
    METHOD_ANCHORS,
    _summary,
    evaluate_validation_checkpoint,
    validate_confirmatory_base_config,
)


def _load_config(run_root: Path) -> dict:
    config = yaml.safe_load((run_root / "resolved_config.yaml").read_text())
    validate_confirmatory_base_config(config)
    config["experiment"]["device"] = "cuda"
    return config


def _worker(run_root: Path, output_root: Path, method: str, seed: int) -> None:
    config = _load_config(run_root)
    matrix = torch.load(
        run_root / "protocol" / "fixed_random_projection.pt",
        map_location="cpu", weights_only=False,
    )
    evaluate_validation_checkpoint(
        config,
        run_root / method / f"seed_{seed}" / "checkpoint.pt",
        output_root / method / f"seed_{seed}",
        method, seed, matrix,
    )


def _launch(run_root: Path, output_root: Path, gpu: int, method: str, seed: int):
    command = [
        sys.executable, "-m", "scripts.run_finalgeo_interim_validation",
        "--stage", "worker", "--run-root", str(run_root),
        "--output-root", str(output_root), "--method", method, "--seed", str(seed),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.perf_counter()
    print(f"[interim validation] {method} seed {seed} on physical GPU {gpu}", flush=True)
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env, check=True)
    return method, seed, time.perf_counter() - started


def run_interim_validation(run_root, output_root, seeds=(3, 4), gpu_ids=(0, 1)) -> dict:
    run_root, output_root = Path(run_root), Path(output_root)
    config = _load_config(run_root)
    config["dataset"]["download"] = True
    build_interim_validation_loaders(config)  # Materialize train split once, never test.

    complete_seeds = [
        int(seed) for seed in seeds
        if all((run_root / method / f"seed_{seed}" / "checkpoint.pt").is_file()
               for method in METHOD_ANCHORS)
    ]
    if not complete_seeds:
        raise FileNotFoundError("No requested seed has all three completed checkpoints")
    jobs = [(method, seed) for seed in complete_seeds for method in METHOD_ANCHORS]
    devices: queue.Queue[int] = queue.Queue()
    for gpu in gpu_ids:
        devices.put(int(gpu))

    def execute(job):
        gpu = devices.get()
        try:
            return _launch(run_root, output_root, gpu, *job)
        finally:
            devices.put(gpu)

    timings = []
    with ThreadPoolExecutor(max_workers=min(len(gpu_ids), len(jobs))) as executor:
        futures = [executor.submit(execute, job) for job in jobs]
        for future in as_completed(futures):
            method, seed, seconds = future.result()
            timings.append({"method": method, "seed": seed, "minutes": seconds / 60})

    metrics = pd.concat([
        pd.read_csv(output_root / method / f"seed_{seed}" / "validation_budget_metrics.csv")
        for seed in complete_seeds for method in METHOD_ANCHORS
    ], ignore_index=True)
    geometry = pd.concat([
        pd.read_csv(
            output_root / method / f"seed_{seed}" /
            "validation_representation_local_geometry.csv"
        )
        for seed in complete_seeds for method in METHOD_ANCHORS
    ], ignore_index=True)
    protocol = json.loads(
        (run_root / "protocol" / "finalgeo_frozen_protocol.json").read_text()
    )
    summary = _summary(metrics, geometry, list(map(float, protocol["common_holdout"])))
    metrics.to_csv(output_root / "interim_validation_dense_all.csv", index=False)
    geometry.to_csv(output_root / "interim_validation_geometry_all.csv", index=False)
    summary.to_csv(output_root / "interim_validation_method_summary.csv", index=False)
    pd.DataFrame(timings).to_csv(output_root / "interim_validation_runtime.csv", index=False)
    note = {
        "status": "INTERIM_VALIDATION_ONLY_NOT_CONFIRMATORY_TEST_RESULT",
        "evaluated_seeds": complete_seeds,
        "test_split_accessed": False,
        "permitted_use": "directional monitoring only; final gates remain sealed",
    }
    (output_root / "interim_validation_status.json").write_text(json.dumps(note, indent=2) + "\n")
    return note


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["worker", "full"], default="full")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--method", choices=list(METHOD_ANCHORS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", nargs="*", type=int, default=[3, 4])
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0, 1])
    args = parser.parse_args()
    if args.stage == "worker":
        if args.method is None or args.seed is None:
            parser.error("worker requires --method and --seed")
        _worker(Path(args.run_root), Path(args.output_root), args.method, args.seed)
    else:
        print(json.dumps(run_interim_validation(
            args.run_root, args.output_root, args.seeds, args.gpu_ids
        ), indent=2))


if __name__ == "__main__":
    main()
