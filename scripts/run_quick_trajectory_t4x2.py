"""T4x2 scheduler for the read-only quick trajectory diagnostic."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from data import build_interim_validation_loaders
from rq2_quick_trajectory_diagnostic import PATHS, merge_trajectory_paths


def run_quick_trajectory_t4x2(ht_root, output_dir, dataset_root, gpu_ids=(0, 1)):
    ht_root, output_dir, dataset_root = map(Path, (ht_root, output_dir, dataset_root))
    gpu_ids = tuple(map(int, gpu_ids))
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2 or torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"Quick trajectory diagnostic requires T4 x2; detected {torch.cuda.device_count()} GPUs"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_root = output_dir / "worker_paths"
    worker_root.mkdir(parents=True, exist_ok=True)

    # Download/materialize once before concurrent read-only workers start.
    config = copy.deepcopy(yaml.safe_load((ht_root / "resolved_config.yaml").read_text()))
    config["dataset"]["root"] = str(dataset_root)
    config["dataset"]["download"] = True
    build_interim_validation_loaders(config)

    processes, started = [], {}
    for path, physical_gpu in zip(PATHS, gpu_ids):
        worker_dir = worker_root / path
        complete = worker_dir / "metadata.json"
        if complete.is_file():
            metadata = json.loads(complete.read_text())
            if metadata.get("status") == "QUICK_TRAJECTORY_PATH_COMPLETE":
                continue
        command = [
            sys.executable, "-m", "scripts.run_quick_trajectory_worker",
            "--ht-root", str(ht_root), "--output", str(worker_dir),
            "--path", path, "--dataset-root", str(dataset_root),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        print(f"[quick trajectory scheduler] {path} starting on physical GPU {physical_gpu}", flush=True)
        started[path] = time.perf_counter()
        processes.append((path, physical_gpu, subprocess.Popen(command, env=environment)))
    failures, runtimes = [], []
    for path, physical_gpu, process in processes:
        return_code = process.wait()
        elapsed = time.perf_counter() - started[path]
        runtimes.append({"path": path, "physical_gpu": physical_gpu, "minutes": elapsed / 60})
        if return_code:
            failures.append((path, physical_gpu, return_code))
        else:
            print(f"[quick trajectory scheduler] {path} finished in {elapsed/60:.1f} min", flush=True)
    if failures:
        raise RuntimeError(f"Quick trajectory worker failures: {failures}")
    pd.DataFrame(runtimes).to_csv(output_dir / "runtime_by_path.csv", index=False)
    result = merge_trajectory_paths(
        [worker_root / path for path in PATHS], output_dir
    )
    return {"root": str(output_dir), "metadata": result, "runtimes": runtimes}
