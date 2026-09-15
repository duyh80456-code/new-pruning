"""Two-GPU scheduler for the read-only cross-subnet interaction probe."""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from data import build_interim_validation_loaders
from rq2_cross_subnet_interaction import merge_cross_subnet_interaction_workers


def run_t4x2_probe(
    checkpoint,
    config_path,
    geometry_marginals_path,
    resource_marginals_path,
    output_dir,
    dataset_root,
    observed_comparison_path=None,
    gpu_ids=(0, 1),
    num_batches=16,
):
    """Download data once, run disjoint shards concurrently, and merge outputs."""
    gpu_ids = tuple(map(int, gpu_ids))
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise ValueError("The optimized scheduler requires exactly two distinct GPU IDs")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"Choose Kaggle T4 x2; detected {torch.cuda.device_count()} GPU(s)")
    if int(num_batches) < 2:
        raise ValueError("Two-GPU execution requires at least two batches")

    output_dir = Path(output_dir)
    worker_root = output_dir / "worker_shards"
    if worker_root.exists():
        shutil.rmtree(worker_root)
    worker_root.mkdir(parents=True, exist_ok=True)

    # Materialize CIFAR-100 before workers start so concurrent download/extraction cannot race.
    config = copy.deepcopy(yaml.safe_load(Path(config_path).read_text()))
    config["dataset"]["root"] = str(dataset_root)
    config["dataset"]["download"] = True
    build_interim_validation_loaders(config)

    first_count = int(num_batches) // 2
    shards = ((0, first_count), (first_count, int(num_batches) - first_count))
    processes = []
    worker_dirs = []
    for worker_index, ((offset, count), physical_gpu) in enumerate(zip(shards, gpu_ids)):
        worker_dir = worker_root / f"gpu_{physical_gpu}"
        worker_dirs.append(worker_dir)
        command = [
            sys.executable, "-m", "scripts.run_cross_subnet_interaction_worker",
            "--checkpoint", str(checkpoint),
            "--config", str(config_path),
            "--geometry-marginals", str(geometry_marginals_path),
            "--resource-marginals", str(resource_marginals_path),
            "--output", str(worker_dir),
            "--dataset-root", str(dataset_root),
            "--batch-offset", str(offset),
            "--num-batches", str(count),
        ]
        if worker_index == 0:
            command.append("--one-step")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        print(
            f"[interaction scheduler] GPU {physical_gpu}: global batches "
            f"{offset}..{offset + count - 1}", flush=True,
        )
        processes.append((physical_gpu, subprocess.Popen(command, env=environment)))
    failures = []
    for physical_gpu, process in processes:
        return_code = process.wait()
        if return_code:
            failures.append((physical_gpu, return_code))
    if failures:
        raise RuntimeError(f"Cross-subnet interaction worker failures: {failures}")
    return merge_cross_subnet_interaction_workers(
        worker_dirs,
        output_dir,
        observed_comparison_path=observed_comparison_path,
        expected_batches=int(num_batches),
    )
