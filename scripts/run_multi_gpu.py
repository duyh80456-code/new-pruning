from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

from geometry.analysis import correlation_summary
from geometry.reporting import write_smoke_report
from research_utils import load_config


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and (value != value or abs(value) == float("inf")):
        return None
    return value


def _finalize(root_output: Path, config: dict) -> Path:
    central_frames = []
    summaries = []
    for seed_value in config["experiment"]["seeds"]:
        seed = int(seed_value)
        seed_dir = root_output / f"seed_{seed}"
        central_frames.append(pd.read_csv(seed_dir / "results" / "central_analysis.csv"))
        with (seed_dir / "results" / "analysis_summary.json").open() as handle:
            summaries.append(json.load(handle))

    combined = pd.concat(central_frames, ignore_index=True)
    combined.to_csv(root_output / "central_analysis_all_seeds.csv", index=False)
    pooled = combined[["local_wasserstein_sensitivity", "local_accuracy_sensitivity"]].dropna()
    aggregate = correlation_summary(
        pooled["local_wasserstein_sensitivity"],
        pooled["local_accuracy_sensitivity"],
        int(config["geometry"]["bootstrap_samples"]),
        int(config["geometry"]["projection_seed"]),
    )
    with (root_output / "aggregate_correlations.json").open("w") as handle:
        json.dump(_json_safe(aggregate), handle, indent=2, allow_nan=False)
    return write_smoke_report(root_output, summaries, combined)


def run_multi_gpu(config_path: str | Path, gpu_ids: list[int]) -> Path:
    """Run one seed per process, dynamically scheduling seeds over visible GPUs."""
    config = load_config(config_path)
    seeds = [int(seed) for seed in config["experiment"]["seeds"]]
    if not seeds:
        raise ValueError("experiment.seeds must not be empty")
    if not gpu_ids:
        raise ValueError("gpu_ids must not be empty")

    root_output = Path(config["experiment"]["output_dir"])
    root_output.mkdir(parents=True, exist_ok=True)
    with (root_output / "resolved_config.yaml").open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    worker_base = root_output.parent / f".{root_output.name}_gpu_workers"
    worker_base.mkdir(parents=True, exist_ok=False)
    project_root = Path(__file__).resolve().parents[1]
    available_gpus: queue.Queue[int] = queue.Queue()
    for gpu_id in gpu_ids:
        available_gpus.put(int(gpu_id))

    def launch(seed: int) -> tuple[int, int, Path]:
        gpu_id = available_gpus.get()
        worker_root = worker_base / f"worker_seed_{seed}"
        worker_config = yaml.safe_load(yaml.safe_dump(config))
        worker_config["experiment"]["seeds"] = [seed]
        worker_config["experiment"]["output_dir"] = str(worker_root)
        worker_config_path = worker_base / f"seed_{seed}.yaml"
        worker_config_path.write_text(yaml.safe_dump(worker_config, sort_keys=False))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[scheduler] seed {seed} starting on physical GPU {gpu_id}", flush=True)
        try:
            subprocess.run(
                [sys.executable, "-m", "scripts.run_experiment", "--config", str(worker_config_path)],
                cwd=project_root,
                env=env,
                check=True,
            )
        finally:
            available_gpus.put(gpu_id)
        print(f"[scheduler] seed {seed} completed on physical GPU {gpu_id}", flush=True)
        return seed, gpu_id, worker_root

    completed: dict[int, Path] = {}
    max_workers = min(len(gpu_ids), len(seeds))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(launch, seed): seed for seed in seeds}
        for future in as_completed(futures):
            seed, _, worker_root = future.result()
            completed[seed] = worker_root

    for seed in seeds:
        source = completed[seed] / f"seed_{seed}"
        destination = root_output / f"seed_{seed}"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
        shutil.move(str(source), str(destination))

    return _finalize(root_output, config)

