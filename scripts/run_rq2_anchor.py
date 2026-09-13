from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
import yaml

from rq2_anchor_placement import (
    CONFIRMATORY_SEEDS,
    UNIFORM_ANCHORS,
    _uniform_source_config,
    evaluate_checkpoint,
    finalize_rq2,
    select_geometry_anchors,
    smoke_geo,
    train_geo_seed,
    validate_rq2_config,
)


def _load(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def _validate_comparator(config: dict, source: dict) -> None:
    for section, keys in {
        "dataset": ("name", "validation_size", "split_seed", "bn_calibration_size", "feature_subset_size"),
        "model": ("backbone", "projection_dim"),
        "training": ("batch_size", "learning_rate", "momentum", "weight_decay", "kd_lambda", "kd_temperature"),
    }.items():
        for key in keys:
            if config[section][key] != source[section][key]:
                raise ValueError(f"RQ2/Uniform comparator mismatch: {section}.{key}")
    if list(map(float, config["compression"]["eval_widths"])) != list(
        map(float, source["compression"]["eval_widths"])
    ):
        raise ValueError("RQ2/Uniform dense evaluation grids differ")


def run_worker(config_path: str | Path, uniform_root: str | Path, stage: str, seed: int | None):
    config = _load(config_path)
    validate_rq2_config(config)
    root = Path(config["experiment"]["output_dir"])
    uniform_root = Path(uniform_root)
    if stage == "smoke":
        return smoke_geo(config, root)
    if seed not in CONFIRMATORY_SEEDS:
        raise ValueError("Train/eval workers are locked to confirmatory seeds 1 and 2")
    if stage == "train":
        return train_geo_seed(config, root, int(seed))
    random_matrix = torch.load(
        root / "protocol" / "fixed_random_projection.pt", map_location="cpu", weights_only=False
    )
    if stage == "eval_geo":
        checkpoint = root / "geo" / f"seed_{seed}" / "checkpoint.pt"
        return evaluate_checkpoint(config, root, checkpoint, int(seed), "geo", random_matrix)
    if stage == "eval_uniform":
        uniform_config = dict(config)
        uniform_config["compression"] = dict(config["compression"])
        uniform_config["compression"]["train_widths"] = list(UNIFORM_ANCHORS)
        checkpoint = uniform_root / "shared" / f"seed_{seed}" / "checkpoint.pt"
        return evaluate_checkpoint(
            uniform_config, root, checkpoint, int(seed), "uniform", random_matrix
        )
    raise ValueError(stage)


def _run_job(config_path: Path, uniform_root: Path, gpu: int, job: dict):
    command = [
        sys.executable, "-m", "scripts.run_rq2_anchor",
        "--config", str(config_path), "--uniform-root", str(uniform_root),
        "--stage", job["stage"],
    ]
    if job.get("seed") is not None:
        command += ["--seed", str(job["seed"])]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.perf_counter()
    print(f"[RQ2 scheduler] {job['name']} starting on physical GPU {gpu}", flush=True)
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env, check=True)
    elapsed = time.perf_counter() - started
    print(f"[RQ2 scheduler] {job['name']} finished in {elapsed/60:.1f} min", flush=True)
    return job["name"], elapsed


def _schedule(config_path: Path, uniform_root: Path, gpu_ids: list[int], jobs: list[dict]):
    pending = [job for job in jobs if not all(Path(path).is_file() for path in job["complete"])]
    if not pending:
        return {}
    available: queue.Queue[int] = queue.Queue()
    for gpu in gpu_ids:
        available.put(int(gpu))

    def launch(job):
        gpu = available.get()
        try:
            return _run_job(config_path, uniform_root, gpu, job)
        finally:
            available.put(gpu)

    timings = {}
    with ThreadPoolExecutor(max_workers=min(len(gpu_ids), len(pending))) as executor:
        futures = [executor.submit(launch, job) for job in pending]
        for future in as_completed(futures):
            name, elapsed = future.result()
            timings[name] = elapsed
    return timings


def run_rq2(config_path: str | Path, uniform_root: str | Path, gpu_ids=(0, 1)) -> dict:
    config = _load(config_path)
    validate_rq2_config(config)
    uniform_root = Path(uniform_root)
    source = _uniform_source_config(uniform_root)
    _validate_comparator(config, source)
    if len(gpu_ids) < 2:
        raise RuntimeError("RQ2 final screening requires two GPUs for parallel seeds 1 and 2")
    root = Path(config["experiment"]["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    protocol = root / "protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    selection = select_geometry_anchors(uniform_root, protocol)
    config["compression"]["train_widths"] = selection["selected_anchors"]
    resolved = root / "resolved_config.yaml"
    resolved.write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copy2(
        uniform_root / "protocol" / "fixed_random_projection.pt",
        protocol / "fixed_random_projection.pt",
    )
    provenance = {
        "uniform_root": str(uniform_root),
        "uniform_resolved_config": str(uniform_root / "resolved_config.yaml"),
        "uniform_protocol": "50+50 weights-only extension",
        "geo_protocol": "50+50 weights-only extension",
        "test_sealed_until_both_geo_checkpoints_complete": True,
    }
    (protocol / "comparison_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if tuple(selection["selected_anchors"]) == UNIFORM_ANCHORS:
        decision = {
            "rq2_screening_pass": False,
            "verdict": "NO INTERVENTION: geometry selected the uniform anchors",
            "selected_anchors": selection["selected_anchors"],
        }
        (root / "rq2_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
        return {"root": root, "selection": selection, "decision": decision, "timings": {}}
    timings = {}
    smoke_job = {
        "name": "smoke_seed_1",
        "stage": "smoke",
        "complete": [root / "development" / "smoke" / "checkpoint.pt"],
    }
    timings.update(_schedule(resolved, uniform_root, [gpu_ids[0]], [smoke_job]))
    train_jobs = [
        {
            "name": f"geo_train_seed_{seed}", "stage": "train", "seed": seed,
            "complete": [
                root / "geo" / f"seed_{seed}" / "checkpoint.pt",
                root / "geo" / f"seed_{seed}" / "training_metrics_1_100.csv",
            ],
        }
        for seed in CONFIRMATORY_SEEDS
    ]
    timings.update(_schedule(resolved, uniform_root, list(gpu_ids), train_jobs))
    checkpoints = [root / "geo" / f"seed_{seed}" / "checkpoint.pt" for seed in CONFIRMATORY_SEEDS]
    if not all(path.is_file() for path in checkpoints):
        raise RuntimeError("Test reveal blocked: both Geometry-4 checkpoints must finish")
    eval_jobs = []
    for method in ("geo", "uniform"):
        for seed in CONFIRMATORY_SEEDS:
            eval_jobs.append({
                "name": f"eval_{method}_seed_{seed}",
                "stage": f"eval_{method}",
                "seed": seed,
                "complete": [
                    root / "evaluation" / method / f"seed_{seed}" / "budget_metrics.csv",
                    root / "evaluation" / method / f"seed_{seed}" / "representation_local_geometry.csv",
                    root / "predictions" / method / f"seed_{seed}" / "predictions_all_widths.csv",
                ],
            })
    timings.update(_schedule(resolved, uniform_root, list(gpu_ids), eval_jobs))
    decision = finalize_rq2(config, root, selection)
    pd.DataFrame([
        {"stage": stage, "minutes": seconds / 60} for stage, seconds in timings.items()
    ]).to_csv(root / "runtime_by_stage.csv", index=False)
    (root / "rq2_complete.json").write_text(json.dumps(decision, indent=2) + "\n")
    return {"root": root, "selection": selection, "decision": decision, "timings": timings}


def main():
    parser = argparse.ArgumentParser(description="RQ2 geometry-guided anchor placement")
    parser.add_argument("--config", required=True)
    parser.add_argument("--uniform-root", required=True)
    parser.add_argument(
        "--stage", default="full",
        choices=["full", "smoke", "train", "eval_geo", "eval_uniform"],
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu-ids", nargs="*", type=int)
    args = parser.parse_args()
    if args.stage == "full":
        result = run_rq2(args.config, args.uniform_root, args.gpu_ids or [0, 1])
        print(result["decision"]["verdict"])
    else:
        run_worker(args.config, args.uniform_root, args.stage, args.seed)


if __name__ == "__main__":
    main()
