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
import yaml

from data import build_confirmatory_loaders
from finalgeo_confirmatory import (
    CONFIRMATORY_SEEDS,
    METHOD_ANCHORS,
    evaluate_method_seed,
    finalize_confirmatory,
    train_method_seed,
    validate_confirmatory_base_config,
    validate_frozen_protocol,
)


def _load(path):
    return yaml.safe_load(Path(path).read_text())


def run_worker(config_path, root, stage, method, seed):
    config = _load(config_path)
    if stage == "train":
        return train_method_seed(config, root, method, seed)
    if stage == "eval":
        return evaluate_method_seed(config, root, method, seed)
    raise ValueError(stage)


def _job(config_path: Path, root: Path, gpu: int, job: dict):
    command = [
        sys.executable, "-m", "scripts.run_finalgeo_confirmatory",
        "--config", str(config_path), "--root", str(root),
        "--stage", job["stage"], "--method", job["method"], "--seed", str(job["seed"]),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.perf_counter()
    print(f"[FinalGeo scheduler] {job['name']} starting on physical GPU {gpu}", flush=True)
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env, check=True)
    elapsed = time.perf_counter() - started
    print(f"[FinalGeo scheduler] {job['name']} finished in {elapsed / 60:.1f} min", flush=True)
    return job["name"], elapsed


def _schedule(config_path: Path, root: Path, gpu_ids, jobs):
    pending = [job for job in jobs if not all(Path(path).is_file() for path in job["complete"])]
    if not pending:
        return {}
    devices: queue.Queue[int] = queue.Queue()
    for gpu in gpu_ids:
        devices.put(int(gpu))

    def launch(job):
        gpu = devices.get()
        try:
            return _job(config_path, root, gpu, job)
        finally:
            devices.put(gpu)

    timings = {}
    with ThreadPoolExecutor(max_workers=min(len(gpu_ids), len(pending))) as executor:
        futures = [executor.submit(launch, job) for job in pending]
        for future in as_completed(futures):
            name, elapsed = future.result()
            timings[name] = elapsed
    return timings


def run_confirmatory(development_root, freeze_dir, output_dir, gpu_ids=(0, 1)):
    if len(gpu_ids) != 2:
        raise RuntimeError("Locked FinalGeo runner requires Kaggle T4x2")
    development_root, freeze_dir, root = map(Path, (development_root, freeze_dir, output_dir))
    protocol = validate_frozen_protocol(development_root, freeze_dir)
    base = _load(development_root / "resolved_config.yaml")
    validate_confirmatory_base_config(base)
    base["experiment"]["output_dir"] = str(root)
    base["experiment"]["device"] = "cuda"
    base["experiment"]["confirmatory_seeds"] = list(CONFIRMATORY_SEEDS)
    # Materialize CIFAR-100 once before isolated GPU workers start. Concurrent
    # torchvision downloads/extractions can corrupt the shared dataset root.
    base["dataset"]["download"] = True
    build_confirmatory_loaders(base, training_seed=CONFIRMATORY_SEEDS[0])
    base["dataset"]["download"] = False
    root.mkdir(parents=True, exist_ok=True)
    protocol_dir = root / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "finalgeo_selector_candidates.csv", "finalgeo_selected_anchors.json",
        "finalgeo_frozen_protocol.json",
    ):
        shutil.copy2(freeze_dir / filename, protocol_dir / filename)
    shutil.copy2(
        development_root / "protocol" / "fixed_random_projection.pt",
        protocol_dir / "fixed_random_projection.pt",
    )
    config_path = root / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(base, sort_keys=False))

    train_jobs = [
        {
            "name": f"train_{method}_seed_{seed}", "stage": "train",
            "method": method, "seed": seed,
            "complete": [
                root / method / f"seed_{seed}" / "checkpoint.pt",
                root / method / f"seed_{seed}" / "training_metrics_1_100.csv",
                root / method / f"seed_{seed}" / "training_provenance.json",
            ],
        }
        for seed in CONFIRMATORY_SEEDS for method in METHOD_ANCHORS
    ]
    timings = _schedule(config_path, root, list(gpu_ids), train_jobs)
    all_checkpoints = [
        root / method / f"seed_{seed}" / "checkpoint.pt"
        for seed in CONFIRMATORY_SEEDS for method in METHOD_ANCHORS
    ]
    if not all(path.is_file() for path in all_checkpoints):
        raise RuntimeError("TEST REMAINS SEALED: all nine checkpoints must complete before evaluation")

    eval_jobs = [
        {
            "name": f"eval_{method}_seed_{seed}", "stage": "eval",
            "method": method, "seed": seed,
            "complete": [
                root / "evaluation" / method / f"seed_{seed}" / "budget_metrics.csv",
                root / "evaluation" / method / f"seed_{seed}" / "representation_local_geometry.csv",
                root / "predictions" / method / f"seed_{seed}" / "predictions_all_widths.csv",
            ],
        }
        for seed in CONFIRMATORY_SEEDS for method in METHOD_ANCHORS
    ]
    timings.update(_schedule(config_path, root, list(gpu_ids), eval_jobs))
    decision = finalize_confirmatory(base, root, protocol)
    pd.DataFrame([
        {"stage": name, "minutes": seconds / 60.0} for name, seconds in timings.items()
    ]).to_csv(root / "finalgeo_runtime_by_job.csv", index=False)
    return {"root": str(root), "decision": decision, "timings": timings}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--root")
    parser.add_argument("--stage", choices=["train", "eval", "full"], default="full")
    parser.add_argument("--method", choices=list(METHOD_ANCHORS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--development-root")
    parser.add_argument("--freeze-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--gpu-ids", nargs="*", type=int)
    args = parser.parse_args()
    if args.stage == "full":
        result = run_confirmatory(
            args.development_root, args.freeze_dir, args.output_dir, args.gpu_ids or [0, 1]
        )
        print(json.dumps(result["decision"], indent=2))
    else:
        if args.config is None or args.root is None or args.method is None or args.seed is None:
            parser.error("worker stages require --config, --root, --method, and --seed")
        run_worker(args.config, args.root, args.stage, args.method, args.seed)


if __name__ == "__main__":
    main()
