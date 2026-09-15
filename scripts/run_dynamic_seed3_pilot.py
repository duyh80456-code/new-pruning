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
import yaml

from data import build_development_train_loaders
from rq2_dynamic_seed3_pilot import (
    DEFAULT_PILOT_SEED,
    GEOMETRY_CONTINUOUS_METHOD,
    METHODS,
    RUN_METHODS,
    evaluate_dynamic_method,
    finalize_pilot,
    freeze_pilot_policies,
    load_frozen_policy,
    simulate_sampler_sanity,
    train_dynamic_method,
    validate_pilot_config,
)


def _load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def run_worker(config_path, root, protocol_dir, stage, method, seed):
    config = _load_config(config_path)
    if stage == "train":
        return train_dynamic_method(config, root, protocol_dir, method, seed)
    if stage == "eval":
        return evaluate_dynamic_method(config, root, method, seed)
    raise ValueError(stage)


def _job(config_path: Path, root: Path, protocol_dir: Path, gpu: int, job: dict):
    command = [
        sys.executable, "-m", "scripts.run_dynamic_seed3_pilot",
        "--config", str(config_path), "--root", str(root),
        "--protocol-dir", str(protocol_dir), "--stage", job["stage"],
        "--method", job["method"], "--seed", str(job["seed"]),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.perf_counter()
    print(f"[Dynamic pilot] {job['name']} starting on physical GPU {gpu}", flush=True)
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env, check=True)
    elapsed = time.perf_counter() - started
    print(f"[Dynamic pilot] {job['name']} finished in {elapsed/60:.1f} min", flush=True)
    return job["name"], elapsed


def _schedule(config_path, root, protocol_dir, gpu_ids, jobs):
    pending = [job for job in jobs if not all(Path(path).is_file() for path in job["complete"])]
    devices: queue.Queue[int] = queue.Queue()
    for gpu in gpu_ids:
        devices.put(int(gpu))
    timings = {}

    def launch(job):
        gpu = devices.get()
        try:
            return _job(config_path, root, protocol_dir, gpu, job)
        finally:
            devices.put(gpu)

    if pending:
        with ThreadPoolExecutor(max_workers=min(len(gpu_ids), len(pending))) as executor:
            futures = [executor.submit(launch, job) for job in pending]
            for future in as_completed(futures):
                name, elapsed = future.result()
                timings[name] = elapsed
    return timings


def run_seed_pilot(
    development_root,
    theory_root,
    preview_root,
    output_dir,
    gpu_ids=(0, 1),
    seed=DEFAULT_PILOT_SEED,
    include_geometry_continuous=False,
):
    if len(gpu_ids) != 2:
        raise RuntimeError("The two dynamic methods should run concurrently on Kaggle T4x2")
    development_root, theory_root, preview_root, root = map(
        Path, (development_root, theory_root, preview_root, output_dir)
    )
    seed = int(seed)
    if seed in (0, 1, 2) or seed < 0:
        raise ValueError("Pilot seed must be nonnegative and outside development seeds 0,1,2")
    active_methods = (
        ("geometry_dynamic", GEOMETRY_CONTINUOUS_METHOD, "resource_dynamic")
        if include_geometry_continuous else METHODS
    )
    base = _load_config(development_root / "resolved_config.yaml")
    base["experiment"]["output_dir"] = str(root)
    base["experiment"]["device"] = "cuda"
    base["experiment"]["confirmatory_seeds"] = [seed]
    base["training"]["epochs"] = 100
    validate_pilot_config(base)
    root.mkdir(parents=True, exist_ok=True)
    protocol_dir = root / "protocol"
    protocol = freeze_pilot_policies(
        theory_root, preview_root, development_root, protocol_dir, pilot_seed=seed
    )
    config_path = root / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(base, sort_keys=False))

    # Materialize CIFAR exactly once before isolated workers to avoid concurrent extraction.
    base["dataset"]["download"] = True
    build_development_train_loaders(base, training_seed=seed)
    base["dataset"]["download"] = False
    config_path.write_text(yaml.safe_dump(base, sort_keys=False))

    sanity = []
    for method in METHODS:
        pairs, pi, flops, _ = load_frozen_policy(protocol_dir, method)
        sanity.append(simulate_sampler_sanity(
            pairs, pi, flops, float(protocol["fixed_endpoint_compute"]), method, seed=seed
        ))
    pd.DataFrame(sanity).to_csv(root / "pretraining_sampler_sanity.csv", index=False)

    train_jobs = [{
        "name": f"train_{method}_seed_{seed}", "stage": "train", "method": method,
        "seed": seed,
        "complete": [
            root / method / f"seed_{seed}" / "epoch_050.pt",
            root / method / f"seed_{seed}" / "epoch_100.pt",
            root / method / f"seed_{seed}" / "training_provenance.json",
        ],
    } for method in active_methods]
    timings = _schedule(config_path, root, protocol_dir, list(gpu_ids), train_jobs)
    if not all((root / method / f"seed_{seed}" / "epoch_100.pt").is_file() for method in active_methods):
        raise RuntimeError("Both epoch-100 checkpoints must exist before validation comparison")

    eval_jobs = [{
        "name": f"eval_{method}_seed_{seed}", "stage": "eval", "method": method,
        "seed": seed,
        "complete": [root / "evaluation" / method / f"seed_{seed}" / "dense_validation_accuracy.csv"],
    } for method in active_methods]
    timings.update(_schedule(config_path, root, protocol_dir, list(gpu_ids), eval_jobs))
    decision = finalize_pilot(root, seed=seed, methods=active_methods)
    pd.DataFrame([
        {"job": name, "minutes": seconds / 60.0} for name, seconds in timings.items()
    ]).to_csv(root / "runtime_by_job.csv", index=False)
    return {"root": str(root), "decision": decision, "timings": timings}


def run_seed3_pilot(development_root, theory_root, preview_root, output_dir, gpu_ids=(0, 1)):
    """Backward-compatible entry point for the original seed-3 notebook."""
    return run_seed_pilot(
        development_root, theory_root, preview_root, output_dir,
        gpu_ids=gpu_ids, seed=3,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--root")
    parser.add_argument("--protocol-dir")
    parser.add_argument("--stage", choices=("train", "eval", "full"), default="full")
    parser.add_argument("--method", choices=RUN_METHODS)
    parser.add_argument("--development-root")
    parser.add_argument("--theory-root")
    parser.add_argument("--preview-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--gpu-ids", nargs="*", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_PILOT_SEED)
    parser.add_argument("--include-geometry-continuous", action="store_true")
    args = parser.parse_args()
    if args.stage == "full":
        result = run_seed_pilot(
            args.development_root, args.theory_root, args.preview_root,
            args.output_dir, args.gpu_ids or [0, 1], seed=args.seed,
            include_geometry_continuous=args.include_geometry_continuous,
        )
        print(json.dumps(result["decision"], indent=2))
    else:
        if not all((args.config, args.root, args.protocol_dir, args.method)):
            parser.error("worker stages require config, root, protocol-dir, and method")
        run_worker(args.config, args.root, args.protocol_dir, args.stage, args.method, args.seed)


if __name__ == "__main__":
    main()
