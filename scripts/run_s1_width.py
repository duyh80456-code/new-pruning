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

import torch
import yaml

from data import build_confirmatory_loaders
from research_utils import budget_tag, load_config, seed_everything
from s1_width import (
    SPECIALIZED_WIDTHS,
    evaluate_shared_checkpoint,
    evaluate_specialized_checkpoints,
    finalize_s1,
    fixed_random_projection,
    make_model,
    read_s0_selection,
    train_shared_reference,
    train_specialized_reference,
)


def validate_protocol(config: dict) -> None:
    if config["dataset"]["name"].lower() != "cifar100":
        raise ValueError("S1 requires CIFAR-100")
    if config["model"]["backbone"] != "slimmable_resnet18":
        raise ValueError("S1 requires Slimmable ResNet-18")
    if list(map(float, config["compression"]["train_widths"])) != [0.25, 0.5, 0.75, 1.0]:
        raise ValueError("S1 anchors are locked to .25,.50,.75,1.0")
    expected = [round(0.25 + 0.05 * index, 2) for index in range(16)]
    if list(map(float, config["compression"]["eval_widths"])) != expected:
        raise ValueError("S1 dense grid must be .25,.30,...,1.0")
    if list(map(float, config["specialization"]["widths"])) != list(SPECIALIZED_WIDTHS):
        raise ValueError("S1 specialized widths are locked to .30,.40,.60,.80")
    if list(map(int, config["experiment"]["seeds"])) != [0, 1, 2]:
        raise ValueError("S1 seeds are locked to 0,1,2")
    if config["specialization"].get("initialization") != "scratch":
        raise ValueError("S1 specialized references must be independently initialized")


def run_worker(config_path: str | Path, stage: str, seed: int, width: float | None = None):
    config = load_config(config_path); validate_protocol(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(int(seed))
    root = Path(config["experiment"]["output_dir"])
    loaders = build_confirmatory_loaders(config, training_seed=int(seed))
    if stage == "train_shared":
        output = root / "shared" / f"seed_{seed}"
        return train_shared_reference(make_model(config, device), loaders.train, config, device, output)
    if stage == "train_specialized":
        if width is None:
            raise ValueError("train_specialized requires --width")
        output = root / "specialized" / f"seed_{seed}" / f"width_{budget_tag(width)}"
        return train_specialized_reference(
            make_model(config, device), float(width), loaders, config, device, int(seed), output
        )
    if stage == "evaluate_shared":
        matrix = torch.load(root / "protocol" / "fixed_random_projection.pt", weights_only=False)
        checkpoint = root / "shared" / f"seed_{seed}" / "checkpoint.pt"
        return evaluate_shared_checkpoint(
            checkpoint, loaders, config, device, int(seed),
            root / "shared" / f"seed_{seed}" / "evaluation", matrix,
        )
    if stage == "evaluate_specialized":
        return evaluate_specialized_checkpoints(root, loaders, config, device, int(seed))
    raise ValueError(f"Unknown S1 worker stage: {stage}")


def _schedule(config_path: Path, gpu_ids: list[int], jobs: list[dict]) -> dict:
    available: queue.Queue[int] = queue.Queue()
    for gpu_id in gpu_ids:
        available.put(int(gpu_id))
    project_root = Path(__file__).resolve().parents[1]

    def launch(job):
        gpu_id = available.get(); started = time.perf_counter()
        command = [sys.executable, "-m", "scripts.run_s1_width", "--config", str(config_path),
                   "--stage", job["stage"], "--seed", str(job["seed"])]
        if "width" in job:
            command += ["--width", str(job["width"])]
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        label = job["name"]
        print(f"[S1 scheduler] {label} starting on physical GPU {gpu_id}", flush=True)
        try:
            subprocess.run(command, cwd=project_root, env=env, check=True)
        finally:
            available.put(gpu_id)
        elapsed = time.perf_counter() - started
        print(f"[S1 scheduler] {label} finished in {elapsed/60:.1f} min", flush=True)
        return label, elapsed

    pending = [job for job in jobs if not Path(job["complete"]).is_file()]
    timings = {}
    with ThreadPoolExecutor(max_workers=min(len(gpu_ids), len(pending) or 1)) as executor:
        futures = [executor.submit(launch, job) for job in pending]
        for future in as_completed(futures):
            name, elapsed = future.result(); timings[name] = elapsed
    return timings


def run_s1(config_path: str | Path, s0_selection_path: str | Path, gpu_ids=(0, 1)) -> dict:
    config_path = Path(config_path)
    config = load_config(config_path); validate_protocol(config)
    selection = read_s0_selection(s0_selection_path)
    config["training"]["epochs"] = selection["selected_horizon"]
    config["specialization"]["epochs"] = selection["selected_horizon"]
    config["horizon_selection"] = selection
    root = Path(config["experiment"]["output_dir"]); root.mkdir(parents=True, exist_ok=True)
    protocol = root / "protocol"; protocol.mkdir(parents=True, exist_ok=True)
    selection_record = protocol / "s0_selection.json"
    if selection_record.is_file():
        previous = json.loads(selection_record.read_text())
        if previous["selected_horizon"] != selection["selected_horizon"]:
            raise RuntimeError(
                "Refusing to resume one RUN_DIR with a different training horizon: "
                f"{previous['selected_horizon']} != {selection['selected_horizon']}"
            )
    resolved = root / "resolved_config.yaml"
    resolved.write_text(yaml.safe_dump(config, sort_keys=False))
    selection_record.write_text(json.dumps(selection, indent=2) + "\n")
    matrix_path = protocol / "fixed_random_projection.pt"
    if not matrix_path.is_file():
        torch.save(fixed_random_projection(512, 128, int(config["representations"]["random_seed"])), matrix_path)

    seeds = list(map(int, config["experiment"]["seeds"]))
    training_jobs = []
    for seed in seeds:
        training_jobs.append({"name": f"shared_seed_{seed}", "stage": "train_shared", "seed": seed,
                              "complete": root / "shared" / f"seed_{seed}" / "checkpoint.pt"})
    for seed in seeds:
        for width in SPECIALIZED_WIDTHS:
            training_jobs.append({
                "name": f"specialized_seed_{seed}_width_{width:.2f}",
                "stage": "train_specialized", "seed": seed, "width": width,
                "complete": root / "specialized" / f"seed_{seed}" /
                            f"width_{budget_tag(width)}" / "training_complete.json",
            })
    timings = _schedule(resolved, list(gpu_ids), training_jobs)
    required = [Path(job["complete"]) for job in training_jobs]
    if not all(path.is_file() for path in required):
        raise RuntimeError("S1 training incomplete; test grid remains sealed")

    # Test data is opened only after all 3 shared and 12 specialized checkpoints exist.
    shared_eval_jobs = [{
        "name": f"evaluate_shared_seed_{seed}", "stage": "evaluate_shared", "seed": seed,
        "complete": root / "shared" / f"seed_{seed}" / "evaluation" / "budget_metrics.csv",
    } for seed in seeds]
    timings.update(_schedule(resolved, list(gpu_ids), shared_eval_jobs))
    specialized_eval_jobs = [{
        "name": f"evaluate_specialized_seed_{seed}", "stage": "evaluate_specialized", "seed": seed,
        "complete": root / "specialized" / f"seed_{seed}" / "specialized_metrics.csv",
    } for seed in seeds]
    timings.update(_schedule(resolved, list(gpu_ids), specialized_eval_jobs))
    result = finalize_s1(root, config)
    (root / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")
    (root / "s1_complete.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True,
                        choices=["train_shared", "train_specialized", "evaluate_shared", "evaluate_specialized"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--width", type=float)
    args = parser.parse_args()
    run_worker(args.config, args.stage, args.seed, args.width)


if __name__ == "__main__":
    main()
