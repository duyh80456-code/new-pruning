from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
import yaml

from data import build_confirmatory_loaders
from research_utils import budget_tag, load_config, seed_everything
from s1_width import (
    SPECIALIZED_WIDTHS,
    evaluate_shared_checkpoint,
    evaluate_specialized_checkpoints,
    finalize_s1,
    make_model,
    train_shared_reference,
    train_specialized_reference,
)
from scripts.run_s1_width import validate_protocol


def _is_complete_source(root: Path) -> bool:
    required = [
        root / "resolved_config.yaml",
        root / "specialization_table.csv",
        root / "shared_dense_metrics_all_seeds.csv",
        root / "protocol" / "fixed_random_projection.pt",
    ]
    required += [root / "shared" / f"seed_{seed}" / "checkpoint.pt" for seed in (0, 1, 2)]
    required += [
        root / "specialized" / f"seed_{seed}" / f"width_{budget_tag(width)}" /
        "best_checkpoint.pt"
        for seed in (0, 1, 2) for width in SPECIALIZED_WIDTHS
    ]
    return all(path.is_file() for path in required)


def materialize_source(input_root: str | Path, destination: str | Path) -> Path:
    """Find a direct S1 output or safely unpack its exported ZIP."""
    input_root, destination = Path(input_root), Path(destination)
    direct = sorted({path.parent for path in input_root.rglob("s1_complete.json")})
    direct = [root for root in direct if _is_complete_source(root)]
    if len(direct) == 1:
        return direct[0]
    if len(direct) > 1:
        raise RuntimeError(f"Multiple complete S1 source runs found: {direct}")
    archives = sorted(input_root.rglob("kaggle-s1-width-*.zip"))
    if len(archives) != 1:
        raise FileNotFoundError(f"Expected exactly one S1 ZIP below {input_root}, found: {archives}")
    if not _is_complete_source(destination):
        destination.mkdir(parents=True, exist_ok=True)
        resolved_destination = destination.resolve()
        with zipfile.ZipFile(archives[0]) as bundle:
            for member in bundle.infolist():
                target = (destination / member.filename).resolve()
                if resolved_destination not in target.parents and target != resolved_destination:
                    raise RuntimeError(f"Unsafe path in S1 archive: {member.filename}")
            bundle.extractall(destination)
    if not _is_complete_source(destination):
        raise RuntimeError(f"Extracted S1 archive is incomplete: {archives[0]}")
    return destination


def _load_source_protocol(source_root: Path, config: dict) -> dict:
    source = yaml.safe_load((source_root / "resolved_config.yaml").read_text())
    expected = int(config["experiment"]["source_horizon"])
    if int(source["training"]["epochs"]) != expected:
        raise ValueError(
            f"Source horizon is {source['training']['epochs']}, expected {expected}"
        )
    for key in ("train_widths", "eval_widths"):
        if source["compression"][key] != config["compression"][key]:
            raise ValueError(f"Source/config compression.{key} differ")
    if source["experiment"]["seeds"] != config["experiment"]["seeds"]:
        raise ValueError("Source/config seeds differ")
    return source


def run_worker(config_path, source_root, stage, seed, width=None):
    config = load_config(config_path); validate_protocol(config)
    source_root = Path(source_root); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(int(seed))
    loaders = build_confirmatory_loaders(config, training_seed=int(seed))
    root = Path(config["experiment"]["output_dir"])
    extension_lr = float(config["experiment"]["extension_learning_rate"])
    if stage == "train_shared":
        source = source_root / "shared" / f"seed_{seed}" / "checkpoint.pt"
        return train_shared_reference(
            make_model(config, device), loaders.train, config, device,
            root / "shared" / f"seed_{seed}", initial_checkpoint=source,
            initial_epoch=int(config["experiment"]["source_horizon"]),
            learning_rate_override=extension_lr,
        )
    if stage == "train_specialized":
        width = float(width)
        source = (
            source_root / "specialized" / f"seed_{seed}" /
            f"width_{budget_tag(width)}" / "best_checkpoint.pt"
        )
        payload = torch.load(source, map_location="cpu", weights_only=False)
        return train_specialized_reference(
            make_model(config, device), width, loaders, config, device, int(seed),
            root / "specialized" / f"seed_{seed}" / f"width_{budget_tag(width)}",
            initial_checkpoint=source, initial_epoch=int(payload["best"]["epoch"]),
            initial_best=payload["best"], learning_rate_override=extension_lr,
        )
    if stage == "evaluate_shared":
        matrix = torch.load(root / "protocol" / "fixed_random_projection.pt", weights_only=False)
        return evaluate_shared_checkpoint(
            root / "shared" / f"seed_{seed}" / "checkpoint.pt", loaders, config, device,
            int(seed), root / "shared" / f"seed_{seed}" / "evaluation", matrix,
        )
    if stage == "evaluate_specialized":
        return evaluate_specialized_checkpoints(root, loaders, config, device, int(seed))
    raise ValueError(stage)


def _schedule(config_path, source_root, gpu_ids, jobs):
    available: queue.Queue[int] = queue.Queue()
    for gpu in gpu_ids: available.put(int(gpu))
    project_root = Path(__file__).resolve().parents[1]

    def launch(job):
        gpu = available.get(); started = time.perf_counter()
        command = [sys.executable, "-m", "scripts.run_s1_extension", "--config", str(config_path),
                   "--source-root", str(source_root), "--stage", job["stage"],
                   "--seed", str(job["seed"])]
        if "width" in job: command += ["--width", str(job["width"])]
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print(f"[S1-100 scheduler] {job['name']} starting on physical GPU {gpu}", flush=True)
        try:
            subprocess.run(command, cwd=project_root, env=env, check=True)
        finally:
            available.put(gpu)
        elapsed = time.perf_counter() - started
        print(f"[S1-100 scheduler] {job['name']} finished in {elapsed/60:.1f} min", flush=True)
        return job["name"], elapsed

    pending = [job for job in jobs if not Path(job["complete"]).is_file()]
    timings = {}
    with ThreadPoolExecutor(max_workers=min(len(gpu_ids), len(pending) or 1)) as executor:
        futures = [executor.submit(launch, job) for job in pending]
        for future in as_completed(futures):
            name, elapsed = future.result(); timings[name] = elapsed
    return timings


def _write_comparison(source_root: Path, root: Path):
    source_shared = pd.read_csv(source_root / "shared_dense_metrics_all_seeds.csv")
    extended_shared = pd.read_csv(root / "shared_dense_metrics_all_seeds.csv")
    shared = source_shared.merge(extended_shared, on=["seed", "budget"], suffixes=("_50", "_100"))
    shared["accuracy_delta_100_minus_50"] = shared["accuracy_100"] - shared["accuracy_50"]
    shared.to_csv(root / "shared_50_to_100_comparison.csv", index=False)
    source_spec = pd.read_csv(source_root / "specialization_table.csv")
    extended_spec = pd.read_csv(root / "specialization_table.csv")
    spec = source_spec.merge(extended_spec, on=["seed", "width"], suffixes=("_50", "_100"))
    spec["gap_delta_100_minus_50"] = spec["specialization_gap_100"] - spec["specialization_gap_50"]
    spec.to_csv(root / "specialization_50_to_100_comparison.csv", index=False)
    with (root / "s1_report.md").open("a") as handle:
        handle.write("\n## 50-to-100 extension provenance\n\n")
        handle.write(
            "This is a weights-only warm-start extension with a reset optimizer and a new "
            "cosine schedule; it is not equivalent to a from-scratch T_max=100 run.\n\n"
        )
        handle.write(spec[["seed", "width", "specialization_gap_50",
                           "specialization_gap_100", "gap_delta_100_minus_50"]].to_markdown(index=False))
        handle.write("\n")


def run_extension(config_path, source_root, gpu_ids=(0, 1)):
    config = load_config(config_path); validate_protocol(config)
    source_root = Path(source_root); _load_source_protocol(source_root, config)
    target = int(config["experiment"]["target_horizon"])
    if int(config["training"]["epochs"]) != target or target != 100:
        raise ValueError("Extension target must be locked to epoch 100")
    config["horizon_selection"] = {"selection_basis": "weights_only_extension",
                                    "selected_horizon": target}
    root = Path(config["experiment"]["output_dir"]); root.mkdir(parents=True, exist_ok=True)
    protocol = root / "protocol"; protocol.mkdir(parents=True, exist_ok=True)
    resolved = root / "resolved_config.yaml"; resolved.write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copy2(source_root / "protocol" / "fixed_random_projection.pt",
                 protocol / "fixed_random_projection.pt")
    provenance = {
        "source_root": str(source_root), "source_horizon": 50, "target_horizon": 100,
        "optimizer_state_reused": False, "scheduler_state_reused": False,
        "extension_learning_rate": float(config["experiment"]["extension_learning_rate"]),
    }
    (protocol / "extension_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    seeds = [0, 1, 2]
    jobs = [{"name": f"shared_seed_{s}", "stage": "train_shared", "seed": s,
             "complete": root / "shared" / f"seed_{s}" / "checkpoint.pt"} for s in seeds]
    jobs += [{"name": f"specialized_seed_{s}_width_{w:.2f}", "stage": "train_specialized",
              "seed": s, "width": w,
              "complete": root / "specialized" / f"seed_{s}" /
                          f"width_{budget_tag(w)}" / "training_complete.json"}
             for s in seeds for w in SPECIALIZED_WIDTHS]
    timings = _schedule(resolved, source_root, list(gpu_ids), jobs)
    if not all(Path(job["complete"]).is_file() for job in jobs):
        raise RuntimeError("Extension training incomplete; test remains sealed")
    shared_eval = [{"name": f"evaluate_shared_seed_{s}", "stage": "evaluate_shared", "seed": s,
                    "complete": root / "shared" / f"seed_{s}" / "evaluation" / "budget_metrics.csv"}
                   for s in seeds]
    timings.update(_schedule(resolved, source_root, list(gpu_ids), shared_eval))
    spec_eval = [{"name": f"evaluate_specialized_seed_{s}", "stage": "evaluate_specialized", "seed": s,
                  "complete": root / "specialized" / f"seed_{s}" / "specialized_metrics.csv"}
                 for s in seeds]
    timings.update(_schedule(resolved, source_root, list(gpu_ids), spec_eval))
    result = finalize_s1(root, config); _write_comparison(source_root, root)
    (root / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")
    (root / "s1_extension_complete.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--source-root", required=True)
    parser.add_argument("--stage", required=True); parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--width", type=float)
    args = parser.parse_args()
    run_worker(args.config, args.source_root, args.stage, args.seed, args.width)


if __name__ == "__main__": main()
