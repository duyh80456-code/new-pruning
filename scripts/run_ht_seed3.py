"""T4x2 scheduler for the seed-3 Geo-HT versus Resource-HT development run."""

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

from data import build_development_train_loaders
from rq2_anchor_placement import _sha256
from rq2_dynamic_seed3_pilot import freeze_pilot_policies
from rq2_ht_seed3 import (
    METHODS,
    SEED,
    evaluate_ht_checkpoints,
    finalize_ht_development,
    load_ht_policy,
    simulate_ht_sanity,
    train_ht_method,
    validate_ht_policy,
)


def _load_config(path):
    return yaml.safe_load(Path(path).read_text())


def run_worker(config_path, root, protocol_dir, stage, method):
    config = _load_config(config_path)
    if stage == "train":
        return train_ht_method(config, root, protocol_dir, method)
    if stage == "eval":
        return evaluate_ht_checkpoints(config, root, method)
    raise ValueError(stage)


def _job(config_path, root, protocol_dir, gpu, job):
    command = [
        sys.executable, "-m", "scripts.run_ht_seed3",
        "--config", str(config_path), "--root", str(root),
        "--protocol-dir", str(protocol_dir), "--stage", job["stage"],
        "--method", job["method"],
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.perf_counter()
    print(f"[HT scheduler] {job['name']} starting on physical GPU {gpu}", flush=True)
    subprocess.run(
        command, cwd=Path(__file__).resolve().parents[1], env=environment, check=True
    )
    elapsed = time.perf_counter() - started
    print(f"[HT scheduler] {job['name']} finished in {elapsed/60:.1f} min", flush=True)
    return job["name"], elapsed


def _schedule(config_path, root, protocol_dir, gpu_ids, jobs):
    pending = [job for job in jobs if not all(Path(path).is_file() for path in job["complete"])]
    devices = queue.Queue()
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


def run_ht_development(
    development_root,
    theory_root,
    preview_root,
    variance_gate_path,
    output_dir,
    gpu_ids=(0, 1),
):
    if len(gpu_ids) != 2 or len(set(map(int, gpu_ids))) != 2:
        raise RuntimeError("Geo-HT and Resource-HT must run concurrently on two GPUs")
    development_root, theory_root, preview_root = map(
        Path, (development_root, theory_root, preview_root)
    )
    variance_gate_path, root = Path(variance_gate_path), Path(output_dir)
    gate = json.loads(variance_gate_path.read_text())
    if (
        gate.get("experiment") != "v3_cross_batch_heldout_variance"
        or gate.get("decision") not in {"STRONG_GO", "WEAK_GO"}
        or gate.get("network_training") is not False
        or gate.get("test_used") is not False
    ):
        raise RuntimeError("Frozen cross-batch variance gate does not authorize HT development")
    root.mkdir(parents=True, exist_ok=True)
    protocol_dir = root / "protocol"
    protocol = freeze_pilot_policies(
        theory_root, preview_root, development_root, protocol_dir, pilot_seed=SEED
    )
    gate_copy = protocol_dir / "variance_gate_metadata.json"
    shutil.copy2(variance_gate_path, gate_copy)
    protocol["variance_gate"] = {
        "decision": gate["decision"], "path": gate_copy.name,
        "sha256": _sha256(gate_copy),
    }
    (protocol_dir / "dynamic_pilot_frozen_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )

    config = _load_config(development_root / "resolved_config.yaml")
    config["experiment"]["output_dir"] = str(root)
    config["experiment"]["device"] = "cuda"
    config["experiment"]["confirmatory_seeds"] = [SEED]
    config["training"]["epochs"] = 100
    config_path = root / "resolved_config.yaml"
    config["dataset"]["download"] = True
    build_development_train_loaders(config, training_seed=SEED)
    config["dataset"]["download"] = False
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    sanity_frames, policy_rows = [], []
    for method in METHODS:
        pairs, pi, flops, _ = load_ht_policy(protocol_dir, method)
        checks = validate_ht_policy(pairs, pi, flops)
        sanity_frames.append(simulate_ht_sanity(pairs, pi, method, draws=100_000))
        policy_rows.append({"method": method, **checks})
    policy_table = pd.DataFrame(policy_rows)
    relative_compute_difference = abs(
        policy_table.expected_interior_flops.iloc[0]
        - policy_table.expected_interior_flops.iloc[1]
    ) / policy_table.expected_interior_flops.iloc[0]
    if relative_compute_difference >= 1e-8:
        raise RuntimeError("Geo-HT and Resource-HT expected compute is not matched")
    policy_table["relative_compute_difference_between_methods"] = relative_compute_difference
    policy_table.to_csv(root / "pretraining_policy_checks.csv", index=False)
    sanity = pd.concat(sanity_frames, ignore_index=True)
    sanity.to_csv(root / "pretraining_ht_sanity.csv", index=False)
    if sanity.effective_weight_error.abs().max() > 0.01:
        raise RuntimeError("100k-draw HT effective-weight sanity check failed")

    train_jobs = [{
        "name": f"train_{method}_seed_3", "stage": "train", "method": method,
        "complete": [
            root / method / "seed_3" / "epoch_100.pt",
            root / method / "seed_3" / "training_provenance.json",
        ],
    } for method in METHODS]
    timings = _schedule(config_path, root, protocol_dir, list(gpu_ids), train_jobs)
    eval_jobs = [{
        "name": f"eval_{method}_seed_3", "stage": "eval", "method": method,
        "complete": [
            root / "evaluation" / method / "seed_3" / "evaluation_complete.json"
        ],
    } for method in METHODS]
    timings.update(_schedule(config_path, root, protocol_dir, list(gpu_ids), eval_jobs))
    decision = finalize_ht_development(root)
    pd.DataFrame([
        {"job": name, "minutes": seconds / 60} for name, seconds in timings.items()
    ]).to_csv(root / "runtime_by_job.csv", index=False)
    return {"root": str(root), "decision": decision, "timings": timings}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--root")
    parser.add_argument("--protocol-dir")
    parser.add_argument("--stage", choices=("train", "eval", "full"), default="full")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--development-root")
    parser.add_argument("--theory-root")
    parser.add_argument("--preview-root")
    parser.add_argument("--variance-gate")
    parser.add_argument("--output-dir")
    parser.add_argument("--gpu-ids", nargs="*", type=int)
    args = parser.parse_args()
    if args.stage == "full":
        result = run_ht_development(
            args.development_root, args.theory_root, args.preview_root,
            args.variance_gate, args.output_dir, args.gpu_ids or [0, 1],
        )
        print(json.dumps(result["decision"], indent=2))
    else:
        if not all((args.config, args.root, args.protocol_dir, args.method)):
            parser.error("worker stages require config, root, protocol-dir, and method")
        run_worker(args.config, args.root, args.protocol_dir, args.stage, args.method)


if __name__ == "__main__":
    main()
