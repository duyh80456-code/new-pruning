"""Two-GPU scheduler for the three-way end-to-end pairwise pilot."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from threading import Lock
from pathlib import Path

import pandas as pd

from rq2_e2e_pairwise_pilot import (
    ALL_METHODS,
    DIAGNOSTIC_STATES,
    METHODS,
    RPGEO_DIAGNOSTIC_STATES,
    extract_diagnostic_state,
    load_config,
    train_branch,
)
from rq2_rpgeo_gate import (
    GATE_STATES,
    run_rpgeo_offline_gate,
    run_rpgeo_retention_probe,
)


_RUNTIME_LOCK = Lock()


def _schedule(jobs, launch, gpu_ids, label):
    available, pending, running, timings = list(map(int, gpu_ids)), list(jobs), {}, []
    if not available:
        raise RuntimeError("At least one GPU is required")
    with ThreadPoolExecutor(max_workers=len(available)) as pool:
        while pending or running:
            while pending and available:
                job, gpu = pending.pop(0), available.pop(0)
                print(f"[{label}] {job} starting on physical GPU {gpu}", flush=True)
                running[pool.submit(launch, job, gpu)] = (job, gpu)
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                job, gpu = running.pop(future)
                code, seconds = future.result(); available.append(gpu)
                timings.append({"job": str(job), "physical_gpu": gpu, "minutes": seconds / 60})
                if code:
                    raise RuntimeError(f"{label} job {job} failed with code {code}")
                print(f"[{label}] {job} finished in {seconds/60:.1f} min", flush=True)
    return pd.DataFrame(timings)


def run_branches(root, dataset_root, gate_a_summary, gpu_ids=(0, 1), methods=METHODS):
    root, dataset_root, gate_a_summary = map(Path, (root, dataset_root, gate_a_summary))
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or not set(methods).issubset(ALL_METHODS):
        raise ValueError(f"methods must be a non-empty unique subset of {ALL_METHODS}, got {methods}")

    def launch(method, gpu):
        command = [
            sys.executable, "-m", "scripts.run_e2e_pairwise_pilot", "--branch-worker",
            "--root", str(root), "--dataset-root", str(dataset_root),
            "--gate-a-summary", str(gate_a_summary), "--method", method,
        ]
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        started = time.perf_counter()
        return subprocess.run(command, env=env).returncode, time.perf_counter() - started

    timings = _schedule(methods, launch, gpu_ids, "e2e branches")
    runtime_path = root / "branch_runtime.csv"
    with _RUNTIME_LOCK:
        if runtime_path.is_file():
            timings = pd.concat([pd.read_csv(runtime_path), timings], ignore_index=True)
            timings = timings.drop_duplicates(subset=["job"], keep="last")
        timings.to_csv(runtime_path, index=False)
    return timings


def run_diagnostics(
    root, dataset_root, gate_a_summary, gpu_ids=(0, 1), jobs=DIAGNOSTIC_STATES,
):
    root, dataset_root, gate_a_summary = map(Path, (root, dataset_root, gate_a_summary))
    jobs = tuple((str(method), int(epoch)) for method, epoch in jobs)
    allowed = set(RPGEO_DIAGNOSTIC_STATES)
    if not jobs or len(set(jobs)) != len(jobs) or not set(jobs).issubset(allowed):
        raise ValueError(f"jobs must be a non-empty unique subset of {RPGEO_DIAGNOSTIC_STATES}")

    def launch(job, gpu):
        method, epoch = job
        output = root / "diagnostics" / f"{method}_E{epoch}"
        metadata = output / "metadata.json"
        if metadata.is_file():
            try:
                if json.loads(metadata.read_text()).get("status") == "E2E_PAIRWISE_DIAGNOSTIC_COMPLETE":
                    return 0, 0.0
            except json.JSONDecodeError:
                pass
        command = [
            sys.executable, "-m", "scripts.run_e2e_pairwise_pilot", "--diagnostic-worker",
            "--root", str(root), "--dataset-root", str(dataset_root),
            "--gate-a-summary", str(gate_a_summary), "--method", method,
            "--epoch", str(epoch), "--output", str(output),
        ]
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        started = time.perf_counter()
        return subprocess.run(command, env=env).returncode, time.perf_counter() - started

    timings = _schedule(jobs, launch, gpu_ids, "e2e diagnostics")
    runtime_path = root / "diagnostic_runtime.csv"
    with _RUNTIME_LOCK:
        if runtime_path.is_file():
            timings = pd.concat([pd.read_csv(runtime_path), timings], ignore_index=True)
            timings = timings.drop_duplicates(subset=["job"], keep="last")
        timings.to_csv(runtime_path, index=False)
    return timings


def run_completion(root, dataset_root, gate_a_summary, gpu_ids=(0, 1)):
    """Train Uniform while the other GPU diagnoses already-complete branches."""
    gpu_ids = tuple(map(int, gpu_ids))
    if len(gpu_ids) != 2:
        raise RuntimeError("Optimized completion requires exactly two GPUs")
    existing_jobs = tuple(
        job for job in DIAGNOSTIC_STATES if job[0] in {"common_warmup", "resource", "pure_sw"}
    )
    uniform_jobs = tuple(job for job in DIAGNOSTIC_STATES if job[0] == "uniform")
    with ThreadPoolExecutor(max_workers=2) as pool:
        branch_future = pool.submit(
            run_branches, root, dataset_root, gate_a_summary, (gpu_ids[0],), ("uniform",)
        )
        diagnostic_future = pool.submit(
            run_diagnostics, root, dataset_root, gate_a_summary, (gpu_ids[1],), existing_jobs
        )
        branch_runtime = branch_future.result()
        diagnostic_future.result()
    diagnostic_runtime = run_diagnostics(
        root, dataset_root, gate_a_summary, gpu_ids, uniform_jobs
    )
    return branch_runtime, diagnostic_runtime


def run_rpgeo_extension(root, dataset_root, gate_a_summary, gpu_ids=(0, 1)):
    """Run Uniform and the gated RP-Geo lane concurrently on two GPUs."""
    root, dataset_root, gate_a_summary = map(Path, (root, dataset_root, gate_a_summary))
    gpu_ids = tuple(map(int, gpu_ids))
    if len(gpu_ids) != 2:
        raise RuntimeError("RP-Geo extension requires exactly two GPUs")
    required = (
        root / "common_warmup" / "epoch_010.pt",
        root / "resource" / "checkpoints" / "epoch_100.pt",
        root / "pure_sw" / "checkpoints" / "epoch_100.pt",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Attach the completed protocol-v2 Resource/Pure-SW root: {missing}")

    def rpgeo_lane():
        run_diagnostics(root, dataset_root, gate_a_summary, (gpu_ids[1],), GATE_STATES)
        gate = run_rpgeo_offline_gate(root, gate_a_summary, root / "rpgeo_offline_gate")
        if gate["decision"] != "GO":
            probe = run_rpgeo_retention_probe(
                root, gate_a_summary, root / "rpgeo_retention_probe"
            )
            return {"gate": gate, "retention_probe": probe}
        run_branches(
            root, dataset_root, gate_a_summary, (gpu_ids[1],), ("resource_geo",)
        )
        return {"gate": gate, "retention_probe": None}

    with ThreadPoolExecutor(max_workers=2) as pool:
        uniform_future = pool.submit(
            run_branches, root, dataset_root, gate_a_summary, (gpu_ids[0],), ("uniform",)
        )
        rpgeo_future = pool.submit(rpgeo_lane)
        uniform_runtime = uniform_future.result()
        lane = rpgeo_future.result()
    gate = lane["gate"]
    if gate["decision"] != "GO":
        return {
            "status": "RPGEO_EXTENSION_STOPPED_AT_GATE",
            "gate": gate,
            "retention_probe": lane["retention_probe"],
            "uniform_complete": True,
            "resource_geo_complete": False,
        }, uniform_runtime, pd.DataFrame()
    remaining = tuple(
        job for job in RPGEO_DIAGNOSTIC_STATES
        if job[0] in {"uniform", "resource_geo"}
    )
    diagnostic_runtime = run_diagnostics(
        root, dataset_root, gate_a_summary, gpu_ids, remaining
    )
    return {
        "status": "RPGEO_EXTENSION_TRAINING_COMPLETE",
        "gate": gate,
        "uniform_complete": True,
        "resource_geo_complete": True,
    }, uniform_runtime, diagnostic_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-worker", action="store_true")
    parser.add_argument("--diagnostic-worker", action="store_true")
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--gate-a-summary", required=True)
    parser.add_argument("--method")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.root)
    config = load_config(root / "resolved_config.yaml", args.dataset_root)
    if args.branch_worker:
        train_branch(config, root, args.gate_a_summary, args.method)
    elif args.diagnostic_worker:
        extract_diagnostic_state(
            root, args.method, args.epoch, args.dataset_root,
            args.gate_a_summary, args.output, "cuda:0",
        )
    else:
        raise RuntimeError("Select one worker mode")


if __name__ == "__main__":
    main()
