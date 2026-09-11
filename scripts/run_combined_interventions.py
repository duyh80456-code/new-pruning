from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from baseline_artifacts import find_confirmatory_root
from cfm_mechanism import run_cfm_mechanism
from scripts.run_geoweighting import finalize_geoweighting, run as run_geoweighting


ANCHORS = {0.25, 0.50, 0.75, 1.00}


def _read_yaml(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def _write_yaml(config: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def _run_module_on_gpu(
    module: str, config_path: Path, gpu_id: int, limit_cpu_threads: bool = False
) -> float:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if limit_cpu_threads:
        # Leave CPU capacity for CIFAR decoding/augmentation in the concurrent
        # A1 lane; the small CFM networks themselves execute on the other GPU.
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
    started = time.perf_counter()
    print(f"[combined] {module} starting on physical GPU {gpu_id}", flush=True)
    subprocess.run(
        [sys.executable, "-m", module, "--config", str(config_path)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
    )
    elapsed = time.perf_counter() - started
    print(
        f"[combined] {module} completed on physical GPU {gpu_id} "
        f"in {elapsed / 60:.1f} minutes",
        flush=True,
    )
    return elapsed


def _write_seed_worker_config(
    full_config: dict, seed: int, worker_root: Path, config_dir: Path
) -> Path:
    config = copy.deepcopy(full_config)
    config["experiment"]["seeds"] = [seed]
    config["experiment"]["output_dir"] = str(worker_root)
    return _write_yaml(config, config_dir / f"a1_seed_{seed}.yaml")


def summarize_dense(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    for seed, seed_frame in frame.groupby("seed"):
        seed_frame = seed_frame.sort_values("budget")
        unseen = seed_frame.loc[~seed_frame["budget"].round(2).isin(ANCHORS)]
        local = seed_frame.loc[seed_frame["local_wasserstein_sensitivity"].notna()]
        cliff = local.loc[local["budget"].between(0.30, 0.45)]
        anchor_rows = seed_frame.loc[seed_frame["budget"].round(2).isin(ANCHORS)]
        log_flops = np.log(seed_frame["flops"].to_numpy(float))
        accuracy = seed_frame["accuracy"].to_numpy(float)
        rows.append(
            {
                "method": method,
                "seed": int(seed),
                "mean_unseen_accuracy": unseen["accuracy"].mean(),
                "worst_unseen_accuracy": unseen["accuracy"].min(),
                "mean_G": local["local_wasserstein_sensitivity"].mean(),
                "max_G": local["local_wasserstein_sensitivity"].max(),
                "cliff_region_max_G": cliff["local_wasserstein_sensitivity"].max(),
                "mean_anchor_accuracy": anchor_rows["accuracy"].mean(),
                "full_width_accuracy": seed_frame.loc[
                    np.isclose(seed_frame["budget"], 1.0), "accuracy"
                ].iloc[0],
                "accuracy_logflops_auc": np.trapz(accuracy, log_flops)
                / (log_flops[-1] - log_flops[0]),
            }
        )
    return pd.DataFrame(rows)


def compare_geoweighting(baseline_root: Path, a1_dir: Path):
    baseline = pd.read_csv(baseline_root / "central_analysis_all_seeds.csv")
    a1 = pd.read_csv(a1_dir / "central_analysis_all_seeds.csv")
    summary = pd.concat(
        [summarize_dense(baseline, "baseline"), summarize_dense(a1, "GeoWeighting A1")],
        ignore_index=True,
    )
    wide = summary.pivot(index="seed", columns="method")
    comparison = pd.DataFrame({"seed": wide.index})
    metrics = [
        "mean_unseen_accuracy",
        "worst_unseen_accuracy",
        "mean_G",
        "max_G",
        "cliff_region_max_G",
        "mean_anchor_accuracy",
        "full_width_accuracy",
        "accuracy_logflops_auc",
    ]
    for metric in metrics:
        comparison[f"baseline_{metric}"] = wide[metric]["baseline"].to_numpy()
        comparison[f"a1_{metric}"] = wide[metric]["GeoWeighting A1"].to_numpy()
        comparison[f"delta_{metric}"] = (
            comparison[f"a1_{metric}"] - comparison[f"baseline_{metric}"]
        )
    checks = {
        "mean_unseen_pass": bool(
            comparison["delta_mean_unseen_accuracy"].mean() > 0
            and (comparison["delta_mean_unseen_accuracy"] > 0).sum() >= 2
        ),
        "worst_pass": bool(
            comparison["delta_worst_unseen_accuracy"].mean() >= -0.002
            and comparison["delta_worst_unseen_accuracy"].min() >= -0.005
        ),
        "geometry_pass": bool(
            comparison["delta_max_G"].mean() < 0
            or comparison["delta_cliff_region_max_G"].mean() < 0
        ),
        "anchor_pass": bool(
            comparison["delta_mean_anchor_accuracy"].mean() >= -0.003
            and comparison["delta_full_width_accuracy"].mean() >= -0.003
        ),
    }
    checks["go"] = all(checks.values())
    summary.to_csv(a1_dir / "baseline_a1_metrics_by_seed.csv", index=False)
    comparison.to_csv(a1_dir / "baseline_a1_paired_comparison.csv", index=False)
    with (a1_dir / "a1_decision.json").open("w") as handle:
        json.dump(checks, handle, indent=2)
    return baseline, a1, summary, comparison, checks


def compare_cfm(results: pd.DataFrame, cfm_dir: Path):
    wide = results.pivot(
        index=["seed", "holdout_width"], columns="method", values="sliced_wasserstein"
    ).reset_index()
    wide["cfm_lt_linear"] = wide["cfm"] < wide["linear_interpolation"]
    wide["cfm_lt_mlp"] = wide["cfm"] < wide["conditional_mlp"]
    wide["cfm_lt_nearest"] = wide["cfm"] < wide["shared_nearest"]
    wide["cfm_vs_linear_reduction"] = (
        wide["linear_interpolation"] - wide["cfm"]
    ) / wide["linear_interpolation"]
    wide["cfm_vs_mlp_reduction"] = (
        wide["conditional_mlp"] - wide["cfm"]
    ) / wide["conditional_mlp"]
    summary = (
        results.groupby(["holdout_width", "method"])[
            ["sliced_wasserstein", "paired_mse", "paired_cosine"]
        ]
        .mean()
        .reset_index()
    )
    checks = {
        "beats_linear_all": bool(wide["cfm_lt_linear"].all()),
        "beats_mlp_all": bool(wide["cfm_lt_mlp"].all()),
        "beats_nearest_all": bool(wide["cfm_lt_nearest"].all()),
        "mean_vs_linear_reduction": float(wide["cfm_vs_linear_reduction"].mean()),
        "mean_vs_mlp_reduction": float(wide["cfm_vs_mlp_reduction"].mean()),
    }
    checks["go"] = checks["beats_linear_all"]
    wide.to_csv(cfm_dir / "cfm_paired_comparison.csv", index=False)
    summary.to_csv(cfm_dir / "cfm_method_summary.csv", index=False)
    with (cfm_dir / "cfm_decision.json").open("w") as handle:
        json.dump(checks, handle, indent=2)
    return wide, summary, checks


def write_combined_report(
    output_dir: Path,
    a1_comparison: pd.DataFrame,
    a1_checks: dict,
    cfm_comparison: pd.DataFrame,
    cfm_checks: dict,
    timings: dict,
) -> Path:
    if a1_checks["go"] and cfm_checks["go"]:
        verdict = "BOTH GO — proceed to A2; CFM is justified for later integration"
    elif a1_checks["go"]:
        verdict = "A1 GO, CFM NO-GO — proceed to A2 only"
    elif cfm_checks["go"]:
        verdict = "CFM GO, A1 NO-GO — inspect A1 and defer A2"
    else:
        verdict = "BOTH NO-GO — inspect the causal intervention story before expanding"
    decision = {
        "combined_verdict": verdict,
        "a1_go": bool(a1_checks["go"]),
        "cfm_go": bool(cfm_checks["go"]),
        **timings,
    }
    with (output_dir / "combined_decision.json").open("w") as handle:
        json.dump(decision, handle, indent=2)
    lines = [
        "# Combined GeoWeighting A1 + CFM mechanism report",
        "",
        f"Decision: **{verdict}**",
        "",
        "The two GO decisions are independent: A1 tests whether frozen geometry is actionable; "
        "CFM tests representation transport without retraining the backbone.",
        "",
        "## GeoWeighting A1 checks",
        "",
        "```json",
        json.dumps(a1_checks, indent=2),
        "```",
        "",
        a1_comparison.to_markdown(index=False),
        "",
        "## CFM mechanism checks",
        "",
        "```json",
        json.dumps(cfm_checks, indent=2),
        "```",
        "",
        cfm_comparison.to_markdown(index=False),
        "",
        "## Runtime",
        "",
        "```json",
        json.dumps(timings, indent=2),
        "```",
    ]
    report = output_dir / "combined_report.md"
    report.write_text("\n".join(lines) + "\n")
    return report


def run_combined(
    a1_config_path: str | Path,
    cfm_config_path: str | Path,
    output_dir: str | Path,
    gpu_ids: list[int] | None = None,
):
    output_dir = Path(output_dir)
    # Re-entering the notebook cell after post-processing/CFM failure must not
    # silently retrain a completed A1 run. Completed stages are reused; a
    # partial A1 directory is preserved and rejected because mixing seed runs
    # would invalidate the paired comparison.
    output_dir.mkdir(parents=True, exist_ok=True)
    a1_dir = output_dir / "geoweighting_a1"
    cfm_dir = output_dir / "cfm_mechanism"
    a1_config = _read_yaml(a1_config_path)
    cfm_config = _read_yaml(cfm_config_path)
    baseline_inputs = {
        str(a1_config["baseline_artifacts"]["input_root"]),
        str(cfm_config["baseline_artifacts"]["input_root"]),
    }
    if len(baseline_inputs) != 1:
        raise ValueError("A1 and CFM must use the same frozen baseline input")
    baseline_root = find_confirmatory_root(baseline_inputs.pop())
    a1_config["experiment"]["output_dir"] = str(a1_dir)
    cfm_config["experiment"]["output_dir"] = str(cfm_dir)
    resolved_dir = output_dir / "resolved_configs"
    a1_resolved = _write_yaml(a1_config, resolved_dir / "geoweighting_a1.yaml")
    cfm_resolved = _write_yaml(cfm_config, resolved_dir / "cfm_mechanism.yaml")

    total_start = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))
    a1_complete = (a1_dir / "central_analysis_all_seeds.csv").is_file() and all(
        (a1_dir / f"seed_{seed}" / "checkpoint.pt").is_file() for seed in (0, 1, 2)
    )
    cfm_results_path = cfm_dir / "cfm_mechanism_results.csv"
    cfm_complete = cfm_results_path.is_file()
    if a1_complete:
        print(f"[combined] reusing completed A1 stage: {a1_dir}", flush=True)
    elif a1_dir.exists():
        raise RuntimeError(
            f"Partial A1 output found at {a1_dir}. Preserve it for diagnosis, "
            "then create a new RUN_NAME before retrying."
        )
    elif len(gpu_ids) >= 2:
        # Static two-lane schedule. CFM is independent of A1 and starts as soon
        # as seed 1 releases GPU 1, overlapping seed 2 on GPU 0.
        worker_base = output_dir / ".a1_seed_workers"
        worker_base.mkdir(parents=True, exist_ok=True)
        worker_roots = {
            seed: worker_base / f"worker_seed_{seed}" for seed in (0, 1, 2)
        }
        worker_configs = {
            seed: _write_seed_worker_config(
                a1_config, seed, worker_roots[seed], resolved_dir
            )
            for seed in (0, 1, 2)
        }

        def run_seed(seed: int, gpu_id: int):
            completed = (
                worker_roots[seed] / f"seed_{seed}" / "checkpoint.pt"
            ).is_file() and (
                worker_roots[seed]
                / f"seed_{seed}"
                / "results"
                / "central_analysis.csv"
            ).is_file()
            if completed:
                print(f"[combined] reusing completed A1 seed {seed} worker", flush=True)
                stage_seconds[f"a1_seed_{seed}"] = 0.0
            else:
                stage_seconds[f"a1_seed_{seed}"] = _run_module_on_gpu(
                    "scripts.run_geoweighting", worker_configs[seed], gpu_id
                )

        def gpu_zero_lane():
            run_seed(0, gpu_ids[0])
            run_seed(2, gpu_ids[0])

        def gpu_one_lane():
            run_seed(1, gpu_ids[1])
            if not cfm_complete:
                stage_seconds["cfm"] = _run_module_on_gpu(
                    "scripts.run_cfm_mechanism",
                    cfm_resolved,
                    gpu_ids[1],
                    limit_cpu_threads=True,
                )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(gpu_zero_lane), executor.submit(gpu_one_lane)]
            for future in futures:
                future.result()

        a1_dir.mkdir(parents=True, exist_ok=False)
        _write_yaml(a1_config, a1_dir / "resolved_config.yaml")
        for seed in (0, 1, 2):
            source = worker_roots[seed] / f"seed_{seed}"
            if not source.is_dir():
                raise FileNotFoundError(f"Missing completed A1 worker output: {source}")
            shutil.move(str(source), str(a1_dir / f"seed_{seed}"))
        finalize_geoweighting(a1_dir, a1_config)
    else:
        a1_start = time.perf_counter()
        run_geoweighting(a1_resolved)
        stage_seconds["a1_sequential"] = time.perf_counter() - a1_start
    baseline, a1, _, a1_comparison, a1_checks = compare_geoweighting(
        baseline_root, a1_dir
    )

    if cfm_results_path.is_file():
        print(f"[combined] reusing completed CFM stage: {cfm_dir}", flush=True)
        cfm_results = pd.read_csv(cfm_results_path)
    else:
        cfm_start = time.perf_counter()
        cfm_results = run_cfm_mechanism(baseline_root, cfm_config, cfm_dir)
        stage_seconds["cfm"] = time.perf_counter() - cfm_start
    cfm_comparison, cfm_summary, cfm_checks = compare_cfm(cfm_results, cfm_dir)
    timings = {
        "schedule": "GPU 0: A1 seed 0 -> seed 2; GPU 1: A1 seed 1 -> CFM",
        "stage_minutes": {
            name: seconds / 60 for name, seconds in sorted(stage_seconds.items())
        },
        "total_hours": (time.perf_counter() - total_start) / 3600,
    }
    report = write_combined_report(
        output_dir, a1_comparison, a1_checks, cfm_comparison, cfm_checks, timings
    )
    return {
        "baseline": baseline,
        "a1": a1,
        "a1_comparison": a1_comparison,
        "a1_checks": a1_checks,
        "cfm_results": cfm_results,
        "cfm_comparison": cfm_comparison,
        "cfm_summary": cfm_summary,
        "cfm_checks": cfm_checks,
        "timings": timings,
        "report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GeoWeighting A1 then cached CFM")
    parser.add_argument("--a1-config", required=True)
    parser.add_argument("--cfm-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-ids", nargs="*", type=int)
    args = parser.parse_args()
    result = run_combined(
        args.a1_config, args.cfm_config, args.output_dir, gpu_ids=args.gpu_ids
    )
    print(f"Completed. Combined report: {result['report']}")


if __name__ == "__main__":
    main()
