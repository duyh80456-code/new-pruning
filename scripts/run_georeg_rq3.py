from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from baseline_artifacts import find_confirmatory_root
from data import build_confirmatory_loaders
from evaluation import evaluate_grid, evaluate_prediction_grid
from geometry.analysis import analyze_seed
from georeg_curvature import (
    ANCHORS,
    evaluate_anchor_checkpoint,
    fixed_projection_directions,
    load_anchor_feature_bank,
    load_training_checkpoint,
    make_training_objects,
    paired_accuracy_bootstrap,
    paired_curvature_bootstrap,
    train_segment,
)
from models import slimmable_resnet18
from research_utils import load_config, seed_everything


UNSEEN_WIDTHS = (0.30, 0.35, 0.40, 0.45, 0.55, 0.60, 0.65, 0.70, 0.80, 0.85, 0.90, 0.95)


def validate_rq3_config(config: dict) -> None:
    if [float(value) for value in config["compression"]["train_widths"]] != list(ANCHORS):
        raise ValueError("RQ3 must train only the four registered anchors")
    if [int(value) for value in config["experiment"]["seeds"]] != [0, 1, 2]:
        raise ValueError("RQ3 seeds must be [0, 1, 2]")
    if int(config["experiment"]["development_seed"]) != 0:
        raise ValueError("Seed 0 must remain the development seed")
    if [int(value) for value in config["experiment"]["confirmatory_seeds"]] != [1, 2]:
        raise ValueError("Seeds 1 and 2 must remain confirmatory")
    if int(config["dataset"].get("num_workers", -1)) != 0:
        raise ValueError("Exact augmentation replay requires dataset.num_workers=0")
    if [float(value) for value in config["georeg"]["sanity_multipliers"]] != [0.0, 0.5, 1.0, 2.0]:
        raise ValueError("Sanity multipliers are preregistered as [0, 0.5, 1, 2]")
    if int(config["georeg"]["sanity_epoch"]) >= int(config["training"]["epochs"]):
        raise ValueError("Sanity epoch must precede the 20-epoch final horizon")
    if int(config["georeg"]["train_projection_seed"]) == int(
        config["georeg"]["heldout_projection_seed"]
    ):
        raise ValueError("Held-out directions must use an independent seed")


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def _write_yaml(config: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _directions(config: dict, kind: str, device="cpu"):
    cfg = config["georeg"]
    return fixed_projection_directions(
        int(config["model"]["projection_dim"]),
        int(cfg[f"{kind}_projection_count"]),
        int(cfg[f"{kind}_projection_seed"]),
        device,
    )


def _save_directions(config: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for kind in ("train", "heldout"):
        tensor = _directions(config, kind)
        path = output_dir / f"{kind}_projection_directions.pt"
        torch.save(tensor, path)
        digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        records.append(
            {
                "kind": kind,
                "count": tensor.shape[1],
                "seed": config["georeg"][f"{kind}_projection_seed"],
                "sha256": digest,
            }
        )
    pd.DataFrame(records).to_csv(output_dir / "projection_direction_manifest.csv", index=False)


def _worker_warmup(config: dict, output_dir: Path) -> None:
    if int(config["dataset"].get("num_workers", -1)) != 0:
        raise ValueError("RQ3 exact branch replay requires dataset.num_workers=0")
    device = torch.device(config["experiment"]["device"])
    seed = int(config["experiment"]["development_seed"])
    loaders = build_confirmatory_loaders(config, training_seed=seed)
    model, optimizer, scheduler = make_training_objects(config, seed, device)
    directions = _directions(config, "train", device)
    _, gradients, checkpoint = train_segment(
        model, optimizer, scheduler, loaders.train, config, device, directions,
        start_epoch=1, end_epoch=1, lambda_geo=0.0, output_dir=output_dir,
        gradient_diagnostic_batches=int(config["georeg"]["gradient_calibration_batches"]),
    )
    finite = gradients.loc[
        np.isfinite(gradients["projection_ratio"]) & (gradients["projection_ratio"] > 0)
    ]
    if len(finite) < 4:
        raise RuntimeError("Too few finite projection-head gradient ratios for calibration")
    ratio = float(finite["projection_ratio"].median())
    lambda_zero = float(config["georeg"]["target_gradient_fraction"]) * ratio
    finite_last_block = finite.loc[np.isfinite(finite["last_block_ratio"]), "last_block_ratio"]
    summary = {
        "lambda_0": lambda_zero,
        "median_projection_task_to_geo_grad_ratio": ratio,
        "median_last_block_task_to_geo_grad_ratio": (
            float(finite_last_block.median()) if len(finite_last_block) else None
        ),
        "calibration_batches": int(len(finite)),
        "warmup_checkpoint": str(checkpoint),
    }
    (output_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[RQ3] calibrated lambda_0={lambda_zero:.8g}", flush=True)


def _worker_sanity(
    config: dict, warmup_checkpoint: Path, multiplier: float, output_dir: Path
) -> None:
    device = torch.device(config["experiment"]["device"])
    seed = int(config["experiment"]["development_seed"])
    loaders = build_confirmatory_loaders(config, training_seed=seed)
    model, optimizer, scheduler = make_training_objects(config, seed, device)
    loaded_epoch = load_training_checkpoint(
        warmup_checkpoint, model, optimizer, scheduler, loaders.train, device
    )
    calibration = json.loads((warmup_checkpoint.parent / "calibration_summary.json").read_text())
    lambda_value = float(multiplier) * float(calibration["lambda_0"])
    _, gradients, checkpoint = train_segment(
        model, optimizer, scheduler, loaders.train, config, device, _directions(config, "train", device),
        start_epoch=loaded_epoch + 1, end_epoch=int(config["georeg"]["sanity_epoch"]),
        lambda_geo=lambda_value, output_dir=output_dir,
        gradient_diagnostic_batches=int(config["georeg"]["gradient_diagnostic_batches"]),
    )
    metrics, geometry = evaluate_anchor_checkpoint(
        checkpoint, config, seed, _directions(config, "train", device),
        output_dir / "anchor_evaluation", device,
    )
    summary = {
        "multiplier": float(multiplier),
        "lambda_geo": lambda_value,
        "checkpoint": str(checkpoint),
        "mean_anchor_validation_accuracy": float(metrics["accuracy"].mean()),
        "full_width_validation_accuracy": float(
            metrics.loc[np.isclose(metrics["width"], 1.0), "accuracy"].iloc[0]
        ),
        **geometry,
        "stable": bool(
            np.isfinite(metrics["accuracy"]).all()
            and all(np.isfinite(value) for value in geometry.values())
            and (gradients.empty or np.isfinite(gradients.select_dtypes("number")).all().all())
        ),
    }
    (output_dir / "sanity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def _worker_final(
    config: dict,
    seed: int,
    lambda_value: float,
    output_dir: Path,
    resume_checkpoint: Path | None,
) -> None:
    device = torch.device(config["experiment"]["device"])
    loaders = build_confirmatory_loaders(config, training_seed=seed)
    model, optimizer, scheduler = make_training_objects(config, seed, device)
    directions = _directions(config, "train", device)
    history_parts = []
    if resume_checkpoint is None:
        warmup_history, _, warmup = train_segment(
            model, optimizer, scheduler, loaders.train, config, device, directions,
            start_epoch=1, end_epoch=1, lambda_geo=0.0,
            output_dir=output_dir / "warmup",
            gradient_diagnostic_batches=int(config["georeg"]["gradient_diagnostic_batches"]),
        )
        loaded_epoch = load_training_checkpoint(
            warmup, model, optimizer, scheduler, loaders.train, device
        )
        history_parts.append(warmup_history)
    else:
        loaded_epoch = load_training_checkpoint(
            resume_checkpoint, model, optimizer, scheduler, loaders.train, device
        )
        development_dir = resume_checkpoint.parents[2]
        for history_path in (
            development_dir / "warmup" / "training_metrics.csv",
            resume_checkpoint.parent / "training_metrics.csv",
        ):
            if history_path.is_file():
                history_parts.append(pd.read_csv(history_path))
    final_history, _, checkpoint = train_segment(
        model, optimizer, scheduler, loaders.train, config, device, directions,
        start_epoch=loaded_epoch + 1, end_epoch=int(config["training"]["epochs"]),
        lambda_geo=lambda_value, output_dir=output_dir,
        gradient_diagnostic_batches=int(config["georeg"]["gradient_diagnostic_batches"]),
    )
    history_parts.append(final_history)
    combined_history = (
        pd.concat(history_parts, ignore_index=True)
        .drop_duplicates(subset=["epoch", "width"], keep="last")
        .sort_values(["epoch", "width"])
    )
    if set(combined_history["epoch"].astype(int)) != set(
        range(1, int(config["training"]["epochs"]) + 1)
    ):
        raise RuntimeError(f"Incomplete final training history for seed {seed}")
    combined_history.to_csv(output_dir / "training_metrics.csv", index=False)
    completed_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    torch.save(
        {"model": completed_payload["model"], "config": config, "epoch": int(config["training"]["epochs"])},
        output_dir / "checkpoint.pt",
    )
    checkpoint.unlink()
    if resume_checkpoint is None and warmup.is_file():
        warmup.unlink()


def _worker_dense(
    config: dict, seed: int, checkpoint: Path, output_dir: Path
) -> None:
    """Reveal the dense grid only after the orchestrator has all final checkpoints."""
    device = torch.device(config["experiment"]["device"])
    loaders = build_confirmatory_loaders(config, training_seed=seed)
    seed_everything(seed)
    model = slimmable_resnet18(
        num_classes=int(config["dataset"]["num_classes"]),
        supported_widths=config["compression"]["eval_widths"],
        projection_dim=int(config["model"]["projection_dim"]),
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    evaluate_grid(
        model, loaders.train, loaders.test, loaders.geometry, config, device, seed,
        output_dir, calibration_loader=loaders.calibration,
        prediction_output_dir=output_dir / "predictions",
    )
    analyze_seed(output_dir, config, seed)


def _worker_baseline_predictions(
    config: dict, baseline_root: Path, seed: int, output_dir: Path
) -> None:
    device = torch.device(config["experiment"]["device"])
    checkpoint = baseline_root / f"seed_{seed}" / "checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Baseline checkpoint required for paired inference: {checkpoint}")
    loaders = build_confirmatory_loaders(config, training_seed=seed)
    seed_everything(seed)
    model = slimmable_resnet18(
        num_classes=int(config["dataset"]["num_classes"]),
        supported_widths=config["compression"]["eval_widths"],
        projection_dim=int(config["model"]["projection_dim"]),
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    evaluate_prediction_grid(
        model, loaders.test, loaders.calibration, config, device, seed, output_dir
    )


def _subprocess_worker(
    config_path: Path,
    stage: str,
    gpu_id: int,
    output_dir: Path,
    extra: list[str] | None = None,
) -> float:
    command = [
        sys.executable, "-m", "scripts.run_georeg_rq3", "--config", str(config_path),
        "--stage", stage, "--worker-output", str(output_dir),
    ] + (extra or [])
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    started = time.perf_counter()
    print(f"[RQ3 scheduler] {stage} starting on physical GPU {gpu_id}", flush=True)
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env, check=True)
    elapsed = time.perf_counter() - started
    print(f"[RQ3 scheduler] {stage} finished in {elapsed/60:.1f} min", flush=True)
    return elapsed


def _schedule_jobs(
    jobs: list[dict], gpu_ids: list[int], tolerate_failures: bool = False
) -> dict[str, float]:
    if not jobs:
        return {}
    available: queue.Queue[int] = queue.Queue()
    for gpu_id in gpu_ids:
        available.put(gpu_id)

    def launch(job):
        gpu_id = available.get()
        try:
            try:
                elapsed = _subprocess_worker(
                    job["config"], job["stage"], gpu_id, job["output"], job.get("extra")
                )
                return job["name"], elapsed
            except Exception as error:
                if not tolerate_failures:
                    raise
                job["output"].mkdir(parents=True, exist_ok=True)
                (job["output"] / "worker_failure.txt").write_text(
                    f"{type(error).__name__}: {error}\n"
                )
                print(f"[RQ3 scheduler] tolerated failed job {job['name']}: {error}", flush=True)
                return job["name"], float("nan")
        finally:
            available.put(gpu_id)

    timings = {}
    with ThreadPoolExecutor(max_workers=min(len(gpu_ids), len(jobs))) as executor:
        futures = [executor.submit(launch, job) for job in jobs]
        for future in as_completed(futures):
            name, elapsed = future.result(); timings[name] = elapsed
    return timings


def _branch_tag(multiplier: float) -> str:
    return "lambda_0" if multiplier == 0 else f"lambda_{str(multiplier).replace('.', 'p')}x"


def select_lambda(config: dict, root: Path, directions: torch.Tensor) -> tuple[dict, pd.DataFrame]:
    rows = []
    multipliers = [float(value) for value in config["georeg"]["sanity_multipliers"]]
    baseline_dir = root / "development" / "sanity" / _branch_tag(0.0)
    baseline_summary_path = baseline_dir / "sanity_summary.json"
    if not baseline_summary_path.is_file():
        failure_path = baseline_dir / "worker_failure.txt"
        detail = failure_path.read_text().strip() if failure_path.is_file() else "no worker failure record"
        raise RuntimeError(
            "The mandatory lambda=0 sanity control failed before producing its summary. "
            f"Worker detail: {detail}"
        )
    baseline_summary = json.loads(baseline_summary_path.read_text())
    if not baseline_summary.get("stable", False):
        raise RuntimeError("The mandatory lambda=0 sanity control was numerically unstable")
    baseline_bank, baseline_ids = load_anchor_feature_bank(
        baseline_dir / "anchor_evaluation"
    )
    for multiplier in multipliers:
        branch_dir = root / "development" / "sanity" / _branch_tag(multiplier)
        summary_path = branch_dir / "sanity_summary.json"
        if not summary_path.is_file():
            if multiplier == 0:
                raise RuntimeError("The lambda=0 epoch-10 control branch did not complete")
            rows.append(
                {
                    "multiplier": multiplier,
                    "lambda_geo": multiplier * json.loads(
                        (root / "development" / "warmup" / "calibration_summary.json").read_text()
                    )["lambda_0"],
                    "stable": False,
                    "geometry_gate": False,
                    "anchor_gate": False,
                    "full_width_gate": False,
                    "eligible": False,
                    "failure": (branch_dir / "worker_failure.txt").read_text().strip()
                    if (branch_dir / "worker_failure.txt").is_file() else "missing output",
                }
            )
            continue
        summary = json.loads(summary_path.read_text())
        if multiplier == 0:
            bootstrap = {
                "baseline_D_geo": summary["D_geo"], "candidate_D_geo": summary["D_geo"],
                "delta_D_geo": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "bootstrap_replicates": 0,
            }
        else:
            candidate_bank, candidate_ids = load_anchor_feature_bank(
                branch_dir / "anchor_evaluation"
            )
            if not torch.equal(baseline_ids, candidate_ids):
                raise RuntimeError("Sanity branch feature IDs/order differ")
            bootstrap = paired_curvature_bootstrap(
                baseline_bank, candidate_bank, directions,
                int(config["georeg"]["bootstrap_replicates"]),
                int(config["georeg"]["bootstrap_seed"]) + int(multiplier * 1000),
                float(config["georeg"]["epsilon"]),
            )
        row = {
            **summary,
            **{f"bootstrap_{key}": value for key, value in bootstrap.items()},
            "delta_mean_anchor_validation_accuracy": (
                summary["mean_anchor_validation_accuracy"]
                - baseline_summary["mean_anchor_validation_accuracy"]
            ),
            "delta_full_width_validation_accuracy": (
                summary["full_width_validation_accuracy"]
                - baseline_summary["full_width_validation_accuracy"]
            ),
        }
        row["geometry_gate"] = bool(multiplier > 0 and bootstrap["ci_high"] < 0)
        row["anchor_gate"] = bool(
            row["delta_mean_anchor_validation_accuracy"]
            >= -float(config["georeg"]["anchor_accuracy_margin"])
        )
        row["full_width_gate"] = bool(
            row["delta_full_width_validation_accuracy"]
            >= -float(config["georeg"]["full_width_accuracy_margin"])
        )
        row["eligible"] = bool(
            summary["stable"] and row["geometry_gate"]
            and row["anchor_gate"] and row["full_width_gate"]
        )
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("multiplier")
    table.to_csv(root / "development" / "sanity_selection.csv", index=False)
    eligible = table.loc[table["eligible"]].sort_values("multiplier")
    if eligible.empty:
        decision = {"go": False, "reason": "No nonzero lambda passed all preregistered sanity gates"}
    else:
        selected = eligible.iloc[0]
        decision = {
            "go": True,
            "selected_multiplier": float(selected["multiplier"]),
            "selected_lambda": float(selected["lambda_geo"]),
            "selected_checkpoint": str(selected["checkpoint"]),
        }
    (root / "development" / "lambda_selection.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    return decision, table


def _dense_summary(frame: pd.DataFrame, method: str):
    rows = []
    anchors = set(ANCHORS)
    for seed, group in frame.groupby("seed"):
        group = group.sort_values("budget")
        unseen = group.loc[group["budget"].round(2).isin(UNSEEN_WIDTHS)]
        local = group.dropna(subset=["local_wasserstein_sensitivity"])
        log_flops = np.log(group["flops"].to_numpy(float))
        accuracy = group["accuracy"].to_numpy(float)
        rows.append(
            {
                "seed": int(seed), "method": method,
                "mean_unseen_accuracy": float(unseen["accuracy"].mean()),
                "worst_unseen_accuracy": float(unseen["accuracy"].min()),
                "mean_G": float(local["local_wasserstein_sensitivity"].mean()),
                "max_G": float(local["local_wasserstein_sensitivity"].max()),
                "mean_anchor_accuracy": float(
                    group.loc[group["budget"].round(2).isin(anchors), "accuracy"].mean()
                ),
                "full_width_accuracy": float(
                    group.loc[np.isclose(group["budget"], 1.0), "accuracy"].iloc[0]
                ),
                "accuracy_logflops_auc": float(
                    np.trapz(accuracy, log_flops) / (log_flops[-1] - log_flops[0])
                ),
            }
        )
    return pd.DataFrame(rows)


def finalize_rq3(config: dict, root: Path, baseline_root: Path, lambda_decision: dict):
    georeg_frames = [
        pd.read_csv(root / "georeg" / f"seed_{seed}" / "results" / "central_analysis.csv")
        for seed in config["experiment"]["seeds"]
    ]
    georeg_central = pd.concat(georeg_frames, ignore_index=True)
    georeg_central.to_csv(root / "georeg_central_analysis_all_seeds.csv", index=False)
    baseline_central = pd.read_csv(baseline_root / "central_analysis_all_seeds.csv")
    summary = pd.concat(
        [_dense_summary(baseline_central, "Baseline"), _dense_summary(georeg_central, "GeoReg-Curvature")],
        ignore_index=True,
    )
    geometry_rows, accuracy_rows = [], []
    train_directions = _directions(config, "train")
    heldout_directions = _directions(config, "heldout")
    for seed in config["experiment"]["seeds"]:
        seed = int(seed)
        baseline_bank, baseline_ids = load_anchor_feature_bank(baseline_root, seed)
        georeg_bank, georeg_ids = load_anchor_feature_bank(root / "georeg", seed)
        if not torch.equal(baseline_ids, georeg_ids):
            raise RuntimeError(f"Baseline/GeoReg feature IDs differ for seed {seed}")
        for kind, directions in (("train", train_directions), ("heldout", heldout_directions)):
            result = paired_curvature_bootstrap(
                baseline_bank, georeg_bank, directions,
                int(config["georeg"]["bootstrap_replicates"]),
                int(config["georeg"]["bootstrap_seed"]) + seed + (10000 if kind == "heldout" else 0),
                float(config["georeg"]["epsilon"]),
            )
            geometry_rows.append({"seed": seed, "directions": kind, **result})
        baseline_predictions = pd.read_csv(
            root / "baseline_predictions" / f"seed_{seed}" / "predictions_all_widths.csv"
        )
        georeg_predictions = pd.read_csv(
            root / "georeg" / f"seed_{seed}" / "predictions" / "predictions_all_widths.csv"
        )
        for predictions, central, label in (
            (baseline_predictions, baseline_central, "baseline"),
            (georeg_predictions, georeg_central, "GeoReg"),
        ):
            measured = predictions.groupby("budget", as_index=False)["correct"].mean()
            recorded = central.loc[central["seed"] == seed, ["budget", "accuracy"]]
            check = measured.merge(recorded, on="budget", validate="one_to_one")
            if len(check) != len(config["compression"]["eval_widths"]):
                raise RuntimeError(f"Incomplete {label} prediction grid for seed {seed}")
            if not np.allclose(check["correct"], check["accuracy"], atol=1e-12):
                raise RuntimeError(
                    f"Per-sample {label} predictions disagree with aggregate metrics for seed {seed}"
                )
        accuracy_rows.append(
            {
                "seed": seed,
                **paired_accuracy_bootstrap(
                    baseline_predictions, georeg_predictions, list(UNSEEN_WIDTHS),
                    int(config["georeg"]["bootstrap_replicates"]),
                    int(config["georeg"]["bootstrap_seed"]) + 20000 + seed,
                ),
            }
        )
    geometry = pd.DataFrame(geometry_rows)
    accuracy_bootstrap = pd.DataFrame(accuracy_rows)
    geometry.to_csv(root / "final_geometry_bootstrap.csv", index=False)
    accuracy_bootstrap.to_csv(root / "final_accuracy_bootstrap.csv", index=False)
    heldout = geometry.loc[geometry["directions"] == "heldout", [
        "seed", "baseline_D_geo", "candidate_D_geo", "delta_D_geo", "ci_low", "ci_high"
    ]].rename(columns={
        "baseline_D_geo": "baseline_D_geo_heldout",
        "candidate_D_geo": "georeg_D_geo_heldout",
        "delta_D_geo": "delta_D_geo_heldout",
        "ci_low": "D_geo_delta_ci_low", "ci_high": "D_geo_delta_ci_high",
    })
    wide = summary.pivot(index="seed", columns="method")
    comparison = pd.DataFrame({"seed": wide.index})
    for metric in [
        "mean_unseen_accuracy", "worst_unseen_accuracy", "mean_G", "max_G",
        "mean_anchor_accuracy", "full_width_accuracy", "accuracy_logflops_auc",
    ]:
        comparison[f"baseline_{metric}"] = wide[metric]["Baseline"].to_numpy()
        comparison[f"georeg_{metric}"] = wide[metric]["GeoReg-Curvature"].to_numpy()
        comparison[f"delta_{metric}"] = comparison[f"georeg_{metric}"] - comparison[f"baseline_{metric}"]
    comparison = comparison.merge(heldout, on="seed", validate="one_to_one").merge(
        accuracy_bootstrap, on="seed", validate="one_to_one", suffixes=("", "_paired")
    )
    summary = summary.merge(
        pd.concat([
            heldout[["seed", "baseline_D_geo_heldout"]].rename(columns={"baseline_D_geo_heldout": "D_geo_heldout"}).assign(method="Baseline"),
            heldout[["seed", "georeg_D_geo_heldout"]].rename(columns={"georeg_D_geo_heldout": "D_geo_heldout"}).assign(method="GeoReg-Curvature"),
        ], ignore_index=True), on=["seed", "method"], validate="one_to_one"
    )
    summary.to_csv(root / "rq3_main_table.csv", index=False)
    comparison.to_csv(root / "rq3_paired_comparison.csv", index=False)
    confirmatory = comparison.loc[
        comparison["seed"].isin(config["experiment"]["confirmatory_seeds"])
    ].copy()
    margin_anchor = float(config["georeg"]["anchor_accuracy_margin"])
    margin_full = float(config["georeg"]["full_width_accuracy_margin"])
    confirmatory["geometry_improved"] = confirmatory["delta_D_geo_heldout"] < 0
    confirmatory["unseen_improved"] = confirmatory["delta_mean_unseen_accuracy"] > 0
    confirmatory["anchors_preserved"] = confirmatory["delta_mean_anchor_accuracy"] >= -margin_anchor
    confirmatory["full_preserved"] = confirmatory["delta_full_width_accuracy"] >= -margin_full
    confirmatory["all_links_hold"] = confirmatory[
        ["geometry_improved", "unseen_improved", "anchors_preserved", "full_preserved"]
    ].all(axis=1)
    confirmatory.to_csv(root / "rq3_confirmatory_seed_checks.csv", index=False)
    supported = bool(len(confirmatory) == 2 and confirmatory["all_links_hold"].all())
    verdict = (
        "RQ3 SUPPORTED IN THIS SETUP" if supported
        else "RQ3 NOT SUPPORTED — stop the intervention branch"
    )
    decision = {
        "sanity_go": True,
        "selected_lambda": lambda_decision["selected_lambda"],
        "selected_multiplier": lambda_decision["selected_multiplier"],
        "rq3_supported": supported,
        "verdict": verdict,
        "confirmatory_seeds": config["experiment"]["confirmatory_seeds"],
    }
    (root / "rq3_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    report = [
        "# RQ3 GeoReg-Curvature report", "", f"Decision: **{verdict}**", "",
        "Seed 0 is development-only. Seeds 1 and 2 are the confirmatory evidence.", "",
        "A1 GeoWeighting is retained as the previously completed negative-control ablation; "
        "it is not rerun or used in the primary Baseline-vs-GeoReg decision.", "",
        "## Main table", "", summary.to_markdown(index=False), "",
        "## Paired comparison", "", comparison.to_markdown(index=False), "",
        "## Confirmatory checks", "", confirmatory.to_markdown(index=False), "",
        "> Intermediate widths were not used for gradients or lambda selection. "
        "Held-out projection directions were not used by the GeoReg loss.",
    ]
    (root / "rq3_report.md").write_text("\n".join(report) + "\n")
    return decision


def run_rq3(config_path: str | Path, gpu_ids: list[int] | None = None):
    config = load_config(config_path)
    validate_rq3_config(config)
    root = Path(config["experiment"]["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    _write_yaml(config, root / "resolved_config.yaml")
    _save_directions(config, root / "protocol")
    baseline_root = find_confirmatory_root(config["baseline_artifacts"]["input_root"])
    for seed in config["experiment"]["seeds"]:
        if not (baseline_root / f"seed_{seed}" / "checkpoint.pt").is_file():
            raise FileNotFoundError(f"Missing baseline checkpoint for seed {seed}")
    gpu_ids = gpu_ids or list(range(torch.cuda.device_count()))
    if not gpu_ids:
        raise RuntimeError("RQ3 requires at least one CUDA GPU")
    config_path = root / "resolved_config.yaml"
    timings = {}
    warmup_dir = root / "development" / "warmup"
    if not (
        (warmup_dir / "calibration_summary.json").is_file()
        and (warmup_dir / "checkpoint_epoch_01.pt").is_file()
    ):
        timings["calibration"] = _subprocess_worker(
            config_path, "warmup", gpu_ids[0], warmup_dir
        )
    else:
        print("[RQ3] reusing completed calibration", flush=True)
    calibration = json.loads((warmup_dir / "calibration_summary.json").read_text())
    warmup_checkpoint = Path(calibration["warmup_checkpoint"])
    sanity_jobs = []
    for multiplier in config["georeg"]["sanity_multipliers"]:
        multiplier = float(multiplier)
        branch_dir = root / "development" / "sanity" / _branch_tag(multiplier)
        if (
            (branch_dir / "sanity_summary.json").is_file()
            and (branch_dir / "checkpoint_epoch_10.pt").is_file()
        ):
            print(f"[RQ3] reusing completed sanity branch {multiplier:g}x", flush=True)
            continue
        sanity_jobs.append(
            {
                "name": f"sanity_{multiplier:g}x", "config": config_path,
                "stage": "sanity", "output": branch_dir,
                "extra": ["--warmup-checkpoint", str(warmup_checkpoint), "--multiplier", str(multiplier)],
            }
        )
    timings.update(_schedule_jobs(sanity_jobs, gpu_ids, tolerate_failures=True))
    selection, sanity_table = select_lambda(
        config, root, torch.load(root / "protocol" / "train_projection_directions.pt", weights_only=False)
    )
    if not selection["go"]:
        decision = {
            "sanity_go": False,
            "rq3_supported": False,
            "verdict": "SANITY NO-GO — final dense grid was not opened",
        }
        (root / "rq3_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
        (root / "rq3_report.md").write_text(
            "# RQ3 GeoReg-Curvature report\n\n"
            "Decision: **SANITY NO-GO**. No nonzero lambda passed all preregistered "
            "anchor-only gates. Dense unseen evaluation was not run.\n"
        )
        pd.DataFrame(
            [{"stage": name, "minutes": seconds / 60} for name, seconds in timings.items()]
        ).to_csv(root / "runtime_by_stage.csv", index=False)
        return {"root": root, "decision": decision, "sanity": sanity_table, "timings": timings}
    selected_checkpoint = Path(selection["selected_checkpoint"])
    selected_lambda = float(selection["selected_lambda"])
    final_jobs = [
        {
            "name": "final_seed_0", "config": config_path, "stage": "final",
            "output": root / "georeg" / "seed_0",
            "extra": ["--seed", "0", "--lambda-value", str(selected_lambda),
                      "--resume-checkpoint", str(selected_checkpoint)],
        },
        *[
            {
                "name": f"final_seed_{seed}", "config": config_path, "stage": "final",
                "output": root / "georeg" / f"seed_{seed}",
                "extra": ["--seed", str(seed), "--lambda-value", str(selected_lambda)],
            }
            for seed in (1, 2)
        ],
    ]
    final_jobs = [
        job for job in final_jobs
        if not (
            (job["output"] / "checkpoint.pt").is_file()
            and (job["output"] / "training_metrics.csv").is_file()
        )
    ]
    timings.update(_schedule_jobs(final_jobs, gpu_ids))
    final_checkpoints = [root / "georeg" / f"seed_{seed}" / "checkpoint.pt" for seed in (0, 1, 2)]
    if not all(path.is_file() for path in final_checkpoints):
        raise RuntimeError("Dense reveal blocked: not all three final checkpoints exist")
    dense_jobs = [
        {
            "name": f"dense_georeg_seed_{seed}", "config": config_path,
            "stage": "dense", "output": root / "georeg" / f"seed_{seed}",
            "extra": ["--seed", str(seed), "--resume-checkpoint", str(final_checkpoints[seed])],
        }
        for seed in (0, 1, 2)
        if not (
            (root / "georeg" / f"seed_{seed}" / "results" / "central_analysis.csv").is_file()
            and (root / "georeg" / f"seed_{seed}" / "predictions" / "predictions_all_widths.csv").is_file()
        )
    ]
    baseline_jobs = [
        {
            "name": f"baseline_predictions_seed_{seed}", "config": config_path,
            "stage": "baseline_predictions", "output": root / "baseline_predictions" / f"seed_{seed}",
            "extra": ["--seed", str(seed), "--baseline-root", str(baseline_root)],
        }
        for seed in (0, 1, 2)
    ]
    baseline_jobs = [
        job for job in baseline_jobs
        if not (job["output"] / "predictions_all_widths.csv").is_file()
    ]
    timings.update(_schedule_jobs(dense_jobs + baseline_jobs, gpu_ids))
    decision = finalize_rq3(config, root, baseline_root, selection)
    pd.DataFrame(
        [{"stage": name, "minutes": seconds / 60} for name, seconds in timings.items()]
    ).to_csv(root / "runtime_by_stage.csv", index=False)
    return {"root": root, "decision": decision, "sanity": sanity_table, "timings": timings}


def main():
    parser = argparse.ArgumentParser(description="Preregistered RQ3 GeoReg-Curvature run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", default="full", choices=["full", "warmup", "sanity", "final", "dense", "baseline_predictions"])
    parser.add_argument("--worker-output")
    parser.add_argument("--warmup-checkpoint")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--baseline-root")
    parser.add_argument("--multiplier", type=float)
    parser.add_argument("--lambda-value", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu-ids", nargs="*", type=int)
    args = parser.parse_args()
    config = _load_yaml(args.config)
    validate_rq3_config(config)
    if args.stage == "full":
        result = run_rq3(args.config, args.gpu_ids)
        print(result["decision"]["verdict"])
    elif args.stage == "warmup":
        _worker_warmup(config, Path(args.worker_output))
    elif args.stage == "sanity":
        _worker_sanity(config, Path(args.warmup_checkpoint), args.multiplier, Path(args.worker_output))
    elif args.stage == "final":
        _worker_final(
            config, args.seed, args.lambda_value, Path(args.worker_output),
            Path(args.resume_checkpoint) if args.resume_checkpoint else None,
        )
    elif args.stage == "dense":
        _worker_dense(
            config, args.seed, Path(args.resume_checkpoint), Path(args.worker_output)
        )
    else:
        _worker_baseline_predictions(config, Path(args.baseline_root), args.seed, Path(args.worker_output))


if __name__ == "__main__":
    main()
