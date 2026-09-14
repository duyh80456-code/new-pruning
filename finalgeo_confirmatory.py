"""Locked RQ2 FinalGeo confirmatory training/evaluation for seeds 3,4,5."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import build_confirmatory_loaders, build_interim_validation_loaders
from evaluation import calibrate_batch_norm, evaluate_width
from profiling import profile_subnet
from research_utils import seed_everything
from rq2_anchor_placement import GRID, UNIFORM_ANCHORS, _sha256, evaluate_checkpoint
from rq2_finalgeo_selector import EXPECTED_PUREGEO, _canonical_hash
from s1_width import (
    analyze_representation_views,
    extract_representation_views,
    make_model,
    train_shared_reference,
)


METHOD_ANCHORS = {
    "uniform": UNIFORM_ANCHORS,
    "puregeo": EXPECTED_PUREGEO,
    "finalgeo": (0.25, 0.40, 0.70, 1.00),
}
CONFIRMATORY_SEEDS = (3, 4, 5)


def validate_confirmatory_base_config(config: dict) -> None:
    """Reject protocol drift before any confirmatory optimization starts."""
    if config["dataset"]["name"].lower() != "cifar100":
        raise ValueError("FinalGeo confirmation requires CIFAR-100")
    if config["model"]["backbone"] != "slimmable_resnet18":
        raise ValueError("FinalGeo confirmation requires Slimmable ResNet-18")
    if tuple(map(float, config["compression"]["eval_widths"])) != GRID:
        raise ValueError("FinalGeo confirmation requires the frozen 16-width dense grid")
    if int(config["training"]["phase_1_epochs"]) != 50 or int(
        config["training"]["total_epochs"]
    ) != 100:
        raise ValueError("FinalGeo confirmation requires the frozen 50+50 horizon")
    if int(config["training"]["anchors_per_batch"]) != 4:
        raise ValueError("FinalGeo confirmation requires four anchor forwards per batch")
    if int(config["dataset"].get("num_workers", -1)) != 0:
        raise ValueError("Exact loader replay requires dataset.num_workers=0")
    if not np.isclose(float(config["training"]["extension_learning_rate"]), 0.01):
        raise ValueError("FinalGeo confirmation requires the registered phase-2 LR=0.01")


def validate_frozen_protocol(
    development_root: str | Path,
    freeze_dir: str | Path,
    repo_root: str | Path | None = None,
    verify_git: bool = True,
) -> dict:
    development_root, freeze_dir = Path(development_root), Path(freeze_dir)
    protocol_path = freeze_dir / "finalgeo_frozen_protocol.json"
    selected_path = freeze_dir / "finalgeo_selected_anchors.json"
    candidates_path = freeze_dir / "finalgeo_selector_candidates.csv"
    missing = [str(path) for path in (protocol_path, selected_path, candidates_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete FinalGeo freeze: {missing}")
    protocol = json.loads(protocol_path.read_text())
    frozen_id = protocol.get("freeze_id_sha256")
    core = dict(protocol)
    core.pop("freeze_id_sha256", None)
    if frozen_id != _canonical_hash(core):
        raise RuntimeError("FinalGeo frozen protocol fingerprint is invalid")
    required = {
        "status": "FROZEN_BEFORE_CONFIRMATORY_SEEDS",
        "K": 4,
        "development_seeds": [0, 1, 2],
        "seeds_confirmatory": [3, 4, 5],
        "uniform_anchors": list(UNIFORM_ANCHORS),
        "puregeo_anchors": list(EXPECTED_PUREGEO),
        "finalgeo_anchors": list(METHOD_ANCHORS["finalgeo"]),
        "dense_eval_grid": list(GRID),
        "candidate_grid": list(GRID),
        "fixed_endpoints": [0.25, 1.0],
        "candidate_count": 91,
        "methods_confirmatory": ["Uniform", "PureGeo", "FinalGeo"],
        "common_holdout": [0.30, 0.35, 0.45, 0.55, 0.65, 0.80, 0.85, 0.90, 0.95],
    }
    for key, value in required.items():
        if protocol.get(key) != value:
            raise RuntimeError(f"Frozen protocol mismatch for {key}: {protocol.get(key)}")
    rule = protocol["selection_rule"]
    if rule.get("objective") != "minimize_R_G" or not rule.get("objective_is_geometry_only"):
        raise RuntimeError("FinalGeo geometry-only selection rule changed")
    if any(rule.get(key) for key in ("accuracy_used", "parameter_exposure_used", "gradient_interference_used")):
        raise RuntimeError("A prohibited development signal entered FinalGeo selection")
    compute = protocol["compute_constraint"]
    constraints = rule.get("constraints", {})
    if (
        float(constraints.get("minimum_compute_ratio_vs_uniform", np.nan)) != 0.90
        or float(constraints.get("maximum_compute_ratio_vs_uniform", np.nan)) != 1.00
    ):
        raise RuntimeError("Frozen FinalGeo compute constraint changed")
    ratio = float(compute["selected_ratio_vs_uniform"])
    if not 0.90 - 1e-12 <= ratio <= 1.00 + 1e-12:
        raise RuntimeError("FinalGeo selected compute is outside the frozen 90-100% constraint")

    source_paths = {
        "rq2_geometry_all.csv": development_root / "rq2_geometry_all.csv",
        "geometry_trajectory_coordinates.csv": development_root / "protocol" / "geometry_trajectory_coordinates.csv",
        "rq2_dense_metrics_all.csv": development_root / "rq2_dense_metrics_all.csv",
        "selected_anchors.json": development_root / "protocol" / "selected_anchors.json",
    }
    expected_hashes = protocol["geometry_source"]["source_artifact_sha256"]
    measured = {name: _sha256(path) for name, path in source_paths.items()}
    if measured != expected_hashes:
        raise RuntimeError("RQ2 development artifacts differ from the frozen FinalGeo sources")
    selected = json.loads(selected_path.read_text())
    if selected.get("freeze_id_sha256") != frozen_id or selected.get("finalgeo_anchors") != required["finalgeo_anchors"]:
        raise RuntimeError("FinalGeo selected-anchor artifact disagrees with frozen protocol")
    candidates = pd.read_csv(candidates_path)
    chosen = candidates.loc[candidates["selected"]]
    if len(candidates) != 91 or len(chosen) != 1 or chosen.iloc[0]["anchors"] != "0.25,0.40,0.70,1.00":
        raise RuntimeError("FinalGeo candidate table does not reproduce the frozen selection")

    if verify_git:
        repo = Path(repo_root or Path(__file__).resolve().parent)
        frozen_commit = protocol.get("git_commit")
        if not frozen_commit or frozen_commit == "UNAVAILABLE":
            raise RuntimeError("Frozen protocol lacks a usable Git commit")
        subprocess.run(["git", "merge-base", "--is-ancestor", frozen_commit, "HEAD"], cwd=repo, check=True)
        changed = subprocess.run(
            ["git", "diff", "--name-only", frozen_commit, "HEAD", "--", "rq2_finalgeo_selector.py"],
            cwd=repo, check=True, capture_output=True, text=True,
        ).stdout.strip()
        if changed:
            raise RuntimeError("FinalGeo selector code changed after the frozen Git commit")
    return protocol


def method_config(base: dict, method: str) -> dict:
    if method not in METHOD_ANCHORS:
        raise ValueError(f"Unknown confirmatory method: {method}")
    config = copy.deepcopy(base)
    config["compression"]["train_widths"] = list(METHOD_ANCHORS[method])
    return config


def train_method_seed(config: dict, root: str | Path, method: str, seed: int) -> Path:
    if int(seed) not in CONFIRMATORY_SEEDS:
        raise ValueError("FinalGeo confirmatory training is locked to seeds 3,4,5")
    root, device = Path(root), torch.device(config["experiment"]["device"])
    config = method_config(config, method)
    output = root / method / f"seed_{seed}"
    final_path = output / "checkpoint.pt"
    if (
        final_path.is_file()
        and (output / "training_metrics_1_100.csv").is_file()
        and (output / "training_provenance.json").is_file()
    ):
        return final_path

    phase_1 = copy.deepcopy(config)
    phase_1["training"]["epochs"] = int(config["training"]["phase_1_epochs"])
    seed_everything(int(seed))
    loaders_1 = build_confirmatory_loaders(phase_1, training_seed=int(seed))
    first = train_shared_reference(
        make_model(phase_1, device), loaders_1.train, phase_1, device, output / "phase_1"
    )
    first_hash = _sha256(first)

    phase_2 = copy.deepcopy(config)
    phase_2["training"]["epochs"] = int(config["training"]["total_epochs"])
    seed_everything(int(seed))
    loaders_2 = build_confirmatory_loaders(phase_2, training_seed=int(seed))
    final = train_shared_reference(
        make_model(phase_2, device), loaders_2.train, phase_2, device, output,
        initial_checkpoint=first,
        initial_epoch=int(config["training"]["phase_1_epochs"]),
        learning_rate_override=float(config["training"]["extension_learning_rate"]),
    )
    first_metrics = pd.read_csv(output / "phase_1" / "training_metrics.csv")
    second_metrics = pd.read_csv(output / "training_metrics.csv")
    combined = pd.concat([first_metrics, second_metrics], ignore_index=True).sort_values(
        ["epoch", "width"]
    )
    if set(combined["epoch"].astype(int)) != set(range(1, 101)):
        raise RuntimeError(f"Incomplete 1-100 training history for {method} seed {seed}")
    combined.insert(0, "seed", int(seed))
    combined.insert(1, "method", method)
    combined.to_csv(output / "training_metrics_1_100.csv", index=False)
    provenance = {
        "method": method,
        "seed": int(seed),
        "anchors": list(METHOD_ANCHORS[method]),
        "source_horizon": 50,
        "target_horizon": 100,
        "extension_learning_rate": float(config["training"]["extension_learning_rate"]),
        "optimizer_state_reused": False,
        "scheduler_state_reused": False,
        "rng_and_loader_restarted_from_seed": True,
        "phase_1_checkpoint_sha256": first_hash,
        "final_checkpoint_sha256": _sha256(final),
    }
    (output / "training_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return final


def evaluate_method_seed(config: dict, root: str | Path, method: str, seed: int) -> Path:
    root = Path(root)
    checkpoint = root / method / f"seed_{seed}" / "checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    random_matrix = torch.load(
        root / "protocol" / "fixed_random_projection.pt", map_location="cpu", weights_only=False
    )
    return evaluate_checkpoint(
        method_config(config, method), root, checkpoint, int(seed), method, random_matrix
    )


def evaluate_validation_checkpoint(
    config: dict,
    checkpoint: str | Path,
    output_dir: str | Path,
    method: str,
    seed: int,
    random_matrix: torch.Tensor,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Interim dense validation/geometry diagnostic with the test split sealed."""
    if method not in METHOD_ANCHORS or int(seed) not in CONFIRMATORY_SEEDS:
        raise ValueError("Interim evaluation is locked to registered methods and seeds 3,4,5")
    config = method_config(config, method)
    device = torch.device(config["experiment"]["device"])
    loaders = build_interim_validation_loaders(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = make_model(config, device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=False)["model"]
    )
    rows = []
    anchors = set(METHOD_ANCHORS[method])
    for width in map(float, config["compression"]["eval_widths"]):
        calibrate_batch_norm(
            model, loaders.calibration, width, device,
            int(config["evaluation"]["bn_calibration_batches"]),
        )
        metrics = evaluate_width(model, loaders.validation, width, device)
        flops, params = profile_subnet(model, width)
        extract_representation_views(
            model, loaders.geometry, width, device, random_matrix,
            output_dir / "validation_representations",
        )
        rows.append({
            "seed": int(seed), "method": method, "split": "validation_5k",
            "budget": width, "is_train_anchor": width in anchors,
            **metrics, "flops": flops, "params": params,
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "validation_budget_metrics.csv", index=False)
    geometry = analyze_representation_views(
        output_dir / "validation_representations", config, int(seed), output_dir
    )
    geometry.insert(1, "method", method)
    geometry.insert(2, "split", "fixed_validation_2k")
    geometry.to_csv(output_dir / "validation_representation_local_geometry.csv", index=False)
    (output_dir / "TEST_SPLIT_NOT_ACCESSED.txt").write_text(
        "Interim diagnostic used only the fixed CIFAR-100 train/validation split.\n"
    )
    return metrics, geometry


def _summary(metrics: pd.DataFrame, geometry: pd.DataFrame, common_holdout: list[float]) -> pd.DataFrame:
    rows = []
    for (method, seed), group in metrics.groupby(["method", "seed"]):
        holdout = group.loc[group["budget"].round(2).isin(common_holdout)]
        low = holdout.loc[holdout["budget"].between(0.30, 0.45)]
        high = holdout.loc[holdout["budget"].between(0.80, 0.95)]
        geo = geometry.loc[
            geometry["method"].eq(method)
            & geometry["seed"].eq(seed)
            & geometry["representation"].eq("learned_projection")
        ]
        rows.append({
            "method": method, "seed": int(seed),
            "mean_common_holdout_accuracy": float(holdout["accuracy"].mean()),
            "worst_common_holdout_accuracy": float(holdout["accuracy"].min()),
            "mean_low_region_accuracy": float(low["accuracy"].mean()),
            "mean_high_region_accuracy": float(high["accuracy"].mean()),
            "full_width_accuracy": float(group.loc[np.isclose(group["budget"], 1.0), "accuracy"].iloc[0]),
            "dense_mean_accuracy": float(group["accuracy"].mean()),
            "mean_G": float(geo["G"].mean()), "max_G": float(geo["G"].max()),
        })
    return pd.DataFrame(rows)


def finalize_confirmatory(config: dict, root: str | Path, protocol: dict) -> dict:
    root = Path(root)
    common_holdout = list(map(float, protocol["common_holdout"]))
    metric_frames, geometry_frames = [], []
    for method in METHOD_ANCHORS:
        for seed in CONFIRMATORY_SEEDS:
            evaluation = root / "evaluation" / method / f"seed_{seed}"
            metrics = pd.read_csv(evaluation / "budget_metrics.csv")
            metrics["method"] = method
            metric_frames.append(metrics)
            geometry = pd.read_csv(evaluation / "representation_local_geometry.csv")
            geometry["method"] = method
            geometry_frames.append(geometry)
    metrics = pd.concat(metric_frames, ignore_index=True)
    geometry = pd.concat(geometry_frames, ignore_index=True)
    metrics.to_csv(root / "finalgeo_dense_metrics_all.csv", index=False)
    geometry.to_csv(root / "finalgeo_geometry_all.csv", index=False)
    summary = _summary(metrics, geometry, common_holdout)
    summary.to_csv(root / "finalgeo_method_summary.csv", index=False)

    wide = summary.pivot(index="seed", columns="method")
    comparison = pd.DataFrame({"seed": list(CONFIRMATORY_SEEDS)})
    for metric in (
        "mean_common_holdout_accuracy", "worst_common_holdout_accuracy",
        "mean_low_region_accuracy", "mean_high_region_accuracy", "full_width_accuracy",
        "dense_mean_accuracy", "mean_G", "max_G",
    ):
        for method in ("puregeo", "finalgeo"):
            comparison[f"{method}_minus_uniform_{metric}"] = [
                float(wide.loc[seed, (metric, method)] - wide.loc[seed, (metric, "uniform")])
                for seed in CONFIRMATORY_SEEDS
            ]
    comparison.to_csv(root / "finalgeo_seed_comparison.csv", index=False)

    gates = protocol["success_gates"]
    mean_delta = comparison["finalgeo_minus_uniform_mean_common_holdout_accuracy"]
    worst_delta = comparison["finalgeo_minus_uniform_worst_common_holdout_accuracy"]
    full_delta = comparison["finalgeo_minus_uniform_full_width_accuracy"]
    high_delta = comparison["finalgeo_minus_uniform_mean_high_region_accuracy"]
    final_low = summary.loc[summary["method"].eq("finalgeo"), "mean_low_region_accuracy"].mean()
    uniform_low = summary.loc[summary["method"].eq("uniform"), "mean_low_region_accuracy"].mean()
    pure_low = summary.loc[summary["method"].eq("puregeo"), "mean_low_region_accuracy"].mean()
    denominator = pure_low - uniform_low
    retention = (final_low - uniform_low) / denominator if denominator > 0 else float("-inf")
    checks = {
        "mean_common_holdout_better_3_of_3": bool((mean_delta > 0).all()),
        "pooled_mean_effect_at_least_0.002": bool(
            mean_delta.mean() >= float(gates["minimum_pooled_mean_common_holdout_effect"])
        ),
        "worst_holdout_margin_all_seeds": bool(
            (worst_delta >= float(gates["worst_common_holdout_accuracy_margin_vs_uniform"])).all()
        ),
        "full_width_margin_all_seeds": bool(
            (full_delta >= float(gates["full_width_accuracy_margin_vs_uniform"])).all()
        ),
        "high_region_margin_all_seeds": bool(
            (high_delta >= float(gates["high_region_accuracy_margin_vs_uniform"])).all()
        ),
        "pooled_low_region_gain_retention_at_least_0.60": bool(
            retention >= float(gates["minimum_low_region_puregeo_gain_retention"])
        ),
    }
    passed = bool(all(checks.values()))
    decision = {
        "status": "CONFIRMATORY_COMPLETE",
        "verdict": "FINALGEO CONFIRMATORY PASS" if passed else "FINALGEO CONFIRMATORY FAIL",
        "all_preregistered_gates_pass": passed,
        "gate_checks": checks,
        "pooled_mean_common_holdout_effect": float(mean_delta.mean()),
        "pooled_low_region_puregeo_gain": float(denominator),
        "pooled_low_region_finalgeo_gain": float(final_low - uniform_low),
        "pooled_low_region_gain_retention": float(retention),
        "frozen_protocol_sha256": protocol["freeze_id_sha256"],
    }
    (root / "finalgeo_confirmatory_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    return decision
