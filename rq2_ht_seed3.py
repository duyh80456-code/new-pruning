"""Development seed-3 Geo-HT versus matched Resource-HT training experiment."""

from __future__ import annotations

import copy
import json
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.nn import functional as F

from data import build_development_train_loaders, build_interim_validation_loaders
from evaluation import calibrate_batch_norm, evaluate_width
from research_utils import seed_everything
from rq2_anchor_placement import GRID, _sha256
from rq2_dynamic_seed3_pilot import (
    _atomic_checkpoint,
    _capture_rng,
    _new_optimizer_scheduler,
    _restore_rng,
    load_frozen_policy,
    sampler_seed,
    validate_pilot_config,
)
from rq2_probabilistic_support import INTERIOR_WIDTHS, pair_marginals
from s1_width import make_model
from training import kd_loss


SEED = 3
METHODS = ("geo_ht", "resource_ht")
POLICY_METHOD = {"geo_ht": "geometry_dynamic", "resource_ht": "resource_dynamic"}
CHECKPOINT_EPOCHS = tuple(range(10, 101, 10))
TARGET_INTERIOR_WEIGHT = 2.0 / len(INTERIOR_WIDTHS)


def ht_coefficients(pi: np.ndarray) -> np.ndarray:
    pi = np.asarray(pi, float)
    if pi.shape != (len(INTERIOR_WIDTHS),) or np.any(pi <= 0):
        raise ValueError("HT marginals must be positive and cover all interior widths")
    return TARGET_INTERIOR_WEIGHT / pi


def ht_weighted_objective(
    loss_full: torch.Tensor,
    loss_small: torch.Tensor,
    sampled_losses: tuple[torch.Tensor, torch.Tensor],
    sampled_widths: tuple[float, float],
    alpha: dict[float, float],
) -> torch.Tensor:
    """Historical mean-of-four scale with exact, unnormalized HT correction."""
    wi, wj = map(float, sampled_widths)
    if wi == wj or wi not in alpha or wj not in alpha:
        raise ValueError("HT objective requires two distinct registered interior widths")
    return (
        loss_full + loss_small
        + float(alpha[wi]) * sampled_losses[0]
        + float(alpha[wj]) * sampled_losses[1]
    ) / 4.0


def load_ht_policy(protocol_dir: str | Path, method: str):
    if method not in METHODS:
        raise ValueError(f"Unknown HT method: {method}")
    return load_frozen_policy(protocol_dir, POLICY_METHOD[method])


def validate_ht_policy(pair_table: pd.DataFrame, pi: np.ndarray, flops: dict) -> dict:
    pi = np.asarray(pi, float)
    probabilities = pair_table.probability.to_numpy(float)
    errors = np.abs(pair_marginals(pair_table) - pi)
    if (
        len(pair_table) != 91 or abs(probabilities.sum() - 1) > 1e-10
        or np.any(probabilities < 0) or abs(pi.sum() - 2) > 1e-8
        or errors.max() > 1e-8 or np.any(pi <= 0) or np.any(pi > 1)
    ):
        raise RuntimeError("Invalid frozen HT pair policy")
    alpha = ht_coefficients(pi)
    expected_effective = pi * alpha
    return {
        "sum_q": float(probabilities.sum()), "sum_pi": float(pi.sum()),
        "max_pair_marginal_error": float(errors.max()),
        "minimum_pi": float(pi.min()), "maximum_pi": float(pi.max()),
        "minimum_ht_weight": float(alpha.min()), "maximum_ht_weight": float(alpha.max()),
        "max_expected_target_weight_error": float(
            np.max(np.abs(expected_effective - TARGET_INTERIOR_WEIGHT))
        ),
        "expected_interior_flops": float(sum(
            flops[width] * pi[index] for index, width in enumerate(INTERIOR_WIDTHS)
        )),
    }


def simulate_ht_sanity(
    pair_table: pd.DataFrame,
    pi: np.ndarray,
    method: str,
    draws: int = 100_000,
) -> pd.DataFrame:
    """Verify realized inclusion and unnormalized HT contribution before training."""
    if method not in METHODS:
        raise ValueError(method)
    rng = np.random.default_rng(sampler_seed(POLICY_METHOD[method], SEED))
    probabilities = pair_table.probability.to_numpy(float)
    selected = rng.choice(len(pair_table), size=int(draws), p=probabilities)
    first = pair_table.width_i.to_numpy(float)[selected]
    second = pair_table.width_j.to_numpy(float)[selected]
    alpha = ht_coefficients(pi)
    rows = []
    for index, width in enumerate(INTERIOR_WIDTHS):
        empirical_pi = float(np.mean((first == width) | (second == width)))
        effective = empirical_pi * alpha[index]
        rows.append({
            "method": method, "width": width, "draws": int(draws),
            "expected_pi": pi[index], "empirical_pi": empirical_pi,
            "pi_error": empirical_pi - pi[index],
            "ht_weight_when_sampled": alpha[index],
            "expected_effective_weight": TARGET_INTERIOR_WEIGHT,
            "empirical_effective_weight": effective,
            "effective_weight_error": effective - TARGET_INTERIOR_WEIGHT,
        })
    return pd.DataFrame(rows)


def _save_model_snapshot(path: Path, model, epoch: int, method: str, policy_hash: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(), "epoch": int(epoch), "method": method,
        "seed": SEED, "policy_sha256": policy_hash, "ht_corrected": True,
    }, temporary)
    temporary.replace(path)


def _train_phase_ht(
    config: dict,
    output: Path,
    method: str,
    pair_table: pd.DataFrame,
    pi: np.ndarray,
    flops: dict,
    protocol: dict,
    first_epoch: int,
    last_epoch: int,
    learning_rate: float,
    phase_dir: Path,
    initial_checkpoint: Path | None = None,
) -> Path:
    device = torch.device(config["experiment"]["device"])
    seed_everything(SEED)
    loaders = build_development_train_loaders(config, training_seed=SEED)
    model = make_model(config, device)
    optimizer, scheduler = _new_optimizer_scheduler(
        model, config, learning_rate, last_epoch - first_epoch + 1
    )
    policy_name = POLICY_METHOD[method]
    policy_hash = protocol["frozen_policy_files"][policy_name]["sha256"]
    sampler_rng = np.random.default_rng(sampler_seed(policy_name, SEED))
    latest = phase_dir / "latest.pt"
    start_epoch = first_epoch
    cumulative_batches = 0
    cumulative_realized_flops = 0.0
    cumulative_counts = {width: 0 for width in INTERIOR_WIDTHS}
    cumulative_effective = {width: 0.0 for width in INTERIOR_WIDTHS}
    cumulative_pair_counts = {
        (round(float(row.width_i), 2), round(float(row.width_j), 2)): 0
        for row in pair_table.itertuples()
    }
    if latest.is_file():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        if payload.get("method") != method or payload.get("policy_sha256") != policy_hash:
            raise RuntimeError("HT resume checkpoint identity mismatch")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scheduler.load_state_dict(payload["scheduler"])
        _restore_rng(payload["rng"], loaders.train, sampler_rng)
        start_epoch = int(payload["epoch"]) + 1
        cumulative_batches = int(payload["cumulative_batches"])
        cumulative_realized_flops = float(payload["cumulative_realized_flops"])
        cumulative_counts.update({float(k): int(v) for k, v in payload["width_counts"].items()})
        cumulative_effective.update({
            float(k): float(v) for k, v in payload["effective_weight_sums"].items()
        })
        cumulative_pair_counts.update({
            tuple(map(float, key.split(","))): int(value)
            for key, value in payload["pair_counts"].items()
        })
    elif initial_checkpoint is not None:
        payload = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
        if payload.get("method") != method or payload.get("policy_sha256") != policy_hash:
            raise RuntimeError("HT phase-1 checkpoint identity mismatch")
        model.load_state_dict(payload["model"])
        # Reset optimizer/scheduler at epoch 51, but continue the exact model,
        # augmentation/data-loader, and pair-sampler RNG streams from epoch 50.
        _restore_rng(payload["rng"], loaders.train, sampler_rng)
        cumulative_batches = int(payload["cumulative_batches"])
        cumulative_realized_flops = float(payload["cumulative_realized_flops"])
        cumulative_counts.update({float(k): int(v) for k, v in payload["width_counts"].items()})
        cumulative_effective.update({
            float(k): float(v) for k, v in payload["effective_weight_sums"].items()
        })
        cumulative_pair_counts.update({
            tuple(map(float, key.split(","))): int(value)
            for key, value in payload["pair_counts"].items()
        })

    metrics_path = output / "training_epoch_metrics.csv"
    width_path = output / "ht_width_metrics_by_epoch.csv"
    pair_path = output / "pair_counts_by_epoch.csv"
    metrics_records = (
        pd.read_csv(metrics_path).loc[lambda x: x.epoch < start_epoch].to_dict("records")
        if metrics_path.is_file() else []
    )
    width_records = (
        pd.read_csv(width_path).loc[lambda x: x.epoch < start_epoch].to_dict("records")
        if width_path.is_file() else []
    )
    pair_records = (
        pd.read_csv(pair_path).loc[lambda x: x.epoch < start_epoch].to_dict("records")
        if pair_path.is_file() else []
    )
    probabilities = pair_table.probability.to_numpy(float)
    pairs = [
        (round(float(row.width_i), 2), round(float(row.width_j), 2))
        for row in pair_table.itertuples()
    ]
    alpha = dict(zip(INTERIOR_WIDTHS, ht_coefficients(pi)))
    pi_by_width = dict(zip(INTERIOR_WIDTHS, pi))
    endpoint_flops = float(protocol["fixed_endpoint_compute"])
    expected_total_flops = float(protocol["expected_total_compute"])

    for epoch in range(start_epoch, last_epoch + 1):
        model.train()
        epoch_loss = 0.0
        epoch_examples = epoch_batches = 0
        epoch_flops = 0.0
        epoch_counts = {width: 0 for width in INTERIOR_WIDTHS}
        epoch_effective = {width: 0.0 for width in INTERIOR_WIDTHS}
        epoch_pair_counts = {pair: 0 for pair in pairs}
        sampled_ht_weights = []
        for images, labels, _ in loaders.train:
            images, labels = images.to(device), labels.to(device)
            wi, wj = pairs[int(sampler_rng.choice(len(pairs), p=probabilities))]
            if wi == wj:
                raise RuntimeError("HT pair sampler selected duplicate widths")
            optimizer.zero_grad(set_to_none=True)
            model.set_width(1.0)
            teacher = model(images)
            teacher_detached = teacher.detach()
            loss_full = F.cross_entropy(teacher, labels)
            model.set_width(0.25)
            small = model(images)
            loss_small = F.cross_entropy(small, labels) + float(
                config["training"]["kd_lambda"]
            ) * kd_loss(small, teacher_detached, float(config["training"]["kd_temperature"]))
            sampled_losses = []
            for width in (wi, wj):
                model.set_width(width)
                logits = model(images)
                subnet = F.cross_entropy(logits, labels) + float(
                    config["training"]["kd_lambda"]
                ) * kd_loss(logits, teacher_detached, float(config["training"]["kd_temperature"]))
                sampled_losses.append(subnet)
                epoch_counts[width] += 1
                cumulative_counts[width] += 1
                epoch_effective[width] += float(alpha[width])
                cumulative_effective[width] += float(alpha[width])
                sampled_ht_weights.append(float(alpha[width]))
            total = ht_weighted_objective(
                loss_full, loss_small, tuple(sampled_losses), (wi, wj), alpha
            )
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite HT loss: method={method}, epoch={epoch}")
            total.backward(); optimizer.step()
            n = labels.numel()
            epoch_loss += float(total.detach()) * n
            epoch_examples += n; epoch_batches += 1; cumulative_batches += 1
            realized = endpoint_flops + flops[wi] + flops[wj]
            epoch_flops += realized; cumulative_realized_flops += realized
            epoch_pair_counts[(wi, wj)] += 1; cumulative_pair_counts[(wi, wj)] += 1

        lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        metrics_records.append({
            "method": method, "seed": SEED, "epoch": epoch,
            "phase": 1 if epoch <= 50 else 2,
            "train_loss": epoch_loss / epoch_examples, "learning_rate": lr,
            "batches": epoch_batches, "cumulative_batches": cumulative_batches,
            "mean_sampled_ht_weight": float(np.mean(sampled_ht_weights)),
            "max_sampled_ht_weight": float(np.max(sampled_ht_weights)),
            "expected_total_flops_per_batch": expected_total_flops,
            "realized_total_flops_per_batch": epoch_flops / epoch_batches,
            "cumulative_realized_total_flops_per_batch": cumulative_realized_flops / cumulative_batches,
            "cumulative_realized_flops": cumulative_realized_flops,
        })
        for width in INTERIOR_WIDTHS:
            width_records.append({
                "method": method, "seed": SEED, "epoch": epoch, "width": width,
                "expected_pi": pi_by_width[width],
                "epoch_empirical_pi": epoch_counts[width] / epoch_batches,
                "cumulative_empirical_pi": cumulative_counts[width] / cumulative_batches,
                "ht_weight_when_sampled": alpha[width],
                "target_effective_weight": TARGET_INTERIOR_WEIGHT,
                "epoch_effective_weight_per_batch": epoch_effective[width] / epoch_batches,
                "cumulative_effective_weight_per_batch": (
                    cumulative_effective[width] / cumulative_batches
                ),
            })
        for pair in pairs:
            pair_records.append({
                "method": method, "seed": SEED, "epoch": epoch,
                "width_i": pair[0], "width_j": pair[1],
                "epoch_count": epoch_pair_counts[pair],
                "cumulative_count": cumulative_pair_counts[pair],
            })
        pd.DataFrame(metrics_records).to_csv(metrics_path, index=False)
        pd.DataFrame(width_records).to_csv(width_path, index=False)
        pd.DataFrame(pair_records).to_csv(pair_path, index=False)
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "method": method,
            "seed": SEED, "phase": 1 if epoch <= 50 else 2,
            "rng": _capture_rng(loaders.train, sampler_rng),
            "cumulative_batches": cumulative_batches,
            "cumulative_realized_flops": cumulative_realized_flops,
            "width_counts": {str(k): v for k, v in cumulative_counts.items()},
            "effective_weight_sums": {str(k): v for k, v in cumulative_effective.items()},
            "pair_counts": {f"{k[0]:.2f},{k[1]:.2f}": v for k, v in cumulative_pair_counts.items()},
            "policy_sha256": policy_hash, "ht_corrected": True,
        }
        _atomic_checkpoint(latest, checkpoint)
        if epoch in CHECKPOINT_EPOCHS:
            _save_model_snapshot(output / f"epoch_{epoch:03d}.pt", model, epoch, method, policy_hash)
        print(
            f"{method} seed=3 epoch {epoch}/100 | loss={epoch_loss/epoch_examples:.4f}, "
            f"lr={lr:.6g}, HT mean/max={np.mean(sampled_ht_weights):.3f}/"
            f"{np.max(sampled_ht_weights):.3f}", flush=True,
        )
    return latest


def train_ht_method(config: dict, root: str | Path, protocol_dir: str | Path, method: str) -> Path:
    validate_pilot_config(config)
    pair_table, pi, flops, protocol = load_ht_policy(protocol_dir, method)
    if int(protocol.get("seed", -1)) != SEED:
        raise RuntimeError("Frozen policy protocol is not declared for development seed 3")
    output = Path(root) / method / "seed_3"
    output.mkdir(parents=True, exist_ok=True)
    final = output / "epoch_100.pt"
    if final.is_file() and (output / "training_provenance.json").is_file():
        return final
    phase_one = _train_phase_ht(
        config, output, method, pair_table, pi, flops, protocol,
        1, 50, 0.1, output / "phase_1",
    )
    _train_phase_ht(
        config, output, method, pair_table, pi, flops, protocol,
        51, 100, 0.01, output, initial_checkpoint=phase_one,
    )
    if not all((output / f"epoch_{epoch:03d}.pt").is_file() for epoch in CHECKPOINT_EPOCHS):
        raise RuntimeError("Missing required 10-epoch HT snapshots")
    shutil.copy2(output / "latest.pt", output / "resumable_final.pt")
    provenance = {
        "experiment": "rq2_v3_ht_development", "method": method, "seed": SEED,
        "epochs": 100, "schedule": "50+50_optimizer_scheduler_reset",
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "target_loss": "L_.25 + L_1.0 + (1/7) sum over 14 interior losses",
        "estimator": "Horvitz-Thompson without clipping or batch renormalization",
        "target_interior_weight": TARGET_INTERIOR_WEIGHT,
        "policy_frozen": True, "online_geometry": False, "gradient_oracle_used": False,
        "accuracy_used_to_build_policy": False, "test_used": False,
        "sampler_seed": sampler_seed(POLICY_METHOD[method], SEED),
        "policy_sha256": protocol["frozen_policy_files"][POLICY_METHOD[method]]["sha256"],
        "variance_gate_decision": protocol["variance_gate"]["decision"],
        "variance_gate_sha256": protocol["variance_gate"]["sha256"],
        "epoch_100_sha256": _sha256(final),
    }
    (output / "training_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return final


def evaluate_ht_checkpoints(config: dict, root: str | Path, method: str) -> Path:
    if method not in METHODS:
        raise ValueError(method)
    root = Path(root)
    method_root = root / method / "seed_3"
    output = root / "evaluation" / method / "seed_3"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "dense_validation_by_checkpoint.csv"
    records = pd.read_csv(result_path).to_dict("records") if result_path.is_file() else []
    completed = set(pd.DataFrame(records).groupby("epoch").size().loc[lambda x: x == 16].index) if records else set()
    device = torch.device(config["experiment"]["device"])
    loaders = build_interim_validation_loaders(config)
    training = pd.read_csv(method_root / "training_epoch_metrics.csv").set_index("epoch")
    for epoch in CHECKPOINT_EPOCHS:
        if epoch in completed:
            continue
        checkpoint = method_root / f"epoch_{epoch:03d}.pt"
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload.get("method") != method or int(payload.get("epoch", -1)) != epoch:
            raise RuntimeError(f"Snapshot identity mismatch: {checkpoint}")
        model = make_model(config, device)
        model.load_state_dict(payload["model"])
        epoch_rows = []
        for width in GRID:
            calibrate_batch_norm(
                model, loaders.calibration, width, device,
                int(config["evaluation"]["bn_calibration_batches"]),
            )
            metrics = evaluate_width(model, loaders.validation, width, device)
            epoch_rows.append({
                "method": method, "seed": SEED, "epoch": epoch,
                "split": "validation_5k", "width": width,
                "cumulative_realized_flops": float(training.loc[epoch, "cumulative_realized_flops"]),
                **metrics,
            })
        records = [row for row in records if int(row["epoch"]) != epoch] + epoch_rows
        pd.DataFrame(records).sort_values(["epoch", "width"]).to_csv(result_path, index=False)
        print(f"{method}: dense validation checkpoint {epoch}/100 complete", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frame = pd.read_csv(result_path)
    if len(frame) != len(CHECKPOINT_EPOCHS) * len(GRID):
        raise RuntimeError("Incomplete dense checkpoint evaluation")
    (output / "evaluation_complete.json").write_text(json.dumps({
        "method": method, "seed": SEED, "epochs": list(CHECKPOINT_EPOCHS),
        "widths": list(GRID), "split": "validation_5k", "test_used": False,
    }, indent=2) + "\n")
    (output / "TEST_SPLIT_NOT_ACCESSED.txt").write_text(
        "HT development evaluation used only the fixed CIFAR-100 train/validation split.\n"
    )
    return result_path


def finalize_ht_development(root: str | Path) -> dict:
    root = Path(root)
    dense = pd.concat([
        pd.read_csv(root / "evaluation" / method / "seed_3" / "dense_validation_by_checkpoint.csv")
        for method in METHODS
    ], ignore_index=True)
    if len(dense) != len(METHODS) * len(CHECKPOINT_EPOCHS) * len(GRID):
        raise RuntimeError("Incomplete HT development evaluation")
    dense.to_csv(root / "dense_validation_by_checkpoint.csv", index=False)
    rows = []
    for (method, epoch), group in dense.groupby(["method", "epoch"]):
        rows.append({
            "method": method, "epoch": int(epoch),
            "cumulative_realized_flops": float(group.cumulative_realized_flops.iloc[0]),
            "dense_mean_accuracy": float(group.accuracy.mean()),
            "worst_accuracy": float(group.accuracy.min()),
            "low_mean_accuracy": float(group.loc[group.width.between(.30, .45), "accuracy"].mean()),
            "mid_mean_accuracy": float(group.loc[group.width.between(.50, .75), "accuracy"].mean()),
            "high_mean_accuracy": float(group.loc[group.width.between(.80, .95), "accuracy"].mean()),
            "width_025_accuracy": float(group.loc[np.isclose(group.width, .25), "accuracy"].iloc[0]),
            "width_100_accuracy": float(group.loc[np.isclose(group.width, 1.0), "accuracy"].iloc[0]),
        })
    summary = pd.DataFrame(rows).sort_values(["epoch", "method"])
    summary.to_csv(root / "checkpoint_family_summary.csv", index=False)
    wide = summary.pivot(index="epoch", columns="method")
    convergence_rows = []
    for epoch in CHECKPOINT_EPOCHS:
        row = {"epoch": epoch}
        for metric in (
            "dense_mean_accuracy", "worst_accuracy", "low_mean_accuracy",
            "mid_mean_accuracy", "high_mean_accuracy", "width_025_accuracy", "width_100_accuracy",
        ):
            geo = float(wide.loc[epoch, (metric, "geo_ht")])
            resource = float(wide.loc[epoch, (metric, "resource_ht")])
            row[f"geo_{metric}"] = geo; row[f"resource_{metric}"] = resource
            row[f"delta_{metric}"] = geo - resource
        row["geo_cumulative_realized_flops"] = float(
            wide.loc[epoch, ("cumulative_realized_flops", "geo_ht")]
        )
        row["resource_cumulative_realized_flops"] = float(
            wide.loc[epoch, ("cumulative_realized_flops", "resource_ht")]
        )
        convergence_rows.append(row)
    convergence = pd.DataFrame(convergence_rows)
    convergence.to_csv(root / "ht_convergence_comparison.csv", index=False)
    curves = {
        method: summary.loc[summary.method.eq(method)].sort_values("cumulative_realized_flops")
        for method in METHODS
    }
    common_compute = np.linspace(
        max(frame.cumulative_realized_flops.min() for frame in curves.values()),
        min(frame.cumulative_realized_flops.max() for frame in curves.values()),
        len(CHECKPOINT_EPOCHS),
    )
    compute_comparison = pd.DataFrame({"cumulative_realized_flops": common_compute})
    for method, frame in curves.items():
        compute_comparison[f"{method}_dense_mean_accuracy"] = np.interp(
            common_compute, frame.cumulative_realized_flops, frame.dense_mean_accuracy
        )
    compute_comparison["geo_minus_resource_dense_mean_accuracy"] = (
        compute_comparison.geo_ht_dense_mean_accuracy
        - compute_comparison.resource_ht_dense_mean_accuracy
    )
    compute_comparison.to_csv(root / "ht_convergence_common_compute.csv", index=False)
    final_dense = dense.loc[dense.epoch.eq(100)]
    final_wide = final_dense.pivot(index="width", columns="method", values="accuracy").reset_index()
    final_wide["geo_minus_resource_accuracy"] = final_wide.geo_ht - final_wide.resource_ht
    final_wide.to_csv(root / "final_width_comparison.csv", index=False)
    fig, axis = plt.subplots(figsize=(8, 4.8))
    for method, frame in summary.groupby("method"):
        axis.plot(frame.epoch, frame.dense_mean_accuracy, marker="o", label=method)
    axis.set(xlabel="Epoch", ylabel="Dense validation mean accuracy")
    axis.grid(alpha=0.25); axis.legend(); fig.tight_layout()
    fig.savefig(root / "ht_convergence_vs_epoch.png", dpi=200); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4.8))
    for method, frame in curves.items():
        axis.plot(frame.cumulative_realized_flops, frame.dense_mean_accuracy,
                  marker="o", label=method)
    axis.set(xlabel="Cumulative realized training FLOPs", ylabel="Dense validation mean accuracy")
    axis.grid(alpha=0.25); axis.legend(); fig.tight_layout()
    fig.savefig(root / "ht_convergence_vs_compute.png", dpi=200); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(final_wide.width, final_wide.geo_minus_resource_accuracy, width=0.035)
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="Width", ylabel="Geo-HT minus Resource-HT accuracy (epoch 100)")
    axis.grid(axis="y", alpha=0.25); fig.tight_layout()
    fig.savefig(root / "final_width_delta.png", dpi=200); plt.close(fig)
    final_row = convergence.loc[convergence.epoch.eq(100)].iloc[0]
    mean_convergence_delta = float(convergence.delta_dense_mean_accuracy.mean())
    decision = {
        "status": "RQ2_V3_HT_SEED3_DEVELOPMENT_COMPLETE",
        "seed": SEED, "methods": list(METHODS), "test_used": False,
        "same_target_objective": True, "expected_compute_matched": True,
        "final_dense_accuracy_delta": float(final_row.delta_dense_mean_accuracy),
        "final_worst_accuracy_delta": float(final_row.delta_worst_accuracy),
        "mean_checkpoint_dense_accuracy_delta": mean_convergence_delta,
        "mean_common_compute_dense_accuracy_delta": float(
            compute_comparison.geo_minus_resource_dense_mean_accuracy.mean()
        ),
        "geo_faster_on_average_checkpoints": bool(mean_convergence_delta > 0),
        "geo_better_final_dense": bool(final_row.delta_dense_mean_accuracy > 0),
        "fresh_confirmatory_seeds_authorized_automatically": False,
        "interpretation": "development evidence only; do not tune HT weights or pi from seed 3",
    }
    (root / "ht_development_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    return decision


def _safe_extract(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    resolved = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != resolved and resolved not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)
    return destination


def find_variance_gate(input_root: str | Path, materialized_root: str | Path) -> Path:
    input_root, materialized_root = Path(input_root), Path(materialized_root)

    def find(root):
        matches = []
        for path in root.rglob("metadata.json"):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("experiment") == "v3_cross_batch_heldout_variance":
                matches.append(path)
        return sorted(set(matches))

    matches = find(input_root)
    if not matches:
        archives = sorted(input_root.rglob("*cross-batch*.zip"))
        if len(archives) == 1:
            matches = find(_safe_extract(archives[0], materialized_root))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one v3 variance-gate metadata file, found {matches}")
    payload = json.loads(matches[0].read_text())
    if payload.get("decision") not in {"STRONG_GO", "WEAK_GO"}:
        raise RuntimeError(f"Variance gate does not authorize development HT training: {payload.get('decision')}")
    return matches[0]
