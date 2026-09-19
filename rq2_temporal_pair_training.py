"""Long-horizon temporal SW pair-policy ablations.

All branches reuse the frozen common Uniform warm-up through epoch 10.  The
training objective, optimizer, scheduler, data order, BN protocol and random
pair-draw stream are inherited from the RQ2 pairwise pilot.  Only q_ij changes.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from data import (
    build_development_train_loaders,
    build_interim_validation_loaders,
    build_policy_geometry_loaders,
)
from research_utils import seed_everything
from rq2_dynamic_seed3_pilot import _atomic_checkpoint
from rq2_e2e_pairwise_pilot import (
    EVAL_EPOCHS,
    PAIR_RNG_SEED,
    PILOT_SEED,
    REFRESH_STATES,
    _audit_policy_subsets,
    _capture_matched_rng,
    _ids_sha256,
    _optimizer_scheduler,
    _preserving_dense_evaluation,
    _restore_matched_rng,
    _sw_matrix,
    _train_batch,
)
from rq2_pair_policies import PairPolicy, TemporalSWPairPolicy, UniformPairPolicy
from rq2_pairwise_surrogate_regret import INTERIOR_WIDTHS, NUM_PAIRS, PAIR_INDICES
from rq2_quick_trajectory_diagnostic import _current_geometry
from s1_width import make_model


TEMPORAL_METHODS = ("sw_continuity", "sw_anneal", "sw_continuity_anneal")
CONTINUITY_STRENGTH_IN_SCORE_STD = 1.0
FINAL_UNIFORM_MIXTURE = 0.8


def uniform_mixture_at_epoch(state_epoch: int) -> float:
    """Cosine anneal from alpha_10=0 to alpha_90=0.8."""
    epoch = int(state_epoch)
    if epoch not in REFRESH_STATES:
        raise ValueError(f"Policy refresh epoch must be one of {REFRESH_STATES}")
    progress = (epoch - REFRESH_STATES[0]) / (REFRESH_STATES[-1] - REFRESH_STATES[0])
    return float(FINAL_UNIFORM_MIXTURE * 0.5 * (1.0 - np.cos(np.pi * progress)))


def temporal_policy(
    method: str,
    sw_matrix: np.ndarray,
    previous_probabilities: np.ndarray,
    state_epoch: int,
) -> TemporalSWPairPolicy:
    if method not in TEMPORAL_METHODS:
        raise ValueError(method)
    use_continuity = method in {"sw_continuity", "sw_continuity_anneal"}
    alpha = uniform_mixture_at_epoch(state_epoch) if method in {
        "sw_anneal", "sw_continuity_anneal"
    } else 0.0
    return TemporalSWPairPolicy(
        method,
        sw_matrix,
        previous_probabilities,
        continuity_strength_in_score_std=CONTINUITY_STRENGTH_IN_SCORE_STD,
        uniform_mixture=alpha,
        use_continuity=use_continuity,
    )


def load_temporal_protocol(root: str | Path) -> dict:
    path = Path(root) / "temporal_pair_protocol.json"
    if not path.is_file():
        raise FileNotFoundError("Frozen temporal_pair_protocol.json is required")
    protocol = json.loads(path.read_text())
    expected = {
        "status": "FROZEN_BEFORE_TEMPORAL_PAIR_TRAINING",
        "accuracy_used_to_select_hyperparameters": False,
        "continuity_divergence": "KL(q||q_previous)",
        "continuity_strength_in_score_std": CONTINUITY_STRENGTH_IN_SCORE_STD,
        "uniform_annealing_schedule": "cosine_alpha_10_0_alpha_90_0.8",
        "final_uniform_mixture": FINAL_UNIFORM_MIXTURE,
        "fixed_marginal": 1.0 / 7.0,
        "common_warmup_epochs": 10,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise RuntimeError(f"Frozen temporal protocol mismatch: {key}")
    return protocol


def _checkpoint(output: Path, epoch: int) -> Path:
    return output / "checkpoints" / f"epoch_{epoch:03d}.pt"


def _policy_sweep(
    model,
    loaders,
    config,
    device,
    method,
    previous_probabilities,
    state_epoch,
    output,
    expected_geometry_ids_sha256,
):
    started = time.perf_counter()
    _, _, representation_pairs, sample_ids = _current_geometry(
        model, loaders, config, device
    )
    if _ids_sha256(list(map(int, sample_ids))) != expected_geometry_ids_sha256:
        raise RuntimeError("Temporal policy geometry sample IDs/order changed")
    sw = _sw_matrix(representation_pairs)
    policy = temporal_policy(method, sw, previous_probabilities, state_epoch)
    policy_dir = output / "sw_policies"
    policy_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        policy_dir / f"epoch_{state_epoch:03d}.npz",
        sw_matrix=sw,
        probabilities=policy.probabilities,
        previous_probabilities=np.asarray(previous_probabilities),
        geometry_sample_ids=np.asarray(sample_ids, dtype=np.int64),
        widths=np.asarray(INTERIOR_WIDTHS),
    )
    policy.frame().to_csv(policy_dir / f"epoch_{state_epoch:03d}.csv", index=False)
    row = {"method": method, "epoch": int(state_epoch), **policy.diagnostics}
    history_path = output / "temporal_policy_history.csv"
    if history_path.is_file():
        history = pd.read_csv(history_path)
        history = history.loc[~history.epoch.eq(state_epoch)]
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    else:
        history = pd.DataFrame([row])
    history.sort_values("epoch").to_csv(history_path, index=False)
    return policy, time.perf_counter() - started


def train_temporal_branch(
    config: dict,
    root: str | Path,
    gate_a_summary: str | Path,
    method: str,
) -> Path:
    if method not in TEMPORAL_METHODS:
        raise ValueError(method)
    root = Path(root)
    protocol = load_temporal_protocol(root)
    output = root / method
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    final = _checkpoint(output, 100)
    provenance_path = output / "training_provenance.json"
    if final.is_file() and provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("temporal_protocol_sha256") != hashlib.sha256(
            (root / "temporal_pair_protocol.json").read_bytes()
        ).hexdigest():
            raise RuntimeError("Existing branch belongs to a different temporal protocol")
        return final

    common = root / "common_warmup" / "epoch_010.pt"
    if not common.is_file():
        raise FileNotFoundError("Common Uniform epoch-10 checkpoint is required")
    device = torch.device(config["experiment"]["device"])
    seed_everything(PILOT_SEED)
    loaders = build_development_train_loaders(config, training_seed=PILOT_SEED)
    evaluation_loaders = build_interim_validation_loaders(config)
    policy_loaders = build_policy_geometry_loaders(config)
    subset_audit = _audit_policy_subsets(policy_loaders, evaluation_loaders, output)
    model = make_model(config, device)
    optimizer, scheduler = _optimizer_scheduler(model, config, 0.1)
    pair_rng = np.random.default_rng(PAIR_RNG_SEED)
    latest = output / "latest.pt"
    source = torch.load(
        latest if latest.is_file() else common, map_location="cpu", weights_only=False
    )
    model.load_state_dict(source["model"])
    optimizer.load_state_dict(source["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value): state[key] = value.to(device)
    scheduler.load_state_dict(source["scheduler"])
    _restore_matched_rng(source["rng"], loaders.train, pair_rng)
    start_epoch = 11
    current_policy = UniformPairPolicy()
    if latest.is_file():
        if source.get("method") != method or int(source.get("seed", -1)) != PILOT_SEED:
            raise RuntimeError("Temporal branch resume identity mismatch")
        if source.get("temporal_protocol_sha256") != hashlib.sha256(
            (root / "temporal_pair_protocol.json").read_bytes()
        ).hexdigest():
            raise RuntimeError("Temporal resume checkpoint uses another frozen protocol")
        start_epoch = int(source["epoch"]) + 1
        current_policy = PairPolicy(method, np.asarray(source["current_q"], float))
        if int(source["epoch"]) >= 51:
            optimizer, scheduler = _optimizer_scheduler(model, config, 0.01)
            optimizer.load_state_dict(source["optimizer"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value): state[key] = value.to(device)
            scheduler.load_state_dict(source["scheduler"])

    def prior(path):
        return (
            pd.read_csv(path).loc[lambda frame: frame.epoch < start_epoch].to_dict("records")
            if path.is_file() else []
        )

    metrics_path = output / "metrics.csv"
    pair_path = output / "pair_stats.csv"
    width_path = output / "width_marginals.csv"
    dense_path = output / "dense_metrics.csv"
    records, pair_rows, width_rows, dense_rows = map(
        prior, (metrics_path, pair_path, width_path, dense_path)
    )
    if not dense_rows:
        common_dense = pd.read_csv(root / "common_warmup" / "dense_metrics_epoch_010.csv")
        common_dense["method"] = method
        dense_rows = common_dense.to_dict("records")
    protocol_hash = hashlib.sha256(
        (root / "temporal_pair_protocol.json").read_bytes()
    ).hexdigest()

    for epoch in range(start_epoch, 101):
        state_epoch = epoch - 1
        policy_seconds = 0.0
        if state_epoch in REFRESH_STATES:
            current_policy, policy_seconds = _policy_sweep(
                model, policy_loaders, config, device, method,
                current_policy.probabilities, state_epoch, output,
                subset_audit["policy_geometry_ids_sha256"],
            )
        if epoch == 51:
            optimizer, scheduler = _optimizer_scheduler(model, config, 0.01)
        model.train()
        started = time.perf_counter()
        pair_counts = np.zeros(NUM_PAIRS, dtype=int)
        width_counts = np.zeros(14, dtype=int)
        losses, draws, sample_ids = [], [], []
        for images, labels, ids in loaders.train:
            sample_ids.extend(torch.as_tensor(ids).cpu().tolist())
            images, labels = images.to(device), labels.to(device)
            draw = float(pair_rng.random()); draws.append(draw)
            wi, wj, pair_index = current_policy.sample(draw)
            optimizer.zero_grad(set_to_none=True)
            loss = _train_batch(model, images, labels, (0.25, wi, wj), config)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss: {method}, epoch {epoch}")
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
            pair_counts[pair_index] += 1
            width_counts[INTERIOR_WIDTHS.index(wi)] += 1
            width_counts[INTERIOR_WIDTHS.index(wj)] += 1
        lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        records.append({
            "method": method, "seed": PILOT_SEED, "epoch": epoch,
            "train_loss": float(np.mean(losses)), "learning_rate": lr,
            "wall_clock_seconds": time.perf_counter() - started,
            "policy_computation_seconds": policy_seconds,
            "pair_uniform_draw_sha256": hashlib.sha256(np.asarray(draws).tobytes()).hexdigest(),
            "train_sample_order_sha256": _ids_sha256(list(map(int, sample_ids))),
        })
        for index, (i, j) in enumerate(PAIR_INDICES):
            pair_rows.append({
                "method": method, "seed": PILOT_SEED, "epoch": epoch,
                "pair_index": index, "width_i": INTERIOR_WIDTHS[i],
                "width_j": INTERIOR_WIDTHS[j], "count": int(pair_counts[index]),
                "policy_probability": float(current_policy.probabilities[index]),
            })
        for index, width in enumerate(INTERIOR_WIDTHS):
            width_rows.append({
                "method": method, "seed": PILOT_SEED, "epoch": epoch, "width": width,
                "count": int(width_counts[index]),
                "realized_marginal": float(width_counts[index] / len(draws)),
                "expected_marginal": 1.0 / 7.0,
            })
        if epoch in EVAL_EPOCHS:
            dense_rows.extend(_preserving_dense_evaluation(
                model, evaluation_loaders, config, device, method, epoch
            ))
        pd.DataFrame(records).to_csv(metrics_path, index=False)
        pd.DataFrame(pair_rows).to_csv(pair_path, index=False)
        pd.DataFrame(width_rows).to_csv(width_path, index=False)
        pd.DataFrame(dense_rows).to_csv(dense_path, index=False)
        checkpoint = {
            "status": "TEMPORAL_PAIR_BRANCH_PROGRESS", "method": method,
            "seed": PILOT_SEED, "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": _capture_matched_rng(loaders.train, pair_rng, device),
            "current_q": current_policy.probabilities.tolist(),
            "common_epoch10_sha256": hashlib.sha256(common.read_bytes()).hexdigest(),
            "temporal_protocol_sha256": protocol_hash,
        }
        _atomic_checkpoint(latest, checkpoint)
        if epoch in (50, 100):
            _atomic_checkpoint(_checkpoint(output, epoch), checkpoint)
        print(
            f"{method} epoch {epoch}/100 | loss={np.mean(losses):.4f}, "
            f"lr={lr:.6g}, policy={policy_seconds/60:.1f}m", flush=True,
        )

    provenance = {
        "status": "TEMPORAL_PAIR_BRANCH_COMPLETE", "method": method,
        "seed": PILOT_SEED, "common_epoch10_sha256": hashlib.sha256(common.read_bytes()).hexdigest(),
        "temporal_protocol_sha256": protocol_hash,
        "pair_rng_seed": PAIR_RNG_SEED, "fixed_marginal": 1.0 / 7.0,
        "refresh_states": list(REFRESH_STATES),
        "continuity_divergence": protocol["continuity_divergence"],
        "continuity_strength_in_score_std": CONTINUITY_STRENGTH_IN_SCORE_STD,
        "uniform_annealing_schedule": protocol["uniform_annealing_schedule"],
        "final_uniform_mixture": FINAL_UNIFORM_MIXTURE,
        "accuracy_used_to_select_hyperparameters": False,
        "validation_used_to_build_policy": False, "test_used_to_build_policy": False,
        "loss_reduction": "mean_of_four_equal_subnet_losses",
        "policy_geometry_source": subset_audit["policy_geometry_source"],
        "policy_geometry_size": subset_audit["policy_geometry_count"],
        "policy_geometry_ids_sha256": subset_audit["policy_geometry_ids_sha256"],
        "bn_calibration_ids_sha256": subset_audit["bn_calibration_ids_sha256"],
        "validation_ids_sha256": subset_audit["validation_ids_sha256"],
        "data_subset_audit_status": subset_audit["status"],
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return final


def summarize_temporal_ablation(root: str | Path) -> dict:
    root = Path(root)
    methods = ("uniform", "resource", "pure_sw", "resource_geo") + TEMPORAL_METHODS
    missing = [
        str(root / method / "dense_metrics.csv") for method in methods
        if not (root / method / "dense_metrics.csv").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Incomplete temporal ablation: {missing}")
    rows = []
    low = {0.30, 0.35, 0.40, 0.45}
    mid = {0.55, 0.60, 0.65, 0.70}
    high = {0.80, 0.85, 0.90, 0.95}
    for method in methods:
        dense = pd.read_csv(root / method / "dense_metrics.csv")
        final = dense.loc[dense.epoch.eq(100)].copy()
        interior = final.loc[final.width.isin(INTERIOR_WIDTHS)]
        rows.append({
            "method": method,
            "dense_mean_accuracy": float(final.accuracy.mean()),
            "interior_mean_accuracy": float(interior.accuracy.mean()),
            "worst_width_accuracy": float(interior.accuracy.min()),
            "low_accuracy": float(final.loc[final.width.isin(low), "accuracy"].mean()),
            "mid_accuracy": float(final.loc[final.width.isin(mid), "accuracy"].mean()),
            "high_accuracy": float(final.loc[final.width.isin(high), "accuracy"].mean()),
            "full_width_accuracy": float(final.loc[final.width.eq(1.0), "accuracy"].iloc[0]),
        })
    table = pd.DataFrame(rows)
    table.to_csv(root / "temporal_pair_ablation_summary.csv", index=False)
    indexed = table.set_index("method")
    comparisons = {}
    for method in TEMPORAL_METHODS:
        comparisons[method] = {
            f"minus_{baseline}_{metric}": float(indexed.loc[method, metric] - indexed.loc[baseline, metric])
            for baseline in ("uniform", "pure_sw")
            for metric in ("dense_mean_accuracy", "interior_mean_accuracy", "worst_width_accuracy")
        }
    decision = {
        "status": "TEMPORAL_PAIR_ABLATION_COMPLETE",
        "seed": PILOT_SEED,
        "methods": list(methods),
        "comparisons": comparisons,
        "continuity_recovers_over_pure_sw": bool(
            indexed.loc["sw_continuity", "interior_mean_accuracy"]
            > indexed.loc["pure_sw", "interior_mean_accuracy"]
        ),
        "annealing_recovers_over_pure_sw": bool(
            indexed.loc["sw_anneal", "interior_mean_accuracy"]
            > indexed.loc["pure_sw", "interior_mean_accuracy"]
        ),
        "combined_beats_uniform": bool(
            indexed.loc["sw_continuity_anneal", "interior_mean_accuracy"]
            > indexed.loc["uniform", "interior_mean_accuracy"]
        ),
        "accuracy_used_to_modify_method_after_freeze": False,
    }
    (root / "temporal_pair_ablation_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    return decision


__all__ = [
    "TEMPORAL_METHODS", "CONTINUITY_STRENGTH_IN_SCORE_STD",
    "FINAL_UNIFORM_MIXTURE", "uniform_mixture_at_epoch", "temporal_policy",
    "load_temporal_protocol", "train_temporal_branch", "summarize_temporal_ablation",
]
