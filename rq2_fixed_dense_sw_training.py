"""Resume-safe seed-3 training for frozen E10 SW and shape-matched control."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import (
    build_development_train_loaders, build_interim_validation_loaders,
    build_policy_geometry_loaders,
)
from research_utils import seed_everything
from rq2_dynamic_seed3_pilot import _atomic_checkpoint
from rq2_e2e_pairwise_pilot import (
    EVAL_EPOCHS, PAIR_RNG_SEED, PILOT_SEED, REFRESH_STATES,
    _audit_policy_subsets, _capture_matched_rng, _ids_sha256,
    _optimizer_scheduler, _preserving_dense_evaluation, _restore_matched_rng,
    _train_batch,
)
from rq2_fixed_dense_sw_gate import METHODS, load_frozen_policy, sha256
from rq2_pair_policies import PairPolicy
from rq2_pairwise_surrogate_regret import INTERIOR_WIDTHS, NUM_PAIRS, PAIR_INDICES
from rq2_quick_trajectory_diagnostic import _current_geometry
from s1_width import make_model


def train_fixed_branch(config: dict, root: str | Path, method: str) -> Path:
    if method not in METHODS:
        raise ValueError(method)
    root = Path(root)
    protocol = load_frozen_policy(root)
    protocol_hash = sha256(root / "fixed_dense_sw_frozen_protocol.json")
    output = root / method
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    final = output / "checkpoints/epoch_100.pt"
    provenance_path = output / "training_provenance.json"
    if final.is_file() and provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("frozen_protocol_sha256") != protocol_hash:
            raise RuntimeError("Completed branch belongs to another frozen policy")
        return final

    device = torch.device(config["experiment"]["device"])
    seed_everything(PILOT_SEED)
    loaders = build_development_train_loaders(config, training_seed=PILOT_SEED)
    evaluation_loaders = build_interim_validation_loaders(config)
    policy_loaders = build_policy_geometry_loaders(config)
    subset_audit = _audit_policy_subsets(policy_loaders, evaluation_loaders, output)
    model = make_model(config, device)
    optimizer, scheduler = _optimizer_scheduler(model, config, 0.1)
    pair_rng = np.random.default_rng(PAIR_RNG_SEED)
    frozen_q = np.asarray(protocol["policy_probabilities"][method], dtype=float)
    policy = PairPolicy(method, frozen_q)
    latest = output / "latest.pt"
    source = torch.load(
        latest if latest.is_file() else root / "common_warmup/epoch_010.pt",
        map_location="cpu", weights_only=False,
    )
    model.load_state_dict(source["model"])
    optimizer.load_state_dict(source["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    scheduler.load_state_dict(source["scheduler"])
    _restore_matched_rng(source["rng"], loaders.train, pair_rng)
    start_epoch = 11
    if latest.is_file():
        if source.get("method") != method or int(source.get("seed", -1)) != PILOT_SEED:
            raise RuntimeError("Resume branch identity mismatch")
        if source.get("frozen_protocol_sha256") != protocol_hash:
            raise RuntimeError("Resume policy freeze mismatch")
        if source.get("common_epoch10_sha256") != protocol["common_e10_sha256"]:
            raise RuntimeError("Resume checkpoint uses another common E10 state")
        if not np.allclose(source["current_q"], frozen_q, atol=1e-12):
            raise RuntimeError("Resume checkpoint no longer uses the frozen policy")
        start_epoch = int(source["epoch"]) + 1
        if int(source["epoch"]) >= 51:
            optimizer, scheduler = _optimizer_scheduler(model, config, 0.01)
            optimizer.load_state_dict(source["optimizer"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
            scheduler.load_state_dict(source["scheduler"])

    def prior(path):
        return (
            pd.read_csv(path).loc[lambda frame: frame.epoch < start_epoch].to_dict("records")
            if path.is_file() else []
        )

    metrics_path = output / "train_log.csv"
    pair_path = output / "pair_stats.csv"
    width_path = output / "width_marginals.csv"
    dense_path = output / "dense_metrics.csv"
    records, pair_rows, width_rows, dense_rows = map(
        prior, (metrics_path, pair_path, width_path, dense_path)
    )
    if not dense_rows:
        common_dense = pd.read_csv(root / "common_warmup/dense_metrics_epoch_010.csv")
        common_dense["method"] = method
        dense_rows = common_dense.to_dict("records")

    for epoch in range(start_epoch, 101):
        state_epoch = epoch - 1
        sweep_seconds = 0.0
        if state_epoch in REFRESH_STATES:
            # Match the extra forward-compute in the original U/R/SW/RG lanes.
            # The sweep is discarded: the E10 policy remains exactly frozen.
            sweep_started = time.perf_counter()
            _, _, _, sample_ids = _current_geometry(
                model, policy_loaders, config, device
            )
            if _ids_sha256(list(map(int, sample_ids))) != subset_audit["policy_geometry_ids_sha256"]:
                raise RuntimeError("Policy-geometry IDs/order changed")
            sweep_seconds = time.perf_counter() - sweep_started
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
            draw = float(pair_rng.random())
            draws.append(draw)
            wi, wj, pair_index = policy.sample(draw)
            optimizer.zero_grad(set_to_none=True)
            loss = _train_batch(model, images, labels, (0.25, wi, wj), config)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss: {method} epoch {epoch}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            pair_counts[pair_index] += 1
            width_counts[INTERIOR_WIDTHS.index(wi)] += 1
            width_counts[INTERIOR_WIDTHS.index(wj)] += 1
        lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        records.append({
            "method": method, "seed": PILOT_SEED, "epoch": epoch,
            "train_loss": float(np.mean(losses)), "learning_rate": lr,
            "wall_clock_seconds": time.perf_counter() - started,
            "matched_geometry_sweep_seconds": sweep_seconds,
            "pair_uniform_draw_sha256": hashlib.sha256(np.asarray(draws).tobytes()).hexdigest(),
            "train_sample_order_sha256": _ids_sha256(list(map(int, sample_ids))),
        })
        for index, (i, j) in enumerate(PAIR_INDICES):
            pair_rows.append({
                "method": method, "seed": PILOT_SEED, "epoch": epoch,
                "pair_index": index, "width_i": INTERIOR_WIDTHS[i],
                "width_j": INTERIOR_WIDTHS[j], "count": int(pair_counts[index]),
                "policy_probability": float(frozen_q[index]),
            })
        for index, width in enumerate(INTERIOR_WIDTHS):
            width_rows.append({
                "method": method, "seed": PILOT_SEED, "epoch": epoch,
                "width": width, "count": int(width_counts[index]),
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
            "status": "FIXED_DENSE_SW_BRANCH_PROGRESS", "method": method,
            "seed": PILOT_SEED, "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": _capture_matched_rng(loaders.train, pair_rng, device),
            "current_q": frozen_q.tolist(),
            "common_epoch10_sha256": protocol["common_e10_sha256"],
            "frozen_protocol_sha256": protocol_hash,
        }
        _atomic_checkpoint(latest, checkpoint)
        if epoch in (50, 100):
            _atomic_checkpoint(output / f"checkpoints/epoch_{epoch:03d}.pt", checkpoint)
        print(f"{method} epoch {epoch}/100 | loss={np.mean(losses):.4f}, "
              f"lr={lr:.6g}, matched_sweep={sweep_seconds/60:.1f}m", flush=True)

    final_dense = pd.DataFrame(dense_rows).loc[lambda frame: frame.epoch.eq(100)]
    summary = {
        "method": method, "seed": PILOT_SEED,
        "policy": protocol["policy_summaries"][method],
        "selected_kappa": protocol["selected_kappa"],
        "fixed_since_epoch": 10,
        "dense_mean_validation_accuracy": float(final_dense.accuracy.mean()),
        "interior_mean_validation_accuracy": float(
            final_dense.loc[final_dense.width.isin(INTERIOR_WIDTHS), "accuracy"].mean()
        ),
        "worst_interior_validation_accuracy": float(
            final_dense.loc[final_dense.width.isin(INTERIOR_WIDTHS), "accuracy"].min()
        ),
        "full_width_validation_accuracy": float(
            final_dense.loc[final_dense.width.eq(1.0), "accuracy"].iloc[0]
        ),
        "realized_pair_counts": (
            pd.DataFrame(pair_rows).groupby("pair_index")["count"].sum()
            .reindex(range(NUM_PAIRS), fill_value=0).astype(int).tolist()
        ),
        "test_used": False,
    }
    (output / "policy_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    provenance = {
        "status": "FIXED_DENSE_SW_BRANCH_COMPLETE", "method": method,
        "seed": PILOT_SEED, "common_epoch10_sha256": protocol["common_e10_sha256"],
        "frozen_protocol_sha256": protocol_hash, "pair_rng_seed": PAIR_RNG_SEED,
        "fixed_marginal": 1.0 / 7.0, "policy_frozen_at_epoch": 10,
        "matched_sweep_epochs": list(REFRESH_STATES),
        "loss_reduction": "mean_of_four_equal_subnet_losses",
        "accuracy_used_to_build_policy": False,
        "validation_used_to_build_policy": False,
        "test_used": False,
        "policy_geometry_ids_sha256": subset_audit["policy_geometry_ids_sha256"],
        "bn_calibration_ids_sha256": subset_audit["bn_calibration_ids_sha256"],
        "validation_ids_sha256": subset_audit["validation_ids_sha256"],
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return final
