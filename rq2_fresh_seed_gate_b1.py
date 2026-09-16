"""Fresh-seed trajectory training and frozen-predictor Gate B1 evaluation."""

from __future__ import annotations

import copy
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn import functional as F

from data import build_development_train_loaders, build_interim_validation_loaders
from profiling import profile_subnet
from research_utils import seed_everything
from rq2_anchor_placement import GRID, UNIFORM_ANCHORS, _sha256
from rq2_cross_subnet_interaction import _state_digest, interaction_parameters
from rq2_dynamic_seed3_pilot import (
    _atomic_checkpoint,
    _capture_rng,
    _restore_rng,
)
from rq2_pairwise_external_gates import run_gate_b1_fresh
from rq2_probabilistic_support import INTERIOR_WIDTHS
from rq2_quick_trajectory_diagnostic import (
    NUM_BATCHES,
    _batch_gradient_matrix,
    _current_geometry,
    _fixed_probe_batches,
    _gram_checks,
)
from s1_width import make_model
from training import kd_loss


FRESH_SEED = 6
CHECKPOINT_EPOCHS = (10, 50, 100)
METHOD = "fresh_uniform_fixed"


def load_fresh_config(config_path: str | Path, dataset_root: str | Path) -> dict:
    config = yaml.safe_load(Path(config_path).read_text())
    config = copy.deepcopy(config)
    config["experiment"]["device"] = "cuda"
    config["experiment"]["seeds"] = [FRESH_SEED]
    config["dataset"]["root"] = str(dataset_root)
    config["dataset"]["download"] = True
    config["dataset"]["num_workers"] = 0
    config["compression"]["train_widths"] = list(UNIFORM_ANCHORS)
    config["compression"]["eval_widths"] = list(GRID)
    config["training"]["epochs"] = 100
    if (
        tuple(map(float, config["compression"]["train_widths"])) != UNIFORM_ANCHORS
        or tuple(map(float, config["compression"]["eval_widths"])) != GRID
        or not np.isclose(float(config["training"]["kd_lambda"]), 1.0)
        or not np.isclose(float(config["training"]["kd_temperature"]), 2.0)
        or int(config["training"]["batch_size"]) != 128
    ):
        raise RuntimeError("Fresh Gate B1 config violates the locked baseline protocol")
    return config


def _optimizer_scheduler(model, config: dict, learning_rate: float):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=float(learning_rate),
        momentum=float(config["training"]["momentum"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    return optimizer, scheduler


def _snapshot(path: Path, model, epoch: int, config: dict) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save({
        "model": model.state_dict(), "epoch": int(epoch), "seed": FRESH_SEED,
        "method": METHOD, "anchors": list(UNIFORM_ANCHORS),
        "schedule": "50+50_optimizer_scheduler_reset",
        "kd_lambda": float(config["training"]["kd_lambda"]),
        "kd_temperature": float(config["training"]["kd_temperature"]),
    }, temporary)
    temporary.replace(path)


def _train_phase(
    config: dict,
    root: Path,
    first_epoch: int,
    last_epoch: int,
    learning_rate: float,
    initial_checkpoint: Path | None,
) -> Path:
    device = torch.device(config["experiment"]["device"])
    seed_everything(FRESH_SEED)
    loaders = build_development_train_loaders(config, training_seed=FRESH_SEED)
    model = make_model(config, device)
    optimizer, scheduler = _optimizer_scheduler(model, config, learning_rate)
    phase = 1 if first_epoch == 1 else 2
    latest = root / f"phase_{phase}_latest.pt"
    dummy_rng = np.random.default_rng(FRESH_SEED + phase)
    start_epoch = first_epoch
    if latest.is_file():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        if payload.get("method") != METHOD or int(payload.get("seed", -1)) != FRESH_SEED:
            raise RuntimeError("Fresh trajectory resume identity mismatch")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scheduler.load_state_dict(payload["scheduler"])
        _restore_rng(payload["rng"], loaders.train, dummy_rng)
        start_epoch = int(payload["epoch"]) + 1
    elif initial_checkpoint is not None:
        payload = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
        if int(payload.get("epoch", -1)) != first_epoch - 1:
            raise RuntimeError("Phase-2 initialization is not the epoch-50 phase-1 state")
        model.load_state_dict(payload["model"])
        _restore_rng(payload["rng"], loaders.train, dummy_rng)
    metrics_path = root / "training_metrics.csv"
    records = (
        pd.read_csv(metrics_path).loc[lambda frame: frame.epoch < start_epoch].to_dict("records")
        if metrics_path.is_file() else []
    )
    anchors = tuple(map(float, config["compression"]["train_widths"]))
    for epoch in range(start_epoch, last_epoch + 1):
        model.train()
        sums = {width: {"loss": 0.0, "correct": 0, "n": 0} for width in anchors}
        for images, labels, _ in loaders.train:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            model.set_width(1.0)
            teacher = model(images)
            losses = {1.0: F.cross_entropy(teacher, labels)}
            logits_by_width = {1.0: teacher}
            total = losses[1.0]
            for width in anchors[:-1]:
                model.set_width(width)
                logits = model(images)
                loss = F.cross_entropy(logits, labels) + float(
                    config["training"]["kd_lambda"]
                ) * kd_loss(logits, teacher, float(config["training"]["kd_temperature"]))
                losses[width] = loss; logits_by_width[width] = logits; total = total + loss
            total = total / len(anchors)
            if not bool(torch.isfinite(total)):
                raise FloatingPointError(f"Non-finite fresh trajectory loss at epoch {epoch}")
            total.backward(); optimizer.step()
            n = labels.numel()
            for width in anchors:
                sums[width]["loss"] += float(losses[width].detach()) * n
                sums[width]["correct"] += int((logits_by_width[width].argmax(1) == labels).sum())
                sums[width]["n"] += n
        lr = float(optimizer.param_groups[0]["lr"])
        records = [row for row in records if not (
            int(row["epoch"]) == epoch and float(row["width"]) in anchors
        )]
        for width in anchors:
            item = sums[width]
            records.append({
                "epoch": epoch, "width": width, "loss": item["loss"] / item["n"],
                "train_accuracy_monitor_only": item["correct"] / item["n"],
                "learning_rate": lr, "phase": phase,
            })
        pd.DataFrame(records).sort_values(["epoch", "width"]).to_csv(metrics_path, index=False)
        scheduler.step()
        payload = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "seed": FRESH_SEED,
            "method": METHOD, "phase": phase,
            "rng": _capture_rng(loaders.train, dummy_rng),
        }
        _atomic_checkpoint(latest, payload)
        if epoch in CHECKPOINT_EPOCHS:
            _snapshot(root / "checkpoints" / f"epoch_{epoch:03d}.pt", model, epoch, config)
        status = ", ".join(
            f"w={width:.2f}: acc={sums[width]['correct']/sums[width]['n']:.4f}"
            for width in anchors
        )
        print(f"fresh seed {FRESH_SEED} epoch {epoch}/100 | {status}", flush=True)
    return latest


def train_fresh_trajectory(config: dict, root: str | Path) -> Path:
    root = Path(root)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    resolved = root / "resolved_config.yaml"
    if not resolved.is_file():
        resolved.write_text(yaml.safe_dump(config, sort_keys=False))
    phase_one = _train_phase(config, root, 1, 50, 0.1, None)
    phase_two = _train_phase(config, root, 51, 100, 0.01, phase_one)
    checkpoints = [root / "checkpoints" / f"epoch_{epoch:03d}.pt" for epoch in CHECKPOINT_EPOCHS]
    if not all(path.is_file() for path in checkpoints):
        raise RuntimeError("Fresh trajectory did not produce all locked checkpoints")
    provenance = {
        "status": "FRESH_TRAJECTORY_TRAINING_COMPLETE",
        "seed": FRESH_SEED, "method": METHOD, "epochs": 100,
        "anchors": list(UNIFORM_ANCHORS),
        "schedule": "50+50_optimizer_scheduler_reset",
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "accuracy_used_for_selection_or_tuning": False,
        "policy_learning_performed": False, "test_used": False,
        "checkpoint_sha256": {str(epoch): _sha256(path) for epoch, path in zip(CHECKPOINT_EPOCHS, checkpoints)},
    }
    (root / "training_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return phase_two


def extract_fresh_state(
    root: str | Path,
    epoch: int,
    dataset_root: str | Path,
    gate_a_summary: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
) -> dict:
    root, output_dir = Path(root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch = int(epoch)
    if epoch not in CHECKPOINT_EPOCHS:
        raise ValueError(epoch)
    config = yaml.safe_load((root / "resolved_config.yaml").read_text())
    config["dataset"]["root"] = str(dataset_root); config["dataset"]["download"] = False
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    checkpoint = root / "checkpoints" / f"epoch_{epoch:03d}.pt"
    payload = torch.load(checkpoint, map_location=torch_device, weights_only=False)
    if payload.get("method") != METHOD or int(payload.get("seed", -1)) != FRESH_SEED:
        raise RuntimeError("Fresh checkpoint identity mismatch")
    model = make_model(config, torch_device); model.load_state_dict(payload["model"]); model.eval()
    parameter_before = _state_digest(model.named_parameters())
    buffer_before = _state_digest(model.named_buffers())
    batches, probe_ids = _fixed_probe_batches(config, Path(dataset_root), torch_device)
    parameters = interaction_parameters(model)
    grams = []
    for batch_index, (cpu_images, cpu_labels) in enumerate(batches):
        matrix = _batch_gradient_matrix(
            model, cpu_images.to(torch_device), cpu_labels.to(torch_device), parameters, config
        )
        interior = matrix[1:-1]
        grams.append((interior @ interior.T).detach().double().cpu().numpy())
        print(f"[fresh Gate B1] epoch={epoch}: gradient batch {batch_index+1}/{NUM_BATCHES}", flush=True)
    grams = np.stack(grams)
    checks = _gram_checks(grams)
    np.save(output_dir / f"gradient_gram_epoch_{epoch:03d}.npy", grams)
    loaders = build_interim_validation_loaders(config)
    _, _, representation_pairs, geometry_ids = _current_geometry(model, loaders, config, torch_device)
    width_to_index = {width: index for index, width in enumerate(INTERIOR_WIDTHS)}
    sw_matrix = np.zeros((len(INTERIOR_WIDTHS), len(INTERIOR_WIDTHS)), dtype=np.float64)
    for row in representation_pairs.itertuples(index=False):
        i, j = width_to_index[round(float(row.width_i), 2)], width_to_index[round(float(row.width_j), 2)]
        sw_matrix[i, j] = sw_matrix[j, i] = float(row.representation_sw)
    np.save(output_dir / f"sw_matrix_epoch_{epoch:03d}.npy", sw_matrix)
    profiled = {width: float(profile_subnet(model, width)[0]) for width in INTERIOR_WIDTHS}
    audited = {
        round(float(key), 2): float(value)
        for key, value in json.loads(Path(gate_a_summary).read_text())["flops"].items()
    }
    relative_error = max(
        abs(profiled[width] - audited[width]) / audited[width] for width in INTERIOR_WIDTHS
    )
    if relative_error > 1e-12:
        raise RuntimeError(f"Fresh model FLOPs disagree with Gate A audit: {relative_error}")
    mean_gram = grams.mean(axis=0)
    rows = []
    for row in representation_pairs.itertuples(index=False):
        wi, wj = round(float(row.width_i), 2), round(float(row.width_j), 2)
        i, j = width_to_index[wi], width_to_index[wj]
        distance2 = max(0.0, mean_gram[i, i] + mean_gram[j, j] - 2 * mean_gram[i, j])
        rows.append({
            "state": f"F{epoch}", "seed": FRESH_SEED, "epoch": epoch,
            "width_i": wi, "width_j": wj,
            "width_distance": abs(wi - wj),
            "log_flops_distance": abs(np.log(profiled[wi]) - np.log(profiled[wj])),
            "representation_sw": float(row.representation_sw),
            "sw2": float(row.representation_sw) ** 2,
            "gradient_euclidean_distance": np.sqrt(distance2),
            "gradient_distance2": distance2,
            "flops_i": profiled[wi], "flops_j": profiled[wj],
        })
    pd.DataFrame(rows).to_csv(output_dir / f"pair_structure_epoch_{epoch:03d}.csv", index=False)
    if parameter_before != _state_digest(model.named_parameters()):
        raise RuntimeError("Fresh state extraction changed model parameters")
    if buffer_before != _state_digest(model.named_buffers()):
        raise RuntimeError("Fresh state extraction did not restore BN buffers")
    metadata = {
        "status": "FRESH_PAIRWISE_STATE_COMPLETE", "seed": FRESH_SEED, "epoch": epoch,
        "checkpoint_sha256": _sha256(checkpoint), "gram_checks": checks,
        "gradient_probe_ids": probe_ids, "geometry_probe_ids": geometry_ids,
        "gradient_scope": "shared convolution/projection/classifier weights; BN excluded",
        "predictor_used_during_extraction": False, "optimizer_steps": 0,
        "weights_unchanged": True, "bn_buffers_unchanged": True,
        "test_used": False, "accuracy_used": False,
        "profiled_flops_match_gate_a": True,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def merge_and_evaluate_fresh_states(
    root: str | Path,
    worker_dirs: list[str | Path],
    frozen_predictor: str | Path,
) -> dict:
    root, workers = Path(root), [Path(path) for path in worker_dirs]
    metadata = [json.loads((path / "metadata.json").read_text()) for path in workers]
    if {int(item["epoch"]) for item in metadata} != set(CHECKPOINT_EPOCHS):
        raise RuntimeError("Fresh workers do not cover epochs 10/50/100 exactly")
    if len({tuple(item["gradient_probe_ids"]) for item in metadata}) != 1:
        raise RuntimeError("Fresh states used different gradient batches/order")
    if len({tuple(item["geometry_probe_ids"]) for item in metadata}) != 1:
        raise RuntimeError("Fresh states used different geometry samples/order")
    pair_frames = []
    (root / "sw_matrices").mkdir(exist_ok=True)
    (root / "gradient_grams").mkdir(exist_ok=True)
    (root / "grams").mkdir(exist_ok=True)
    for worker, item in zip(workers, metadata):
        epoch = int(item["epoch"]); state = f"seed_{FRESH_SEED}_epoch_{epoch:03d}"
        pair_frames.append(pd.read_csv(worker / f"pair_structure_epoch_{epoch:03d}.csv"))
        shutil.copy2(worker / f"sw_matrix_epoch_{epoch:03d}.npy", root / "sw_matrices" / f"epoch_{epoch:03d}.npy")
        source_gram = worker / f"gradient_gram_epoch_{epoch:03d}.npy"
        shutil.copy2(source_gram, root / "gradient_grams" / f"epoch_{epoch:03d}.npy")
        shutil.copy2(source_gram, root / "grams" / f"{state}.npy")
    pairs = pd.concat(pair_frames, ignore_index=True).sort_values(["epoch", "width_i", "width_j"])
    pairs.to_csv(root / "pair_structure.csv", index=False)
    pairs.to_csv(root / "fresh_pair_structure.csv", index=False)
    fresh_metadata = {
        "status": "FRESH_PAIRWISE_STATES_COMPLETE", "seeds": [FRESH_SEED],
        "epochs": list(CHECKPOINT_EPOCHS), "states": [f"F{x}" for x in CHECKPOINT_EPOCHS],
        "trajectory": METHOD, "predictor_used_during_extraction": False,
        "gradient_used_for_predictor_fit": False, "test_used": False,
    }
    (root / "state_metadata.json").write_text(json.dumps(fresh_metadata, indent=2) + "\n")
    (root / "fresh_state_metadata.json").write_text(json.dumps(fresh_metadata, indent=2) + "\n")
    evaluation = root / "evaluation"
    summary = run_gate_b1_fresh(root, frozen_predictor, evaluation)
    results = pd.read_csv(evaluation / "gate_b1_fresh_exact_variance.csv")
    metrics = pd.read_csv(evaluation / "gate_b1_fresh_metrics.csv")
    metric_wide = metrics.pivot(index="state", columns="model", values=["mae", "spearman_rho"])
    metric_wide.columns = [
        f"{metric}_{'R' if model == 'Resource' else 'R_plus_SW'}"
        for metric, model in metric_wide.columns
    ]
    results = results.merge(metric_wide.reset_index(), on="state", how="left", validate="one_to_one")
    results["State"] = results.epoch.map(lambda value: f"F{int(value)}")
    results = results[[
        "State", "seed", "epoch", "V_resource", "V_resource_plus_sw", "V_oracle",
        "delta_V_hybrid_minus_resource", "relative_delta_V",
        "mae_R", "mae_R_plus_SW", "spearman_rho_R", "spearman_rho_R_plus_SW",
        "resource_oracle_gap_captured", "hybrid_variance_win",
    ]]
    results.to_csv(evaluation / "resource_vs_hybrid.csv", index=False)
    wins = int(results.hybrid_variance_win.sum())
    mean_delta = float(results.delta_V_hybrid_minus_resource.mean())
    early_mid = results.loc[results.epoch.isin([10, 50]), "hybrid_variance_win"].all()
    screening = (
        "NO_GO" if mean_delta > 0 or wins == 0
        else "STRONG_POSITIVE" if wins == 3
        else "PROMISING_EARLY_MID" if wins == 2 and early_mid
        else "PROMISING" if wins == 2
        else "WEAK"
    )
    summary.update({
        "screening_decision": screening,
        "early_mid_both_win": bool(early_mid),
        "one_seed_screening_only": True,
        "second_fresh_seed_required_before_gate_c": screening in {"STRONG_POSITIVE", "PROMISING_EARLY_MID", "PROMISING"},
        "gate_c_authorized": False,
    })
    (evaluation / "gate_b1_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


__all__ = [
    "FRESH_SEED", "CHECKPOINT_EPOCHS", "load_fresh_config", "train_fresh_trajectory",
    "extract_fresh_state", "merge_and_evaluate_fresh_states",
]
