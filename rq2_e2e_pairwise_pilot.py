"""Three-way end-to-end fixed-marginal pair-allocation pilot.

One Uniform-Pair warm-up is shared through epoch 10.  Uniform, raw Resource,
and online Pure-SW branches then train independently from exactly the same
model/optimizer/scheduler/RNG checkpoint.  The only intervention is q_ij.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn import functional as F

from data import (
    build_development_train_loaders,
    build_interim_validation_loaders,
    build_policy_geometry_loaders,
)
from evaluation import calibrate_batch_norm, evaluate_width
from research_utils import seed_everything
from rq2_anchor_placement import GRID, _sha256
from rq2_cross_subnet_interaction import _state_digest, interaction_parameters
from rq2_dynamic_seed3_pilot import _atomic_checkpoint, _capture_rng, _restore_rng
from rq2_fresh_puresw_replication import find_cifar100_root, find_gate_a_summary
from rq2_gradient_variance_v3 import importance_corrected_variance
from rq2_pair_policies import (
    PairPolicy,
    ResourcePairPolicy,
    SWPairPolicy,
    UniformPairPolicy,
    pair_table,
)
from rq2_pairwise_surrogate_regret import (
    INTERIOR_WIDTHS,
    NUM_PAIRS,
    PAIR_INDICES,
    TARGET_WEIGHTS,
    UNIFORM_PI,
    gram_pair_scores,
    solve_pair_lp,
)
from rq2_quick_trajectory_diagnostic import (
    NUM_BATCHES,
    _batch_gradient_matrix,
    _current_geometry,
    _fixed_probe_batches,
    _gram_checks,
)
from s1_width import make_model
from training import kd_loss


PILOT_SEED = 3
METHODS = ("uniform", "resource", "pure_sw")
REFRESH_STATES = (10, 20, 30, 40, 50, 60, 70, 80, 90)
EVAL_EPOCHS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
DIAGNOSTIC_STATES = (("common_warmup", 10),) + tuple(
    (method, epoch) for method in METHODS for epoch in (50, 100)
)
PAIR_RNG_SEED = 310003


def find_seed7_pass_summary(input_root: str | Path) -> Path:
    candidates = []
    for path in Path(input_root).rglob("fresh_seed7_puresw_replication_summary.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("status") == "FRESH_SEED7_PURE_SW_REPLICATION_COMPLETE"
            and payload.get("decision") == "PASS"
            and int(payload.get("fresh_seed", -1)) == 7
        ):
            candidates.append(path)
    by_hash: dict[str, list[Path]] = {}
    for path in candidates:
        by_hash.setdefault(_sha256(path), []).append(path)
    if len(by_hash) != 1:
        raise FileNotFoundError(
            "The 3-way pilot is locked until one content-unique fresh seed-7 Pure-SW summary "
            f"records decision=PASS; found={candidates}"
        )
    return sorted(next(iter(by_hash.values())), key=lambda path: (len(str(path)), str(path)))[0]


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


def materialize_progress(
    input_root: str | Path,
    destination: str | Path,
    extraction_root: str | Path,
) -> Path:
    """Restore one matching pilot tree, or return an empty destination."""
    input_root, destination, extraction_root = map(
        Path, (input_root, destination, extraction_root)
    )
    if (destination / "frozen_protocol.json").is_file():
        return destination
    experiment_name = destination.name
    candidates = sorted({
        path.parent for path in input_root.rglob("frozen_protocol.json")
        if path.parent.name == experiment_name
    })
    if not candidates:
        archives = []
        for archive in input_root.rglob("*.zip"):
            try:
                with zipfile.ZipFile(archive) as bundle:
                    if any(name.endswith(
                        f"{experiment_name}/frozen_protocol.json"
                    ) for name in bundle.namelist()):
                        archives.append(archive)
            except (OSError, zipfile.BadZipFile):
                continue
        if len(archives) == 1:
            extracted = _safe_extract(archives[0], extraction_root)
            candidates = sorted({
                path.parent for path in extracted.rglob("frozen_protocol.json")
                if path.parent.name == experiment_name
            })
        elif len(archives) > 1:
            raise RuntimeError(f"Multiple pilot resume archives attached: {archives}")
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple pilot progress trees attached: {candidates}")
    if candidates:
        shutil.copytree(candidates[0], destination, dirs_exist_ok=True)
    else:
        destination.mkdir(parents=True, exist_ok=True)
    return destination


def load_config(config_path: str | Path, dataset_root: str | Path) -> dict:
    config = copy.deepcopy(yaml.safe_load(Path(config_path).read_text()))
    config["experiment"]["device"] = "cuda"
    config["experiment"]["seeds"] = [PILOT_SEED]
    config["dataset"]["root"] = str(dataset_root)
    config["dataset"]["download"] = False
    config["dataset"]["num_workers"] = 0
    config["compression"]["eval_widths"] = list(GRID)
    config["training"]["epochs"] = 100
    required = {
        "batch_size": 128, "momentum": 0.9, "weight_decay": 5e-4,
        "kd_lambda": 1.0, "kd_temperature": 2.0,
    }
    for key, expected in required.items():
        actual = float(config["training"][key])
        if not np.isclose(actual, expected):
            raise RuntimeError(f"Locked pilot requires training.{key}={expected}, got {actual}")
    if tuple(map(float, config["compression"]["eval_widths"])) != GRID:
        raise RuntimeError("Locked pilot requires the 16-width dense grid")
    if int(config["geometry"]["num_projections"]) != 128:
        raise RuntimeError("Locked pilot requires 128 SW projections")
    return config


def _flops(gate_a_summary: str | Path) -> dict[float, float]:
    payload = json.loads(Path(gate_a_summary).read_text())
    values = {round(float(key), 2): float(value) for key, value in payload["flops"].items()}
    if set(values) != set(INTERIOR_WIDTHS) or any(value <= 0 for value in values.values()):
        raise RuntimeError("Gate A must contain positive FLOPs for all 14 interior widths")
    return values


def _optimizer_scheduler(model, config: dict, learning_rate: float):
    training = config["training"]
    optimizer = torch.optim.SGD(
        model.parameters(), lr=float(learning_rate),
        momentum=float(training["momentum"]), weight_decay=float(training["weight_decay"]),
    )
    return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)


def _ordered_loader_ids(loader) -> list[int]:
    ids = []
    for _, _, batch_ids in loader:
        ids.extend(torch.as_tensor(batch_ids).cpu().tolist())
    return list(map(int, ids))


def _ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


def _audit_policy_subsets(policy_loaders, evaluation_loaders, output: Path) -> dict:
    calibration_ids = _ordered_loader_ids(policy_loaders.calibration)
    policy_ids = _ordered_loader_ids(policy_loaders.geometry)
    validation_ids = _ordered_loader_ids(evaluation_loaders.validation)
    sets = map(set, (calibration_ids, policy_ids, validation_ids))
    calibration_set, policy_set, validation_set = sets
    if calibration_set & policy_set:
        raise RuntimeError("Policy geometry overlaps BN calibration")
    if calibration_set & validation_set or policy_set & validation_set:
        raise RuntimeError("Online-policy inputs overlap validation")
    audit = {
        "status": "POLICY_GEOMETRY_SUBSETS_AUDITED",
        "policy_geometry_source": "training_split_disjoint_from_bn_calibration",
        "bn_calibration_count": len(calibration_ids),
        "policy_geometry_count": len(policy_ids),
        "validation_count": len(validation_ids),
        "bn_calibration_ids_sha256": _ids_sha256(calibration_ids),
        "policy_geometry_ids_sha256": _ids_sha256(policy_ids),
        "validation_ids_sha256": _ids_sha256(validation_ids),
        "bn_calibration_policy_geometry_disjoint": True,
        "bn_calibration_validation_disjoint": True,
        "policy_geometry_validation_disjoint": True,
    }
    path = output / "data_subset_audit.json"
    if path.is_file():
        previous = json.loads(path.read_text())
        if previous != audit:
            raise RuntimeError("Policy subset identity changed across resume")
    else:
        path.write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def _capture_matched_rng(loader, pair_rng: np.random.Generator, device: torch.device) -> dict:
    """Capture one logical CUDA stream so a GPU-0 checkpoint restores on any worker GPU."""
    state = _capture_rng(loader, pair_rng)
    if torch.cuda.is_available():
        state["cuda"] = [torch.cuda.get_rng_state(device)]
    return state


def _restore_matched_rng(state: dict, loader, pair_rng: np.random.Generator) -> None:
    if torch.cuda.is_available() and len(state.get("cuda", [])) != 1:
        raise RuntimeError("Matched pilot checkpoints must contain exactly one logical CUDA RNG")
    _restore_rng(state, loader, pair_rng)


def _policy_from_method(method: str, resource: PairPolicy, current_sw: np.ndarray | None):
    if method == "uniform":
        return UniformPairPolicy()
    if method == "resource":
        return resource
    if method == "pure_sw" and current_sw is not None:
        return SWPairPolicy(current_sw)
    raise RuntimeError(f"Cannot construct policy {method}")


def _sw_matrix(representation_pairs: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((14, 14), dtype=np.float64)
    indices = {width: index for index, width in enumerate(INTERIOR_WIDTHS)}
    if len(representation_pairs) != NUM_PAIRS:
        raise RuntimeError("Geometry sweep did not produce all 91 interior pairs")
    for row in representation_pairs.itertuples(index=False):
        i = indices[round(float(row.width_i), 2)]
        j = indices[round(float(row.width_j), 2)]
        matrix[i, j] = matrix[j, i] = float(row.representation_sw)
    if np.any(matrix < 0) or not np.isfinite(matrix).all():
        raise RuntimeError("Invalid SW matrix")
    return matrix


def _policy_sweep(
    model,
    loaders,
    config: dict,
    device: torch.device,
    method: str,
    resource_policy: PairPolicy,
    state_epoch: int,
    output_dir: Path,
    expected_geometry_ids_sha256: str,
) -> tuple[PairPolicy, float]:
    """Run the identical calibration/representation sweep for every method."""
    started = time.perf_counter()
    _, _, representation_pairs, sample_ids = _current_geometry(
        model, loaders, config, device
    )
    if _ids_sha256(list(map(int, sample_ids))) != expected_geometry_ids_sha256:
        raise RuntimeError("Policy geometry sample IDs/order changed during training")
    sw = _sw_matrix(representation_pairs)
    policy = _policy_from_method(method, resource_policy, sw)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"epoch_{state_epoch:03d}.npz",
        sw_matrix=sw,
        probabilities=policy.probabilities,
        geometry_sample_ids=np.asarray(sample_ids, dtype=np.int64),
        widths=np.asarray(INTERIOR_WIDTHS),
    )
    policy.frame().to_csv(output_dir / f"epoch_{state_epoch:03d}.csv", index=False)
    return policy, time.perf_counter() - started


def _preserving_dense_evaluation(
    model,
    loaders,
    config: dict,
    device: torch.device,
    method: str,
    epoch: int,
) -> list[dict]:
    parameter_before = _state_digest(model.named_parameters())
    buffer_before = _state_digest(model.named_buffers())
    original_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    rows = []
    try:
        for width in GRID:
            calibrate_batch_norm(
                model, loaders.calibration, width, device,
                int(config["evaluation"]["bn_calibration_batches"]),
            )
            result = evaluate_width(model, loaders.validation, width, device)
            rows.append({
                "method": method, "seed": PILOT_SEED, "epoch": epoch,
                "split": "validation_5k", "width": width, **result,
            })
    finally:
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(original_buffers[name])
    if parameter_before != _state_digest(model.named_parameters()):
        raise RuntimeError("Dense evaluation changed model parameters")
    if buffer_before != _state_digest(model.named_buffers()):
        raise RuntimeError("Dense evaluation did not restore BN buffers")
    return rows


def _train_batch(model, images, labels, widths, config: dict):
    model.set_width(1.0)
    teacher_logits = model(images)
    teacher = teacher_logits.detach()
    losses = [F.cross_entropy(teacher_logits, labels)]
    for width in widths:
        model.set_width(float(width))
        logits = model(images)
        losses.append(
            F.cross_entropy(logits, labels)
            + float(config["training"]["kd_lambda"])
            * kd_loss(logits, teacher, float(config["training"]["kd_temperature"]))
        )
    # This is the historical shared-model reduction: four equal subnet terms,
    # with no 1/pi, HT, geometry weight, or pair-specific coefficient.
    return torch.stack(losses).mean()


def train_common_warmup(config: dict, root: str | Path) -> Path:
    root = Path(root)
    output = root / "common_warmup"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "epoch_010.pt"
    if checkpoint.is_file():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("status") == "COMMON_WARMUP_EPOCH10" and int(payload.get("seed", -1)) == PILOT_SEED:
            return checkpoint
    device = torch.device(config["experiment"]["device"])
    seed_everything(PILOT_SEED)
    loaders = build_development_train_loaders(config, training_seed=PILOT_SEED)
    evaluation_loaders = build_interim_validation_loaders(config)
    model = make_model(config, device)
    optimizer, scheduler = _optimizer_scheduler(model, config, 0.1)
    pair_rng = np.random.default_rng(PAIR_RNG_SEED)
    policy = UniformPairPolicy()
    latest = output / "latest.pt"
    records, pair_rows, width_rows = [], [], []
    start_epoch = 1
    if latest.is_file():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        if payload.get("status") != "COMMON_WARMUP_PROGRESS":
            raise RuntimeError("Common warm-up resume identity mismatch")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scheduler.load_state_dict(payload["scheduler"])
        _restore_matched_rng(payload["rng"], loaders.train, pair_rng)
        start_epoch = int(payload["epoch"]) + 1
    metrics_path = output / "metrics.csv"
    if metrics_path.is_file():
        records = pd.read_csv(metrics_path).loc[lambda x: x.epoch < start_epoch].to_dict("records")
    pair_path, width_path = output / "pair_stats.csv", output / "width_marginals.csv"
    if pair_path.is_file():
        pair_rows = pd.read_csv(pair_path).loc[lambda x: x.epoch < start_epoch].to_dict("records")
    if width_path.is_file():
        width_rows = pd.read_csv(width_path).loc[lambda x: x.epoch < start_epoch].to_dict("records")
    for epoch in range(start_epoch, 11):
        model.train(); started = time.perf_counter()
        pair_counts = np.zeros(NUM_PAIRS, dtype=int); width_counts = np.zeros(14, dtype=int)
        losses = []; draws = []; epoch_sample_ids = []
        for images, labels, sample_ids in loaders.train:
            epoch_sample_ids.extend(torch.as_tensor(sample_ids).cpu().tolist())
            images, labels = images.to(device), labels.to(device)
            draw = float(pair_rng.random()); draws.append(draw)
            wi, wj, pair_index = policy.sample(draw)
            optimizer.zero_grad(set_to_none=True)
            loss = _train_batch(model, images, labels, (0.25, wi, wj), config)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite common warm-up loss")
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
            pair_counts[pair_index] += 1
            width_counts[INTERIOR_WIDTHS.index(wi)] += 1
            width_counts[INTERIOR_WIDTHS.index(wj)] += 1
        lr = float(optimizer.param_groups[0]["lr"]); scheduler.step()
        records.append({
            "method": "common_uniform_warmup", "seed": PILOT_SEED, "epoch": epoch,
            "train_loss": float(np.mean(losses)), "learning_rate": lr,
            "wall_clock_seconds": time.perf_counter() - started,
            "pair_uniform_draw_sha256": hashlib.sha256(np.asarray(draws).tobytes()).hexdigest(),
            "train_sample_order_sha256": _ids_sha256(list(map(int, epoch_sample_ids))),
        })
        for index, (i, j) in enumerate(PAIR_INDICES):
            pair_rows.append({
                "epoch": epoch, "pair_index": index, "width_i": INTERIOR_WIDTHS[i],
                "width_j": INTERIOR_WIDTHS[j], "count": int(pair_counts[index]),
            })
        for index, width in enumerate(INTERIOR_WIDTHS):
            width_rows.append({
                "epoch": epoch, "width": width, "count": int(width_counts[index]),
                "realized_marginal": float(width_counts[index] / len(draws)),
                "expected_marginal": 1.0 / 7.0,
            })
        pd.DataFrame(records).to_csv(metrics_path, index=False)
        pd.DataFrame(pair_rows).to_csv(pair_path, index=False)
        pd.DataFrame(width_rows).to_csv(width_path, index=False)
        payload = {
            "status": "COMMON_WARMUP_PROGRESS", "seed": PILOT_SEED, "epoch": epoch,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": _capture_matched_rng(loaders.train, pair_rng, device),
        }
        _atomic_checkpoint(latest, payload)
        print(f"common warm-up epoch {epoch}/10 | loss={np.mean(losses):.4f}", flush=True)
    dense = pd.DataFrame(_preserving_dense_evaluation(
        model, evaluation_loaders, config, device, "common_warmup", 10
    ))
    dense.to_csv(output / "dense_metrics_epoch_010.csv", index=False)
    final = torch.load(latest, map_location="cpu", weights_only=False)
    final["status"] = "COMMON_WARMUP_EPOCH10"
    _atomic_checkpoint(checkpoint, final)
    return checkpoint


def _branch_checkpoint(output: Path, epoch: int) -> Path:
    return output / "checkpoints" / f"epoch_{epoch:03d}.pt"


def train_branch(
    config: dict,
    root: str | Path,
    gate_a_summary: str | Path,
    method: str,
) -> Path:
    if method not in METHODS:
        raise ValueError(method)
    root = Path(root); output = root / method
    output.mkdir(parents=True, exist_ok=True); (output / "checkpoints").mkdir(exist_ok=True)
    final = _branch_checkpoint(output, 100)
    if final.is_file() and (output / "training_provenance.json").is_file():
        return final
    common_path = root / "common_warmup" / "epoch_010.pt"
    if not common_path.is_file():
        raise FileNotFoundError("Common epoch-10 warm-up must complete before branches")
    device = torch.device(config["experiment"]["device"])
    seed_everything(PILOT_SEED)
    loaders = build_development_train_loaders(config, training_seed=PILOT_SEED)
    evaluation_loaders = build_interim_validation_loaders(config)
    policy_loaders = build_policy_geometry_loaders(config)
    subset_audit = _audit_policy_subsets(policy_loaders, evaluation_loaders, output)
    model = make_model(config, device)
    optimizer, scheduler = _optimizer_scheduler(model, config, 0.1)
    pair_rng = np.random.default_rng(PAIR_RNG_SEED)
    resource_policy = ResourcePairPolicy(_flops(gate_a_summary))
    current_policy = UniformPairPolicy() if method == "uniform" else resource_policy
    latest = output / "latest.pt"; start_epoch = 11
    source = torch.load(latest if latest.is_file() else common_path, map_location="cpu", weights_only=False)
    model.load_state_dict(source["model"])
    optimizer.load_state_dict(source["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    scheduler.load_state_dict(source["scheduler"])
    _restore_matched_rng(source["rng"], loaders.train, pair_rng)
    if latest.is_file():
        if source.get("method") != method or int(source.get("seed", -1)) != PILOT_SEED:
            raise RuntimeError("Branch resume identity mismatch")
        start_epoch = int(source["epoch"]) + 1
        current_policy = PairPolicy(method, np.asarray(source["current_q"], float))
        if int(source["epoch"]) >= 51:
            optimizer, scheduler = _optimizer_scheduler(model, config, 0.01)
            optimizer.load_state_dict(source["optimizer"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
            scheduler.load_state_dict(source["scheduler"])

    def prior_records(path: Path):
        return pd.read_csv(path).loc[lambda x: x.epoch < start_epoch].to_dict("records") if path.is_file() else []
    metrics_path = output / "metrics.csv"; pair_path = output / "pair_stats.csv"
    width_path = output / "width_marginals.csv"; dense_path = output / "dense_metrics.csv"
    records, pair_rows, width_rows, dense_rows = map(prior_records, (
        metrics_path, pair_path, width_path, dense_path,
    ))
    if not dense_rows:
        common_dense = pd.read_csv(root / "common_warmup" / "dense_metrics_epoch_010.csv")
        common_dense["method"] = method
        dense_rows = common_dense.to_dict("records")

    for epoch in range(start_epoch, 101):
        state_epoch = epoch - 1
        policy_seconds = 0.0
        if state_epoch in REFRESH_STATES:
            current_policy, policy_seconds = _policy_sweep(
                model, policy_loaders, config, device, method, resource_policy,
                state_epoch, output / "sw_policies",
                subset_audit["policy_geometry_ids_sha256"],
            )
        if epoch == 51:
            optimizer, scheduler = _optimizer_scheduler(model, config, 0.01)
        model.train(); started = time.perf_counter()
        pair_counts = np.zeros(NUM_PAIRS, dtype=int); width_counts = np.zeros(14, dtype=int)
        losses, draws, epoch_sample_ids = [], [], []
        for images, labels, sample_ids in loaders.train:
            epoch_sample_ids.extend(torch.as_tensor(sample_ids).cpu().tolist())
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
        lr = float(optimizer.param_groups[0]["lr"]); scheduler.step()
        records.append({
            "method": method, "seed": PILOT_SEED, "epoch": epoch,
            "train_loss": float(np.mean(losses)), "learning_rate": lr,
            "wall_clock_seconds": time.perf_counter() - started,
            "policy_computation_seconds": policy_seconds,
            "pair_uniform_draw_sha256": hashlib.sha256(np.asarray(draws).tobytes()).hexdigest(),
            "train_sample_order_sha256": _ids_sha256(list(map(int, epoch_sample_ids))),
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
            "status": "E2E_PAIRWISE_BRANCH_PROGRESS", "method": method,
            "seed": PILOT_SEED, "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": _capture_matched_rng(loaders.train, pair_rng, device),
            "current_q": current_policy.probabilities.tolist(),
            "common_epoch10_sha256": _sha256(common_path),
        }
        _atomic_checkpoint(latest, checkpoint)
        if epoch in (50, 100):
            _atomic_checkpoint(_branch_checkpoint(output, epoch), checkpoint)
        print(
            f"{method} epoch {epoch}/100 | loss={np.mean(losses):.4f}, lr={lr:.6g}, "
            f"policy={policy_seconds/60:.1f}m", flush=True,
        )
    provenance = {
        "status": "E2E_PAIRWISE_BRANCH_COMPLETE", "method": method,
        "seed": PILOT_SEED, "common_epoch10_sha256": _sha256(common_path),
        "pair_rng_seed": PAIR_RNG_SEED, "fixed_marginal": 1.0 / 7.0,
        "refresh_states": list(REFRESH_STATES), "equal_calibration_sweeps": True,
        "loss_reduction": "mean_of_four_equal_subnet_losses",
        "importance_or_ht_weighting": False, "learned_hybrid": False,
        "accuracy_used_to_build_policy": False, "test_used": False,
        "validation_used_to_build_policy": False,
        "test_used_to_build_policy": False,
        "policy_geometry_source": subset_audit["policy_geometry_source"],
        "policy_geometry_size": subset_audit["policy_geometry_count"],
        "policy_geometry_ids_sha256": subset_audit["policy_geometry_ids_sha256"],
        "bn_calibration_ids_sha256": subset_audit["bn_calibration_ids_sha256"],
        "validation_ids_sha256": subset_audit["validation_ids_sha256"],
    }
    (output / "training_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return final


def _diagnostic_checkpoint(root: Path, method: str, epoch: int) -> Path:
    if method == "common_warmup" and epoch == 10:
        return root / "common_warmup" / "epoch_010.pt"
    return _branch_checkpoint(root / method, epoch)


def extract_diagnostic_state(
    root: str | Path,
    method: str,
    epoch: int,
    dataset_root: str | Path,
    gate_a_summary: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
) -> dict:
    root, output_dir = Path(root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((root / "resolved_config.yaml").read_text())
    config["dataset"]["root"] = str(dataset_root); config["dataset"]["download"] = False
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    checkpoint = _diagnostic_checkpoint(root, method, int(epoch))
    payload = torch.load(checkpoint, map_location=torch_device, weights_only=False)
    model = make_model(config, torch_device); model.load_state_dict(payload["model"]); model.eval()
    parameter_before = _state_digest(model.named_parameters())
    buffer_before = _state_digest(model.named_buffers())
    batches, probe_ids = _fixed_probe_batches(config, Path(dataset_root), torch_device)
    parameters = interaction_parameters(model)
    grams = []
    for batch_index, (images, labels) in enumerate(batches):
        matrix = _batch_gradient_matrix(
            model, images.to(torch_device), labels.to(torch_device), parameters, config
        )[1:-1]
        grams.append((matrix @ matrix.T).detach().double().cpu().numpy())
        print(f"[e2e diagnostic] {method}-{epoch}: batch {batch_index+1}/{NUM_BATCHES}", flush=True)
    grams = np.stack(grams); checks = _gram_checks(grams); mean_gram = grams.mean(axis=0)
    policy_loaders = build_policy_geometry_loaders(config)
    _, _, representation_pairs, geometry_ids = _current_geometry(
        model, policy_loaders, config, torch_device
    )
    sw = _sw_matrix(representation_pairs)
    resource = ResourcePairPolicy(_flops(gate_a_summary))
    policies = {
        "uniform": UniformPairPolicy().probabilities,
        "resource": resource.probabilities,
        "pure_sw": SWPairPolicy(sw).probabilities,
        "oracle": solve_pair_lp(gram_pair_scores(mean_gram)[0], maximize=True),
    }
    variances = {
        name: float(importance_corrected_variance(
            mean_gram, TARGET_WEIGHTS, UNIFORM_PI, pair_table(q, name)
        )["importance_corrected_variance"])
        for name, q in policies.items()
    }
    np.save(output_dir / "gradient_grams.npy", grams)
    np.save(output_dir / "sw_matrix.npy", sw)
    pd.concat([pair_table(q, name) for name, q in policies.items()]).to_csv(
        output_dir / "pair_policies.csv", index=False
    )
    row = {
        "state": f"{method}_E{epoch}", "method": method, "epoch": int(epoch),
        "V_uniform": variances["uniform"], "V_resource": variances["resource"],
        "V_SW": variances["pure_sw"], "V_oracle": variances["oracle"],
        "SW_beats_uniform": bool(variances["pure_sw"] < variances["uniform"]),
        "SW_beats_resource": bool(variances["pure_sw"] < variances["resource"]),
    }
    pd.DataFrame([row]).to_csv(output_dir / "variance.csv", index=False)
    if parameter_before != _state_digest(model.named_parameters()) or buffer_before != _state_digest(model.named_buffers()):
        raise RuntimeError("Diagnostic changed model state")
    metadata = {
        "status": "E2E_PAIRWISE_DIAGNOSTIC_COMPLETE", "method": method,
        "epoch": int(epoch), "probe_ids": probe_ids, "geometry_ids": geometry_ids,
        "gram_checks": checks, "optimizer_steps": 0,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def finalize(root: str | Path) -> dict:
    root = Path(root)
    training_metrics = pd.concat([
        pd.read_csv(root / method / "metrics.csv") for method in METHODS
    ], ignore_index=True)
    draw_audit = training_metrics.pivot(
        index="epoch", columns="method", values="pair_uniform_draw_sha256"
    )
    if set(draw_audit.columns) != set(METHODS) or not draw_audit.nunique(axis=1).eq(1).all():
        raise RuntimeError("The three branches did not consume the same pair-uniform draw stream")
    data_audit = training_metrics.pivot(
        index="epoch", columns="method", values="train_sample_order_sha256"
    )
    if set(data_audit.columns) != set(METHODS) or not data_audit.nunique(axis=1).eq(1).all():
        raise RuntimeError("Training sample order differed across methods")
    training_metrics.to_csv(root / "training_metrics_all_methods.csv", index=False)
    dense = pd.concat([pd.read_csv(root / method / "dense_metrics.csv") for method in METHODS])
    dense.to_csv(root / "dense_metrics_all_methods.csv", index=False)
    final = dense.loc[dense.epoch.eq(100)].copy()
    rows = []
    for method, group in final.groupby("method"):
        rows.append({
            "method": method,
            "dense_mean_accuracy": float(group.accuracy.mean()),
            "interior_mean_accuracy": float(
                group.loc[group.width.between(0.30, 0.95), "accuracy"].mean()
            ),
            "worst_accuracy": float(group.accuracy.min()),
            "low_mean_accuracy": float(group.loc[group.width.between(0.30, 0.45), "accuracy"].mean()),
            "mid_mean_accuracy": float(group.loc[group.width.between(0.50, 0.75), "accuracy"].mean()),
            "high_mean_accuracy": float(group.loc[group.width.between(0.80, 0.95), "accuracy"].mean()),
            "full_width_accuracy": float(group.loc[np.isclose(group.width, 1.0), "accuracy"].iloc[0]),
        })
    summary_table = pd.DataFrame(rows).sort_values("method")
    summary_table.to_csv(root / "method_summary.csv", index=False)
    indexed = summary_table.set_index("method")
    comparisons = {}
    for baseline in ("uniform", "resource"):
        comparisons[f"pure_sw_minus_{baseline}"] = {
            column: float(indexed.loc["pure_sw", column] - indexed.loc[baseline, column])
            for column in summary_table.columns if column != "method"
        }
    diagnostics = pd.concat([
        pd.read_csv(root / "diagnostics" / f"{method}_E{epoch}" / "variance.csv")
        for method, epoch in DIAGNOSTIC_STATES
    ], ignore_index=True)
    diagnostics.to_csv(root / "trajectory_variance_diagnostics.csv", index=False)
    primary = comparisons["pure_sw_minus_uniform"]["dense_mean_accuracy"] > 0
    decision = {
        "status": "E2E_PAIRWISE_PILOT_COMPLETE", "seed": PILOT_SEED,
        "primary_dense_accuracy_pass": bool(primary),
        "secondary_worst_not_lower": bool(
            comparisons["pure_sw_minus_uniform"]["worst_accuracy"] >= 0
        ),
        "full_width_delta_vs_uniform": comparisons["pure_sw_minus_uniform"]["full_width_accuracy"],
        "comparisons": comparisons, "test_used": False,
        "matched_pair_uniform_draw_stream_verified": True,
        "matched_training_sample_order_verified": True,
        "single_development_seed_only": True,
        "confirmatory_claim_authorized": False,
    }
    (root / "summary.json").write_text(json.dumps(decision, indent=2) + "\n")
    return decision


__all__ = [
    "PILOT_SEED", "METHODS", "DIAGNOSTIC_STATES", "load_config",
    "find_cifar100_root", "find_gate_a_summary", "train_common_warmup",
    "find_seed7_pass_summary", "materialize_progress", "train_branch",
    "extract_diagnostic_state", "finalize",
]
