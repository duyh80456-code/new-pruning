"""Read-only trajectory diagnostic for Geo-HT versus Resource-HT seed 3."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import build_interim_validation_loaders
from rq2_anchor_placement import GRID, _sha256
from rq2_cross_subnet_interaction import _flat_gradient, _state_digest, interaction_parameters
from rq2_gradient_variance_v3 import importance_corrected_variance
from rq2_probabilistic_support import INTERIOR_WIDTHS, pair_marginals
from s1_width import make_model
from training import kd_loss


PATHS = ("geo_ht", "resource_ht")
EPOCHS = (10, 50, 100)
NUM_BATCHES = 8
PROBE_BATCH_SIZE = 128
TARGET_INTERIOR_WEIGHT = 1.0 / 7.0


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


def _is_complete_ht_root(root: Path) -> bool:
    decision = root / "ht_development_decision.json"
    required = [
        root / "resolved_config.yaml",
        root / "protocol" / "frozen_dynamic_marginals.csv",
        root / "protocol" / "geometry_pair_distribution.csv",
        root / "protocol" / "resource_pair_distribution.csv",
    ]
    required += [
        root / path / "seed_3" / f"epoch_{epoch:03d}.pt"
        for path in PATHS for epoch in EPOCHS
    ]
    if not all(path.is_file() for path in required):
        return False
    # The six immutable snapshots plus the frozen protocol are sufficient for
    # this read-only probe. A missing finalizer decision must not hide a usable
    # result when Kaggle stopped after training/evaluation.
    if not decision.is_file():
        return True
    try:
        payload = json.loads(decision.read_text())
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "RQ2_V3_HT_SEED3_DEVELOPMENT_COMPLETE"


def find_ht_development_root(input_root: str | Path, materialized_root: str | Path) -> Path:
    """Resolve one completed HT development result, extracting its ZIP when needed."""
    input_root, materialized_root = Path(input_root), Path(materialized_root)
    candidates = {
        path.parents[2]
        for path in input_root.rglob("epoch_100.pt")
        if len(path.parents) >= 3 and path.parent.name == "seed_3"
        and path.parent.parent.name in PATHS
    }
    candidates.update(path.parent for path in input_root.rglob("ht_development_decision.json"))
    candidates = sorted(path for path in candidates if _is_complete_ht_root(path))
    if not candidates:
        # Kaggle may rename notebook-output archives. Inspect ZIP central
        # directories rather than depending on one exact filename.
        matching_archives = []
        for archive in sorted(input_root.rglob("*.zip")):
            try:
                with zipfile.ZipFile(archive) as bundle:
                    names = {name.rstrip("/") for name in bundle.namelist()}
            except (OSError, zipfile.BadZipFile):
                continue
            suffixes = (
                "geo_ht/seed_3/epoch_010.pt", "geo_ht/seed_3/epoch_050.pt",
                "geo_ht/seed_3/epoch_100.pt", "resource_ht/seed_3/epoch_010.pt",
                "resource_ht/seed_3/epoch_050.pt", "resource_ht/seed_3/epoch_100.pt",
                "protocol/frozen_dynamic_marginals.csv",
            )
            if all(any(name.endswith(suffix) for name in names) for suffix in suffixes):
                matching_archives.append(archive)
        if len(matching_archives) == 1:
            extracted = _safe_extract(matching_archives[0], materialized_root)
            candidates = sorted({
                path.parents[2] for path in extracted.rglob("epoch_100.pt")
                if len(path.parents) >= 3 and path.parent.name == "seed_3"
                and path.parent.parent.name in PATHS and _is_complete_ht_root(path.parents[2])
            })
    if len(candidates) != 1:
        visible_zips = sorted(str(path) for path in input_root.rglob("*.zip"))
        visible_checkpoints = sorted(str(path) for path in input_root.rglob("epoch_100.pt"))
        raise FileNotFoundError(
            "Expected one HT root containing both methods at epochs 10/50/100 plus the frozen "
            f"protocol; candidates={candidates}; visible_zips={visible_zips}; "
            f"visible_epoch100_checkpoints={visible_checkpoints}"
        )
    return candidates[0]


def _policy_inputs(root: Path):
    protocol = root / "protocol"
    marginals = pd.read_csv(protocol / "frozen_dynamic_marginals.csv").sort_values("width")
    if tuple(marginals.width.round(2)) != INTERIOR_WIDTHS:
        raise RuntimeError("Frozen marginals do not cover all 14 interior widths")
    policies = {
        "geo": (
            marginals.pi_geometry.to_numpy(float),
            pd.read_csv(protocol / "geometry_pair_distribution.csv"),
        ),
        "resource": (
            marginals.pi_resource.to_numpy(float),
            pd.read_csv(protocol / "resource_pair_distribution.csv"),
        ),
    }
    for name, (pi, pairs) in policies.items():
        if (
            abs(pi.sum() - 2.0) > 1e-8 or np.any(pi <= 0) or np.any(pi > 1)
            or abs(pairs.probability.sum() - 1.0) > 1e-10
            or np.max(np.abs(pair_marginals(pairs) - pi)) > 1e-6
        ):
            raise RuntimeError(f"Invalid frozen {name} policy")
    return policies


def _fixed_probe_batches(config: dict, dataset_root: Path, device: torch.device):
    config = json.loads(json.dumps(config))
    config["dataset"]["root"] = str(dataset_root)
    config["dataset"]["download"] = False
    loaders = build_interim_validation_loaders(config)
    loader = DataLoader(
        loaders.calibration.dataset,
        batch_size=PROBE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    batches, ids = [], []
    for batch_index, (images, labels, sample_ids) in enumerate(loader):
        if batch_index == NUM_BATCHES:
            break
        batches.append((images, labels))
        ids.extend(torch.as_tensor(sample_ids).tolist())
    if len(batches) != NUM_BATCHES or len(ids) != NUM_BATCHES * PROBE_BATCH_SIZE:
        raise RuntimeError("Could not materialize the fixed eight full training batches")
    return batches, ids


def _batch_gradient_matrix(model, images, labels, parameters, config) -> torch.Tensor:
    model.set_width(1.0)
    teacher = model(images)
    teacher_detached = teacher.detach()
    vectors = {1.0: _flat_gradient(F.cross_entropy(teacher, labels), parameters)}
    for width in GRID[:-1]:
        model.set_width(width)
        logits = model(images)
        loss = F.cross_entropy(logits, labels) + float(config["training"]["kd_lambda"]) * kd_loss(
            logits, teacher_detached, float(config["training"]["kd_temperature"])
        )
        vectors[width] = _flat_gradient(loss, parameters)
    matrix = torch.stack([vectors[width] for width in GRID])
    if not bool(torch.isfinite(matrix).all()):
        raise FloatingPointError("Non-finite trajectory gradient")
    return matrix


def _gram_checks(grams: np.ndarray) -> dict:
    if grams.shape != (NUM_BATCHES, len(INTERIOR_WIDTHS), len(INTERIOR_WIDTHS)):
        raise RuntimeError(f"Unexpected interior Gram shape {grams.shape}")
    scale = max(float(np.max(np.abs(grams))), 1e-30)
    symmetry_error = float(np.max(np.abs(grams - grams.transpose(0, 2, 1)))) / scale
    symmetric = 0.5 * (grams + grams.transpose(0, 2, 1))
    eigenvalues = np.linalg.eigvalsh(symmetric)
    psd_relative_floor = float(eigenvalues.min()) / max(float(eigenvalues.max()), 1e-30)
    if not np.isfinite(grams).all() or symmetry_error > 1e-4:
        raise RuntimeError("Interior Gram is non-finite or materially asymmetric")
    if np.any(np.diagonal(symmetric, axis1=1, axis2=2) < 0) or psd_relative_floor < -1e-4:
        raise RuntimeError("Interior Gram violates PSD beyond FP32 tolerance")
    return {
        "max_relative_symmetry_error": symmetry_error,
        "minimum_relative_eigenvalue": psd_relative_floor,
    }


def probe_trajectory_path(
    ht_root: str | Path,
    output_dir: str | Path,
    path: str,
    dataset_root: str | Path,
    device: str = "cuda:0",
) -> dict:
    """Probe epochs 10/50/100 for one trajectory without changing model state."""
    if path not in PATHS:
        raise ValueError(path)
    ht_root, output_dir, dataset_root = Path(ht_root), Path(output_dir), Path(dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ht_root / "resolved_config.yaml").read_text())
    if (
        tuple(map(float, config["compression"]["eval_widths"])) != GRID
        or not np.isclose(float(config["training"]["kd_lambda"]), 1.0)
        or not np.isclose(float(config["training"]["kd_temperature"]), 2.0)
    ):
        raise RuntimeError("HT trajectory config does not match the locked loss/grid")
    policies = _policy_inputs(ht_root)
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    batches, probe_ids = _fixed_probe_batches(config, dataset_root, torch_device)
    pd.DataFrame({"order": range(len(probe_ids)), "sample_id": probe_ids}).to_csv(
        output_dir / "fixed_training_subset_ids.csv", index=False
    )
    target_weights = np.full(len(INTERIOR_WIDTHS), TARGET_INTERIOR_WEIGHT)
    trajectory_rows, total_rows, drift_rows, distance_rows = [], [], [], []
    checkpoint_hashes, gram_checks = {}, {}
    for epoch in EPOCHS:
        checkpoint = ht_root / path / "seed_3" / f"epoch_{epoch:03d}.pt"
        payload = torch.load(checkpoint, map_location=torch_device, weights_only=False)
        if payload.get("method") != path or int(payload.get("epoch", -1)) != epoch:
            raise RuntimeError(f"Checkpoint identity mismatch: {checkpoint}")
        model = make_model(config, torch_device)
        model.load_state_dict(payload["model"])
        model.eval()
        parameters = interaction_parameters(model)
        parameter_before = _state_digest(model.named_parameters())
        buffers_before = _state_digest(model.named_buffers())
        gram_batches = []
        sum_target = None
        sum_target_norm2 = 0.0
        for batch_id, (cpu_images, cpu_labels) in enumerate(batches):
            images = cpu_images.to(torch_device, non_blocking=True)
            labels = cpu_labels.to(torch_device, non_blocking=True)
            matrix = _batch_gradient_matrix(model, images, labels, parameters, config)
            interior = matrix[1:-1]
            gram = (interior @ interior.T).detach().double().cpu().numpy()
            gram_batches.append(gram)
            target = matrix[0] + matrix[-1] + TARGET_INTERIOR_WEIGHT * interior.sum(dim=0)
            if sum_target is None:
                sum_target = torch.zeros_like(target)
            sum_target.add_(target)
            target_double = target.double()
            sum_target_norm2 += float(torch.dot(target_double, target_double))
            del matrix, interior, target, target_double
            print(f"[quick trajectory] {path} epoch={epoch}: batch {batch_id + 1}/{NUM_BATCHES}", flush=True)
        grams = np.stack(gram_batches).astype(np.float64, copy=False)
        checks = _gram_checks(grams)
        gram_checks[str(epoch)] = checks
        np.save(output_dir / f"interior_grams_epoch_{epoch:03d}.npy", grams)
        batch_variances = {}
        for policy_name, (pi, pairs) in policies.items():
            batch_variances[policy_name] = np.asarray([
                importance_corrected_variance(gram, target_weights, pi, pairs)[
                    "importance_corrected_variance"
                ]
                for gram in grams
            ])
        v_geo = float(batch_variances["geo"].mean())
        v_resource = float(batch_variances["resource"].mean())
        trajectory_rows.append({
            "path": path, "epoch": epoch, "V_geo": v_geo, "V_resource": v_resource,
            "delta_geo_resource": v_geo - v_resource,
            "ratio_geo_resource": v_geo / v_resource,
        })
        mean_target = sum_target / NUM_BATCHES
        v_data = max(0.0, sum_target_norm2 / NUM_BATCHES - float(
            torch.dot(mean_target.double(), mean_target.double())
        ))
        total_geo, total_resource = v_data + v_geo, v_data + v_resource
        total_rows.append({
            "path": path, "epoch": epoch, "V_data": v_data,
            "V_subnet_geo": v_geo, "V_subnet_resource": v_resource,
            "V_total_geo": total_geo, "V_total_resource": total_resource,
            "subnet_fraction_geo": v_geo / total_geo,
            "subnet_fraction_resource": v_resource / total_resource,
            "total_relative_gain_geo": (total_resource - total_geo) / total_resource,
        })
        moments = np.diagonal(grams, axis1=1, axis2=2).mean(axis=0)
        oracle = 2.0 * np.sqrt(moments) / np.sqrt(moments).sum()
        if abs(oracle.sum() - 2.0) > 1e-10 or np.any(oracle <= 0) or np.any(oracle > 1):
            raise RuntimeError("Invalid trajectory gradient oracle marginal")
        for index, width in enumerate(INTERIOR_WIDTHS):
            drift_rows.append({
                "path": path, "epoch": epoch, "width": width,
                "pi_geo": policies["geo"][0][index],
                "pi_resource": policies["resource"][0][index],
                "pi_oracle": oracle[index], "gradient_second_moment": moments[index],
            })
        distance_rows.append({
            "path": path, "epoch": epoch,
            "tv_geo_oracle": 0.5 * float(np.abs(policies["geo"][0] - oracle).sum()),
            "tv_resource_oracle": 0.5 * float(np.abs(policies["resource"][0] - oracle).sum()),
        })
        if parameter_before != _state_digest(model.named_parameters()):
            raise RuntimeError("Trajectory probe changed model weights")
        if buffers_before != _state_digest(model.named_buffers()):
            raise RuntimeError("Trajectory probe changed BN buffers")
        checkpoint_hashes[str(epoch)] = _sha256(checkpoint)
        del model, sum_target, mean_target
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
    pd.DataFrame(trajectory_rows).to_csv(output_dir / "quick_trajectory_variance.csv", index=False)
    pd.DataFrame(total_rows).to_csv(output_dir / "quick_total_variance.csv", index=False)
    pd.DataFrame(drift_rows).to_csv(output_dir / "quick_oracle_drift.csv", index=False)
    pd.DataFrame(distance_rows).to_csv(output_dir / "quick_policy_distance.csv", index=False)
    metadata = {
        "status": "QUICK_TRAJECTORY_PATH_COMPLETE", "path": path,
        "epochs": list(EPOCHS), "num_fixed_training_batches": NUM_BATCHES,
        "batch_size": PROBE_BATCH_SIZE, "probe_batch_ids": probe_ids,
        "checkpoint_sha256": checkpoint_hashes, "gram_checks": gram_checks,
        "training_performed": False, "optimizer_steps": 0, "accuracy_computed": False,
        "test_used": False, "policy_updated": False, "model_eval": True,
        "weights_unchanged": True, "bn_buffers_unchanged": True,
        "gradient_scope": "shared convolution/projection/classifier weights; BN affine/buffers excluded",
        "probe_data": "fixed first 8 batches of the deterministic 45k training split; no augmentation",
        "variance_scale": "unscaled target-family loss; historical divide-by-4 multiplies all variances by 1/16",
        "loss": {"full": "CE", "smaller": "CE+KD", "kd_lambda": 1.0, "temperature": 2.0},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def _plot_outputs(output_dir: Path, trajectory: pd.DataFrame, total: pd.DataFrame,
                  distance: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    for path, frame in trajectory.groupby("path"):
        axes[0].plot(frame.epoch, frame.V_geo, marker="o", label=f"{path}: Geo policy")
        axes[0].plot(frame.epoch, frame.V_resource, marker="s", linestyle="--",
                     label=f"{path}: Resource policy")
        axes[1].plot(frame.epoch, frame.delta_geo_resource, marker="o", label=path)
    axes[0].set(xlabel="Epoch", ylabel="Conditional subnet variance")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(xlabel="Epoch", ylabel="V_G - V_R")
    for axis in axes: axis.grid(alpha=0.25); axis.legend()
    fig.tight_layout(); fig.savefig(output_dir / "quick_trajectory_variance.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for path, frame in total.groupby("path"):
        axes[0].plot(frame.epoch, frame.V_total_geo, marker="o", label=f"{path}: Geo")
        axes[0].plot(frame.epoch, frame.V_total_resource, marker="s", linestyle="--",
                     label=f"{path}: Resource")
        axes[1].plot(frame.epoch, frame.total_relative_gain_geo, marker="o", label=path)
    axes[0].set(xlabel="Epoch", ylabel="Total gradient variance")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(xlabel="Epoch", ylabel="Relative total-variance gain of Geo")
    for axis in axes: axis.grid(alpha=0.25); axis.legend()
    fig.tight_layout(); fig.savefig(output_dir / "quick_total_variance.png", dpi=200); plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.8))
    for path, frame in distance.groupby("path"):
        axis.plot(frame.epoch, frame.tv_geo_oracle, marker="o", label=f"{path}: Geo")
        axis.plot(frame.epoch, frame.tv_resource_oracle, marker="s", linestyle="--",
                  label=f"{path}: Resource")
    axis.set(xlabel="Epoch", ylabel="TV distance to trajectory oracle")
    axis.grid(alpha=0.25); axis.legend(); fig.tight_layout()
    fig.savefig(output_dir / "quick_oracle_drift.png", dpi=200); plt.close(fig)


def merge_trajectory_paths(worker_dirs: list[str | Path], output_dir: str | Path) -> dict:
    worker_dirs, output_dir = [Path(path) for path in worker_dirs], Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = [json.loads((path / "metadata.json").read_text()) for path in worker_dirs]
    if {item["path"] for item in metadata} != set(PATHS):
        raise RuntimeError("Trajectory workers must cover Geo-HT and Resource-HT exactly once")
    if len({tuple(item["probe_batch_ids"]) for item in metadata}) != 1:
        raise RuntimeError("Trajectory workers did not use identical fixed sample IDs/order")
    for item in metadata:
        if (
            item["training_performed"] or item["optimizer_steps"] != 0 or item["test_used"]
            or not item["weights_unchanged"] or not item["bn_buffers_unchanged"]
        ):
            raise RuntimeError("A trajectory worker violated the read-only contract")
    names = (
        "quick_trajectory_variance.csv", "quick_total_variance.csv",
        "quick_oracle_drift.csv", "quick_policy_distance.csv",
    )
    merged = {}
    for name in names:
        frame = pd.concat([pd.read_csv(path / name) for path in worker_dirs], ignore_index=True)
        frame = frame.sort_values(["path", "epoch", *( ["width"] if "width" in frame else [])])
        frame.to_csv(output_dir / name, index=False)
        merged[name] = frame
    _plot_outputs(
        output_dir, merged["quick_trajectory_variance.csv"],
        merged["quick_total_variance.csv"], merged["quick_policy_distance.csv"],
    )
    trajectory = merged["quick_trajectory_variance.csv"]
    total = merged["quick_total_variance.csv"]
    result = {
        "status": "RQ2_V3_QUICK_TRAJECTORY_DIAGNOSTIC_COMPLETE",
        "paths": list(PATHS), "epochs": list(EPOCHS),
        "execution": "two_gpu_one_trajectory_per_gpu",
        "training_performed": False, "optimizer_steps": 0, "accuracy_computed": False,
        "test_used": False, "policy_updated": False,
        "same_fixed_batch_ids_and_order": True,
        "geo_lower_conditional_variance_count": int((trajectory.delta_geo_resource < 0).sum()),
        "comparisons": len(trajectory),
        "mean_total_relative_gain_geo": float(total.total_relative_gain_geo.mean()),
        "interpretation": "mechanism diagnostic only; no policy selection or accuracy claim",
    }
    (output_dir / "metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
