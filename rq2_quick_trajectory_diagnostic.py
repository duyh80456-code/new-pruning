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
from scipy.stats import pearsonr, spearmanr
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import build_interim_validation_loaders
from evaluation import calibrate_batch_norm
from rq2_anchor_placement import GRID, _sha256
from rq2_cross_subnet_interaction import _flat_gradient, _state_digest, interaction_parameters
from rq2_gradient_variance_v3 import importance_corrected_variance
from rq2_probabilistic_support import INTERIOR_WIDTHS, maximum_entropy_pairs, pair_marginals
from s1_width import make_model
from training import kd_loss


PATHS = ("geo_ht", "resource_ht")
EPOCHS = (10, 50, 100)
NUM_BATCHES = 8
PROBE_BATCH_SIZE = 128
TARGET_INTERIOR_WEIGHT = 1.0 / 7.0
PROTOCOL_VERSION = 2


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


def _sliced_projection_bank(feature_bank: dict, num_projections: int, seed: int):
    first = np.asarray(feature_bank[GRID[0]], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    directions = rng.normal(size=(int(num_projections), first.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-12)
    projected = {}
    for width in GRID:
        features = np.asarray(feature_bank[width], dtype=np.float64)
        if features.shape != first.shape or not np.isfinite(features).all():
            raise RuntimeError("Trajectory representation banks are not aligned and finite")
        projected[width] = np.sort(features @ directions.T, axis=0)
    return projected


def _sw_from_projected(projected_a: np.ndarray, projected_b: np.ndarray) -> float:
    # Equal-size/equal-weight 1-D Wasserstein is the mean absolute difference
    # between sorted samples; average over the frozen sliced directions.
    return float(np.mean(np.abs(projected_a - projected_b)))


def _current_geometry(model, loaders, config: dict, device: torch.device):
    """Recompute learned-projection SW geometry using the registered protocol."""
    parameter_before = _state_digest(model.named_parameters())
    buffer_before = _state_digest(model.named_buffers())
    original_buffers = {
        name: tensor.detach().clone() for name, tensor in model.named_buffers()
    }
    feature_bank, reference_ids = {}, None
    try:
        for width in GRID:
            calibrate_batch_norm(
                model, loaders.calibration, width, device,
                int(config["evaluation"]["bn_calibration_batches"]),
            )
            model.set_width(width); model.eval()
            features, sample_ids = [], []
            with torch.no_grad():
                for images, _, ids in loaders.geometry:
                    z = F.normalize(model.forward_features(images.to(device)), p=2, dim=1)
                    features.append(z.cpu())
                    sample_ids.append(torch.as_tensor(ids).cpu())
            features = torch.cat(features)
            ids = torch.cat(sample_ids)
            if reference_ids is None:
                reference_ids = ids
            elif not torch.equal(reference_ids, ids):
                raise RuntimeError("Geometry sample IDs/order changed across widths")
            if not bool(torch.isfinite(features).all()) or float(features.std()) <= 1e-8:
                raise RuntimeError(f"Invalid learned-projection features at width {width}")
            feature_bank[width] = features.numpy()
    finally:
        with torch.no_grad():
            for name, tensor in model.named_buffers():
                tensor.copy_(original_buffers[name])
    if parameter_before != _state_digest(model.named_parameters()):
        raise RuntimeError("Geometry extraction changed model parameters")
    if buffer_before != _state_digest(model.named_buffers()):
        raise RuntimeError("Geometry extraction did not restore BN buffers")

    projected = _sliced_projection_bank(
        feature_bank, int(config["geometry"]["num_projections"]),
        int(config["geometry"]["projection_seed"]),
    )
    edge_rows, edges = [], []
    for left, right in zip(GRID[:-1], GRID[1:]):
        value = _sw_from_projected(projected[left], projected[right])
        edges.append(value)
        edge_rows.append({
            "budget_start": left, "budget_end": right,
            "wasserstein_jump": value, "G": value / (right - left),
        })
    pair_rows = []
    for i, left in enumerate(INTERIOR_WIDTHS):
        for right in INTERIOR_WIDTHS[i + 1:]:
            pair_rows.append({
                "width_i": left, "width_j": right,
                "representation_sw": _sw_from_projected(projected[left], projected[right]),
            })
    cells = pd.DataFrame({
        "width": INTERIOR_WIDTHS,
        "current_functional_mass": [0.5 * (edges[i] + edges[i + 1]) for i in range(14)],
    })
    cells["current_functional_mass_normalized"] = (
        cells.current_functional_mass / cells.current_functional_mass.sum()
    )
    return cells, pd.DataFrame(edge_rows), pd.DataFrame(pair_rows), reference_ids.tolist()


def _correlation_rows(path: str, epoch: int, cells: pd.DataFrame, pairs: pd.DataFrame):
    rows = []
    comparisons = (
        ("current_a_vs_m", cells.current_functional_mass, cells.gradient_second_moment),
        ("frozen_a_vs_m", cells.frozen_functional_mass_normalized, cells.gradient_second_moment),
        ("current_a_vs_oracle_pi", cells.current_functional_mass, cells.pi_oracle),
    )
    for name, x, y in comparisons:
        for metric, function in (("Pearson", pearsonr), ("Spearman", spearmanr)):
            value = function(np.asarray(x, float), np.asarray(y, float))
            rows.append({
                "path": path, "epoch": epoch, "analysis": name, "metric": metric,
                "coefficient": float(value.statistic), "pvalue": float(value.pvalue),
                "n": len(x),
            })
    pair_rows = []
    for target in ("gradient_euclidean_distance", "gradient_cosine_dissimilarity"):
        for metric, function in (("Pearson", pearsonr), ("Spearman", spearmanr)):
            value = function(pairs.representation_sw, pairs[target])
            pair_rows.append({
                "path": path, "epoch": epoch,
                "analysis": f"representation_SW_vs_{target}", "metric": metric,
                "coefficient": float(value.statistic), "pvalue": float(value.pvalue),
                "n_pairs": len(pairs),
            })
    return rows, pair_rows


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
    geometry_config = json.loads(json.dumps(config))
    geometry_config["dataset"]["root"] = str(dataset_root)
    geometry_config["dataset"]["download"] = False
    geometry_loaders = build_interim_validation_loaders(geometry_config)
    pd.DataFrame({"order": range(len(probe_ids)), "sample_id": probe_ids}).to_csv(
        output_dir / "fixed_training_subset_ids.csv", index=False
    )
    target_weights = np.full(len(INTERIOR_WIDTHS), TARGET_INTERIOR_WEIGHT)
    trajectory_rows, total_rows, drift_rows, distance_rows = [], [], [], []
    geometry_rows, edge_rows, pair_rows = [], [], []
    correlation_rows, pair_correlation_rows = [], []
    geometry_probe_ids = None
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

        cells, current_edges, representation_pairs, current_geometry_ids = _current_geometry(
            model, geometry_loaders, geometry_config, torch_device
        )
        if geometry_probe_ids is None:
            geometry_probe_ids = current_geometry_ids
            pd.DataFrame({
                "order": range(len(geometry_probe_ids)), "sample_id": geometry_probe_ids,
            }).to_csv(output_dir / "fixed_geometry_subset_ids.csv", index=False)
        elif geometry_probe_ids != current_geometry_ids:
            raise RuntimeError("Geometry sample IDs/order changed across checkpoints")
        frozen_mass = np.square(policies["geo"][0])
        frozen_mass = frozen_mass / frozen_mass.sum()
        current_score = np.sqrt(cells.current_functional_mass.to_numpy(float))
        current_pi = 2.0 * current_score / current_score.sum()
        if np.any(current_pi <= 0) or np.any(current_pi > 1) or abs(current_pi.sum() - 2) > 1e-9:
            raise RuntimeError("Current-geometry allocation is infeasible")
        current_pairs = maximum_entropy_pairs(current_pi)[0]
        oracle_pairs = maximum_entropy_pairs(oracle)[0]
        dynamic_variance = float(np.mean([
            importance_corrected_variance(gram, target_weights, current_pi, current_pairs)[
                "importance_corrected_variance"
            ] for gram in grams
        ]))
        oracle_variance = float(np.mean([
            importance_corrected_variance(gram, target_weights, oracle, oracle_pairs)[
                "importance_corrected_variance"
            ] for gram in grams
        ]))
        trajectory_rows.append({
            "path": path, "epoch": epoch, "V_geo": v_geo, "V_resource": v_resource,
            "V_current_geometry": dynamic_variance,
            "V_gradient_oracle_marginal": oracle_variance,
            "delta_geo_resource": v_geo - v_resource,
            "ratio_geo_resource": v_geo / v_resource,
        })
        cells["path"] = path; cells["epoch"] = epoch
        cells["frozen_functional_mass_normalized"] = frozen_mass
        cells["gradient_second_moment"] = moments
        cells["pi_frozen_geo"] = policies["geo"][0]
        cells["pi_resource"] = policies["resource"][0]
        cells["pi_current_geometry"] = current_pi
        cells["pi_oracle"] = oracle
        geometry_rows.extend(cells.to_dict("records"))
        current_edges["path"] = path; current_edges["epoch"] = epoch
        edge_rows.extend(current_edges.to_dict("records"))

        mean_gram = grams.mean(axis=0)
        width_index = {width: index for index, width in enumerate(INTERIOR_WIDTHS)}
        pair_metrics = []
        for row in representation_pairs.itertuples(index=False):
            i, j = width_index[round(float(row.width_i), 2)], width_index[round(float(row.width_j), 2)]
            norm_i, norm_j, cross = mean_gram[i, i], mean_gram[j, j], mean_gram[i, j]
            distance_sq = max(0.0, norm_i + norm_j - 2.0 * cross)
            pair_metrics.append({
                "path": path, "epoch": epoch,
                "width_i": row.width_i, "width_j": row.width_j,
                "representation_sw": row.representation_sw,
                "gradient_dot": cross,
                "gradient_euclidean_distance": np.sqrt(distance_sq),
                "gradient_cosine_dissimilarity": 1.0 - cross / max(np.sqrt(norm_i * norm_j), 1e-30),
            })
        pair_frame = pd.DataFrame(pair_metrics)
        pair_rows.extend(pair_metrics)
        current_correlations, current_pair_correlations = _correlation_rows(
            path, epoch, cells, pair_frame
        )
        correlation_rows.extend(current_correlations)
        pair_correlation_rows.extend(current_pair_correlations)
        for index, width in enumerate(INTERIOR_WIDTHS):
            drift_rows.append({
                "path": path, "epoch": epoch, "width": width,
                "pi_geo": policies["geo"][0][index],
                "pi_resource": policies["resource"][0][index],
                "pi_current_geometry": current_pi[index],
                "pi_oracle": oracle[index], "gradient_second_moment": moments[index],
            })
        distance_rows.append({
            "path": path, "epoch": epoch,
            "tv_geo_oracle": 0.5 * float(np.abs(policies["geo"][0] - oracle).sum()),
            "tv_resource_oracle": 0.5 * float(np.abs(policies["resource"][0] - oracle).sum()),
            "tv_current_geometry_oracle": 0.5 * float(np.abs(current_pi - oracle).sum()),
            "tv_current_geometry_frozen_geo": 0.5 * float(
                np.abs(current_pi - policies["geo"][0]).sum()
            ),
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
    pd.DataFrame(geometry_rows).to_csv(output_dir / "quick_dynamic_geometry_by_width.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(output_dir / "quick_dynamic_geometry_edges.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(
        output_dir / "quick_geometry_gradient_correlations.csv", index=False
    )
    pd.DataFrame(pair_rows).to_csv(output_dir / "quick_pair_structure.csv", index=False)
    pd.DataFrame(pair_correlation_rows).to_csv(
        output_dir / "quick_pair_structure_correlations.csv", index=False
    )
    metadata = {
        "status": "QUICK_TRAJECTORY_PATH_COMPLETE", "path": path,
        "protocol_version": PROTOCOL_VERSION,
        "epochs": list(EPOCHS), "num_fixed_training_batches": NUM_BATCHES,
        "batch_size": PROBE_BATCH_SIZE, "probe_batch_ids": probe_ids,
        "geometry_subset_ids": geometry_probe_ids,
        "checkpoint_sha256": checkpoint_hashes, "gram_checks": gram_checks,
        "training_performed": False, "optimizer_steps": 0, "accuracy_computed": False,
        "test_used": False, "policy_updated": False, "model_eval": True,
        "weights_unchanged": True, "bn_buffers_unchanged": True,
        "gradient_scope": "shared convolution/projection/classifier weights; BN affine/buffers excluded",
        "probe_data": "fixed first 8 batches of the deterministic 45k training split; no augmentation",
        "geometry_protocol": {
            "representation": "learned_projection_128d_l2",
            "feature_subset": "fixed 2000-image validation subset",
            "bn_recalibration": True,
            "num_projections": int(config["geometry"]["num_projections"]),
            "projection_seed": int(config["geometry"]["projection_seed"]),
            "buffers_restored_after_each_checkpoint": True,
            "frozen_mass_reconstruction": "normalized a_frozen proportional to pi_geo^2 (p=1)",
        },
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


def _plot_geometry_tracking(output_dir: Path, geometry: pd.DataFrame,
                            correlations: pd.DataFrame,
                            pair_correlations: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False, sharey=False)
    for row_index, path in enumerate(PATHS):
        for column_index, epoch in enumerate(EPOCHS):
            axis = axes[row_index, column_index]
            frame = geometry.loc[geometry.path.eq(path) & geometry.epoch.eq(epoch)]
            axis.scatter(frame.current_functional_mass_normalized,
                         frame.gradient_second_moment, s=35)
            for row in frame.itertuples():
                axis.annotate(f"{row.width:.2f}",
                              (row.current_functional_mass_normalized,
                               row.gradient_second_moment), fontsize=7)
            axis.set_title(f"{path}, epoch {epoch}")
            axis.grid(alpha=0.2)
    fig.supxlabel("Current normalized functional mass a_i(theta_t)")
    fig.supylabel("Current gradient second moment m_i(theta_t)")
    fig.tight_layout(); fig.savefig(
        output_dir / "quick_current_geometry_vs_m.png", dpi=200
    ); plt.close(fig)

    selected = correlations.loc[
        correlations.metric.eq("Spearman")
        & correlations.analysis.isin(("current_a_vs_m", "frozen_a_vs_m"))
    ]
    fig, axis = plt.subplots(figsize=(9, 5))
    for (path, analysis), frame in selected.groupby(["path", "analysis"]):
        axis.plot(frame.epoch, frame.coefficient, marker="o", label=f"{path}: {analysis}")
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="Epoch", ylabel="Spearman correlation with current m_i")
    axis.grid(alpha=0.25); axis.legend(); fig.tight_layout()
    fig.savefig(output_dir / "quick_geometry_gradient_tracking.png", dpi=200); plt.close(fig)

    selected_pairs = pair_correlations.loc[pair_correlations.metric.eq("Spearman")]
    fig, axis = plt.subplots(figsize=(9, 5))
    for (path, analysis), frame in selected_pairs.groupby(["path", "analysis"]):
        axis.plot(frame.epoch, frame.coefficient, marker="o", label=f"{path}: {analysis}")
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="Epoch", ylabel="Spearman pair-structure correlation")
    axis.grid(alpha=0.25); axis.legend(fontsize=8); fig.tight_layout()
    fig.savefig(output_dir / "quick_pair_structure_tracking.png", dpi=200); plt.close(fig)


def merge_trajectory_paths(worker_dirs: list[str | Path], output_dir: str | Path) -> dict:
    worker_dirs, output_dir = [Path(path) for path in worker_dirs], Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = [json.loads((path / "metadata.json").read_text()) for path in worker_dirs]
    if {item["path"] for item in metadata} != set(PATHS):
        raise RuntimeError("Trajectory workers must cover Geo-HT and Resource-HT exactly once")
    if len({tuple(item["probe_batch_ids"]) for item in metadata}) != 1:
        raise RuntimeError("Trajectory workers did not use identical fixed sample IDs/order")
    if len({tuple(item["geometry_subset_ids"]) for item in metadata}) != 1:
        raise RuntimeError("Trajectory workers did not use identical geometry sample IDs/order")
    for item in metadata:
        if (
            item["training_performed"] or item["optimizer_steps"] != 0 or item["test_used"]
            or not item["weights_unchanged"] or not item["bn_buffers_unchanged"]
            or int(item.get("protocol_version", -1)) != PROTOCOL_VERSION
        ):
            raise RuntimeError("A trajectory worker violated the read-only contract")
    names = (
        "quick_trajectory_variance.csv", "quick_total_variance.csv",
        "quick_oracle_drift.csv", "quick_policy_distance.csv",
        "quick_dynamic_geometry_by_width.csv", "quick_dynamic_geometry_edges.csv",
        "quick_geometry_gradient_correlations.csv", "quick_pair_structure.csv",
        "quick_pair_structure_correlations.csv",
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
    _plot_geometry_tracking(
        output_dir, merged["quick_dynamic_geometry_by_width.csv"],
        merged["quick_geometry_gradient_correlations.csv"],
        merged["quick_pair_structure_correlations.csv"],
    )
    trajectory = merged["quick_trajectory_variance.csv"]
    total = merged["quick_total_variance.csv"]
    correlations = merged["quick_geometry_gradient_correlations.csv"]
    spearman = correlations.loc[
        correlations.metric.eq("Spearman")
        & correlations.analysis.isin(("current_a_vs_m", "frozen_a_vs_m"))
    ].pivot(index=["path", "epoch"], columns="analysis", values="coefficient")
    result = {
        "status": "RQ2_V3_QUICK_TRAJECTORY_DIAGNOSTIC_COMPLETE",
        "protocol_version": PROTOCOL_VERSION,
        "paths": list(PATHS), "epochs": list(EPOCHS),
        "execution": "two_gpu_one_trajectory_per_gpu",
        "training_performed": False, "optimizer_steps": 0, "accuracy_computed": False,
        "test_used": False, "policy_updated": False,
        "same_fixed_batch_ids_and_order": True,
        "geo_lower_conditional_variance_count": int((trajectory.delta_geo_resource < 0).sum()),
        "comparisons": len(trajectory),
        "mean_total_relative_gain_geo": float(total.total_relative_gain_geo.mean()),
        "current_geometry_better_than_frozen_count": int(
            (spearman.current_a_vs_m > spearman.frozen_a_vs_m).sum()
        ),
        "geometry_gradient_comparisons": int(len(spearman)),
        "interpretation": "mechanism diagnostic only; no policy selection or accuracy claim",
    }
    (output_dir / "metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
