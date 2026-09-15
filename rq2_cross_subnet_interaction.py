"""Read-only cross-subnet gradient interaction probe on a common checkpoint."""

from __future__ import annotations

import hashlib
import json
import shutil
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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import build_interim_validation_loaders
from evaluation import calibrate_batch_norm
from models.slimmable_ops import SharedProjection, SlimmableConv2d
from rq2_anchor_placement import GRID, _sha256
from rq2_probabilistic_support import INTERIOR_WIDTHS
from s1_width import make_model
from training import kd_loss


PROBE_BATCHES = 16
PROBE_BATCH_SIZE = 128
ONE_STEP_SOURCES = (0.40, 0.50, 0.70, 0.90)
ONE_STEP_EPSILON = 1e-5


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


def find_uniform_seed3_checkpoint(
    input_root: str | Path, materialized_root: str | Path
) -> Path:
    """Resolve exactly one Uniform/seed_3 epoch-100 checkpoint from prior output."""
    input_root, materialized_root = Path(input_root), Path(materialized_root)

    def candidates_below(root):
        return sorted({
            path for path in root.rglob("checkpoint.pt")
            if len(path.parts) >= 3 and path.parts[-3:-1] == ("uniform", "seed_3")
        })

    candidates = candidates_below(input_root)
    if not candidates:
        archives = sorted(input_root.rglob("finalgeo-confirmatory-results.zip"))
        if len(archives) == 1:
            candidates = candidates_below(_safe_extract(archives[0], Path(materialized_root)))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one Uniform seed-3 checkpoint, found: {candidates}")
    return candidates[0]


def find_observed_seed3_comparison(input_root: str | Path) -> Path | None:
    candidates = sorted(Path(input_root).rglob("dynamic_pilot_width_comparison.csv"))
    return candidates[0] if len(candidates) == 1 else None


def interaction_parameters(model):
    """Select all shared convolution/linear weights, excluding BN affine/buffers."""
    selected = []
    for module_name, module in model.named_modules():
        if isinstance(module, (SlimmableConv2d, SharedProjection, nn.Linear)):
            weight = getattr(module, "weight", None)
            if weight is not None and weight.requires_grad:
                name = f"{module_name}.weight" if module_name else "weight"
                selected.append((name, weight))
    if not selected or len({id(parameter) for _, parameter in selected}) != len(selected):
        raise RuntimeError("Could not construct a unique convolution/linear weight scope")
    return selected


def _state_digest(named_tensors) -> str:
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        digest.update(name.encode())
        value = tensor.detach().cpu().contiguous()
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _flat_gradient(loss: torch.Tensor, parameters) -> torch.Tensor:
    tensors = [parameter for _, parameter in parameters]
    gradients = torch.autograd.grad(loss, tensors, allow_unused=True)
    flat = torch.cat([
        torch.zeros_like(parameter).reshape(-1)
        if gradient is None else gradient.detach().reshape(-1)
        for parameter, gradient in zip(tensors, gradients)
    ])
    if not torch.isfinite(flat).all():
        raise FloatingPointError("Non-finite gradient in interaction probe")
    return flat


def _batch_gradients(model, images, labels, parameters, config):
    model.set_width(1.0)
    teacher = model(images)
    teacher_detached = teacher.detach()
    losses = {1.0: F.cross_entropy(teacher, labels)}
    vectors = {1.0: _flat_gradient(losses[1.0], parameters)}
    for width in GRID[:-1]:
        model.set_width(width)
        logits = model(images)
        losses[width] = F.cross_entropy(logits, labels) + float(
            config["training"]["kd_lambda"]
        ) * kd_loss(logits, teacher_detached, float(config["training"]["kd_temperature"]))
        vectors[width] = _flat_gradient(losses[width], parameters)
    matrix = torch.stack([vectors[width] for width in GRID])
    dot = matrix @ matrix.T
    norms = torch.linalg.vector_norm(matrix, dim=1)
    cosine = dot / (norms[:, None] * norms[None, :]).clamp_min(1e-20)
    if not torch.isfinite(dot).all() or not torch.isfinite(cosine).all() or torch.any(norms <= 0):
        raise FloatingPointError("Invalid dot/cosine/norm matrix")
    return matrix, dot, cosine, norms, teacher_detached


@torch.no_grad()
def _loss_value(model, images, labels, width, frozen_teacher, config) -> float:
    model.set_width(width)
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    if width < 1.0:
        loss = loss + float(config["training"]["kd_lambda"]) * kd_loss(
            logits, frozen_teacher, float(config["training"]["kd_temperature"])
        )
    return float(loss)


def one_step_transfer(
    model, images, labels, parameters, gradient_matrix, dot_matrix,
    frozen_teacher, config, epsilon=ONE_STEP_EPSILON,
) -> pd.DataFrame:
    """Validate first-order dot-product signs on one fixed minibatch."""
    originals = [parameter.detach().clone() for _, parameter in parameters]
    baseline = {
        width: _loss_value(model, images, labels, width, frozen_teacher, config)
        for width in GRID
    }
    rows = []
    try:
        for source in ONE_STEP_SOURCES:
            source_index = GRID.index(source)
            flat_source = gradient_matrix[source_index]
            offset = 0
            with torch.no_grad():
                for _, parameter in parameters:
                    count = parameter.numel()
                    parameter.add_(flat_source[offset:offset + count].view_as(parameter), alpha=-epsilon)
                    offset += count
            for target_index, target in enumerate(GRID):
                observed = _loss_value(model, images, labels, target, frozen_teacher, config) - baseline[target]
                predicted = -epsilon * float(dot_matrix[target_index, source_index])
                rows.append({
                    "source_width": source, "target_width": target, "epsilon": epsilon,
                    "predicted_loss_change": predicted, "observed_loss_change": observed,
                    "sign_match": bool(np.sign(predicted) == np.sign(observed)),
                })
            with torch.no_grad():
                for (_, parameter), original in zip(parameters, originals):
                    parameter.copy_(original)
    finally:
        with torch.no_grad():
            for (_, parameter), original in zip(parameters, originals):
                parameter.copy_(original)
    return pd.DataFrame(rows)


def _matrix_frame(matrix: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(matrix, columns=[f"{width:.2f}" for width in GRID])
    frame.insert(0, "width", GRID)
    return frame


def policy_expected_benefits(
    dot_matrix: np.ndarray,
    pi_geometry: np.ndarray,
    pi_resource: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return B_j=<g_j,gbar> for two policies and their Geo-Resource delta."""
    dot_matrix = np.asarray(dot_matrix, dtype=float)
    pi_geometry = np.asarray(pi_geometry, dtype=float)
    pi_resource = np.asarray(pi_resource, dtype=float)
    if dot_matrix.shape != (len(GRID), len(GRID)):
        raise ValueError(f"Expected a {len(GRID)}x{len(GRID)} dot matrix")
    if pi_geometry.shape != (len(INTERIOR_WIDTHS),) or pi_resource.shape != (
        len(INTERIOR_WIDTHS),
    ):
        raise ValueError("Policy marginals must cover the 14 interior widths")
    geometry_weights = np.asarray([1.0, *pi_geometry, 1.0])
    resource_weights = np.asarray([1.0, *pi_resource, 1.0])
    geometry = dot_matrix @ geometry_weights
    resource = dot_matrix @ resource_weights
    return geometry, resource, geometry - resource


def _plots(output_dir: Path, cosine: np.ndarray, effect: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(cosine, cmap="coolwarm", vmin=-1, vmax=1)
    labels = [f"{width:.2f}" for width in GRID]
    axis.set_xticks(range(16), labels, rotation=45, ha="right")
    axis.set_yticks(range(16), labels)
    axis.set(xlabel="Source width i", ylabel="Target width j",
             title="Common Uniform seed-3 checkpoint: gradient cosine")
    fig.colorbar(image, ax=axis, label="mean cosine")
    fig.tight_layout(); fig.savefig(output_dir / "interaction_heatmap.png", dpi=200); plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.axhline(0, color="black", linewidth=1)
    axis.plot(effect.width, effect.delta_benefit, marker="o", label="Delta benefit: Geo - Resource")
    if effect["observed_delta_accuracy_seed3"].notna().all():
        secondary = axis.twinx()
        secondary.plot(effect.width, effect.observed_delta_accuracy_seed3,
                       marker="s", color="tab:orange", label="Observed delta accuracy")
        secondary.set_ylabel("Observed accuracy difference")
    axis.set(xlabel="Target width", ylabel="First-order delta benefit")
    axis.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(output_dir / "policy_effect_by_width.png", dpi=200); plt.close(fig)


def run_cross_subnet_interaction_probe(
    checkpoint: str | Path,
    config_path: str | Path,
    geometry_marginals_path: str | Path,
    resource_marginals_path: str | Path,
    output_dir: str | Path,
    dataset_root: str | Path,
    observed_comparison_path: str | Path | None = None,
    device: str = "cuda",
    num_batches: int = PROBE_BATCHES,
    batch_offset: int = 0,
    run_one_step: bool = True,
) -> dict:
    """Run the common-reference interaction diagnostic without updating weights."""
    checkpoint, config_path = Path(checkpoint), Path(config_path)
    geometry_marginals_path = Path(geometry_marginals_path)
    resource_marginals_path = Path(resource_marginals_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(num_batches) not in range(1, 65):
        raise ValueError("num_batches must be between 1 and 64")
    batch_offset = int(batch_offset)
    if batch_offset < 0 or batch_offset + int(num_batches) > 64:
        raise ValueError("batch_offset must select batches within the first 64")
    config = yaml.safe_load(config_path.read_text())
    config["dataset"]["root"] = str(dataset_root)
    config["dataset"]["download"] = True
    if tuple(map(float, config["compression"]["eval_widths"])) != GRID:
        raise ValueError("Checkpoint config must expose the full 16-width grid")
    if not np.isclose(float(config["training"]["kd_lambda"]), 1.0) or not np.isclose(
        float(config["training"]["kd_temperature"]), 2.0
    ):
        raise ValueError("Interaction probe requires the historical KD lambda=1, T=2 loss")

    geometry_frame = pd.read_csv(geometry_marginals_path)
    if "p" in geometry_frame:
        geometry_frame = geometry_frame.loc[np.isclose(geometry_frame["p"], 1.0)]
    geometry_frame = geometry_frame.sort_values("width")
    resource_frame = pd.read_csv(resource_marginals_path).sort_values("width")
    if (
        tuple(geometry_frame["width"].round(2)) != INTERIOR_WIDTHS
        or tuple(resource_frame["width"].round(2)) != INTERIOR_WIDTHS
    ):
        raise RuntimeError("Both policy marginal tables must cover all 14 interior widths")
    geometry_column = "pi" if "pi" in geometry_frame else "pi_geometry"
    resource_column = (
        "pi_resource_matched_compute"
        if "pi_resource_matched_compute" in resource_frame else "pi_resource"
    )
    pi_geometry = geometry_frame[geometry_column].to_numpy(float)
    pi_resource = resource_frame[resource_column].to_numpy(float)
    if abs(pi_geometry.sum() - 2) > 1e-8 or abs(pi_resource.sum() - 2) > 1e-8:
        raise RuntimeError("Both dynamic policies must allocate exactly two interior slots")

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    loaders = build_interim_validation_loaders(config)
    probe_loader = DataLoader(
        loaders.calibration.dataset,
        batch_size=PROBE_BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=torch_device.type == "cuda",
    )
    probe_ids = []
    for batch_index, (_, _, sample_ids) in enumerate(probe_loader):
        if batch_index < batch_offset:
            continue
        if batch_index >= batch_offset + num_batches:
            break
        probe_ids.extend(torch.as_tensor(sample_ids).tolist())
    pd.DataFrame({"order": range(len(probe_ids)), "sample_id": probe_ids}).to_csv(
        output_dir / "fixed_training_subset_ids.csv", index=False
    )

    model = make_model(config, torch_device)
    payload = torch.load(checkpoint, map_location=torch_device, weights_only=False)
    if int(payload.get("epoch", -1)) != 100:
        raise RuntimeError("Common reference checkpoint must be the completed epoch-100 model")
    model.load_state_dict(payload["model"])
    for width in GRID:
        calibrate_batch_norm(
            model, loaders.calibration, width, torch_device,
            int(config["evaluation"]["bn_calibration_batches"]),
        )
    model.eval()
    parameters = interaction_parameters(model)
    parameter_digest_before = _state_digest(model.named_parameters())
    buffer_digest_before = _state_digest(model.named_buffers())

    dot_sum = torch.zeros((16, 16), dtype=torch.float64)
    cosine_sum = torch.zeros((16, 16), dtype=torch.float64)
    norm_rows, effect_rows = [], []
    one_step = None
    for batch_index, (images, labels, _) in enumerate(probe_loader):
        if batch_index < batch_offset:
            continue
        if batch_index >= batch_offset + num_batches:
            break
        images, labels = images.to(torch_device), labels.to(torch_device)
        matrix, dot, cosine, norms, frozen_teacher = _batch_gradients(
            model, images, labels, parameters, config
        )
        dot_cpu = dot.detach().double().cpu()
        cosine_cpu = cosine.detach().double().cpu()
        dot_sum += dot_cpu
        cosine_sum += cosine_cpu
        benefit_geometry, benefit_resource, delta_benefit = policy_expected_benefits(
            dot_cpu.numpy(), pi_geometry, pi_resource
        )
        for index, width in enumerate(GRID):
            norm_rows.append({
                "batch": batch_index, "width": width,
                "gradient_norm": float(norms[index]),
            })
            effect_rows.append({
                "batch": batch_index, "width": width,
                "benefit_geometry": benefit_geometry[index],
                "benefit_resource": benefit_resource[index],
                "delta_benefit": delta_benefit[index],
            })
        if run_one_step and batch_index == batch_offset:
            one_step = one_step_transfer(
                model, images, labels, parameters, matrix, dot, frozen_teacher,
                config, epsilon=ONE_STEP_EPSILON,
            )
        del matrix, dot, cosine, norms, frozen_teacher
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
        completed = batch_index - batch_offset + 1
        print(f"[cross-subnet interaction] batch {completed}/{num_batches} "
              f"(global batch {batch_index})", flush=True)

    if len(effect_rows) != num_batches * 16:
        raise RuntimeError("Diagnostic loader ended before the requested fixed minibatches")
    parameter_digest_after = _state_digest(model.named_parameters())
    buffer_digest_after = _state_digest(model.named_buffers())
    if parameter_digest_before != parameter_digest_after or buffer_digest_before != buffer_digest_after:
        raise RuntimeError("Read-only interaction probe modified model weights or BN buffers")

    mean_dot = (dot_sum / num_batches).numpy()
    mean_cosine = (cosine_sum / num_batches).numpy()
    _matrix_frame(mean_dot).to_csv(output_dir / "gradient_dot_matrix.csv", index=False)
    _matrix_frame(mean_cosine).to_csv(output_dir / "gradient_cosine_matrix.csv", index=False)
    norm_frame = pd.DataFrame(norm_rows)
    norm_frame.to_csv(output_dir / "gradient_norms_by_batch.csv", index=False)
    norm_frame.groupby("width", as_index=False).agg(
        mean_gradient_norm=("gradient_norm", "mean"),
        std_gradient_norm=("gradient_norm", "std"),
        min_gradient_norm=("gradient_norm", "min"),
        max_gradient_norm=("gradient_norm", "max"),
    ).to_csv(output_dir / "gradient_norms.csv", index=False)
    effects_by_batch = pd.DataFrame(effect_rows)
    effects_by_batch.to_csv(output_dir / "policy_expected_effect_by_batch.csv", index=False)
    effect = effects_by_batch.groupby("width", as_index=False).agg(
        benefit_geometry=("benefit_geometry", "mean"),
        benefit_resource=("benefit_resource", "mean"),
        delta_benefit=("delta_benefit", "mean"),
        delta_benefit_std=("delta_benefit", "std"),
    )
    effect["observed_delta_accuracy_seed3"] = np.nan
    observed_path = Path(observed_comparison_path) if observed_comparison_path else None
    if observed_path is not None and observed_path.is_file():
        observed = pd.read_csv(observed_path)
        observed_width = "width" if "width" in observed else "budget"
        observed_delta = (
            "geometry_minus_resource_accuracy"
            if "geometry_minus_resource_accuracy" in observed
            else "delta_accuracy"
        )
        observed = observed[[observed_width, observed_delta]].rename(columns={
                observed_width: "width", observed_delta: "observed_delta_accuracy_seed3"
            })
        observed["width"] = observed["width"].astype(float).round(2)
        effect["width"] = effect["width"].astype(float).round(2)
        effect = effect.drop(columns="observed_delta_accuracy_seed3").merge(
            observed, on="width", how="left", validate="one_to_one",
        )
    effect.to_csv(output_dir / "policy_expected_effect.csv", index=False)
    if one_step is not None:
        one_step.to_csv(output_dir / "one_step_transfer.csv", index=False)
    _plots(output_dir, mean_cosine, effect)

    correlations = None
    complete_observed = effect["observed_delta_accuracy_seed3"].notna().all()
    if complete_observed:
        pearson = pearsonr(effect["delta_benefit"], effect["observed_delta_accuracy_seed3"])
        spearman = spearmanr(effect["delta_benefit"], effect["observed_delta_accuracy_seed3"])
        correlations = {
            "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
        }
    metadata = {
        "status": "POST_HOC_MECHANISTIC_DIAGNOSTIC_COMPLETE",
        "independent_of_seed4_training": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "test_used": False,
        "checkpoint_role": "common_reference_uniform_seed3_epoch100",
        "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
        "geometry_marginals_path": str(geometry_marginals_path),
        "geometry_marginals_sha256": _sha256(geometry_marginals_path),
        "resource_marginals_path": str(resource_marginals_path),
        "resource_marginals_sha256": _sha256(resource_marginals_path),
        "observed_comparison_path": str(observed_path) if observed_path else None,
        "num_fixed_training_batches": int(num_batches),
        "num_fixed_training_samples": len(probe_ids),
        "batch_offset": batch_offset,
        "global_batch_indices": list(range(batch_offset, batch_offset + int(num_batches))),
        "widths": list(GRID),
        "loss": {"full_width": "CE", "smaller_widths": "CE + KD", "kd_lambda": 1.0, "temperature": 2.0},
        "gradient_scope": "all shared trainable convolution/projection/classifier linear weights; BN excluded",
        "gradient_parameter_tensors": len(parameters),
        "gradient_parameter_scalars": int(sum(parameter.numel() for _, parameter in parameters)),
        "bn_calibrated_before_probe": True,
        "model_eval_during_probe": True,
        "weights_unchanged": parameter_digest_before == parameter_digest_after,
        "bn_buffers_unchanged_during_probe": buffer_digest_before == buffer_digest_after,
        "one_step_sources": list(ONE_STEP_SOURCES),
        "one_step_epsilon": ONE_STEP_EPSILON,
        "one_step_performed": bool(one_step is not None),
        "one_step_sign_match_rate": (
            float(one_step["sign_match"].mean()) if one_step is not None else None
        ),
        "observed_seed3_accuracy_available": bool(complete_observed),
        "delta_benefit_vs_observed_accuracy": correlations,
        "endpoint_delta_benefit": {
            "0.25": float(effect.loc[np.isclose(effect.width, 0.25), "delta_benefit"].iloc[0]),
            "1.00": float(effect.loc[np.isclose(effect.width, 1.0), "delta_benefit"].iloc[0]),
        },
        "width_0.40_delta_benefit": float(
            effect.loc[np.isclose(effect.width, 0.40), "delta_benefit"].iloc[0]
        ),
        "policy_updated": False,
        "seed4_training_authorized_or_started": False,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def merge_cross_subnet_interaction_workers(
    worker_dirs: list[str | Path],
    output_dir: str | Path,
    observed_comparison_path: str | Path | None = None,
    expected_batches: int = PROBE_BATCHES,
) -> dict:
    """Merge disjoint batch shards produced concurrently on separate GPUs."""
    worker_dirs = [Path(path) for path in worker_dirs]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(worker_dirs) < 2:
        raise ValueError("Multi-GPU merge requires at least two worker directories")
    worker_metadata = [json.loads((path / "metadata.json").read_text()) for path in worker_dirs]
    for item in worker_metadata:
        if item["training_performed"] or item["optimizer_steps"] != 0 or item["test_used"]:
            raise RuntimeError("A worker violated the read-only diagnostic contract")
        if not item["weights_unchanged"] or not item["bn_buffers_unchanged_during_probe"]:
            raise RuntimeError("A worker changed model state")
    for key in (
        "checkpoint_sha256", "geometry_marginals_sha256", "resource_marginals_sha256"
    ):
        if len({item[key] for item in worker_metadata}) != 1:
            raise RuntimeError(f"Workers disagree on {key}")
    batch_indices = sorted(
        index for item in worker_metadata for index in item["global_batch_indices"]
    )
    if batch_indices != list(range(int(expected_batches))):
        raise RuntimeError(f"Worker shards do not cover batches 0..{expected_batches - 1}")

    def read_matrix(root: Path, name: str) -> np.ndarray:
        frame = pd.read_csv(root / name)
        if tuple(frame["width"].round(2)) != GRID:
            raise RuntimeError(f"Invalid width ordering in {root / name}")
        return frame.drop(columns="width").to_numpy(float)

    counts = np.asarray([item["num_fixed_training_batches"] for item in worker_metadata])
    mean_dot = sum(
        count * read_matrix(root, "gradient_dot_matrix.csv")
        for root, count in zip(worker_dirs, counts)
    ) / counts.sum()
    mean_cosine = sum(
        count * read_matrix(root, "gradient_cosine_matrix.csv")
        for root, count in zip(worker_dirs, counts)
    ) / counts.sum()
    _matrix_frame(mean_dot).to_csv(output_dir / "gradient_dot_matrix.csv", index=False)
    _matrix_frame(mean_cosine).to_csv(output_dir / "gradient_cosine_matrix.csv", index=False)

    norms = pd.concat(
        [pd.read_csv(root / "gradient_norms_by_batch.csv") for root in worker_dirs],
        ignore_index=True,
    ).sort_values(["batch", "width"])
    effects_by_batch = pd.concat(
        [pd.read_csv(root / "policy_expected_effect_by_batch.csv") for root in worker_dirs],
        ignore_index=True,
    ).sort_values(["batch", "width"])
    if len(norms) != int(expected_batches) * len(GRID) or len(effects_by_batch) != len(norms):
        raise RuntimeError("Merged worker row count is incomplete")
    norms.to_csv(output_dir / "gradient_norms_by_batch.csv", index=False)
    norms.groupby("width", as_index=False).agg(
        mean_gradient_norm=("gradient_norm", "mean"),
        std_gradient_norm=("gradient_norm", "std"),
        min_gradient_norm=("gradient_norm", "min"),
        max_gradient_norm=("gradient_norm", "max"),
    ).to_csv(output_dir / "gradient_norms.csv", index=False)
    effects_by_batch.to_csv(output_dir / "policy_expected_effect_by_batch.csv", index=False)
    effect = effects_by_batch.groupby("width", as_index=False).agg(
        benefit_geometry=("benefit_geometry", "mean"),
        benefit_resource=("benefit_resource", "mean"),
        delta_benefit=("delta_benefit", "mean"),
        delta_benefit_std=("delta_benefit", "std"),
    )
    effect["observed_delta_accuracy_seed3"] = np.nan
    observed_path = Path(observed_comparison_path) if observed_comparison_path else None
    if observed_path is not None and observed_path.is_file():
        observed = pd.read_csv(observed_path)
        observed_width = "width" if "width" in observed else "budget"
        observed_delta = (
            "geometry_minus_resource_accuracy"
            if "geometry_minus_resource_accuracy" in observed else "delta_accuracy"
        )
        observed = observed[[observed_width, observed_delta]].rename(columns={
            observed_width: "width", observed_delta: "observed_delta_accuracy_seed3"
        })
        observed["width"] = observed["width"].astype(float).round(2)
        effect["width"] = effect["width"].astype(float).round(2)
        effect = effect.drop(columns="observed_delta_accuracy_seed3").merge(
            observed, on="width", how="left", validate="one_to_one"
        )
    effect.to_csv(output_dir / "policy_expected_effect.csv", index=False)

    one_step_sources = [root / "one_step_transfer.csv" for root in worker_dirs]
    one_step_sources = [path for path in one_step_sources if path.is_file()]
    if len(one_step_sources) != 1:
        raise RuntimeError("Exactly one worker must perform one-step transfer")
    shutil.copy2(one_step_sources[0], output_dir / "one_step_transfer.csv")
    one_step = pd.read_csv(output_dir / "one_step_transfer.csv")
    id_frames = [pd.read_csv(root / "fixed_training_subset_ids.csv") for root in worker_dirs]
    ids = pd.concat(id_frames, ignore_index=True).drop(columns="order")
    ids.insert(0, "order", range(len(ids)))
    ids.to_csv(output_dir / "fixed_training_subset_ids.csv", index=False)
    _plots(output_dir, mean_cosine, effect)

    complete_observed = effect["observed_delta_accuracy_seed3"].notna().all()
    correlations = None
    if complete_observed:
        pearson = pearsonr(effect["delta_benefit"], effect["observed_delta_accuracy_seed3"])
        spearman = spearmanr(effect["delta_benefit"], effect["observed_delta_accuracy_seed3"])
        correlations = {
            "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
        }
    first = worker_metadata[0]
    metadata = {
        "status": "POST_HOC_MECHANISTIC_DIAGNOSTIC_COMPLETE",
        "execution": "two_gpu_disjoint_batch_shards",
        "gpu_workers": len(worker_dirs),
        "worker_batch_indices": [item["global_batch_indices"] for item in worker_metadata],
        "independent_of_seed4_training": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "test_used": False,
        "checkpoint_role": first["checkpoint_role"],
        "checkpoint": first["checkpoint"],
        "checkpoint_sha256": first["checkpoint_sha256"],
        "geometry_marginals_sha256": first["geometry_marginals_sha256"],
        "resource_marginals_sha256": first["resource_marginals_sha256"],
        "num_fixed_training_batches": int(counts.sum()),
        "num_fixed_training_samples": len(ids),
        "widths": list(GRID),
        "loss": first["loss"],
        "gradient_scope": first["gradient_scope"],
        "gradient_parameter_tensors": first["gradient_parameter_tensors"],
        "gradient_parameter_scalars": first["gradient_parameter_scalars"],
        "bn_calibrated_independently_on_each_worker": True,
        "model_eval_during_probe": True,
        "weights_unchanged": True,
        "bn_buffers_unchanged_during_probe": True,
        "one_step_sources": list(ONE_STEP_SOURCES),
        "one_step_epsilon": ONE_STEP_EPSILON,
        "one_step_sign_match_rate": float(one_step["sign_match"].mean()),
        "observed_seed3_accuracy_available": bool(complete_observed),
        "delta_benefit_vs_observed_accuracy": correlations,
        "endpoint_delta_benefit": {
            "0.25": float(effect.loc[np.isclose(effect.width, 0.25), "delta_benefit"].iloc[0]),
            "1.00": float(effect.loc[np.isclose(effect.width, 1.0), "delta_benefit"].iloc[0]),
        },
        "width_0.40_delta_benefit": float(
            effect.loc[np.isclose(effect.width, 0.40), "delta_benefit"].iloc[0]
        ),
        "policy_updated": False,
        "seed4_training_authorized_or_started": False,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
