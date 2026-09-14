"""Gradient-interference diagnostics for the frozen RQ2-v1 checkpoints.

The diagnostic never updates model weights.  Each checkpoint is BN-calibrated
with the locked training calibration split, switched to eval mode, and then
measured on the fixed validation geometry subset.
"""

from __future__ import annotations

import hashlib
import json
import math
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from data import build_confirmatory_loaders
from evaluation import calibrate_batch_norm
from s1_width import make_model
from training import kd_loss


DIAGNOSTIC_WIDTHS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.75, 0.80, 0.90, 1.00)
FOCUS_PAIRS = ((0.40, 0.75), (0.40, 1.00), (0.60, 0.75), (0.60, 1.00))
SCOPES = ("backbone", "backbone_classifier")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_state_digest(named_tensors) -> str:
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def resolve_checkpoints(uniform_root: str | Path, rq2_root: str | Path) -> dict[tuple[str, int], Path]:
    """Resolve and validate the four preregistered RQ2-v1 checkpoints."""
    uniform_root, rq2_root = Path(uniform_root), Path(rq2_root)
    checkpoints = {
        ("Uniform", 1): uniform_root / "shared" / "seed_1" / "checkpoint.pt",
        ("Uniform", 2): uniform_root / "shared" / "seed_2" / "checkpoint.pt",
        ("PureGeo", 1): rq2_root / "geo" / "seed_1" / "checkpoint.pt",
        ("PureGeo", 2): rq2_root / "geo" / "seed_2" / "checkpoint.pt",
    }
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing RQ2-v1 checkpoints: {missing}")
    return checkpoints


def diagnostic_parameters(model):
    """Return the fixed aligned parameter vector and the primary prefix length.

    Primary excludes both learned projection and classifier.  The sensitivity
    scope includes backbone and classifier but continues to exclude projection.
    Parameters are returned in backbone-then-classifier order.
    """
    backbone, classifier = [], []
    for name, parameter in model.named_parameters():
        if name.startswith("projection."):
            continue
        if name.startswith("classifier."):
            classifier.append((name, parameter))
        else:
            backbone.append((name, parameter))
    if not backbone or not classifier:
        raise ValueError("Expected non-empty backbone and classifier parameter groups")
    selected = backbone + classifier
    backbone_numel = sum(parameter.numel() for _, parameter in backbone)
    return selected, backbone_numel


def _flatten_gradients(loss, parameters) -> torch.Tensor:
    tensors = [parameter for _, parameter in parameters]
    gradients = torch.autograd.grad(loss, tensors, allow_unused=True)
    flat = [
        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.detach().reshape(-1)
        for gradient, parameter in zip(gradients, tensors)
    ]
    result = torch.cat(flat)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Non-finite diagnostic gradient")
    return result


def _scope_statistics(matrix: torch.Tensor, backbone_numel: int):
    """Compute cosine matrices/norms without copying the large backbone prefix."""
    backbone = matrix[:, :backbone_numel]
    classifier = matrix[:, backbone_numel:]
    primary_dot = backbone @ backbone.T
    sensitivity_dot = primary_dot + classifier @ classifier.T
    result = {}
    for scope, dot in (("backbone", primary_dot), ("backbone_classifier", sensitivity_dot)):
        norms = torch.sqrt(torch.clamp(torch.diagonal(dot), min=0.0))
        denominator = norms[:, None] * norms[None, :]
        cosine = torch.where(denominator > 0, dot / denominator.clamp_min(1e-20), torch.nan)
        result[scope] = (cosine.detach().cpu().numpy(), norms.detach().cpu().numpy())
    return result


def diagnose_model(
    model,
    loader,
    config: dict,
    device: torch.device,
    method: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure batchwise gradient cosines for one already BN-calibrated model."""
    model.eval()
    selected, backbone_numel = diagnostic_parameters(model)
    kd_lambda = float(config["training"]["kd_lambda"])
    temperature = float(config["training"]["kd_temperature"])
    pair_rows, norm_rows = [], []
    for batch_index, (images, labels, sample_ids) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        # Exactly one full-width teacher forward. Its detached logits are reused
        # for every small-width loss, while its CE graph yields g_1.0.
        model.set_width(1.0)
        teacher_logits = model(images)
        teacher_detached = teacher_logits.detach()
        full_loss = F.cross_entropy(teacher_logits, labels)
        vectors = {1.0: _flatten_gradients(full_loss, selected)}
        losses = {1.0: float(full_loss.detach())}

        for width in DIAGNOSTIC_WIDTHS:
            if width == 1.0:
                continue
            model.set_width(width)
            logits = model(images)
            loss = F.cross_entropy(logits, labels) + kd_lambda * kd_loss(
                logits, teacher_detached, temperature
            )
            vectors[width] = _flatten_gradients(loss, selected)
            losses[width] = float(loss.detach())

        matrix = torch.stack([vectors[width] for width in DIAGNOSTIC_WIDTHS])
        stats = _scope_statistics(matrix, backbone_numel)
        first_id = int(torch.as_tensor(sample_ids)[0])
        last_id = int(torch.as_tensor(sample_ids)[-1])
        batch_size = int(labels.numel())
        for scope, (cosines, norms) in stats.items():
            if not np.isfinite(cosines).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
                raise FloatingPointError(
                    f"Invalid gradient cosine/norm for {method} seed={seed} "
                    f"batch={batch_index} scope={scope}"
                )
            for index, width in enumerate(DIAGNOSTIC_WIDTHS):
                norm_rows.append({
                    "method": method, "seed": int(seed), "scope": scope,
                    "batch": batch_index, "batch_size": batch_size,
                    "first_sample_id": first_id, "last_sample_id": last_id,
                    "width": width, "loss": losses[width],
                    "gradient_norm": float(norms[index]),
                })
            for left in range(len(DIAGNOSTIC_WIDTHS)):
                for right in range(left + 1, len(DIAGNOSTIC_WIDTHS)):
                    cosine = float(cosines[left, right])
                    pair_rows.append({
                        "method": method, "seed": int(seed), "scope": scope,
                        "batch": batch_index, "batch_size": batch_size,
                        "first_sample_id": first_id, "last_sample_id": last_id,
                        "width_i": DIAGNOSTIC_WIDTHS[left],
                        "width_j": DIAGNOSTIC_WIDTHS[right],
                        "cosine": cosine,
                        "is_conflict": bool(cosine < 0),
                        "negative_cosine": max(-cosine, 0.0),
                    })
        del matrix, vectors, teacher_logits, teacher_detached
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[gradient diagnostic] {method} seed={seed} "
            f"batch={batch_index + 1}/{len(loader)}",
            flush=True,
        )
    return pd.DataFrame(pair_rows), pd.DataFrame(norm_rows)


def summarize_pairwise(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "seed", "scope", "width_i", "width_j"]
    rows = []
    for values, group in frame.groupby(keys, sort=True):
        negative = group.loc[group["cosine"] < 0, "cosine"]
        rows.append({
            **dict(zip(keys, values)),
            "n_batches": len(group),
            "mean_cosine": float(group["cosine"].mean()),
            "median_cosine": float(group["cosine"].median()),
            "cosine_std": float(group["cosine"].std(ddof=1)),
            "conflict_rate": float(group["is_conflict"].mean()),
            "negative_cosine_mass": float(group["negative_cosine"].mean()),
            "mean_cosine_given_conflict": float(negative.mean()) if len(negative) else 0.0,
        })
    return pd.DataFrame(rows)


def summarize_norms(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "seed", "scope", "width"]
    return frame.groupby(keys, as_index=False).agg(
        n_batches=("batch", "count"),
        mean_loss=("loss", "mean"),
        mean_gradient_norm=("gradient_norm", "mean"),
        median_gradient_norm=("gradient_norm", "median"),
        gradient_norm_std=("gradient_norm", "std"),
    )


def compare_methods(summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "scope", "width_i", "width_j"]
    values = [
        "mean_cosine", "median_cosine", "conflict_rate",
        "negative_cosine_mass", "mean_cosine_given_conflict",
    ]
    uniform = summary.loc[summary["method"].eq("Uniform"), keys + values].rename(
        columns={value: f"uniform_{value}" for value in values}
    )
    puregeo = summary.loc[summary["method"].eq("PureGeo"), keys + values].rename(
        columns={value: f"puregeo_{value}" for value in values}
    )
    result = uniform.merge(puregeo, on=keys, how="inner", validate="one_to_one")
    expected = len(uniform)
    if len(result) != expected or len(puregeo) != expected:
        raise RuntimeError("Uniform/PureGeo pairwise summaries are not perfectly matched")
    for value in values:
        result[f"delta_{value}_puregeo_minus_uniform"] = (
            result[f"puregeo_{value}"] - result[f"uniform_{value}"]
        )
    return result


def _heatmap(summary: pd.DataFrame, method: str, seed: int, scope: str, path: Path) -> None:
    values = np.eye(len(DIAGNOSTIC_WIDTHS), dtype=float)
    subset = summary.loc[
        summary["method"].eq(method) & summary["seed"].eq(seed) & summary["scope"].eq(scope)
    ]
    lookup = {
        (round(row.width_i, 2), round(row.width_j, 2)): row.mean_cosine
        for row in subset.itertuples()
    }
    for left, wi in enumerate(DIAGNOSTIC_WIDTHS):
        for right, wj in enumerate(DIAGNOSTIC_WIDTHS):
            if left == right:
                continue
            key = (round(min(wi, wj), 2), round(max(wi, wj), 2))
            values[left, right] = lookup.get(key, np.nan)
    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(values, vmin=-1, vmax=1, cmap="coolwarm")
    labels = [f"{width:.2f}" for width in DIAGNOSTIC_WIDTHS]
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Width j")
    axis.set_ylabel("Width i")
    axis.set_title(f"{method} seed {seed}: mean backbone gradient cosine")
    fig.colorbar(image, ax=axis, label="cosine")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _delta_heatmap(comparison: pd.DataFrame, scope: str, path: Path) -> None:
    averaged = comparison.loc[comparison["scope"].eq(scope)].groupby(
        ["width_i", "width_j"], as_index=False
    )["delta_mean_cosine_puregeo_minus_uniform"].mean()
    values = np.zeros((len(DIAGNOSTIC_WIDTHS), len(DIAGNOSTIC_WIDTHS)), dtype=float)
    for row in averaged.itertuples():
        left = DIAGNOSTIC_WIDTHS.index(round(float(row.width_i), 2))
        right = DIAGNOSTIC_WIDTHS.index(round(float(row.width_j), 2))
        values[left, right] = values[right, left] = row.delta_mean_cosine_puregeo_minus_uniform
    limit = max(float(np.nanmax(np.abs(values))), 1e-6)
    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(values, vmin=-limit, vmax=limit, cmap="coolwarm")
    labels = [f"{width:.2f}" for width in DIAGNOSTIC_WIDTHS]
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Width j")
    axis.set_ylabel("Width i")
    axis.set_title("PureGeo − Uniform mean backbone gradient cosine")
    fig.colorbar(image, ax=axis, label="delta cosine")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    clean = frame.copy()
    for column in clean:
        clean[column] = clean[column].map(
            lambda value: f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)
        )
    header = "| " + " | ".join(map(str, clean.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(clean.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in clean.to_numpy()]
    return "\n".join([header, separator, *rows])


def _write_report(output_dir: Path, focus: pd.DataFrame, manifest: pd.DataFrame) -> None:
    primary = focus.loc[focus["scope"].eq("backbone")].copy()
    columns = [
        "seed", "width_i", "width_j", "uniform_mean_cosine", "puregeo_mean_cosine",
        "delta_mean_cosine_puregeo_minus_uniform", "uniform_conflict_rate",
        "puregeo_conflict_rate", "delta_conflict_rate_puregeo_minus_uniform",
    ]
    worsened = int((primary["delta_mean_cosine_puregeo_minus_uniform"] < 0).sum())
    report = f"""# RQ2-v1 gradient-interference diagnostic

No checkpoint was trained or updated. BN banks were recalibrated on the locked
45k-train calibration subset and then frozen. Gradients were measured on the
fixed 2,000-image validation geometry subset with batch size recorded in the CSV.

## Primary focus pairs (backbone only)

{_markdown_table(primary[columns])}

PureGeo has lower mean cosine than Uniform in {worsened}/{len(primary)} matched
seed/pair comparisons. This is descriptive mechanism evidence; interpret it
together with the accuracy trade-off rather than as a standalone causal test.

## Scope definitions

- `backbone`: conv stem, shared BN affine parameters, and residual layers; projection and classifier excluded.
- `backbone_classifier`: the primary scope plus classifier parameters; learned projection remains excluded.
- Inactive subnet parameter entries are retained as aligned zeros.
- `negative_cosine_mass` is mean `max(-cosine, 0)` over all minibatches.
- `mean_cosine_given_conflict` averages cosine only where cosine is negative.

## Checkpoint provenance

{_markdown_table(manifest[["method", "seed", "checkpoint", "sha256"]])}
"""
    (output_dir / "gradient_interference_summary.md").write_text(report)


def _checkpoint_job(
    method: str,
    seed: int,
    checkpoint: Path,
    config: dict,
    device: torch.device,
    output_dir: Path,
    batch_size: int,
):
    model = make_model(config, device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    loaders = build_confirmatory_loaders(config, training_seed=seed)
    diagnostic_loader = DataLoader(
        loaders.geometry.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    print(f"[gradient diagnostic] calibrating BN: {method} seed={seed}", flush=True)
    for width in DIAGNOSTIC_WIDTHS:
        calibrate_batch_norm(
            model, loaders.calibration, width, device,
            int(config["evaluation"]["bn_calibration_batches"]),
        )
    model.eval()
    parameter_digest_before = _tensor_state_digest(model.named_parameters())
    buffer_digest_before = _tensor_state_digest(model.named_buffers())
    pairwise, norms = diagnose_model(model, diagnostic_loader, config, device, method, seed)
    parameter_digest_after = _tensor_state_digest(model.named_parameters())
    buffer_digest_after = _tensor_state_digest(model.named_buffers())
    if parameter_digest_before != parameter_digest_after:
        raise RuntimeError(f"Diagnostic modified weights for {method} seed={seed}")
    if buffer_digest_before != buffer_digest_after:
        raise RuntimeError(f"Diagnostic modified BN buffers for {method} seed={seed}")
    target = output_dir / "workers" / f"{method.lower()}_seed_{seed}"
    target.mkdir(parents=True, exist_ok=True)
    pairwise.to_csv(target / "gradient_pairwise_by_batch.csv", index=False)
    norms.to_csv(target / "gradient_norms_by_batch.csv", index=False)
    (target / "read_only_integrity.json").write_text(json.dumps({
        "parameter_sha256_before": parameter_digest_before,
        "parameter_sha256_after": parameter_digest_after,
        "buffer_sha256_before": buffer_digest_before,
        "buffer_sha256_after": buffer_digest_after,
        "weights_unchanged": True,
        "bn_buffers_unchanged_during_gradient_measurement": True,
    }, indent=2) + "\n")
    return pairwise, norms


def run_gradient_interference(
    uniform_root: str | Path,
    rq2_root: str | Path,
    output_dir: str | Path,
    dataset_root: str | Path,
    gpu_ids: tuple[int, ...] | list[int] = (0, 1),
    batch_size: int = 64,
) -> dict:
    """Run the complete read-only diagnostic and export tables/figures."""
    uniform_root, rq2_root, output_dir = Path(uniform_root), Path(rq2_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = resolve_checkpoints(uniform_root, rq2_root)
    config_path = rq2_root / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing RQ2 resolved config: {config_path}")
    config = yaml.safe_load(config_path.read_text())
    config["dataset"]["root"] = str(dataset_root)
    config["dataset"]["download"] = True
    if int(config["dataset"]["feature_subset_size"]) != 2000:
        raise ValueError("The locked RQ2 geometry subset must contain 2,000 validation images")
    if not set(DIAGNOSTIC_WIDTHS).issubset(set(map(float, config["compression"]["eval_widths"]))):
        raise ValueError("Diagnostic widths are missing from the checkpoint model grid")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    # Download/materialize once before concurrent workers touch the dataset.
    initial_loaders = build_confirmatory_loaders(config, training_seed=1)
    geometry_subset = initial_loaders.geometry.dataset
    sample_ids = list(map(int, geometry_subset.indices))
    pd.DataFrame({"order": range(len(sample_ids)), "sample_id": sample_ids}).to_csv(
        output_dir / "diagnostic_subset_ids.csv", index=False
    )
    config["dataset"]["download"] = False
    (output_dir / "resolved_diagnostic_config.yaml").write_text(
        yaml.safe_dump({
            **config,
            "gradient_diagnostic": {
                "widths": list(DIAGNOSTIC_WIDTHS),
                "focus_pairs": [list(pair) for pair in FOCUS_PAIRS],
                "batch_size": int(batch_size),
                "weights_updated": False,
                "bn_recalibrated_before_measurement": True,
                "bn_frozen_during_measurement": True,
                "teacher_forward_per_minibatch": 1,
                "teacher_detached_for_small_width_kd": True,
                "primary_scope": "backbone",
                "sensitivity_scope": "backbone_classifier",
            },
        }, sort_keys=False)
    )

    if not gpu_ids or not torch.cuda.is_available():
        devices = [torch.device("cpu")]
    else:
        available = torch.cuda.device_count()
        invalid = [gpu for gpu in gpu_ids if gpu < 0 or gpu >= available]
        if invalid:
            raise ValueError(f"Unavailable GPU ids {invalid}; detected {available} GPUs")
        devices = [torch.device(f"cuda:{gpu}") for gpu in gpu_ids]
    device_queue: queue.Queue[torch.device] = queue.Queue()
    for device in devices:
        device_queue.put(device)

    started = time.perf_counter()
    pair_frames, norm_frames = [], []

    def launch(item):
        device = device_queue.get()
        try:
            (method, seed), checkpoint = item
            return _checkpoint_job(
                method, seed, checkpoint, config, device, output_dir, int(batch_size)
            )
        finally:
            device_queue.put(device)

    with ThreadPoolExecutor(max_workers=min(len(devices), len(checkpoints))) as executor:
        futures = [executor.submit(launch, item) for item in checkpoints.items()]
        for future in as_completed(futures):
            pairwise, norms = future.result()
            pair_frames.append(pairwise)
            norm_frames.append(norms)

    pairwise = pd.concat(pair_frames, ignore_index=True).sort_values(
        ["method", "seed", "scope", "batch", "width_i", "width_j"]
    )
    norms = pd.concat(norm_frames, ignore_index=True).sort_values(
        ["method", "seed", "scope", "batch", "width"]
    )
    pairwise.to_csv(output_dir / "gradient_pairwise_by_batch.csv", index=False)
    norms.to_csv(output_dir / "gradient_norms_by_batch.csv", index=False)
    summary = summarize_pairwise(pairwise)
    norm_summary = summarize_norms(norms)
    comparison = compare_methods(summary)
    focus = comparison.loc[
        comparison.apply(
            lambda row: (round(row.width_i, 2), round(row.width_j, 2)) in FOCUS_PAIRS,
            axis=1,
        )
    ].copy()
    summary.to_csv(output_dir / "gradient_pairwise_summary.csv", index=False)
    norm_summary.to_csv(output_dir / "gradient_norms_summary.csv", index=False)
    comparison.to_csv(output_dir / "gradient_puregeo_minus_uniform.csv", index=False)
    focus.to_csv(output_dir / "gradient_focus_pairs.csv", index=False)

    for method in ("Uniform", "PureGeo"):
        for seed in (1, 2):
            _heatmap(
                summary, method, seed, "backbone",
                output_dir / f"{method.lower()}_seed{seed}_cosine.png",
            )
    _delta_heatmap(comparison, "backbone", output_dir / "puregeo_minus_uniform_cosine.png")

    manifest = pd.DataFrame([
        {
            "method": method, "seed": seed, "checkpoint": str(path),
            "size_bytes": path.stat().st_size, "sha256": _sha256(path),
        }
        for (method, seed), path in checkpoints.items()
    ])
    manifest.to_csv(output_dir / "checkpoint_manifest.csv", index=False)
    _write_report(output_dir, focus, manifest)
    elapsed = time.perf_counter() - started
    completion = {
        "status": "complete",
        "elapsed_minutes": elapsed / 60.0,
        "checkpoints": len(checkpoints),
        "validation_samples": len(sample_ids),
        "minibatches_per_checkpoint": math.ceil(len(sample_ids) / batch_size),
        "widths": list(DIAGNOSTIC_WIDTHS),
        "weights_updated": False,
    }
    (output_dir / "gradient_interference_complete.json").write_text(
        json.dumps(completion, indent=2) + "\n"
    )
    return completion
