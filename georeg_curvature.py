from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from data import build_confirmatory_loaders
from evaluation import calibrate_batch_norm, evaluate_width, extract_features
from models import slimmable_resnet18
from research_utils import budget_tag, seed_everything
from training import kd_loss


ANCHORS = (0.25, 0.50, 0.75, 1.00)


def fixed_projection_directions(
    feature_dim: int, count: int, seed: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    directions = torch.randn(feature_dim, count, generator=generator)
    directions = F.normalize(directions, dim=0)
    return directions.to(device)


def curvature_components(
    features: dict[float, torch.Tensor], directions: torch.Tensor, epsilon: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tuple(sorted(features)) != ANCHORS:
        raise ValueError(f"GeoReg requires exactly the four anchors {ANCHORS}")
    quantiles = []
    for anchor in ANCHORS:
        norms = torch.linalg.vector_norm(features[anchor], dim=1, keepdim=True)
        normalized = features[anchor] / (norms + float(epsilon))
        projected = normalized @ directions
        quantiles.append(torch.sort(projected, dim=0).values)
    displacements = [quantiles[index + 1] - quantiles[index] for index in range(3)]
    curvature = ((displacements[1] - displacements[0]).square().mean() +
                 (displacements[2] - displacements[1]).square().mean())
    motion = torch.stack([value.square().mean() for value in displacements]).sum()
    normalized_curvature = curvature / (motion.detach() + float(epsilon))
    return normalized_curvature, curvature, motion


def _forward_anchor_objectives(model, images, labels, config, directions):
    features, values = {}, {}
    model.set_width(1.0)
    teacher_features = model.forward_features(images)
    teacher_logits = model.classifier(teacher_features)
    teacher_ce = F.cross_entropy(teacher_logits, labels)
    features[1.0] = teacher_features
    values[1.0] = (teacher_ce, teacher_ce, teacher_ce.new_zeros(()), teacher_logits)
    task_loss = teacher_ce
    for width in ANCHORS[:-1]:
        model.set_width(width)
        representation = model.forward_features(images)
        logits = model.classifier(representation)
        ce = F.cross_entropy(logits, labels)
        kd = kd_loss(logits, teacher_logits, float(config["training"]["kd_temperature"]))
        loss = ce + float(config["training"]["kd_lambda"]) * kd
        task_loss = task_loss + loss
        features[width] = representation
        values[width] = (loss, ce, kd, logits)
    task_loss = task_loss / len(ANCHORS)
    geo_loss, curvature, motion = curvature_components(
        features, directions, float(config["georeg"]["epsilon"])
    )
    return task_loss, geo_loss, curvature, motion, values


def _gradient_norm(grads) -> float:
    finite = [gradient.detach().float().square().sum() for gradient in grads if gradient is not None]
    if not finite:
        return 0.0
    return float(torch.sqrt(torch.stack(finite).sum()).cpu())


def _capture_rng(loader) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader_generator": loader.generator.get_state() if loader.generator else None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict, loader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if loader.generator is not None and state.get("loader_generator") is not None:
        loader.generator.set_state(state["loader_generator"])


def make_training_objects(config: dict, seed: int, device: torch.device):
    seed_everything(seed)
    model = slimmable_resnet18(
        num_classes=int(config["dataset"]["num_classes"]),
        supported_widths=config["compression"]["eval_widths"],
        projection_dim=int(config["model"]["projection_dim"]),
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        momentum=float(config["training"]["momentum"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["training"]["epochs"])
    )
    return model, optimizer, scheduler


def save_training_checkpoint(path, model, optimizer, scheduler, loader, epoch, config):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": _capture_rng(loader),
            "epoch": int(epoch),
            "config": config,
        },
        path,
    )


def load_training_checkpoint(path, model, optimizer, scheduler, loader, device):
    # Keep RNG and DataLoader-generator ByteTensors on CPU. Loading the entire
    # payload with map_location="cuda" also moves those states and makes
    # torch.set_rng_state fail before a resumed branch can start.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    scheduler.load_state_dict(payload["scheduler"])
    _restore_rng(payload["rng"], loader)
    return int(payload["epoch"])


def train_segment(
    model,
    optimizer,
    scheduler,
    loader,
    config: dict,
    device: torch.device,
    directions: torch.Tensor,
    start_epoch: int,
    end_epoch: int,
    lambda_geo: float,
    output_dir: str | Path,
    gradient_diagnostic_batches: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Train inclusive 1-based epochs, preserving exact resumable state."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    records, gradients = [], []
    projection_params = list(model.projection.parameters())
    last_block_params = list(model.layer4[-1].parameters())
    diagnostic_count = 0
    for epoch in range(int(start_epoch), int(end_epoch) + 1):
        model.train()
        sums = {
            width: {"loss": 0.0, "ce": 0.0, "kd": 0.0, "correct": 0, "n": 0}
            for width in ANCHORS
        }
        geo_sum = curvature_sum = motion_sum = task_sum = 0.0
        sample_count = 0
        for batch_index, (images, labels, _) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            task_loss, geo_loss, curvature, motion, values = _forward_anchor_objectives(
                model, images, labels, config, directions
            )
            if diagnostic_count < int(gradient_diagnostic_batches):
                joint_params = projection_params + last_block_params
                task_grads = torch.autograd.grad(
                    task_loss, joint_params, retain_graph=True, allow_unused=True
                )
                geo_grads = torch.autograd.grad(
                    geo_loss, joint_params, retain_graph=True, allow_unused=True
                )
                split = len(projection_params)
                task_projection = _gradient_norm(task_grads[:split])
                geo_projection = _gradient_norm(geo_grads[:split])
                task_block = _gradient_norm(task_grads[split:])
                geo_block = _gradient_norm(geo_grads[split:])
                gradients.append(
                    {
                        "epoch": epoch,
                        "batch": batch_index,
                        "task_projection_grad_norm": task_projection,
                        "geo_projection_grad_norm": geo_projection,
                        "projection_ratio": task_projection / max(geo_projection, 1e-12),
                        "task_last_block_grad_norm": task_block,
                        "geo_last_block_grad_norm": geo_block,
                        "last_block_ratio": task_block / max(geo_block, 1e-12),
                    }
                )
                diagnostic_count += 1
            total_loss = task_loss + float(lambda_geo) * geo_loss
            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError(
                    f"Non-finite GeoReg loss at epoch={epoch}, batch={batch_index}"
                )
            total_loss.backward()
            optimizer.step()
            n = labels.numel(); sample_count += n
            task_sum += float(task_loss.detach()) * n
            geo_sum += float(geo_loss.detach()) * n
            curvature_sum += float(curvature.detach()) * n
            motion_sum += float(motion.detach()) * n
            for width, (loss, ce, kd, logits) in values.items():
                item = sums[width]
                item["loss"] += float(loss.detach()) * n
                item["ce"] += float(ce.detach()) * n
                item["kd"] += float(kd.detach()) * n
                item["correct"] += int((logits.argmax(1) == labels).sum())
                item["n"] += n
        current_lr = optimizer.param_groups[0]["lr"]
        for width in ANCHORS:
            item = sums[width]
            records.append(
                {
                    "epoch": epoch,
                    "width": width,
                    "loss": item["loss"] / item["n"],
                    "ce_loss": item["ce"] / item["n"],
                    "kd_loss": item["kd"] / item["n"],
                    "accuracy": item["correct"] / item["n"],
                    "task_loss": task_sum / sample_count,
                    "geo_loss": geo_sum / sample_count,
                    "curvature": curvature_sum / sample_count,
                    "motion": motion_sum / sample_count,
                    "lambda_geo": float(lambda_geo),
                    "learning_rate": current_lr,
                }
            )
        scheduler.step()
        pd.DataFrame(records).to_csv(output_dir / "training_metrics.csv", index=False)
        pd.DataFrame(gradients).to_csv(output_dir / "gradient_diagnostics.csv", index=False)
        accuracy_log = ", ".join(
            f"w={width:.2f}: acc={sums[width]['correct'] / sums[width]['n']:.4f}"
            for width in ANCHORS
        )
        print(
            f"epoch {epoch}/{config['training']['epochs']} | task={task_sum/sample_count:.4f}, "
            f"geo={geo_sum/sample_count:.4f}, lambda={lambda_geo:.6g} | {accuracy_log}",
            flush=True,
        )
    checkpoint = output_dir / f"checkpoint_epoch_{end_epoch:02d}.pt"
    save_training_checkpoint(
        checkpoint, model, optimizer, scheduler, loader, end_epoch, config
    )
    return pd.DataFrame(records), pd.DataFrame(gradients), checkpoint


@torch.no_grad()
def evaluate_anchor_checkpoint(
    checkpoint_path: str | Path,
    config: dict,
    seed: int,
    directions: torch.Tensor,
    output_dir: str | Path,
    device: torch.device,
) -> tuple[pd.DataFrame, dict]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    loaders = build_confirmatory_loaders(config, training_seed=seed)
    seed_everything(seed)
    model = slimmable_resnet18(
        num_classes=int(config["dataset"]["num_classes"]),
        supported_widths=config["compression"]["eval_widths"],
        projection_dim=int(config["model"]["projection_dim"]),
    ).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    feature_bank, rows = {}, []
    for width in ANCHORS:
        calibrate_batch_norm(
            model, loaders.calibration, width, device,
            int(config["evaluation"]["bn_calibration_batches"]),
        )
        metrics = evaluate_width(model, loaders.validation, width, device)
        extracted = extract_features(
            model, loaders.geometry, width, device, "l2",
            output_dir / "features" / f"features_budget_{budget_tag(width)}.pt",
        )
        feature_bank[width] = extracted["features"].to(device)
        rows.append({"seed": seed, "width": width, **metrics})
    dgeo, curvature, motion = curvature_components(
        feature_bank, directions.to(device), float(config["georeg"]["epsilon"])
    )
    geometry = {
        "D_geo": float(dgeo.cpu()),
        "L_curvature": float(curvature.cpu()),
        "L_motion": float(motion.cpu()),
    }
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "anchor_validation_metrics.csv", index=False)
    (output_dir / "anchor_geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")
    return table, geometry


def load_anchor_feature_bank(root: str | Path, seed: int | None = None):
    root = Path(root)
    feature_dir = root / f"seed_{seed}" / "features" if seed is not None else root / "features"
    bank, sample_ids = {}, None
    for width in ANCHORS:
        payload = torch.load(
            feature_dir / f"features_budget_{budget_tag(width)}.pt",
            map_location="cpu", weights_only=False,
        )
        ids = payload["sample_ids"].long()
        if sample_ids is None:
            sample_ids = ids
        elif not torch.equal(sample_ids, ids):
            raise RuntimeError("Anchor feature IDs/order differ")
        bank[width] = payload["features"].float()
    return bank, sample_ids


def _project_bank(bank, directions, epsilon):
    directions = directions.cpu()
    projected = {}
    for width, features in bank.items():
        normalized = features / (
            torch.linalg.vector_norm(features, dim=1, keepdim=True) + float(epsilon)
        )
        projected[width] = (normalized @ directions).numpy()
    return projected


def _curvature_from_projected(projected: dict[float, np.ndarray], indices=None, epsilon=1e-8):
    quantiles = []
    for width in ANCHORS:
        values = projected[width] if indices is None else projected[width][indices]
        quantiles.append(np.sort(values, axis=0))
    displacements = [quantiles[index + 1] - quantiles[index] for index in range(3)]
    curvature = np.mean((displacements[1] - displacements[0]) ** 2) + np.mean(
        (displacements[2] - displacements[1]) ** 2
    )
    motion = np.sum([np.mean(value ** 2) for value in displacements])
    return float(curvature / (motion + epsilon)), float(curvature), float(motion)


def paired_curvature_bootstrap(
    baseline_bank,
    candidate_bank,
    directions,
    replicates: int,
    seed: int,
    epsilon: float = 1e-8,
):
    baseline_projected = _project_bank(baseline_bank, directions, epsilon)
    candidate_projected = _project_bank(candidate_bank, directions, epsilon)
    count = next(iter(baseline_projected.values())).shape[0]
    if any(value.shape[0] != count for value in candidate_projected.values()):
        raise ValueError("Baseline/candidate feature counts differ")
    baseline_point = _curvature_from_projected(baseline_projected, epsilon=epsilon)[0]
    candidate_point = _curvature_from_projected(candidate_projected, epsilon=epsilon)[0]
    rng = np.random.default_rng(seed)
    deltas = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        indices = rng.integers(0, count, count)
        baseline_value = _curvature_from_projected(
            baseline_projected, indices, epsilon
        )[0]
        candidate_value = _curvature_from_projected(
            candidate_projected, indices, epsilon
        )[0]
        deltas[replicate] = candidate_value - baseline_value
    return {
        "baseline_D_geo": baseline_point,
        "candidate_D_geo": candidate_point,
        "delta_D_geo": candidate_point - baseline_point,
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "bootstrap_replicates": int(replicates),
    }


def paired_accuracy_bootstrap(
    baseline_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
    unseen_widths: list[float],
    replicates: int,
    seed: int,
):
    def matrix(frame):
        selected = frame.loc[frame["budget"].round(2).isin(unseen_widths)]
        return selected.pivot(index="sample_id", columns="budget", values="correct").sort_index()
    baseline = matrix(baseline_predictions)
    candidate = matrix(candidate_predictions)
    if not baseline.index.equals(candidate.index) or not baseline.columns.equals(candidate.columns):
        raise RuntimeError("Prediction IDs/budgets are not paired")
    baseline_values = baseline.to_numpy(float)
    candidate_values = candidate.to_numpy(float)
    point = float(candidate_values.mean() - baseline_values.mean())
    rng = np.random.default_rng(seed)
    deltas = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        indices = rng.integers(0, len(baseline_values), len(baseline_values))
        deltas[replicate] = candidate_values[indices].mean() - baseline_values[indices].mean()
    return {
        "delta_mean_unseen_accuracy": point,
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "bootstrap_replicates": int(replicates),
    }
