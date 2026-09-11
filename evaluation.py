from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

from models.slimmable_ops import SlimmableBatchNorm2d
from profiling import profile_subnet
from research_utils import budget_tag


@torch.no_grad()
def calibrate_batch_norm(model, loader, width: float, device: torch.device, max_batches: int) -> None:
    model.set_width(width)
    model.train()
    banks = []
    for module in model.modules():
        if isinstance(module, SlimmableBatchNorm2d):
            bank = module._bank()
            module.reset_current_stats()
            banks.append((bank, bank.momentum))
            bank.momentum = None
    for batch_idx, (images, _, _) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        logits = model(images.to(device))
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError(
                f"Non-finite logits during BN calibration at width={width:g}, batch={batch_idx}"
            )
    for bank, momentum in banks:
        bank.momentum = momentum
    model.eval()


@torch.no_grad()
def evaluate_width(model, loader, width: float, device: torch.device) -> dict[str, float]:
    model.set_width(width)
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for batch_idx, (images, labels, _) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError(
                f"Non-finite evaluation logits at width={width:g}, batch={batch_idx}"
            )
        loss = F.cross_entropy(logits, labels, reduction="sum")
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"Non-finite evaluation loss at width={width:g}, batch={batch_idx}"
            )
        loss_sum += float(loss)
        correct += int((logits.argmax(1) == labels).sum())
        count += labels.numel()
    return {"accuracy": correct / count, "loss": loss_sum / count}


@torch.no_grad()
def extract_features(
    model,
    loader,
    width: float,
    device: torch.device,
    normalize: str,
    output_path: Path,
) -> dict:
    model.set_width(width)
    model.eval()
    features, labels, sample_ids = [], [], []
    for batch_idx, (images, target, ids) in enumerate(loader):
        z = model.forward_features(images.to(device)).cpu()
        if not bool(torch.isfinite(z).all()):
            raise FloatingPointError(
                f"Non-finite extracted features at width={width:g}, batch={batch_idx}"
            )
        if normalize == "l2":
            z = F.normalize(z, p=2, dim=1)
        elif normalize != "raw":
            raise ValueError("features.normalize must be 'raw' or 'l2'")
        features.append(z)
        labels.append(target.cpu())
        sample_ids.append(torch.as_tensor(ids).cpu())
    payload = {
        "features": torch.cat(features),
        "labels": torch.cat(labels),
        "sample_ids": torch.cat(sample_ids),
        "budget": float(width),
        "normalization": normalize,
    }
    if float(payload["features"].std()) <= 1e-8:
        raise FloatingPointError(f"Degenerate near-constant features at width={width:g}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return payload


def evaluate_grid(
    model,
    train_loader,
    val_loader,
    feature_loader,
    config: dict,
    device: torch.device,
    seed: int,
    output_dir: Path,
) -> pd.DataFrame:
    widths = [float(w) for w in config["compression"]["eval_widths"]]
    anchors = set(float(w) for w in config["compression"]["train_widths"])
    records = []
    reference_ids = None
    for width in widths:
        calibrate_batch_norm(
            model,
            train_loader,
            width,
            device,
            int(config["evaluation"]["bn_calibration_batches"]),
        )
        metrics = evaluate_width(model, val_loader, width, device)
        macs, params = profile_subnet(model, width)
        payload = extract_features(
            model,
            feature_loader,
            width,
            device,
            config["features"]["normalize"],
            output_dir / "features" / f"features_budget_{budget_tag(width)}.pt",
        )
        if reference_ids is None:
            reference_ids = payload["sample_ids"]
        elif not torch.equal(reference_ids, payload["sample_ids"]):
            raise RuntimeError("Feature sample IDs/order changed across budgets")
        records.append(
            {
                "seed": seed,
                "budget": width,
                "is_train_anchor": width in anchors,
                **metrics,
                "flops": macs,
                "params": params,
            }
        )
    frame = pd.DataFrame(records)
    result_dir = output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(result_dir / "budget_metrics.csv", index=False)
    return frame
