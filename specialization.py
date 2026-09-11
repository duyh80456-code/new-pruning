from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

from evaluation import calibrate_batch_norm, evaluate_width
from profiling import profile_subnet
from research_utils import budget_tag, seed_everything


def select_best_specialized_checkpoint(
    shared_model,
    width: float,
    train_loader,
    validation_loader,
    calibration_loader,
    config: dict,
    device: torch.device,
    seed: int,
    output_dir: Path,
) -> dict:
    """Fine-tune one width and select a checkpoint without accessing test data."""
    seed_everything(seed)
    if getattr(train_loader, "generator", None) is not None:
        train_loader.generator.manual_seed(seed)
    model = copy.deepcopy(shared_model).to(device)
    model.set_width(width)
    epochs = int(config["specialization"]["epochs"])
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config["specialization"]["learning_rate"]),
        momentum=float(config["training"]["momentum"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    checkpoint_dir = output_dir / "specialized"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"best_budget_{budget_tag(width)}.pt"

    calibrate_batch_norm(
        model,
        calibration_loader,
        width,
        device,
        int(config["evaluation"]["bn_calibration_batches"]),
    )
    initial_validation = evaluate_width(model, validation_loader, width, device)
    best = {
        "epoch": 0,
        "validation_accuracy": initial_validation["accuracy"],
        "validation_loss": initial_validation["loss"],
    }
    torch.save(
        {"model": model.state_dict(), "width": width, "seed": seed, "best": best},
        checkpoint_path,
    )
    history = [
        {
            "epoch": 0,
            "width": width,
            "train_loss": float("nan"),
            "train_accuracy": float("nan"),
            "validation_loss": initial_validation["loss"],
            "validation_accuracy": initial_validation["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "is_best": True,
        }
    ]

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        learning_rate = optimizer.param_groups[0]["lr"]
        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            model.set_width(width)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite specialization loss width={width:g}, "
                    f"epoch={epoch}, batch={batch_idx}"
                )
            loss.backward()
            optimizer.step()
            n = labels.numel()
            loss_sum += float(loss.detach()) * n
            correct += int((logits.argmax(1) == labels).sum())
            count += n
        scheduler.step()
        calibrate_batch_norm(
            model,
            calibration_loader,
            width,
            device,
            int(config["evaluation"]["bn_calibration_batches"]),
        )
        validation = evaluate_width(model, validation_loader, width, device)
        is_best = validation["accuracy"] > best["validation_accuracy"]
        if is_best:
            best = {
                "epoch": epoch,
                "validation_accuracy": validation["accuracy"],
                "validation_loss": validation["loss"],
            }
            torch.save(
                {"model": model.state_dict(), "width": width, "seed": seed, "best": best},
                checkpoint_path,
            )
        history.append(
            {
                "epoch": epoch,
                "width": width,
                "train_loss": loss_sum / count,
                "train_accuracy": correct / count,
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "learning_rate": learning_rate,
                "is_best": is_best,
            }
        )
        print(
            f"specialized width={width:.2f} epoch {epoch}/{epochs} | "
            f"train_loss={loss_sum / count:.4f}, train_acc={correct / count:.4f}, "
            f"val_acc={validation['accuracy']:.4f}, best_epoch={best['epoch']}",
            flush=True,
        )

    history_frame = pd.DataFrame(history)
    result_dir = output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    history_frame.to_csv(
        result_dir / f"specialization_history_budget_{budget_tag(width)}.csv", index=False
    )
    return {"checkpoint": checkpoint_path, **best}


def select_all_specialized_checkpoints(
    shared_model,
    loaders,
    config: dict,
    device: torch.device,
    seed: int,
    output_dir: Path,
) -> pd.DataFrame:
    records = []
    anchors = {float(width) for width in config["compression"]["train_widths"]}
    for width_value in config["specialization"]["widths"]:
        width = float(width_value)
        if width in anchors:
            raise ValueError(f"Specialized width {width} is a training anchor")
        selected = select_best_specialized_checkpoint(
            shared_model,
            width,
            loaders.train,
            loaders.validation,
            loaders.calibration,
            config,
            device,
            seed,
            output_dir,
        )
        records.append({"seed": seed, "width": width, **selected})
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "results" / "selected_specialized_checkpoints.csv", index=False)
    return frame


def evaluate_selected_specialized_models(
    shared_model,
    selected: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    test_loader,
    calibration_loader,
    config: dict,
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    """Evaluate test data only after validation-based checkpoint selection is complete."""
    records = []
    for row in selected.itertuples(index=False):
        width = float(row.width)
        payload = torch.load(row.checkpoint, map_location=device, weights_only=False)
        model = copy.deepcopy(shared_model).to(device)
        model.load_state_dict(payload["model"])
        calibrate_batch_norm(
            model,
            calibration_loader,
            width,
            device,
            int(config["evaluation"]["bn_calibration_batches"]),
        )
        specialized_metrics = evaluate_width(model, test_loader, width, device)
        baseline = baseline_metrics.loc[
            (baseline_metrics["budget"].astype(float) - width).abs() < 1e-8
        ].iloc[0]
        flops, params = profile_subnet(model, width)
        records.append(
            {
                "seed": int(row.seed),
                "width": width,
                "shared_test_accuracy": float(baseline["accuracy"]),
                "specialized_test_accuracy": specialized_metrics["accuracy"],
                "specialization_gap": specialized_metrics["accuracy"]
                - float(baseline["accuracy"]),
                "specialized_test_loss": specialized_metrics["loss"],
                "best_epoch": int(row.epoch),
                "best_validation_accuracy": float(row.validation_accuracy),
                "best_validation_loss": float(row.validation_loss),
                "flops": flops,
                "params": params,
            }
        )
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "results" / "specialized_metrics.csv", index=False)
    return frame
