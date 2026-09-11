from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from profiling import profile_subnet


def kd_loss(student: Tensor, teacher: Tensor, temperature: float) -> Tensor:
    t = float(temperature)
    return F.kl_div(
        F.log_softmax(student / t, dim=1),
        F.softmax(teacher.detach() / t, dim=1),
        reduction="batchmean",
    ) * (t * t)


def train_shared_model(model, loader, config: dict, device: torch.device, output_dir: Path) -> pd.DataFrame:
    cfg = config["training"]
    widths = [float(w) for w in config["compression"]["train_widths"]]
    width_loss_reduction = cfg.get("width_loss_reduction", "mean")
    if width_loss_reduction not in {"mean", "sum"}:
        raise ValueError("training.width_loss_reduction must be 'mean' or 'sum'")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]))
    ce_fn = nn.CrossEntropyLoss()
    profiles = {w: profile_subnet(model, w) for w in widths}
    records: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(int(cfg["epochs"])):
        model.train()
        sums = {w: {"loss": 0.0, "ce": 0.0, "kd": 0.0, "correct": 0, "n": 0} for w in widths}
        for batch_idx, (images, labels, _) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            model.set_width(1.0)
            teacher_logits = model(images)
            teacher_ce = ce_fn(teacher_logits, labels)
            total_loss = teacher_ce
            batch_values = {1.0: (teacher_ce, teacher_ce, teacher_ce.new_zeros(()), teacher_logits)}
            for width in widths:
                if width == 1.0:
                    continue
                model.set_width(width)
                logits = model(images)
                ce = ce_fn(logits, labels)
                kd = kd_loss(logits, teacher_logits, float(cfg["kd_temperature"]))
                loss = ce + float(cfg["kd_lambda"]) * kd
                total_loss = total_loss + loss
                batch_values[width] = (loss, ce, kd, logits)
            # The configured objective is the expectation over training widths.
            # Averaging preserves that objective while preventing the effective
            # step size from growing linearly with the number of anchors.
            if width_loss_reduction == "mean":
                total_loss = total_loss / len(widths)
            if not bool(torch.isfinite(total_loss)):
                details = {
                    width: {
                        "loss": float(loss.detach()),
                        "ce": float(ce.detach()),
                        "kd": float(kd.detach()),
                    }
                    for width, (loss, ce, kd, _) in batch_values.items()
                }
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={epoch}, batch={batch_idx}: {details}"
                )
            total_loss.backward()
            optimizer.step()
            for width, (loss, ce, kd, logits) in batch_values.items():
                item = sums[width]
                n = labels.numel()
                item["loss"] += float(loss.detach()) * n
                item["ce"] += float(ce.detach()) * n
                item["kd"] += float(kd.detach()) * n
                item["correct"] += int((logits.argmax(1) == labels).sum())
                item["n"] += n
        lr = optimizer.param_groups[0]["lr"]
        for width in widths:
            item = sums[width]
            macs, params = profiles[width]
            records.append(
                {
                    "epoch": epoch,
                    "width": width,
                    "loss": item["loss"] / item["n"],
                    "ce_loss": item["ce"] / item["n"],
                    "kd_loss": item["kd"] / item["n"],
                    "accuracy": item["correct"] / item["n"],
                    "learning_rate": lr,
                    "flops": macs,
                    "params": params,
                }
            )
        # Persist partial diagnostics each epoch and expose progress in remote logs.
        pd.DataFrame(records).to_csv(output_dir / "training_metrics.csv", index=False)
        epoch_rows = records[-len(widths) :]
        summary = ", ".join(
            f"w={row['width']:.2f}: loss={row['loss']:.4f}, acc={row['accuracy']:.4f}"
            for row in epoch_rows
        )
        print(f"epoch {epoch + 1}/{int(cfg['epochs'])} | {summary}", flush=True)
        scheduler.step()
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "training_metrics.csv", index=False)
    torch.save({"model": model.state_dict(), "config": config}, output_dir / "checkpoint.pt")
    return frame


def fine_tune_oracle(
    model, width: float, loader, config: dict, device: torch.device
) -> pd.DataFrame:
    model.set_width(width)
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]) * 0.1,
        momentum=float(config["training"]["momentum"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    oracle_epochs = int(config["oracle"]["epochs"])
    records: list[dict] = []
    for epoch in range(oracle_epochs):
        loss_sum = 0.0
        count = 0
        correct = 0
        for batch_idx, (images, labels, _) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            model.set_width(width)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite oracle loss at width={width:g}, epoch={epoch}, batch={batch_idx}"
                )
            loss.backward()
            optimizer.step()
            n = labels.numel()
            loss_sum += float(loss.detach()) * n
            correct += int((logits.argmax(1) == labels).sum())
            count += n
        epoch_loss = loss_sum / count
        epoch_accuracy = correct / count
        records.append(
            {
                "epoch": epoch,
                "width": width,
                "loss": epoch_loss,
                "accuracy": epoch_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"oracle width={width:.2f} epoch {epoch + 1}/{oracle_epochs} | "
            f"loss={epoch_loss:.4f}, acc={epoch_accuracy:.4f}",
            flush=True,
        )
    return pd.DataFrame(records)
