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
    for epoch in range(int(cfg["epochs"])):
        model.train()
        sums = {w: {"loss": 0.0, "ce": 0.0, "kd": 0.0, "correct": 0, "n": 0} for w in widths}
        for images, labels, _ in loader:
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
        scheduler.step()
    frame = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "training_metrics.csv", index=False)
    torch.save({"model": model.state_dict(), "config": config}, output_dir / "checkpoint.pt")
    return frame


def fine_tune_oracle(model, width: float, loader, config: dict, device: torch.device) -> None:
    model.set_width(width)
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]) * 0.1,
        momentum=float(config["training"]["momentum"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    for _ in range(int(config["oracle"]["epochs"])):
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            model.set_width(width)
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
