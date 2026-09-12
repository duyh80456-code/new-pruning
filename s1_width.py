from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.nn import functional as F

from evaluation import calibrate_batch_norm, evaluate_width, profile_subnet
from models import slimmable_resnet18
from research_utils import budget_tag, seed_everything
from training import kd_loss
from geometry.distances import compute_distribution_distance


ANCHORS = (0.25, 0.50, 0.75, 1.00)
SPECIALIZED_WIDTHS = (0.30, 0.40, 0.60, 0.80)
REPRESENTATIONS = ("learned_projection", "backbone_padded", "fixed_random_projection")


def read_s0_selection(path: str | Path) -> dict:
    """Read a passed S0 decision without silently falling back to 20 epochs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"S0 selection artifact not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        payload = yaml.safe_load(path.read_text())
    else:
        payload = json.loads(path.read_text())
    passed = payload.get("s0_pass", payload.get("passed", payload.get("pass")))
    horizon = payload.get("selected_horizon", payload.get("horizon", payload.get("epochs")))
    if passed is not True:
        raise RuntimeError("S1 is locked: the supplied S0 artifact does not record s0_pass=true")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("S0 artifact must contain a positive integer selected_horizon")
    return {"s0_pass": True, "selected_horizon": int(horizon), "source": str(path)}


def fixed_random_projection(in_dim: int, out_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    matrix = torch.randn(int(in_dim), int(out_dim), generator=generator)
    return F.normalize(matrix, p=2, dim=0)


def _capture_rng(loader) -> dict:
    result = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader_generator": loader.generator.get_state() if loader.generator else None,
    }
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def _restore_rng(state: dict, loader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if loader.generator is not None and state.get("loader_generator") is not None:
        loader.generator.set_state(state["loader_generator"])


def _save_state(path, model, optimizer, scheduler, loader, epoch, extra=None):
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "rng": _capture_rng(loader),
        "epoch": int(epoch), **(extra or {}),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_state(path, model, optimizer, scheduler, loader, device) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    scheduler.load_state_dict(payload["scheduler"])
    _restore_rng(payload["rng"], loader)
    return payload


def make_model(config: dict, device: torch.device):
    return slimmable_resnet18(
        num_classes=int(config["dataset"]["num_classes"]),
        supported_widths=config["compression"]["eval_widths"],
        projection_dim=int(config["model"]["projection_dim"]),
    ).to(device)


def train_shared_reference(model, loader, config, device, output_dir: Path) -> Path:
    """Anchor-only CE+KD training with epoch-level exact resume."""
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path, last_path = output_dir / "checkpoint.pt", output_dir / "last_checkpoint.pt"
    if final_path.is_file():
        return final_path
    cfg = config["training"]
    epochs = int(cfg["epochs"])
    optimizer = torch.optim.SGD(
        model.parameters(), lr=float(cfg["learning_rate"]),
        momentum=float(cfg["momentum"]), weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    start_epoch = 1
    if last_path.is_file():
        start_epoch = int(_load_state(last_path, model, optimizer, scheduler, loader, device)["epoch"]) + 1
    records_path = output_dir / "training_metrics.csv"
    records = pd.read_csv(records_path).to_dict("records") if records_path.is_file() else []
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        sums = {width: {"loss": 0.0, "correct": 0, "n": 0} for width in ANCHORS}
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            model.set_width(1.0)
            teacher = model(images)
            teacher_ce = F.cross_entropy(teacher, labels)
            values = {1.0: (teacher_ce, teacher)}
            total = teacher_ce
            for width in ANCHORS[:-1]:
                model.set_width(width)
                logits = model(images)
                loss = F.cross_entropy(logits, labels) + float(cfg["kd_lambda"]) * kd_loss(
                    logits, teacher, float(cfg["kd_temperature"])
                )
                total = total + loss
                values[width] = (loss, logits)
            total = total / len(ANCHORS)
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite shared loss at seed epoch {epoch}")
            total.backward(); optimizer.step()
            n = labels.numel()
            for width, (loss, logits) in values.items():
                sums[width]["loss"] += float(loss.detach()) * n
                sums[width]["correct"] += int((logits.argmax(1) == labels).sum())
                sums[width]["n"] += n
        lr = optimizer.param_groups[0]["lr"]
        for width in ANCHORS:
            item = sums[width]
            records.append({
                "epoch": epoch, "width": width, "loss": item["loss"] / item["n"],
                "accuracy": item["correct"] / item["n"], "learning_rate": lr,
            })
        scheduler.step()
        pd.DataFrame(records).to_csv(records_path, index=False)
        _save_state(last_path, model, optimizer, scheduler, loader, epoch)
        status = ", ".join(
            f"w={width:.2f}: acc={sums[width]['correct']/sums[width]['n']:.4f}"
            for width in ANCHORS
        )
        print(f"shared epoch {epoch}/{epochs} | {status}", flush=True)
    _save_state(final_path, model, optimizer, scheduler, loader, epochs)
    return final_path


def train_specialized_reference(model, width, loaders, config, device, seed, output_dir: Path) -> Path:
    """Train an independently initialized fixed-width reference and select by validation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = output_dir / "best_checkpoint.pt", output_dir / "last_checkpoint.pt"
    complete_path = output_dir / "training_complete.json"
    if complete_path.is_file() and best_path.is_file():
        return best_path
    cfg = config["training"]
    epochs = int(cfg["epochs"])
    optimizer = torch.optim.SGD(
        model.parameters(), lr=float(cfg["learning_rate"]),
        momentum=float(cfg["momentum"]), weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history_path = output_dir / "training_metrics.csv"
    history = pd.read_csv(history_path).to_dict("records") if history_path.is_file() else []
    best = {"epoch": -1, "validation_accuracy": -1.0, "validation_loss": float("inf")}
    start_epoch = 1
    if last_path.is_file():
        payload = _load_state(last_path, model, optimizer, scheduler, loaders.train, device)
        start_epoch = int(payload["epoch"]) + 1
        best = payload["best"]
    for epoch in range(start_epoch, epochs + 1):
        model.train(); model.set_width(width)
        loss_sum = correct = count = 0
        for images, labels, _ in loaders.train:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True); model.set_width(width)
            logits = model(images); loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite specialized loss width={width}, epoch={epoch}")
            loss.backward(); optimizer.step()
            n = labels.numel(); count += n
            loss_sum += float(loss.detach()) * n
            correct += int((logits.argmax(1) == labels).sum())
        lr = optimizer.param_groups[0]["lr"]; scheduler.step()
        calibrate_batch_norm(
            model, loaders.calibration, width, device,
            int(config["evaluation"]["bn_calibration_batches"]),
        )
        validation = evaluate_width(model, loaders.validation, width, device)
        improved = (
            validation["accuracy"] > best["validation_accuracy"]
            or (
                validation["accuracy"] == best["validation_accuracy"]
                and validation["loss"] < best["validation_loss"]
            )
        )
        if improved:
            best = {"epoch": epoch, "validation_accuracy": validation["accuracy"],
                    "validation_loss": validation["loss"]}
            torch.save({"model": model.state_dict(), "seed": seed, "width": float(width),
                        "best": best}, best_path)
        history.append({
            "seed": seed, "width": float(width), "epoch": epoch,
            "train_loss": loss_sum / count, "train_accuracy": correct / count,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"], "learning_rate": lr,
            "is_best": improved,
        })
        pd.DataFrame(history).to_csv(history_path, index=False)
        _save_state(last_path, model, optimizer, scheduler, loaders.train, epoch, {"best": best})
        print(
            f"specialized seed={seed} width={width:.2f} epoch {epoch}/{epochs} | "
            f"train_acc={correct/count:.4f}, val_acc={validation['accuracy']:.4f}, "
            f"best={best['epoch']}", flush=True,
        )
    complete_path.write_text(json.dumps(best, indent=2) + "\n")
    return best_path


@torch.no_grad()
def extract_representation_views(model, loader, width, device, random_matrix, output_dir):
    output_dir = Path(output_dir)
    values = {name: [] for name in REPRESENTATIONS}; labels = []; sample_ids = []
    model.set_width(width); model.eval(); random_matrix = random_matrix.to(device)
    for images, target, ids in loader:
        backbone = model.forward_backbone_features(images.to(device))
        learned = model.projection(backbone)
        padded = F.pad(backbone, (0, 512 - backbone.shape[1]))
        views = {
            "learned_projection": learned,
            "backbone_padded": padded,
            "fixed_random_projection": padded @ random_matrix,
        }
        for name, tensor in views.items():
            values[name].append(F.normalize(tensor, p=2, dim=1).cpu())
        labels.append(target.cpu()); sample_ids.append(torch.as_tensor(ids).cpu())
    labels = torch.cat(labels); sample_ids = torch.cat(sample_ids)
    for name in REPRESENTATIONS:
        path = output_dir / name / f"features_budget_{budget_tag(width)}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": torch.cat(values[name]), "labels": labels,
                    "sample_ids": sample_ids, "budget": float(width),
                    "normalization": "l2"}, path)


def evaluate_shared_checkpoint(checkpoint_path, loaders, config, device, seed, output_dir, random_matrix):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    model = make_model(config, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["model"])
    rows = []
    for width in map(float, config["compression"]["eval_widths"]):
        calibrate_batch_norm(model, loaders.calibration, width, device,
                             int(config["evaluation"]["bn_calibration_batches"]))
        metrics = evaluate_width(model, loaders.test, width, device)
        flops, params = profile_subnet(model, width)
        extract_representation_views(model, loaders.geometry, width, device,
                                     random_matrix, output_dir / "representations")
        rows.append({"seed": seed, "budget": width, "is_train_anchor": width in ANCHORS,
                     **metrics, "flops": flops, "params": params})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "budget_metrics.csv", index=False)
    analyze_representation_views(output_dir / "representations", config, seed, output_dir)
    return frame


def analyze_representation_views(feature_root, config, seed, output_dir):
    budgets = list(map(float, config["compression"]["eval_widths"]))
    rows = []
    for view in REPRESENTATIONS:
        features = []
        reference_ids = None
        for width in budgets:
            payload = torch.load(Path(feature_root) / view /
                                 f"features_budget_{budget_tag(width)}.pt",
                                 map_location="cpu", weights_only=False)
            if reference_ids is None:
                reference_ids = payload["sample_ids"]
            elif not torch.equal(reference_ids, payload["sample_ids"]):
                raise RuntimeError("Representation sample IDs/order changed across widths")
            features.append(payload["features"].numpy())
        jumps = [
            compute_distribution_distance(
                features[index], features[index + 1], method="sliced_wasserstein",
                num_projections=int(config["geometry"]["num_projections"]),
                seed=int(config["geometry"]["projection_seed"]),
            )
            for index in range(len(budgets) - 1)
        ]
        view_dir = Path(output_dir) / "geometry" / view; view_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"budget_start": budgets[:-1], "budget_end": budgets[1:],
                      "wasserstein_jump": jumps}).to_csv(
            view_dir / "adjacent_wasserstein.csv", index=False
        )
        for index, delta in enumerate(np.diff(budgets)):
            rows.append({"seed": seed, "representation": view,
                         "budget_start": budgets[index], "budget_end": budgets[index + 1],
                         "wasserstein_jump": jumps[index],
                         "G": jumps[index] / delta})
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(output_dir) / "representation_local_geometry.csv", index=False)
    return frame


def evaluate_specialized_checkpoints(root, loaders, config, device, seed):
    shared = pd.read_csv(Path(root) / "shared" / f"seed_{seed}" / "evaluation" / "budget_metrics.csv")
    rows = []
    for width in SPECIALIZED_WIDTHS:
        checkpoint = Path(root) / "specialized" / f"seed_{seed}" / f"width_{budget_tag(width)}" / "best_checkpoint.pt"
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model = make_model(config, device); model.load_state_dict(payload["model"])
        calibrate_batch_norm(model, loaders.calibration, width, device,
                             int(config["evaluation"]["bn_calibration_batches"]))
        metrics = evaluate_width(model, loaders.test, width, device)
        base = shared.loc[np.isclose(shared["budget"], width)].iloc[0]
        flops, params = profile_subnet(model, width)
        rows.append({
            "seed": seed, "width": width, "shared_test_accuracy": float(base["accuracy"]),
            "specialized_test_accuracy": metrics["accuracy"],
            "specialization_gap": metrics["accuracy"] - float(base["accuracy"]),
            "best_epoch": int(payload["best"]["epoch"]),
            "best_validation_accuracy": float(payload["best"]["validation_accuracy"]),
            "flops": flops, "params": params,
        })
    frame = pd.DataFrame(rows)
    path = Path(root) / "specialized" / f"seed_{seed}" / "specialized_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(path, index=False)
    return frame


def _safe_corr(x, y, kind):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float((pearsonr(x, y) if kind == "pearson" else spearmanr(x, y)).statistic)


def finalize_s1(root: str | Path, config: dict) -> dict:
    root = Path(root); seeds = list(map(int, config["experiment"]["seeds"]))
    shared = pd.concat([
        pd.read_csv(root / "shared" / f"seed_{seed}" / "evaluation" / "budget_metrics.csv")
        for seed in seeds
    ], ignore_index=True)
    specialized = pd.concat([
        pd.read_csv(root / "specialized" / f"seed_{seed}" / "specialized_metrics.csv")
        for seed in seeds
    ], ignore_index=True)
    geometry = pd.concat([
        pd.read_csv(root / "shared" / f"seed_{seed}" / "evaluation" /
                    "representation_local_geometry.csv") for seed in seeds
    ], ignore_index=True)
    shared.to_csv(root / "shared_dense_metrics_all_seeds.csv", index=False)
    specialized.to_csv(root / "specialization_table.csv", index=False)
    geometry.to_csv(root / "representation_local_geometry_all_seeds.csv", index=False)
    gap_rows = geometry.merge(
        specialized[["seed", "width", "specialization_gap"]],
        left_on=["seed", "budget_start"], right_on=["seed", "width"], how="inner",
    )
    gap_rows.to_csv(root / "rq1_representation_gap_table.csv", index=False)
    correlations = []
    for view, group in gap_rows.groupby("representation"):
        correlations.append({"scope": "pooled", "representation": view,
                             "n": len(group),
                             "pearson_G_gap": _safe_corr(group["G"], group["specialization_gap"], "pearson"),
                             "spearman_G_gap": _safe_corr(group["G"], group["specialization_gap"], "spearman")})
        for seed, seed_group in group.groupby("seed"):
            correlations.append({"scope": f"seed_{seed}", "representation": view,
                                 "n": len(seed_group),
                                 "pearson_G_gap": _safe_corr(seed_group["G"], seed_group["specialization_gap"], "pearson"),
                                 "spearman_G_gap": _safe_corr(seed_group["G"], seed_group["specialization_gap"], "spearman")})
    corr = pd.DataFrame(correlations); corr.to_csv(root / "rq1_representation_correlations.csv", index=False)
    robustness = []
    for scope, group in [("pooled", geometry), *[(f"seed_{s}", geometry[geometry.seed == s]) for s in seeds]]:
        pivot = group.pivot(index=["seed", "budget_start"], columns="representation", values="G")
        for view in REPRESENTATIONS[1:]:
            robustness.append({"scope": scope, "comparison": f"learned_projection vs {view}",
                               "n": len(pivot),
                               "pearson": _safe_corr(pivot["learned_projection"], pivot[view], "pearson"),
                               "spearman": _safe_corr(pivot["learned_projection"], pivot[view], "spearman")})
    pd.DataFrame(robustness).to_csv(root / "representation_pattern_robustness.csv", index=False)
    report = [
        "# S1 Width controlled experiment", "",
        f"S0-selected horizon: **{config['training']['epochs']} epochs**.", "",
        "Specialized references were independently initialized, selected on validation, and evaluated on test only after selection.", "",
        "## Specialization gaps", "", specialized.to_markdown(index=False), "",
        "## Representation robustness", "", pd.DataFrame(robustness).to_markdown(index=False), "",
        "## G versus specialization gap", "", corr.to_markdown(index=False), "",
    ]
    (root / "s1_report.md").write_text("\n".join(report))
    return {"report": str(root / "s1_report.md"), "rows": len(gap_rows)}
