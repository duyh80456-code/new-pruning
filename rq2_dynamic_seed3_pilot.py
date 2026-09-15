"""Seed-3 pilot for frozen Geometry-Dynamic and matched Resource-Dynamic policies."""

from __future__ import annotations

import copy
import json
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from data import build_development_train_loaders, build_interim_validation_loaders
from evaluation import calibrate_batch_norm, evaluate_width
from research_utils import seed_everything
from rq2_anchor_placement import GRID, _sha256
from rq2_probabilistic_support import INTERIOR_WIDTHS, pair_marginals
from s1_width import make_model
from training import kd_loss


DEFAULT_PILOT_SEED = 3
METHODS = ("geometry_dynamic", "resource_dynamic")
SAMPLER_SEED_OFFSETS = {"geometry_dynamic": 100000, "resource_dynamic": 200000}
PHASES = ((1, 50, 0.1), (51, 100, 0.01))
UPDATE_GEOMETRY_DURING_TRAINING = False
UPDATE_POLICY_DURING_TRAINING = False


def sampler_seed(method: str, seed: int) -> int:
    if method not in METHODS:
        raise ValueError(f"Unknown dynamic pilot method: {method}")
    return SAMPLER_SEED_OFFSETS[method] + int(seed)


def validate_pilot_config(config: dict) -> None:
    if config["dataset"]["name"].lower() != "cifar100" or config["dataset"].get("fake_data"):
        raise ValueError("Dynamic pilot requires real CIFAR-100")
    if config["model"]["backbone"] != "slimmable_resnet18":
        raise ValueError("Dynamic pilot requires Slimmable ResNet-18")
    if tuple(map(float, config["compression"]["eval_widths"])) != GRID:
        raise ValueError("Dynamic pilot requires the frozen 16-width grid")
    required = {
        "phase_1_epochs": 50, "total_epochs": 100, "anchors_per_batch": 4,
        "batch_size": 128,
    }
    for key, value in required.items():
        if int(config["training"][key]) != value:
            raise ValueError(f"training.{key} must equal {value}")
    numeric = {
        "learning_rate": 0.1, "extension_learning_rate": 0.01,
        "momentum": 0.9, "weight_decay": 5e-4,
        "kd_lambda": 1.0, "kd_temperature": 2.0,
    }
    for key, value in numeric.items():
        if not np.isclose(float(config["training"][key]), value):
            raise ValueError(f"training.{key} must equal {value}")
    if int(config["evaluation"]["batch_size"]) != 256:
        raise ValueError("Evaluation batch size must equal 256")
    if int(config["dataset"].get("num_workers", -1)) != 0:
        raise ValueError("Exact RNG replay requires dataset.num_workers=0")


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


def find_frozen_policy_roots(input_root: str | Path, materialized_root: str | Path):
    """Locate exactly one theory root and one support-preview root, unpacking if needed."""
    input_root, materialized_root = Path(input_root), Path(materialized_root)
    theory = sorted({path.parent for path in input_root.rglob("theory_probe_summary.json")})
    preview = sorted({path.parent for path in input_root.rglob("support_allocation_diagnostics.json")})
    if not theory:
        archives = sorted(input_root.rglob("theory-allocation-probe.zip"))
        if len(archives) == 1:
            root = _safe_extract(archives[0], materialized_root / "theory")
            theory = sorted({path.parent for path in root.rglob("theory_probe_summary.json")})
    if not preview:
        archives = sorted(input_root.rglob("rq2-probabilistic-support-preview.zip"))
        if len(archives) == 1:
            root = _safe_extract(archives[0], materialized_root / "preview")
            preview = sorted({path.parent for path in root.rglob("support_allocation_diagnostics.json")})
    if len(theory) != 1 or len(preview) != 1:
        raise FileNotFoundError(
            f"Expected exactly one frozen theory root and preview root; theory={theory}, preview={preview}"
        )
    return theory[0], preview[0]


def _validate_pair_table(table: pd.DataFrame, expected_pi: np.ndarray) -> pd.DataFrame:
    required = {"width_i", "width_j", "probability"}
    if not required.issubset(table.columns) or len(table) != 91:
        raise RuntimeError("Frozen pair table must contain all 91 distinct interior pairs")
    table = table.copy()
    table["width_i"] = table["width_i"].round(2)
    table["width_j"] = table["width_j"].round(2)
    probabilities = table["probability"].to_numpy(float)
    if np.any(probabilities < 0) or abs(probabilities.sum() - 1.0) >= 1e-10:
        raise RuntimeError("Frozen pair probabilities are invalid")
    if np.any(table["width_i"] >= table["width_j"]):
        raise RuntimeError("Every frozen pair must contain two ordered distinct widths")
    if set(table["width_i"]) | set(table["width_j"]) != set(INTERIOR_WIDTHS):
        raise RuntimeError("Frozen pair table contains an invalid interior width")
    error = float(np.max(np.abs(pair_marginals(table) - expected_pi)))
    if error >= 1e-8:
        raise RuntimeError(f"Pair marginals disagree with frozen pi: max error={error}")
    return table


def freeze_pilot_policies(
    theory_root: str | Path,
    preview_root: str | Path,
    development_root: str | Path,
    output_dir: str | Path,
    pilot_seed: int = DEFAULT_PILOT_SEED,
) -> dict:
    """Validate/copy the two frozen policies before seed-3 training starts."""
    theory_root, preview_root = Path(theory_root), Path(preview_root)
    development_root, output_dir = Path(development_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    theory_summary = json.loads((theory_root / "theory_probe_summary.json").read_text())
    preview_summary = json.loads((preview_root / "support_allocation_diagnostics.json").read_text())
    if (
        theory_summary.get("accuracy_used") is not False
        or theory_summary.get("test_used") is not False
        or theory_summary.get("training_authorized") is not False
        or theory_summary.get("development_geometry_seeds") != [0, 1, 2]
        or theory_summary.get("primary_theory_policy", {}).get("p") != 1.0
    ):
        raise RuntimeError("Theory artifact is not the frozen no-accuracy p=1 policy")
    required_theory_checks = [key for key in theory_summary if key.endswith("_pass")]
    if not required_theory_checks or not all(theory_summary[key] for key in required_theory_checks):
        raise RuntimeError("Theory artifact did not pass all mathematical checks")
    if (
        preview_summary.get("accuracy_used") is not False
        or preview_summary.get("training_authorized") is not False
        or preview_summary.get("development_geometry_seeds") != [0, 1, 2]
        or not all(preview_summary.get("assertions", {}).values())
    ):
        raise RuntimeError("Resource preview is not a valid frozen no-accuracy policy")

    theory_marginals = pd.read_csv(theory_root / "policy_family_marginals.csv")
    geometry_rows = theory_marginals.loc[np.isclose(theory_marginals["p"], 1.0)].sort_values("width")
    preview_marginals = pd.read_csv(preview_root / "support_allocation_marginals.csv").sort_values("width")
    if tuple(geometry_rows["width"].round(2)) != INTERIOR_WIDTHS:
        raise RuntimeError("Frozen p=1 geometry marginals do not cover all interior widths")
    if tuple(preview_marginals["width"].round(2)) != INTERIOR_WIDTHS:
        raise RuntimeError("Frozen resource marginals do not cover all interior widths")
    pi_geometry = geometry_rows["pi"].to_numpy(float)
    pi_resource = preview_marginals["pi_resource_matched_compute"].to_numpy(float)
    geometry_pairs_path = theory_root / "pair_distribution_p100.csv"
    resource_pairs_path = preview_root / "resource_pair_distribution.csv"
    geometry_pairs = _validate_pair_table(pd.read_csv(geometry_pairs_path), pi_geometry)
    resource_pairs = _validate_pair_table(pd.read_csv(resource_pairs_path), pi_resource)

    metrics = pd.read_csv(
        development_root / "rq2_dense_metrics_all.csv", usecols=["method", "budget", "flops"]
    )
    uniform = metrics.loc[metrics["method"].eq("uniform")]
    flops = {
        round(float(width), 2): float(value)
        for width, value in uniform.groupby("budget")["flops"].first().items()
    }
    F = np.asarray([flops[width] for width in INTERIOR_WIDTHS], float)
    geometry_interior_compute = float(F @ pi_geometry)
    resource_interior_compute = float(F @ pi_resource)
    if abs(geometry_interior_compute - resource_interior_compute) / geometry_interior_compute >= 1e-8:
        raise RuntimeError("Geometry and resource policies are not expected-compute matched")
    endpoint_compute = float(flops[0.25] + flops[1.0])
    uniform_interior_compute = float(flops[0.50] + flops[0.75])
    expected_total = endpoint_compute + geometry_interior_compute
    uniform_total = endpoint_compute + uniform_interior_compute

    geometry_out = output_dir / "geometry_pair_distribution.csv"
    resource_out = output_dir / "resource_pair_distribution.csv"
    geometry_pairs.to_csv(geometry_out, index=False)
    resource_pairs.to_csv(resource_out, index=False)
    marginal_table = pd.DataFrame({
        "width": INTERIOR_WIDTHS,
        "flops": F,
        "pi_geometry": pi_geometry,
        "pi_resource": pi_resource,
    })
    marginals_out = output_dir / "frozen_dynamic_marginals.csv"
    marginal_table.to_csv(marginals_out, index=False)
    metadata = {
        "status": "FROZEN_BEFORE_DECLARED_DYNAMIC_PILOT_SEED",
        "experiment": "rq2_dynamic_seed_pilot",
        "seed": int(pilot_seed),
        "methods": list(METHODS),
        "development_geometry_seeds": [0, 1, 2],
        "accuracy_used_to_build_policy": False,
        "test_used": False,
        "geometry_p": 1.0,
        "fixed_endpoints": [0.25, 1.0],
        "interior_slots": 2,
        "subnets_per_batch": 4,
        "expected_interior_compute": geometry_interior_compute,
        "fixed_endpoint_compute": endpoint_compute,
        "expected_total_compute": expected_total,
        "uniform_total_compute": uniform_total,
        "expected_total_compute_ratio_vs_uniform": expected_total / uniform_total,
        "policy_sources": {
            "theory_probe_summary": _sha256(theory_root / "theory_probe_summary.json"),
            "support_allocation_diagnostics": _sha256(
                preview_root / "support_allocation_diagnostics.json"
            ),
            "geometry_pair_distribution": _sha256(geometry_pairs_path),
            "resource_pair_distribution": _sha256(resource_pairs_path),
        },
        "frozen_policy_files": {
            "geometry_dynamic": {"path": geometry_out.name, "sha256": _sha256(geometry_out)},
            "resource_dynamic": {"path": resource_out.name, "sha256": _sha256(resource_out)},
        },
        "frozen_marginals": {"path": marginals_out.name, "sha256": _sha256(marginals_out)},
        "update_geometry_during_training": UPDATE_GEOMETRY_DURING_TRAINING,
        "update_policy_during_training": UPDATE_POLICY_DURING_TRAINING,
        "training_authorized_only_for_declared_pilot_seed": True,
    }
    (output_dir / "dynamic_pilot_frozen_protocol.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return metadata


def load_frozen_policy(protocol_dir: str | Path, method: str):
    if method not in METHODS:
        raise ValueError(f"Unknown dynamic pilot method: {method}")
    protocol_dir = Path(protocol_dir)
    protocol = json.loads((protocol_dir / "dynamic_pilot_frozen_protocol.json").read_text())
    entry = protocol["frozen_policy_files"][method]
    path = protocol_dir / entry["path"]
    if _sha256(path) != entry["sha256"]:
        raise RuntimeError(f"Frozen {method} pair distribution hash mismatch")
    marginals_path = protocol_dir / protocol["frozen_marginals"]["path"]
    if _sha256(marginals_path) != protocol["frozen_marginals"]["sha256"]:
        raise RuntimeError("Frozen dynamic marginal table hash mismatch")
    marginals = pd.read_csv(marginals_path).sort_values("width")
    pi_column = "pi_geometry" if method == "geometry_dynamic" else "pi_resource"
    pi = marginals[pi_column].to_numpy(float)
    pair_table = _validate_pair_table(pd.read_csv(path), pi)
    flops = dict(zip(marginals["width"].round(2), marginals["flops"].astype(float)))
    return pair_table, pi, flops, protocol


def simulate_sampler_sanity(
    pair_table: pd.DataFrame,
    expected_pi: np.ndarray,
    flops: dict[float, float],
    endpoint_flops: float,
    method: str,
    seed: int = DEFAULT_PILOT_SEED,
    draws: int = 100_000,
) -> dict:
    isolated_sampler_seed = sampler_seed(method, seed)
    rng = np.random.default_rng(isolated_sampler_seed)
    choices = rng.choice(
        len(pair_table), size=int(draws), p=pair_table["probability"].to_numpy(float)
    )
    first = pair_table["width_i"].to_numpy(float)[choices]
    second = pair_table["width_j"].to_numpy(float)[choices]
    empirical = np.asarray([np.mean((first == width) | (second == width)) for width in INTERIOR_WIDTHS])
    expected_compute = endpoint_flops + sum(
        flops[width] * expected_pi[index] for index, width in enumerate(INTERIOR_WIDTHS)
    )
    realized_compute = endpoint_flops + float(np.mean([
        flops[round(float(a), 2)] + flops[round(float(b), 2)] for a, b in zip(first, second)
    ]))
    max_error = float(np.max(np.abs(empirical - expected_pi)))
    # Six-sigma family-wise implementation gate plus a small numeric allowance.
    monte_carlo_limit = float(
        6.0 * np.max(np.sqrt(expected_pi * (1 - expected_pi) / draws)) + 1e-3
    )
    result = {
        "method": method, "seed": int(seed), "draws": int(draws),
        "sampler_seed": isolated_sampler_seed,
        "maximum_marginal_absolute_error": max_error,
        "marginal_error_limit": monte_carlo_limit,
        "expected_total_flops": expected_compute,
        "simulated_total_flops": realized_compute,
        "relative_compute_error": abs(realized_compute - expected_compute) / expected_compute,
        "passed": bool(max_error <= monte_carlo_limit),
    }
    if not result["passed"]:
        raise RuntimeError(f"Pre-training sampler sanity failed: {result}")
    return result


def _capture_rng(loader, sampler_rng: np.random.Generator) -> dict:
    state = {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader_generator": loader.generator.get_state() if loader.generator else None,
        "sampler_bit_generator": copy.deepcopy(sampler_rng.bit_generator.state),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict, loader, sampler_rng: np.random.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if loader.generator is not None and state.get("loader_generator") is not None:
        loader.generator.set_state(state["loader_generator"])
    sampler_rng.bit_generator.state = state["sampler_bit_generator"]


def _atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _new_optimizer_scheduler(model, config: dict, learning_rate: float, epochs: int):
    cfg = config["training"]
    optimizer = torch.optim.SGD(
        model.parameters(), lr=float(learning_rate), momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs))
    return optimizer, scheduler


def _train_phase(
    config: dict,
    output: Path,
    method: str,
    pair_table: pd.DataFrame,
    pi: np.ndarray,
    flops: dict[float, float],
    protocol: dict,
    first_epoch: int,
    last_epoch: int,
    learning_rate: float,
    phase_dir: Path,
    initial_checkpoint: Path | None = None,
    seed: int = DEFAULT_PILOT_SEED,
) -> Path:
    """Train one phase with exact epoch resume and an isolated pair-sampler RNG."""
    device = torch.device(config["experiment"]["device"])
    seed = int(seed)
    seed_everything(seed)
    loaders = build_development_train_loaders(config, training_seed=seed)
    model = make_model(config, device)
    optimizer, scheduler = _new_optimizer_scheduler(
        model, config, learning_rate, last_epoch - first_epoch + 1
    )
    isolated_sampler_seed = sampler_seed(method, seed)
    sampler_rng = np.random.default_rng(isolated_sampler_seed)
    latest = phase_dir / "latest.pt"
    start_epoch = first_epoch
    cumulative_batches = 0
    cumulative_realized_flops = 0.0
    cumulative_width_counts = {width: 0 for width in GRID}
    cumulative_pair_counts = {
        (round(float(row.width_i), 2), round(float(row.width_j), 2)): 0
        for row in pair_table.itertuples()
    }
    if latest.is_file():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        if (
            payload.get("method") != method or int(payload.get("seed", -1)) != seed
            or payload.get("policy_sha256") != protocol["frozen_policy_files"][method]["sha256"]
        ):
            raise RuntimeError("Resume checkpoint method/seed/frozen-policy identity mismatch")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scheduler.load_state_dict(payload["scheduler"])
        _restore_rng(payload["rng"], loaders.train, sampler_rng)
        start_epoch = int(payload["epoch"]) + 1
        cumulative_batches = int(payload["cumulative_batches"])
        cumulative_realized_flops = float(payload["cumulative_realized_flops"])
        cumulative_width_counts.update({float(k): int(v) for k, v in payload["width_counts"].items()})
        cumulative_pair_counts.update({
            tuple(map(float, key.split(","))): int(value)
            for key, value in payload["pair_counts"].items()
        })
    elif initial_checkpoint is not None:
        payload = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
        if (
            payload.get("method") != method or int(payload.get("seed", -1)) != seed
            or payload.get("policy_sha256") != protocol["frozen_policy_files"][method]["sha256"]
        ):
            raise RuntimeError("Phase-1 checkpoint method/seed/frozen-policy identity mismatch")
        model.load_state_dict(payload["model"])
        # Preserve the pair-sampling stream across the phase boundary while
        # intentionally resetting optimizer/scheduler/model-data RNG as in 50+50.
        sampler_rng.bit_generator.state = payload["rng"]["sampler_bit_generator"]
        cumulative_batches = int(payload["cumulative_batches"])
        cumulative_realized_flops = float(payload["cumulative_realized_flops"])
        cumulative_width_counts.update({float(k): int(v) for k, v in payload["width_counts"].items()})
        cumulative_pair_counts.update({
            tuple(map(float, key.split(","))): int(value)
            for key, value in payload["pair_counts"].items()
        })

    metrics_path = output / "training_epoch_metrics.csv"
    width_path = output / "width_inclusion_by_epoch.csv"
    pair_path = output / "pair_counts_by_epoch.csv"
    metrics_records = (
        pd.read_csv(metrics_path).loc[lambda frame: frame.epoch < start_epoch].to_dict("records")
        if metrics_path.is_file() else []
    )
    width_records = (
        pd.read_csv(width_path).loc[lambda frame: frame.epoch < start_epoch].to_dict("records")
        if width_path.is_file() else []
    )
    pair_records = (
        pd.read_csv(pair_path).loc[lambda frame: frame.epoch < start_epoch].to_dict("records")
        if pair_path.is_file() else []
    )
    probabilities = pair_table["probability"].to_numpy(float)
    pair_widths = [
        (round(float(row.width_i), 2), round(float(row.width_j), 2))
        for row in pair_table.itertuples()
    ]
    endpoint_flops = float(protocol["fixed_endpoint_compute"])
    expected_total_flops = float(protocol["expected_total_compute"])

    for epoch in range(start_epoch, last_epoch + 1):
        model.train()
        epoch_loss = 0.0
        epoch_examples = 0
        epoch_batches = 0
        epoch_flops = 0.0
        epoch_width_counts = {width: 0 for width in GRID}
        epoch_pair_counts = {pair: 0 for pair in pair_widths}
        for images, labels, _ in loaders.train:
            images, labels = images.to(device), labels.to(device)
            pair = pair_widths[int(sampler_rng.choice(len(pair_widths), p=probabilities))]
            wi, wj = pair
            widths = (0.25, wi, wj, 1.0)
            if wi == wj or len(set(widths)) != 4:
                raise RuntimeError(f"Invalid sampled four-subnet set: {widths}")
            optimizer.zero_grad(set_to_none=True)
            model.set_width(1.0)
            teacher = model(images)
            total = F.cross_entropy(teacher, labels)
            for width in (0.25, wi, wj):
                model.set_width(width)
                logits = model(images)
                total = total + F.cross_entropy(logits, labels) + float(
                    config["training"]["kd_lambda"]
                ) * kd_loss(logits, teacher, float(config["training"]["kd_temperature"]))
            # Historical shared-reference protocol averages four subnet losses.
            total = total / 4.0
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite dynamic loss at method={method}, epoch={epoch}")
            total.backward()
            optimizer.step()

            batch_size = labels.numel()
            epoch_loss += float(total.detach()) * batch_size
            epoch_examples += batch_size
            epoch_batches += 1
            realized = endpoint_flops + flops[wi] + flops[wj]
            epoch_flops += realized
            cumulative_realized_flops += realized
            cumulative_batches += 1
            for width in widths:
                epoch_width_counts[width] += 1
                cumulative_width_counts[width] += 1
            epoch_pair_counts[pair] += 1
            cumulative_pair_counts[pair] += 1

        lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        epoch_realized = epoch_flops / epoch_batches
        metrics_records.append({
            "method": method, "seed": seed, "epoch": epoch,
            "phase": 1 if epoch <= 50 else 2,
            "train_loss": epoch_loss / epoch_examples, "learning_rate": lr,
            "batches": epoch_batches,
            "expected_total_flops_per_batch": expected_total_flops,
            "realized_total_flops_per_batch": epoch_realized,
            "realized_compute_relative_error": abs(epoch_realized - expected_total_flops) / expected_total_flops,
            "cumulative_realized_total_flops_per_batch": cumulative_realized_flops / cumulative_batches,
        })
        for index, width in enumerate(GRID):
            expected = 1.0 if width in (0.25, 1.0) else float(pi[INTERIOR_WIDTHS.index(width)])
            width_records.append({
                "method": method, "seed": seed, "epoch": epoch, "width": width,
                "epoch_inclusion_count": epoch_width_counts[width],
                "epoch_empirical_pi": epoch_width_counts[width] / epoch_batches,
                "cumulative_inclusion_count": cumulative_width_counts[width],
                "cumulative_empirical_pi": cumulative_width_counts[width] / cumulative_batches,
                "expected_pi": expected,
            })
        for pair in pair_widths:
            pair_records.append({
                "method": method, "seed": seed, "epoch": epoch,
                "width_i": pair[0], "width_j": pair[1],
                "epoch_count": epoch_pair_counts[pair],
                "cumulative_count": cumulative_pair_counts[pair],
            })
        pd.DataFrame(metrics_records).to_csv(metrics_path, index=False)
        pd.DataFrame(width_records).to_csv(width_path, index=False)
        pd.DataFrame(pair_records).to_csv(pair_path, index=False)
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "method": method,
            "seed": seed, "phase": 1 if epoch <= 50 else 2,
            "rng": _capture_rng(loaders.train, sampler_rng),
            "cumulative_batches": cumulative_batches,
            "cumulative_realized_flops": cumulative_realized_flops,
            "width_counts": {str(key): value for key, value in cumulative_width_counts.items()},
            "pair_counts": {f"{key[0]:.2f},{key[1]:.2f}": value for key, value in cumulative_pair_counts.items()},
            "policy_sha256": protocol["frozen_policy_files"][method]["sha256"],
        }
        _atomic_checkpoint(latest, checkpoint)
        print(
            f"{method} seed={seed} epoch {epoch}/100 | loss={epoch_loss/epoch_examples:.4f}, "
            f"lr={lr:.6g}, realized/expected={epoch_realized/expected_total_flops:.4f}",
            flush=True,
        )
    target = output / ("epoch_050.pt" if last_epoch == 50 else "epoch_100.pt")
    shutil.copy2(latest, target)
    return target


def train_dynamic_method(
    config: dict,
    root: str | Path,
    protocol_dir: str | Path,
    method: str,
    seed: int = DEFAULT_PILOT_SEED,
) -> Path:
    seed = int(seed)
    if seed in (0, 1, 2) or seed < 0:
        raise ValueError("Pilot seed must be nonnegative and outside development geometry seeds 0,1,2")
    validate_pilot_config(config)
    pair_table, pi, interior_flops, protocol = load_frozen_policy(protocol_dir, method)
    if int(protocol.get("seed", -1)) != seed:
        raise RuntimeError("Frozen dynamic policy protocol was declared for a different pilot seed")
    output = Path(root) / method / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    final = output / "epoch_100.pt"
    if final.is_file() and (output / "training_provenance.json").is_file():
        return final
    phase_1 = _train_phase(
        config, output, method, pair_table, pi, interior_flops, protocol,
        1, 50, 0.1, output / "phase_1", seed=seed,
    )
    final = _train_phase(
        config, output, method, pair_table, pi, interior_flops, protocol,
        51, 100, 0.01, output, initial_checkpoint=phase_1, seed=seed,
    )
    shutil.copy2(final, output / "checkpoint.pt")
    provenance = {
        "experiment": "rq2_dynamic_seed_pilot", "seed": seed, "method": method,
        "epochs": 100, "phase1_epochs": 50, "phase2_epochs": 50,
        "phase1_learning_rate": 0.1, "phase2_learning_rate": 0.01,
        "optimizer_state_reused_at_phase_boundary": False,
        "scheduler_state_reused_at_phase_boundary": False,
        "model_data_rng_restarted_at_phase_boundary": True,
        "sampler_rng_continued_at_phase_boundary": True,
        "sampler_seed": sampler_seed(method, seed),
        "endpoints": [0.25, 1.0], "interior_slots": 2, "subnets_per_batch": 4,
        "geometry_p": 1.0 if method == "geometry_dynamic" else None,
        "policy_frozen": True, "online_geometry": False,
        "accuracy_used_to_build_policy": False, "test_used": False,
        "development_geometry_seeds": [0, 1, 2],
        "kd_lambda": 1.0, "kd_temperature": 2.0,
        "expected_total_compute_ratio_vs_uniform": protocol["expected_total_compute_ratio_vs_uniform"],
        "policy_source": (
            "p1_linear_expected_geometry_weighted_waiting_debt"
            if method == "geometry_dynamic" else "maximum_entropy_resource_matched_compute"
        ),
        "policy_sha256": protocol["frozen_policy_files"][method]["sha256"],
        "epoch_050_sha256": _sha256(output / "epoch_050.pt"),
        "epoch_100_sha256": _sha256(final),
    }
    (output / "training_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return final


def evaluate_dynamic_method(
    config: dict,
    root: str | Path,
    method: str,
    seed: int = DEFAULT_PILOT_SEED,
) -> Path:
    """Evaluate dense validation only after epoch 100; never construct the test dataset."""
    seed = int(seed)
    if method not in METHODS or seed in (0, 1, 2) or seed < 0:
        raise ValueError("Dynamic pilot evaluation requires a registered method and non-development seed")
    root = Path(root)
    checkpoint = root / method / f"seed_{seed}" / "epoch_100.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = root / "evaluation" / method / f"seed_{seed}"
    final_csv = output / "dense_validation_accuracy.csv"
    if final_csv.is_file():
        return final_csv
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config["experiment"]["device"])
    loaders = build_interim_validation_loaders(config)
    model = make_model(config, device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    rows = []
    for width in GRID:
        calibrate_batch_norm(
            model, loaders.calibration, width, device,
            int(config["evaluation"]["bn_calibration_batches"]),
        )
        metrics = evaluate_width(model, loaders.validation, width, device)
        rows.append({
            "method": method, "seed": seed, "split": "validation_5k",
            "width": width, **metrics,
        })
        print(f"{method} seed={seed} validation width={width:.2f}: acc={metrics['accuracy']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(final_csv, index=False)
    (output / "TEST_SPLIT_NOT_ACCESSED.txt").write_text(
        f"Seed-{seed} dynamic pilot used only the fixed CIFAR-100 train/validation split.\n"
    )
    return final_csv


def _longest_negative_run(widths: np.ndarray, deltas: np.ndarray) -> int:
    longest = current = 0
    previous = None
    for width, delta in sorted(zip(widths, deltas)):
        consecutive = previous is not None and np.isclose(width - previous, 0.05)
        current = current + 1 if delta < 0 and consecutive else int(delta < 0)
        longest = max(longest, current)
        previous = width
    return longest


def finalize_pilot(root: str | Path, seed: int = DEFAULT_PILOT_SEED) -> dict:
    root = Path(root)
    seed = int(seed)
    frames = [
        pd.read_csv(root / "evaluation" / method / f"seed_{seed}" / "dense_validation_accuracy.csv")
        for method in METHODS
    ]
    metrics = pd.concat(frames, ignore_index=True)
    if len(metrics) != 32 or set(metrics["split"]) != {"validation_5k"}:
        raise RuntimeError("Expected 16 validation rows for each dynamic method")
    metrics.to_csv(root / "dense_validation_accuracy.csv", index=False)
    rows = []
    for method, group in metrics.groupby("method"):
        group = group.sort_values("width")
        rows.append({
            "method": method, "seed": seed,
            "dense_mean_accuracy": float(group["accuracy"].mean()),
            "worst_accuracy": float(group["accuracy"].min()),
            "low_mean_accuracy": float(group.loc[group.width.between(0.30, 0.45), "accuracy"].mean()),
            "mid_mean_accuracy": float(group.loc[group.width.between(0.50, 0.75), "accuracy"].mean()),
            "high_mean_accuracy": float(group.loc[group.width.between(0.80, 0.95), "accuracy"].mean()),
            "full_width_accuracy": float(group.loc[np.isclose(group.width, 1.0), "accuracy"].iloc[0]),
        })
    summary = pd.DataFrame(rows).sort_values("method")
    summary.to_csv(root / "dynamic_pilot_method_summary.csv", index=False)
    wide = metrics.pivot(index="width", columns="method", values="accuracy").reset_index()
    wide["geometry_minus_resource_accuracy"] = (
        wide["geometry_dynamic"] - wide["resource_dynamic"]
    )
    wide.to_csv(root / "dynamic_pilot_width_comparison.csv", index=False)
    indexed = summary.set_index("method")
    deltas = {
        metric: float(indexed.loc["geometry_dynamic", metric] - indexed.loc["resource_dynamic", metric])
        for metric in (
            "dense_mean_accuracy", "worst_accuracy", "low_mean_accuracy",
            "mid_mean_accuracy", "high_mean_accuracy", "full_width_accuracy",
        )
    }
    mid_high = wide.loc[wide.width.between(0.50, 0.95)]
    longest_negative = _longest_negative_run(
        mid_high["width"].to_numpy(float),
        mid_high["geometry_minus_resource_accuracy"].to_numpy(float),
    )
    if deltas["low_mean_accuracy"] > 0 and deltas["dense_mean_accuracy"] > 0:
        verdict = "DEVELOPMENT GO" if longest_negative < 3 else "PARTIAL SUPPORT"
    elif bool((wide["geometry_minus_resource_accuracy"] < 0).all()):
        verdict = "DEVELOPMENT NO-GO"
    else:
        verdict = "MIXED — MANUAL DEVELOPMENT REVIEW"
    decision = {
        "status": f"SEED{seed}_VALIDATION_PILOT_COMPLETE",
        "verdict": verdict,
        "seed": seed,
        "test_used": False,
        "geometry_minus_resource": deltas,
        "longest_consecutive_negative_mid_high_width_run": int(longest_negative),
        "sustained_negative_region_definition": "at least 3 consecutive 0.05-spaced widths",
        "seeds_6_7_8_authorized_automatically": False,
        "note": f"The seed-{seed} gate is development evidence, not confirmatory inference.",
    }
    (root / "dynamic_pilot_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    return decision
