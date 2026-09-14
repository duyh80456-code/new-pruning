from __future__ import annotations

import copy
import hashlib
import itertools
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

from data import build_confirmatory_loaders
from evaluation import evaluate_prediction_grid
from georeg_curvature import paired_accuracy_bootstrap
from research_utils import seed_everything
from s1_width import (
    evaluate_shared_checkpoint,
    make_model,
    train_shared_reference,
)


GRID = tuple(round(0.25 + 0.05 * index, 2) for index in range(16))
UNIFORM_ANCHORS = (0.25, 0.50, 0.75, 1.00)
PRIMARY_REPRESENTATION = "learned_projection"
CONFIRMATORY_SEEDS = (1, 2)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_uniform_100_root(root: Path) -> bool:
    required = [
        root / "resolved_config.yaml",
        root / "shared_dense_metrics_all_seeds.csv",
        root / "representation_local_geometry_all_seeds.csv",
        root / "protocol" / "fixed_random_projection.pt",
        *[root / "shared" / f"seed_{seed}" / "checkpoint.pt" for seed in (0, 1, 2)],
    ]
    if not all(path.is_file() for path in required):
        return False
    try:
        config = yaml.safe_load((root / "resolved_config.yaml").read_text())
        return (
            int(config["training"]["epochs"]) == 100
            and tuple(map(float, config["compression"]["train_widths"])) == UNIFORM_ANCHORS
        )
    except (KeyError, TypeError, ValueError):
        return False


def find_uniform_100_root(input_root: str | Path, destination: str | Path) -> Path:
    """Find one complete direct Uniform-100 result or safely unpack its archive."""
    input_root, destination = Path(input_root), Path(destination)
    candidates = sorted({path.parent for path in input_root.rglob("resolved_config.yaml")})
    candidates = [root for root in candidates if _is_uniform_100_root(root)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple Uniform-100 roots found: {candidates}")
    archives = sorted(input_root.rglob("kaggle-s1-extension-100-*.zip"))
    if len(archives) != 1:
        raise FileNotFoundError(
            f"Expected one complete Uniform-100 root or archive below {input_root}; "
            f"archives={archives}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(archives[0]) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if resolved_destination not in target.parents and target != resolved_destination:
                raise RuntimeError(f"Unsafe path in Uniform-100 archive: {member.filename}")
        bundle.extractall(destination)
    candidates = sorted({path.parent for path in destination.rglob("resolved_config.yaml")})
    candidates = [root for root in candidates if _is_uniform_100_root(root)]
    if len(candidates) != 1:
        raise RuntimeError(f"Extracted archive does not contain exactly one Uniform-100 root: {candidates}")
    return candidates[0]


def validate_rq2_config(config: dict) -> None:
    if config["dataset"]["name"].lower() != "cifar100":
        raise ValueError("RQ2 requires CIFAR-100")
    if config["model"]["backbone"] != "slimmable_resnet18":
        raise ValueError("RQ2 requires Slimmable ResNet-18")
    if tuple(map(float, config["compression"]["eval_widths"])) != GRID:
        raise ValueError("RQ2 dense grid must be 0.25,0.30,...,1.00")
    if tuple(map(int, config["experiment"]["confirmatory_seeds"])) != CONFIRMATORY_SEEDS:
        raise ValueError("RQ2 confirmatory seeds are locked to 1 and 2")
    if int(config["experiment"]["development_seed"]) != 0:
        raise ValueError("RQ2 development seed is locked to 0")
    if int(config["training"]["phase_1_epochs"]) != 50 or int(
        config["training"]["total_epochs"]
    ) != 100:
        raise ValueError("RQ2 training must use the locked 50+50 horizon")
    if int(config["training"]["anchors_per_batch"]) != 4:
        raise ValueError("RQ2 requires four subnet forwards per batch")
    if int(config["dataset"].get("num_workers", -1)) != 0:
        raise ValueError("RQ2 exact loader replay requires dataset.num_workers=0")


def _uniform_source_config(root: Path) -> dict:
    config = yaml.safe_load((root / "resolved_config.yaml").read_text())
    if int(config["training"]["epochs"]) != 100:
        raise ValueError("Uniform comparator is not an epoch-100 result")
    if tuple(map(float, config["compression"]["train_widths"])) != UNIFORM_ANCHORS:
        raise ValueError("Uniform comparator does not use registered uniform anchors")
    provenance_path = root / "protocol" / "extension_provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError("Uniform comparator lacks 50+50 extension provenance")
    provenance = json.loads(provenance_path.read_text())
    expected = {
        "source_horizon": 50,
        "target_horizon": 100,
        "optimizer_state_reused": False,
        "scheduler_state_reused": False,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"Uniform provenance mismatch for {key}: {provenance.get(key)}")
    if not np.isclose(float(provenance.get("extension_learning_rate", np.nan)), 0.01):
        raise ValueError("Uniform comparator extension learning rate is not the registered 0.01")
    return config


def select_geometry_anchors(uniform_root: str | Path, output_dir: str | Path) -> dict:
    uniform_root, output_dir = Path(uniform_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = uniform_root / "representation_local_geometry_all_seeds.csv"
    geometry = pd.read_csv(geometry_path)
    selected = geometry.loc[
        (geometry["seed"].astype(int) == 0)
        & geometry["representation"].eq(PRIMARY_REPRESENTATION)
    ].sort_values("budget_start")
    if len(selected) != len(GRID) - 1:
        raise RuntimeError(f"Expected 15 seed-0 adjacent geometry edges, got {len(selected)}")
    starts = tuple(selected["budget_start"].round(2))
    ends = tuple(selected["budget_end"].round(2))
    if starts != GRID[:-1] or ends != GRID[1:]:
        raise RuntimeError("Geometry edges do not exactly cover the registered dense grid")
    edge_lengths = 0.05 * selected["G"].to_numpy(float)
    coordinates = np.concatenate([[0.0], np.cumsum(edge_lengths)])
    coordinate = dict(zip(GRID, coordinates))
    rows = []
    for a1, a2 in itertools.combinations(GRID[1:-1], 2):
        anchors = (GRID[0], a1, a2, GRID[-1])
        distances = [min(abs(coordinate[c] - coordinate[a]) for a in anchors) for c in GRID]
        rows.append({
            "a1": a1,
            "a2": a2,
            "anchors": ",".join(f"{value:.2f}" for value in anchors),
            "max_geometry_radius": max(distances),
            "mean_geometry_distance": float(np.mean(distances)),
        })
    table = pd.DataFrame(rows).sort_values(
        ["max_geometry_radius", "mean_geometry_distance", "a1", "a2"],
        kind="mergesort",
    ).reset_index(drop=True)
    best = table.iloc[0]
    anchors = (0.25, float(best["a1"]), float(best["a2"]), 1.0)
    table["selected"] = (
        np.isclose(table["a1"], anchors[1]) & np.isclose(table["a2"], anchors[2])
    )
    table.to_csv(output_dir / "geometry_anchor_selection.csv", index=False)
    coordinate_table = pd.DataFrame({
        "width": GRID,
        "geometry_coordinate": coordinates,
        "incoming_edge_length": np.concatenate([[np.nan], edge_lengths]),
    })
    coordinate_table.to_csv(output_dir / "geometry_trajectory_coordinates.csv", index=False)
    metrics = pd.read_csv(uniform_root / "shared_dense_metrics_all_seeds.csv")
    flops = {
        round(float(width), 2): float(value)
        for width, value in metrics.groupby("budget")["flops"].first().items()
    }
    compute_rows = []
    for method, method_anchors in (("Uniform-4", UNIFORM_ANCHORS), ("Geometry-4", anchors)):
        total = float(sum(flops[anchor] for anchor in method_anchors))
        compute_rows.append({
            "method": method,
            "anchors": ",".join(f"{value:.2f}" for value in method_anchors),
            "forwards_per_batch": 4,
            "subnet_flops_per_batch": total,
        })
    compute = pd.DataFrame(compute_rows)
    uniform_compute = float(compute.loc[compute["method"] == "Uniform-4", "subnet_flops_per_batch"].iloc[0])
    compute["relative_to_uniform"] = compute["subnet_flops_per_batch"] / uniform_compute
    compute.to_csv(output_dir / "anchor_training_compute.csv", index=False)
    common_holdout = sorted(set(GRID) - set(UNIFORM_ANCHORS) - set(anchors))
    checkpoint = uniform_root / "shared" / "seed_0" / "checkpoint.pt"
    artifact = {
        "selection_seed": 0,
        "source_horizon": 100,
        "source_protocol": "50+50_optimizer_scheduler_reset",
        "representation": PRIMARY_REPRESENTATION,
        "feature_subset": "fixed_2000_validation_samples",
        "candidate_grid": list(GRID),
        "uniform_anchors": list(UNIFORM_ANCHORS),
        "selected_anchors": list(anchors),
        "common_holdout": common_holdout,
        "objective": "minimax k-center on cumulative adjacent SW path length",
        "tie_break": ["mean_geometry_distance", "lexicographic_a1_a2"],
        "source_checkpoint_sha256": _sha256(checkpoint),
        "geometry_csv_sha256": _sha256(geometry_path),
        "state_count_matched": True,
        "subnet_compute_ratio_geo_to_uniform": float(
            compute.loc[compute["method"] == "Geometry-4", "relative_to_uniform"].iloc[0]
        ),
    }
    selection_path = output_dir / "selected_anchors.json"
    if selection_path.is_file():
        previous = json.loads(selection_path.read_text())
        immutable = ["selected_anchors", "common_holdout", "source_checkpoint_sha256", "geometry_csv_sha256"]
        if any(previous.get(key) != artifact.get(key) for key in immutable):
            raise RuntimeError("Refusing to change a previously frozen RQ2 anchor selection")
    selection_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def _path_coordinate_from_geometry(edge_frame: pd.DataFrame) -> dict[float, float]:
    edge_frame = edge_frame.sort_values("budget_start")
    starts = tuple(edge_frame["budget_start"].round(2))
    ends = tuple(edge_frame["budget_end"].round(2))
    if starts != GRID[:-1] or ends != GRID[1:]:
        raise RuntimeError("Geometry edges do not exactly cover the registered dense grid")
    edge_lengths = 0.05 * edge_frame["G"].to_numpy(float)
    return dict(zip(GRID, np.concatenate([[0.0], np.cumsum(edge_lengths)])))


def _radius(anchors, coordinate: dict[float, float]) -> tuple[float, float]:
    distances = [
        min(abs(coordinate[width] - coordinate[anchor]) for anchor in anchors)
        for width in GRID
    ]
    return float(max(distances)), float(np.mean(distances))


def _pareto_mask(first: np.ndarray, second: np.ndarray, tolerance=1e-12) -> np.ndarray:
    result = np.ones(len(first), dtype=bool)
    for index in range(len(first)):
        weakly_better = (first <= first[index] + tolerance) & (second <= second[index] + tolerance)
        strictly_better = (first < first[index] - tolerance) | (second < second[index] - tolerance)
        if np.any(weakly_better & strictly_better):
            result[index] = False
    return result


def select_hybrid_anchors(uniform_root: str | Path, output_dir: str | Path) -> dict:
    """Freeze Hybrid-v2 using only resource metadata and Uniform validation geometry."""
    uniform_root, output_dir = Path(uniform_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = uniform_root / "representation_local_geometry_all_seeds.csv"
    metrics_path = uniform_root / "shared_dense_metrics_all_seeds.csv"
    geometry = pd.read_csv(geometry_path)
    learned = geometry.loc[
        geometry["representation"].eq(PRIMARY_REPRESENTATION)
        & geometry["seed"].astype(int).isin([0, 1, 2])
    ].copy()
    counts = learned.groupby(["budget_start", "budget_end"])["seed"].nunique()
    if len(counts) != 15 or not (counts == 3).all():
        raise RuntimeError("Hybrid selection requires all 15 learned-projection edges for seeds 0,1,2")
    median_edges = learned.groupby(
        ["budget_start", "budget_end"], as_index=False
    )["G"].median()
    geometry_coordinate = _path_coordinate_from_geometry(median_edges)
    seed_zero_coordinate = _path_coordinate_from_geometry(
        learned.loc[learned["seed"].astype(int) == 0]
    )
    metrics = pd.read_csv(metrics_path)
    flops = {
        round(float(width), 2): float(value)
        for width, value in metrics.groupby("budget")["flops"].first().items()
    }
    if set(flops) != set(GRID) or any(value <= 0 for value in flops.values()):
        raise RuntimeError("A positive FLOPs value is required at every registered width")
    log_flops = np.asarray([np.log(flops[width]) for width in GRID])
    span = float(log_flops.max() - log_flops.min())
    if span <= 0:
        raise RuntimeError("Degenerate log-FLOPs resource coordinate")
    resource_coordinate = dict(zip(GRID, (log_flops - log_flops.min()) / span))
    uniform_R_G, _ = _radius(UNIFORM_ANCHORS, geometry_coordinate)
    uniform_R_R, _ = _radius(UNIFORM_ANCHORS, resource_coordinate)
    if uniform_R_G <= 0 or uniform_R_R <= 0:
        raise RuntimeError("Uniform normalization radii must be positive")
    rows = []
    for a1, a2 in itertools.combinations(GRID[1:-1], 2):
        anchors = (0.25, a1, a2, 1.0)
        R_G, mean_G = _radius(anchors, geometry_coordinate)
        R_R, mean_R = _radius(anchors, resource_coordinate)
        normalized_G = R_G / uniform_R_G
        normalized_R = R_R / uniform_R_R
        rows.append({
            "a1": a1,
            "a2": a2,
            "anchors": ",".join(f"{value:.2f}" for value in anchors),
            "R_G": R_G,
            "R_R": R_R,
            "mean_geometry_distance": mean_G,
            "mean_resource_distance": mean_R,
            "normalized_R_G": normalized_G,
            "normalized_R_R": normalized_R,
            "hybrid_objective": max(normalized_G, normalized_R),
            "normalized_radius_sum": normalized_G + normalized_R,
        })
    candidates = pd.DataFrame(rows)
    candidates["pareto_optimal"] = _pareto_mask(
        candidates["R_G"].to_numpy(float), candidates["R_R"].to_numpy(float)
    )
    # Rounding defines numerical ties before the documented secondary and
    # lexicographic rules; no accuracy result participates in selection.
    candidates["_objective_key"] = candidates["hybrid_objective"].round(12)
    candidates["_sum_key"] = candidates["normalized_radius_sum"].round(12)
    candidates = candidates.sort_values(
        ["_objective_key", "_sum_key", "a1", "a2"], kind="mergesort"
    ).reset_index(drop=True)
    hybrid = (0.25, float(candidates.iloc[0]["a1"]), float(candidates.iloc[0]["a2"]), 1.0)
    candidates["selected"] = (
        np.isclose(candidates["a1"], hybrid[1]) & np.isclose(candidates["a2"], hybrid[2])
    )
    candidates = candidates.drop(columns=["_objective_key", "_sum_key"])
    candidates.to_csv(output_dir / "hybrid_anchor_candidates.csv", index=False)
    candidates.loc[candidates["pareto_optimal"]].sort_values(
        ["R_G", "R_R"]
    ).to_csv(output_dir / "hybrid_pareto_frontier.csv", index=False)
    seed_zero_rows = []
    for a1, a2 in itertools.combinations(GRID[1:-1], 2):
        anchors = (0.25, a1, a2, 1.0)
        radius, mean_distance = _radius(anchors, seed_zero_coordinate)
        seed_zero_rows.append((radius, mean_distance, a1, a2))
    _, _, pure_a1, pure_a2 = min(seed_zero_rows)
    pure_geo = (0.25, pure_a1, pure_a2, 1.0)
    common_holdout = sorted(set(GRID) - set(UNIFORM_ANCHORS) - set(pure_geo) - set(hybrid))
    coordinate_table = pd.DataFrame({
        "width": GRID,
        "median_geometry_coordinate": [geometry_coordinate[width] for width in GRID],
        "normalized_log_flops": [resource_coordinate[width] for width in GRID],
        "flops": [flops[width] for width in GRID],
    })
    coordinate_table.to_csv(output_dir / "hybrid_coordinates.csv", index=False)
    compute_rows = []
    for method, anchors in (
        ("Uniform-4", UNIFORM_ANCHORS), ("PureGeo-4", pure_geo), ("Hybrid-4", hybrid)
    ):
        compute_rows.append({
            "method": method,
            "anchors": ",".join(f"{value:.2f}" for value in anchors),
            "forwards_per_batch": 4,
            "subnet_flops_per_batch": sum(flops[width] for width in anchors),
        })
    compute = pd.DataFrame(compute_rows)
    uniform_compute = float(compute.loc[compute["method"].eq("Uniform-4"), "subnet_flops_per_batch"].iloc[0])
    compute["relative_to_uniform"] = compute["subnet_flops_per_batch"] / uniform_compute
    compute.to_csv(output_dir / "hybrid_training_compute.csv", index=False)
    selected_row = candidates.loc[candidates["selected"]].iloc[0]
    artifact = {
        "selector": "Hybrid-v2 normalized minimax resource-functional coverage",
        "development_seeds": [0, 1, 2],
        "confirmatory_seeds": [3, 4, 5],
        "geometry_source": "edge-wise median of Uniform-100 learned_projection validation geometry",
        "resource_coordinate": "min-max normalized log FLOPs over the fixed dense grid",
        "accuracy_or_specialization_gap_used_for_selection": False,
        "uniform_anchors": list(UNIFORM_ANCHORS),
        "pure_geo_anchors": list(pure_geo),
        "hybrid_anchors": list(hybrid),
        "common_holdout_three_methods": common_holdout,
        "R_G_uniform": uniform_R_G,
        "R_R_uniform": uniform_R_R,
        "selected_R_G": float(selected_row["R_G"]),
        "selected_R_R": float(selected_row["R_R"]),
        "selected_normalized_R_G": float(selected_row["normalized_R_G"]),
        "selected_normalized_R_R": float(selected_row["normalized_R_R"]),
        "selected_hybrid_objective": float(selected_row["hybrid_objective"]),
        "tie_break": ["rounded_12dp_normalized_radius_sum", "lexicographic_a1_a2"],
        "geometry_csv_sha256": _sha256(geometry_path),
        "dense_metrics_csv_sha256": _sha256(metrics_path),
    }
    artifact_path = output_dir / "hybrid_selected_anchors.json"
    if artifact_path.is_file():
        previous = json.loads(artifact_path.read_text())
        immutable = [
            "hybrid_anchors", "pure_geo_anchors", "common_holdout_three_methods",
            "geometry_csv_sha256", "dense_metrics_csv_sha256",
        ]
        if any(previous.get(key) != artifact.get(key) for key in immutable):
            raise RuntimeError("Refusing to overwrite a previously frozen Hybrid-v2 selection")
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
    gates = {
        "confirmatory_seeds": [3, 4, 5],
        "run_all_confirmatory_seeds_regardless_of_seed_3_result": True,
        "methods_required_per_seed": ["Uniform-4", "PureGeo-4", "Hybrid-4"],
        "primary_common_holdout": common_holdout,
        "low_region": [width for width in common_holdout if 0.30 <= width <= 0.45],
        "high_region": [width for width in common_holdout if 0.80 <= width <= 0.95],
        "hybrid_mean_H_better_than_uniform_required_seeds": "3/3",
        "minimum_pooled_mean_H_effect": 0.002,
        "worst_H_margin": -0.005,
        "full_width_margin": -0.005,
        "minimum_low_region_puregeo_gain_retention": 0.60,
        "high_region_margin_vs_uniform": -0.002,
        "image_bootstrap_is_conditional_not_seed_level_inference": True,
    }
    (output_dir / "hybrid_v2_preregistered_gates.json").write_text(
        json.dumps(gates, indent=2) + "\n"
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    frontier = candidates.loc[candidates["pareto_optimal"]].sort_values("R_G")
    ax.scatter(candidates["R_G"], candidates["R_R"], alpha=0.28, label="all 91 sets")
    ax.plot(frontier["R_G"], frontier["R_R"], marker="o", color="black", label="Pareto frontier")
    for label, anchors, color in (
        ("Uniform", UNIFORM_ANCHORS, "steelblue"),
        ("PureGeo", pure_geo, "crimson"),
        ("Hybrid", hybrid, "darkgreen"),
    ):
        row = candidates.loc[
            np.isclose(candidates["a1"], anchors[1]) & np.isclose(candidates["a2"], anchors[2])
        ].iloc[0]
        ax.scatter(row["R_G"], row["R_R"], s=120, color=color, label=label, zorder=5)
        ax.annotate(label, (row["R_G"], row["R_R"]), xytext=(5, 5), textcoords="offset points")
    ax.set(xlabel="Functional radius $R_G$", ylabel="Resource radius $R_R$",
           title="Hybrid-v2 anchor-set Pareto frontier")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / "hybrid_anchor_pareto.png", dpi=200); plt.close(fig)
    report = [
        "# Hybrid-v2 CPU selection", "",
        f"Frozen Hybrid anchors: **{list(hybrid)}**", "",
        f"PureGeo-v1 anchors reconstructed from seed 0: `{list(pure_geo)}`", "",
        f"Three-method common holdout: `{common_holdout}`", "",
        "No accuracy, test prediction, or specialization gap was read by the selector.", "",
        "## Training-compute diagnostic", "", _markdown_table(compute), "",
        "## Selected candidate", "", _markdown_table(candidates.loc[candidates["selected"]]), "",
        "Seeds 0,1,2 are development-only. The locked confirmatory seeds are 3,4,5, and all "
        "three must be run regardless of the seed-3 result.",
    ]
    (output_dir / "hybrid_v2_selection_report.md").write_text("\n".join(report) + "\n")
    return artifact


def train_geo_seed(config: dict, root: Path, seed: int) -> Path:
    device = torch.device(config["experiment"]["device"])
    phase_1 = copy.deepcopy(config)
    phase_1["training"]["epochs"] = int(config["training"]["phase_1_epochs"])
    seed_everything(int(seed))
    phase_1_loaders = build_confirmatory_loaders(phase_1, training_seed=int(seed))
    first_dir = root / "geo" / f"seed_{seed}" / "phase_1"
    first = train_shared_reference(
        make_model(phase_1, device), phase_1_loaders.train, phase_1, device, first_dir
    )
    phase_2 = copy.deepcopy(config)
    phase_2["training"]["epochs"] = int(config["training"]["total_epochs"])
    # Match the existing Uniform extension: phase 2 starts in a fresh process,
    # so RNG and the shuffled loader restart from the registered model seed.
    seed_everything(int(seed))
    phase_2_loaders = build_confirmatory_loaders(phase_2, training_seed=int(seed))
    final_dir = root / "geo" / f"seed_{seed}"
    final = train_shared_reference(
        make_model(phase_2, device), phase_2_loaders.train, phase_2, device, final_dir,
        initial_checkpoint=first,
        initial_epoch=int(config["training"]["phase_1_epochs"]),
        learning_rate_override=float(config["training"]["extension_learning_rate"]),
    )
    first_metrics = pd.read_csv(first_dir / "training_metrics.csv")
    second_metrics = pd.read_csv(final_dir / "training_metrics.csv")
    combined = pd.concat([first_metrics, second_metrics], ignore_index=True)
    if sorted(combined["epoch"].unique()) != list(range(1, 101)):
        raise RuntimeError(f"Seed {seed} does not contain the complete 1..100 training history")
    combined.to_csv(final_dir / "training_metrics_1_100.csv", index=False)
    return final


def smoke_geo(config: dict, root: Path) -> Path:
    smoke = copy.deepcopy(config)
    smoke["training"]["epochs"] = int(config["training"]["smoke_epochs"])
    device = torch.device(config["experiment"]["device"])
    seed = CONFIRMATORY_SEEDS[0]
    seed_everything(seed)
    loaders = build_confirmatory_loaders(smoke, training_seed=seed)
    return train_shared_reference(
        make_model(smoke, device), loaders.train, smoke, device, root / "development" / "smoke"
    )


def evaluate_checkpoint(
    config: dict, root: Path, checkpoint: Path, seed: int, method: str, random_matrix: torch.Tensor
) -> Path:
    device = torch.device(config["experiment"]["device"])
    seed_everything(int(seed))
    loaders = build_confirmatory_loaders(config, training_seed=int(seed))
    output = root / "evaluation" / method / f"seed_{seed}"
    evaluate_shared_checkpoint(checkpoint, loaders, config, device, seed, output, random_matrix)
    model = make_model(config, device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    predictions = evaluate_prediction_grid(
        model, loaders.test, loaders.calibration, config, device, seed,
        root / "predictions" / method / f"seed_{seed}",
    )
    recorded = pd.read_csv(output / "budget_metrics.csv")
    measured = predictions.groupby("budget", as_index=False)["correct"].mean()
    check = recorded[["budget", "accuracy"]].merge(measured, on="budget", validate="one_to_one")
    if not np.allclose(check["accuracy"], check["correct"], atol=1e-12):
        raise RuntimeError(f"Aggregate/prediction accuracy mismatch for {method} seed {seed}")
    return output / "budget_metrics.csv"


def _summary(metrics: pd.DataFrame, geometry: pd.DataFrame, holdout: list[float]) -> pd.DataFrame:
    rows = []
    for (method, seed), group in metrics.groupby(["method", "seed"]):
        group = group.sort_values("budget")
        unseen = group.loc[group["budget"].round(2).isin(holdout)]
        local = geometry.loc[
            geometry["method"].eq(method)
            & geometry["seed"].eq(seed)
            & geometry["representation"].eq(PRIMARY_REPRESENTATION)
        ]
        log_flops = np.log(group["flops"].to_numpy(float))
        rows.append({
            "method": method,
            "seed": int(seed),
            "common_holdout_count": len(unseen),
            "mean_accuracy_H": float(unseen["accuracy"].mean()),
            "worst_accuracy_H": float(unseen["accuracy"].min()),
            "dense_mean_accuracy": float(group["accuracy"].mean()),
            "accuracy_logflops_auc": float(
                np.trapz(group["accuracy"].to_numpy(float), log_flops)
                / (log_flops[-1] - log_flops[0])
            ),
            "accuracy_025": float(group.loc[np.isclose(group["budget"], 0.25), "accuracy"].iloc[0]),
            "accuracy_100": float(group.loc[np.isclose(group["budget"], 1.00), "accuracy"].iloc[0]),
            "mean_G": float(local["G"].mean()),
            "max_G": float(local["G"].max()),
        })
    return pd.DataFrame(rows)


def _paired_by_width(baseline: pd.DataFrame, candidate: pd.DataFrame, holdout, replicates, seed):
    rows = []
    for index, width in enumerate(holdout):
        result = paired_accuracy_bootstrap(
            baseline, candidate, [width], replicates, seed + index
        )
        rows.append({"width": width, **result})
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    values = frame.reset_index(drop=True).copy()
    for column in values:
        values[column] = values[column].map(
            lambda value: f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)
        )
    header = "| " + " | ".join(map(str, values.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(values.columns)) + " |"
    rows = [
        "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"
        for row in values.astype(str).to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def finalize_rq2(config: dict, root: Path, selection: dict) -> dict:
    holdout = list(map(float, selection["common_holdout"]))
    metric_frames, geometry_frames = [], []
    for method in ("uniform", "geo"):
        for seed in CONFIRMATORY_SEEDS:
            metrics = pd.read_csv(root / "evaluation" / method / f"seed_{seed}" / "budget_metrics.csv")
            metrics["method"] = method
            metric_frames.append(metrics)
            geometry = pd.read_csv(
                root / "evaluation" / method / f"seed_{seed}" / "representation_local_geometry.csv"
            )
            geometry["method"] = method
            geometry_frames.append(geometry)
    metrics = pd.concat(metric_frames, ignore_index=True)
    geometry = pd.concat(geometry_frames, ignore_index=True)
    metrics.to_csv(root / "rq2_dense_metrics_all.csv", index=False)
    geometry.to_csv(root / "rq2_geometry_all.csv", index=False)
    summary = _summary(metrics, geometry, holdout)
    summary.to_csv(root / "rq2_method_summary.csv", index=False)
    wide = summary.pivot(index="seed", columns="method")
    comparison = pd.DataFrame({"seed": list(wide.index)})
    for metric in (
        "mean_accuracy_H", "worst_accuracy_H", "dense_mean_accuracy",
        "accuracy_logflops_auc", "accuracy_025", "accuracy_100", "mean_G", "max_G",
    ):
        comparison[f"uniform_{metric}"] = wide[metric]["uniform"].to_numpy()
        comparison[f"geo_{metric}"] = wide[metric]["geo"].to_numpy()
        comparison[f"delta_{metric}"] = comparison[f"geo_{metric}"] - comparison[f"uniform_{metric}"]
    bootstrap_rows, width_rows = [], []
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    for seed in CONFIRMATORY_SEEDS:
        uniform_predictions = pd.read_csv(
            root / "predictions" / "uniform" / f"seed_{seed}" / "predictions_all_widths.csv"
        )
        geo_predictions = pd.read_csv(
            root / "predictions" / "geo" / f"seed_{seed}" / "predictions_all_widths.csv"
        )
        bootstrap_rows.append({
            "seed": seed,
            **paired_accuracy_bootstrap(
                uniform_predictions, geo_predictions, holdout, replicates, bootstrap_seed + seed
            ),
        })
        by_width = _paired_by_width(
            uniform_predictions, geo_predictions, holdout, replicates,
            bootstrap_seed + 10000 * seed,
        )
        by_width.insert(0, "seed", seed)
        width_rows.append(by_width)
    bootstrap = pd.DataFrame(bootstrap_rows)
    by_width = pd.concat(width_rows, ignore_index=True)
    bootstrap.to_csv(root / "rq2_paired_bootstrap_common_holdout.csv", index=False)
    by_width.to_csv(root / "rq2_paired_bootstrap_by_width.csv", index=False)
    comparison = comparison.merge(bootstrap, on="seed", validate="one_to_one")
    thresholds = config["decision"]
    comparison["mean_H_improved"] = comparison["delta_mean_accuracy_H"] > 0
    comparison["worst_H_preserved"] = comparison["delta_worst_accuracy_H"] >= -float(
        thresholds["worst_accuracy_margin"]
    )
    comparison["full_width_preserved"] = comparison["delta_accuracy_100"] >= -float(
        thresholds["full_width_accuracy_margin"]
    )
    comparison.to_csv(root / "rq2_seed_comparison.csv", index=False)
    supported = bool(
        len(comparison) == 2
        and comparison["mean_H_improved"].all()
        and comparison["worst_H_preserved"].all()
        and comparison["full_width_preserved"].all()
        and comparison["delta_mean_accuracy_H"].mean() >= float(thresholds["minimum_mean_effect"])
    )
    verdict = "RQ2 SCREENING PASS" if supported else "RQ2 SCREENING NO-GO"
    decision = {
        "rq2_screening_pass": supported,
        "verdict": verdict,
        "selected_anchors": selection["selected_anchors"],
        "common_holdout": holdout,
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "mean_delta_accuracy_H": float(comparison["delta_mean_accuracy_H"].mean()),
        "state_count_matched": True,
        "subnet_compute_ratio_geo_to_uniform": selection["subnet_compute_ratio_geo_to_uniform"],
        "inference_note": "Image bootstrap is conditional on two trained seed pairs; it does not replace training-seed replication.",
    }
    (root / "rq2_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(9, 6))
    for (method, seed), group in metrics.groupby(["method", "seed"]):
        ax.plot(group["budget"], group["accuracy"], marker="o", alpha=0.75,
                label=f"{method} seed {seed}")
    for width in holdout:
        ax.axvline(width, color="grey", alpha=0.04)
    ax.set(xlabel="Width", ylabel="Test accuracy", title="RQ2 dense accuracy curves")
    ax.grid(alpha=0.25); ax.legend(ncol=2); fig.tight_layout()
    fig.savefig(root / "rq2_dense_accuracy_curves.png", dpi=200); plt.close(fig)
    report = [
        "# RQ2 geometry-guided anchor placement", "", f"Decision: **{verdict}**", "",
        f"Geometry anchors: `{selection['selected_anchors']}`", "",
        f"Common holdout: `{holdout}`", "",
        "The primary comparison is state-count matched (four subnet forwards per batch). "
        f"Geometry/Uniform subnet-FLOPs ratio is `{selection['subnet_compute_ratio_geo_to_uniform']:.4f}`; "
        "therefore the result is not described as strictly FLOPs-matched unless this ratio is near one.", "",
        "Seed 0 was used only for anchor selection. Seeds 1 and 2 are screening evidence.", "",
        "## Per-seed comparison", "", _markdown_table(comparison), "",
        "## Conditional paired image bootstrap", "", _markdown_table(bootstrap), "",
        "> Image-level confidence intervals condition on the trained model pairs and do not quantify "
        "training-seed uncertainty. A third confirmatory seed is required before a strong paper claim.",
    ]
    (root / "rq2_report.md").write_text("\n".join(report) + "\n")
    return decision
