"""CPU-only FinalGeo selector and immutable confirmatory protocol freeze."""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from rq2_anchor_placement import GRID, UNIFORM_ANCHORS, _sha256
from rq2_parameter_exposure import _coordinates, _radius


EXPECTED_PUREGEO = (0.25, 0.40, 0.60, 1.00)
MIN_COMPUTE_RATIO = 0.90
MAX_COMPUTE_RATIO = 1.00
CONFIRMATORY_SEEDS = (3, 4, 5)


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_provenance(repo_root: str | Path | None = None) -> dict:
    root = Path(repo_root or Path(__file__).resolve().parent)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        return {"git_commit": commit, "git_worktree_clean": not bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "UNAVAILABLE", "git_worktree_clean": False}


def enumerate_finalgeo_candidates(development_root: str | Path) -> tuple[pd.DataFrame, dict]:
    """Enumerate all 91 candidates using geometry and FLOPs only."""
    root = Path(development_root)
    geometry_coordinate, _, flops, pure_geo, source_hashes = _coordinates(root)
    if tuple(pure_geo) != EXPECTED_PUREGEO:
        raise RuntimeError(
            f"Frozen PureGeo-v1 anchors {pure_geo} do not match required {EXPECTED_PUREGEO}"
        )
    uniform_compute = float(sum(flops[width] for width in UNIFORM_ANCHORS))
    uniform_R_G, uniform_mean_G = _radius(UNIFORM_ANCHORS, geometry_coordinate)
    if uniform_compute <= 0 or uniform_R_G <= 0:
        raise RuntimeError("Uniform compute and geometry radius must be positive")

    rows = []
    for a1, a2 in itertools.combinations(GRID[1:-1], 2):
        anchors = (GRID[0], a1, a2, GRID[-1])
        R_G, mean_G = _radius(anchors, geometry_coordinate)
        compute = float(sum(flops[width] for width in anchors))
        ratio = compute / uniform_compute
        rows.append({
            "a1": a1,
            "a2": a2,
            "anchors": ",".join(f"{width:.2f}" for width in anchors),
            "R_G": R_G,
            "mean_geometry_distance": mean_G,
            "normalized_R_G_vs_uniform": R_G / uniform_R_G,
            "geometry_radius_improvement_vs_uniform": 1.0 - R_G / uniform_R_G,
            "subnet_flops_per_batch": compute,
            "compute_ratio_vs_uniform": ratio,
            "compute_lower_bound": MIN_COMPUTE_RATIO * uniform_compute,
            "compute_upper_bound": MAX_COMPUTE_RATIO * uniform_compute,
            "meets_compute_lower_bound": bool(ratio >= MIN_COMPUTE_RATIO - 1e-12),
            "meets_compute_upper_bound": bool(ratio <= MAX_COMPUTE_RATIO + 1e-12),
        })
    candidates = pd.DataFrame(rows)
    if len(candidates) != 91:
        raise RuntimeError(f"Expected 91 endpoint-locked K=4 candidates, got {len(candidates)}")
    candidates["compute_feasible"] = (
        candidates["meets_compute_lower_bound"] & candidates["meets_compute_upper_bound"]
    )
    feasible = candidates.loc[candidates["compute_feasible"]].copy()
    if feasible.empty:
        raise RuntimeError("No candidate satisfies the frozen compute constraint")

    # R_G is the only objective. Mean geometry distance is a geometry-only
    # deterministic tie-break; compute never ranks candidates after filtering.
    feasible["_R_G_key"] = feasible["R_G"].round(12)
    feasible["_mean_G_key"] = feasible["mean_geometry_distance"].round(12)
    feasible = feasible.sort_values(
        ["_R_G_key", "_mean_G_key", "a1", "a2"], kind="mergesort"
    )
    feasible["selection_rank_among_feasible"] = np.arange(1, len(feasible) + 1)
    rank = feasible.set_index(["a1", "a2"])["selection_rank_among_feasible"]
    candidates["selection_rank_among_feasible"] = [
        int(rank.loc[(row.a1, row.a2)]) if row.compute_feasible else pd.NA
        for row in candidates.itertuples()
    ]
    selected = feasible.iloc[0]
    finalgeo = (0.25, float(selected["a1"]), float(selected["a2"]), 1.0)
    candidates["selected"] = (
        np.isclose(candidates["a1"], finalgeo[1])
        & np.isclose(candidates["a2"], finalgeo[2])
    )
    candidates["is_uniform"] = (
        np.isclose(candidates["a1"], UNIFORM_ANCHORS[1])
        & np.isclose(candidates["a2"], UNIFORM_ANCHORS[2])
    )
    candidates["is_puregeo"] = (
        np.isclose(candidates["a1"], EXPECTED_PUREGEO[1])
        & np.isclose(candidates["a2"], EXPECTED_PUREGEO[2])
    )
    candidates = candidates.sort_values(["a1", "a2"], kind="mergesort").reset_index(drop=True)
    metadata = {
        "geometry_coordinate": geometry_coordinate,
        "flops": flops,
        "source_hashes": source_hashes,
        "uniform_compute": uniform_compute,
        "uniform_R_G": uniform_R_G,
        "uniform_mean_geometry_distance": uniform_mean_G,
        "puregeo_anchors": pure_geo,
        "finalgeo_anchors": finalgeo,
    }
    return candidates, metadata


def _success_gates(common_holdout: list[float]) -> dict:
    """Carry forward the already preregistered Hybrid-v2 evaluation gates."""
    return {
        "run_all_confirmatory_seeds_regardless_of_seed_3_result": True,
        "finalgeo_mean_common_holdout_better_than_uniform_required_seeds": "3/3",
        "minimum_pooled_mean_common_holdout_effect": 0.002,
        "worst_common_holdout_accuracy_margin_vs_uniform": -0.005,
        "full_width_accuracy_margin_vs_uniform": -0.005,
        "minimum_low_region_puregeo_gain_retention": 0.60,
        "high_region_accuracy_margin_vs_uniform": -0.002,
        "low_region": [width for width in common_holdout if 0.30 <= width <= 0.45],
        "high_region": [width for width in common_holdout if 0.80 <= width <= 0.95],
        "image_bootstrap_is_conditional_not_seed_level_inference": True,
    }


def freeze_finalgeo_selector(
    development_root: str | Path,
    output_dir: str | Path,
    repo_root: str | Path | None = None,
) -> dict:
    """Select FinalGeo once and write the immutable pre-confirmatory artifacts."""
    root, output_dir = Path(development_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates, metadata = enumerate_finalgeo_candidates(root)
    finalgeo = tuple(metadata["finalgeo_anchors"])
    puregeo = tuple(metadata["puregeo_anchors"])
    selected = candidates.loc[candidates["selected"]]
    if len(selected) != 1:
        raise RuntimeError(f"Expected exactly one selected FinalGeo candidate, got {len(selected)}")
    selected_row = selected.iloc[0]
    common_holdout = sorted(
        set(GRID) - set(UNIFORM_ANCHORS) - set(puregeo) - set(finalgeo)
    )
    git = _git_provenance(repo_root)
    selection_rule = {
        "objective": "minimize_R_G",
        "objective_is_geometry_only": True,
        "constraints": {
            "minimum_compute_ratio_vs_uniform": MIN_COMPUTE_RATIO,
            "maximum_compute_ratio_vs_uniform": MAX_COMPUTE_RATIO,
            "inclusive_bounds": True,
        },
        "tie_break": [
            "R_G_rounded_to_12_decimal_places",
            "mean_geometry_distance_rounded_to_12_decimal_places",
            "lexicographic_a1_then_a2",
        ],
        "accuracy_used": False,
        "parameter_exposure_used": False,
        "gradient_interference_used": False,
    }
    freeze_core = {
        "status": "FROZEN_BEFORE_CONFIRMATORY_SEEDS",
        "selector_name": "FinalGeo compute-preserving functional geometry selector",
        "uniform_anchors": list(UNIFORM_ANCHORS),
        "puregeo_anchors": list(puregeo),
        "finalgeo_anchors": list(finalgeo),
        "selection_rule": selection_rule,
        "geometry_source": {
            "representation": "learned_projection",
            "aggregation": "edge-wise median of Uniform geometry over development seeds 0,1,2",
            "feature_subset": "fixed 2000-image validation subset",
            "source_artifact_sha256": metadata["source_hashes"],
        },
        "compute_constraint": {
            "definition": "C(A)=sum FLOPs(a) over four anchor forwards per batch",
            "uniform_compute": metadata["uniform_compute"],
            "lower_bound": MIN_COMPUTE_RATIO * metadata["uniform_compute"],
            "upper_bound": MAX_COMPUTE_RATIO * metadata["uniform_compute"],
            "selected_compute": float(selected_row["subnet_flops_per_batch"]),
            "selected_ratio_vs_uniform": float(selected_row["compute_ratio_vs_uniform"]),
        },
        "selected_geometry": {
            "R_G": float(selected_row["R_G"]),
            "uniform_R_G": metadata["uniform_R_G"],
            "normalized_R_G_vs_uniform": float(selected_row["normalized_R_G_vs_uniform"]),
            "geometry_radius_improvement_vs_uniform": float(
                selected_row["geometry_radius_improvement_vs_uniform"]
            ),
        },
        "candidate_grid": list(GRID),
        "K": 4,
        "fixed_endpoints": [0.25, 1.0],
        "candidate_count": 91,
        "compute_feasible_candidate_count": int(candidates["compute_feasible"].sum()),
        "development_seeds": [0, 1, 2],
        "seeds_confirmatory": list(CONFIRMATORY_SEEDS),
        "methods_confirmatory": ["Uniform", "PureGeo", "FinalGeo"],
        "dense_eval_grid": list(GRID),
        "common_holdout_definition": "dense grid minus union of Uniform, PureGeo, and FinalGeo anchors",
        "common_holdout": common_holdout,
        "success_gates": _success_gates(common_holdout),
        **git,
    }
    freeze_id = _canonical_hash(freeze_core)
    protocol = {**freeze_core, "freeze_id_sha256": freeze_id}
    selected_payload = {
        "status": "FROZEN_BEFORE_CONFIRMATORY_SEEDS",
        "finalgeo_anchors": list(finalgeo),
        "R_G": float(selected_row["R_G"]),
        "mean_geometry_distance": float(selected_row["mean_geometry_distance"]),
        "compute_ratio_vs_uniform": float(selected_row["compute_ratio_vs_uniform"]),
        "selection_rank_among_feasible": int(selected_row["selection_rank_among_feasible"]),
        "freeze_id_sha256": freeze_id,
    }

    candidates_path = output_dir / "finalgeo_selector_candidates.csv"
    selected_path = output_dir / "finalgeo_selected_anchors.json"
    protocol_path = output_dir / "finalgeo_frozen_protocol.json"
    if protocol_path.is_file():
        previous = json.loads(protocol_path.read_text())
        if previous.get("freeze_id_sha256") != freeze_id or previous != protocol:
            raise RuntimeError(
                "FinalGeo protocol is already frozen and differs from the requested selector/source/revision"
            )
        if not candidates_path.is_file() or not selected_path.is_file():
            raise RuntimeError("Frozen FinalGeo protocol exists but companion artifacts are missing")
        if json.loads(selected_path.read_text()) != selected_payload:
            raise RuntimeError("Frozen FinalGeo selected-anchor artifact was modified")
        return protocol

    candidates.to_csv(candidates_path, index=False)
    selected_path.write_text(json.dumps(selected_payload, indent=2) + "\n")
    # Written last: its presence marks the selector as permanently frozen.
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol

