from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def find_confirmatory_root(input_root: str | Path) -> Path:
    """Find the complete confirmatory run inside an arbitrarily nested Kaggle input."""
    input_root = Path(input_root)
    candidates = []
    for central_path in input_root.rglob("central_analysis_all_seeds.csv"):
        root = central_path.parent
        required = [
            root / "confirmatory_specialization_table.csv",
            *(root / f"seed_{seed}" / "results" / "local_sensitivity.csv" for seed in range(3)),
            *(root / f"seed_{seed}" / "features" for seed in range(3)),
        ]
        if all(path.exists() for path in required):
            candidates.append(root)
    if not candidates:
        raise FileNotFoundError(
            f"No complete confirmatory run found below {input_root}. "
            "Expected central analysis, specialization table, and seed feature folders."
        )
    candidates.sort(key=lambda path: (len(path.parts), str(path)))
    return candidates[0]


def compute_anchor_geometry_prior(
    baseline_root: str | Path,
    anchors: list[float],
    alpha: float = 1.0,
    beta: float = 0.5,
    epsilon: float = 1e-8,
) -> tuple[pd.DataFrame, dict[float, float]]:
    """Aggregate baseline G by nearest anchor and return mean-one frozen weights."""
    baseline_root = Path(baseline_root)
    frames = []
    for seed in range(3):
        frame = pd.read_csv(baseline_root / f"seed_{seed}" / "results" / "local_sensitivity.csv")
        frame = frame.rename(columns={"budget_start": "budget", "G_width": "G"})
        frame["seed"] = seed
        frames.append(frame[["seed", "budget", "G"]])
    local = pd.concat(frames, ignore_index=True)
    anchor_array = np.asarray(anchors, dtype=float)
    # np.argmin deterministically assigns exact Voronoi ties to the lower anchor.
    local["nearest_anchor"] = [
        float(anchor_array[np.argmin(np.abs(anchor_array - float(budget)))])
        for budget in local["budget"]
    ]
    scores = local.groupby("nearest_anchor", as_index=False)["G"].median()
    scores = scores.rename(columns={"nearest_anchor": "anchor", "G": "geometry_score"})
    region_metadata = (
        local.groupby("nearest_anchor")["budget"]
        .agg(
            interval_count="count",
            assigned_budgets=lambda values: ",".join(
                f"{value:.2f}" for value in sorted(set(values.astype(float)))
            ),
        )
        .reset_index()
        .rename(columns={"nearest_anchor": "anchor"})
    )
    scores = scores.merge(region_metadata, on="anchor", validate="one_to_one")
    if set(scores["anchor"]) != set(anchor_array):
        raise ValueError("Every anchor must receive at least one local-geometry interval")
    raw = np.power(scores["geometry_score"].to_numpy(float) + epsilon, alpha)
    geo_weight = raw / raw.mean()
    final_weight = (1.0 - beta) + beta * geo_weight
    scores["geo_weight"] = geo_weight
    scores["final_weight"] = final_weight
    scores["alpha"] = alpha
    scores["beta"] = beta
    scores["epsilon"] = epsilon
    weights = {
        float(row.anchor): float(row.final_weight) for row in scores.itertuples(index=False)
    }
    if not np.isclose(np.mean(list(weights.values())), 1.0):
        raise AssertionError("Anchor weights must have mean one")
    return scores.sort_values("anchor").reset_index(drop=True), weights
