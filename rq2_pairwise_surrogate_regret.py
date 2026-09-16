"""CPU-only pairwise surrogate-regret test on six HT trajectory states."""

from __future__ import annotations

import json
import zipfile
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from rq2_gradient_variance_v3 import importance_corrected_variance
from rq2_probabilistic_support import INTERIOR_WIDTHS, pair_marginals


PATHS = ("geo_ht", "resource_ht")
EPOCHS = (10, 50, 100)
NUM_WIDTHS = len(INTERIOR_WIDTHS)
UNIFORM_PI = np.full(NUM_WIDTHS, 1.0 / 7.0)
TARGET_WEIGHTS = np.full(NUM_WIDTHS, 1.0 / 7.0)
PAIR_INDICES = tuple(combinations(range(NUM_WIDTHS), 2))
NUM_PAIRS = len(PAIR_INDICES)
SHUFFLE_DRAWS = 256
SHUFFLE_SEED = 20260917


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


def _is_quick_root(root: Path) -> bool:
    required = (
        root / "metadata.json", root / "quick_pair_structure.csv",
        root / "quick_dynamic_geometry_by_width.csv",
    )
    required += tuple(
        root / "worker_paths" / path / f"interior_grams_epoch_{epoch:03d}.npy"
        for path in PATHS for epoch in EPOCHS
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        metadata = json.loads((root / "metadata.json").read_text())
    except json.JSONDecodeError:
        return False
    return metadata.get("status") == "RQ2_V3_QUICK_TRAJECTORY_DIAGNOSTIC_COMPLETE"


def find_quick_trajectory_root(input_root: str | Path, materialized_root: str | Path) -> Path:
    input_root, materialized_root = Path(input_root), Path(materialized_root)
    candidates = sorted({
        path.parent for path in input_root.rglob("quick_pair_structure.csv")
        if _is_quick_root(path.parent)
    })
    if not candidates:
        matching = []
        for archive in input_root.rglob("*.zip"):
            try:
                with zipfile.ZipFile(archive) as bundle:
                    names = bundle.namelist()
            except (OSError, zipfile.BadZipFile):
                continue
            if (
                any(name.endswith("quick_pair_structure.csv") for name in names)
                and all(any(name.endswith(
                    f"worker_paths/{path}/interior_grams_epoch_{epoch:03d}.npy"
                ) for name in names) for path in PATHS for epoch in EPOCHS)
            ):
                matching.append(archive)
        if len(matching) == 1:
            extracted = _safe_extract(matching[0], materialized_root)
            candidates = sorted({
                path.parent for path in extracted.rglob("quick_pair_structure.csv")
                if _is_quick_root(path.parent)
            })
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one complete dynamic-geometry quick diagnostic, found {candidates}"
        )
    return candidates[0]


def incidence_matrix() -> np.ndarray:
    matrix = np.zeros((NUM_WIDTHS, NUM_PAIRS), dtype=float)
    for column, (i, j) in enumerate(PAIR_INDICES):
        matrix[i, column] = 1.0
        matrix[j, column] = 1.0
    return matrix


def solve_pair_lp(scores: np.ndarray, maximize: bool = True) -> np.ndarray:
    """Optimize a linear pair score under fixed uniform inclusion marginals."""
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (NUM_PAIRS,) or not np.isfinite(scores).all():
        raise ValueError("Pair scores must be a finite aligned 91-vector")
    scale = max(float(np.max(np.abs(scores))), 1.0)
    objective = (-scores if maximize else scores) / scale
    result = linprog(
        objective, A_eq=incidence_matrix(), b_eq=UNIFORM_PI,
        bounds=[(0.0, None)] * NUM_PAIRS, method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Pair LP failed: {result.message}")
    q = np.clip(result.x, 0.0, None)
    if (
        abs(q.sum() - 1.0) > 1e-8
        or np.max(np.abs(incidence_matrix() @ q - UNIFORM_PI)) > 1e-8
    ):
        raise RuntimeError("Pair LP solution violates uniform marginals")
    return q


def pair_table(q: np.ndarray) -> pd.DataFrame:
    q = np.asarray(q, float)
    frame = pd.DataFrame([
        {
            "pair_index": index, "width_i": INTERIOR_WIDTHS[i],
            "width_j": INTERIOR_WIDTHS[j], "probability": q[index],
        }
        for index, (i, j) in enumerate(PAIR_INDICES)
    ])
    if np.max(np.abs(pair_marginals(frame) - UNIFORM_PI)) > 1e-8:
        raise RuntimeError("Constructed pair table has incorrect marginals")
    return frame


def gram_pair_scores(gram: np.ndarray):
    gram = np.asarray(gram, dtype=float)
    if gram.shape != (NUM_WIDTHS, NUM_WIDTHS):
        raise ValueError("Expected a 14x14 Gram matrix")
    distance, second_moment = [], []
    for i, j in PAIR_INDICES:
        distance.append(max(0.0, gram[i, i] + gram[j, j] - 2.0 * gram[i, j]))
        second_moment.append(gram[i, i] + gram[j, j] + 2.0 * gram[i, j])
    return np.asarray(distance), np.asarray(second_moment)


def minimax_positive_scale(gradient_distance_sq: np.ndarray, sw_sq: np.ndarray):
    d = np.asarray(gradient_distance_sq, float)
    s = np.asarray(sw_sq, float)
    if d.shape != (NUM_PAIRS,) or s.shape != (NUM_PAIRS,) or np.any(s < 0):
        raise ValueError("Minimax scale requires aligned nonnegative pair scores")
    # d - alpha*s <= epsilon; alpha*s - d <= epsilon.
    a_ub = np.vstack([
        np.column_stack([-s, -np.ones(NUM_PAIRS)]),
        np.column_stack([s, -np.ones(NUM_PAIRS)]),
    ])
    b_ub = np.concatenate([-d, d])
    result = linprog(
        np.asarray([0.0, 1.0]), A_ub=a_ub, b_ub=b_ub,
        bounds=((1e-15, None), (0.0, None)), method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Minimax scale LP failed: {result.message}")
    alpha, epsilon = map(float, result.x)
    direct = float(np.max(np.abs(d - alpha * s)))
    if abs(direct - epsilon) > 1e-7 * max(1.0, direct):
        raise RuntimeError("Minimax scale epsilon does not match direct residual")
    return alpha, direct


def _variance(gram: np.ndarray, q: np.ndarray) -> float:
    return float(importance_corrected_variance(
        gram, TARGET_WEIGHTS, UNIFORM_PI, pair_table(q)
    )["importance_corrected_variance"])


def _resource_flops(root: Path, optional_ht_root: str | Path | None) -> np.ndarray:
    geometry = pd.read_csv(root / "quick_dynamic_geometry_by_width.csv")
    if "flops" in geometry:
        values = (
            geometry.sort_values(["path", "epoch", "width"])
            .drop_duplicates("width").set_index("width").reindex(INTERIOR_WIDTHS).flops
            .to_numpy(float)
        )
        if np.isfinite(values).all() and np.all(values > 0):
            return values
    if optional_ht_root is not None:
        frame = pd.read_csv(Path(optional_ht_root) / "protocol" / "frozen_dynamic_marginals.csv")
        values = frame.set_index(frame.width.round(2)).reindex(INTERIOR_WIDTHS).flops.to_numpy(float)
        if np.isfinite(values).all() and np.all(values > 0):
            return values
    raise FileNotFoundError(
        "FLOPs are absent from the quick diagnostic; supply the HT root or rerun protocol v2"
    )


def _plots(output_dir: Path, main: pd.DataFrame, bound: pd.DataFrame,
           controls: pd.DataFrame) -> None:
    labels = [f"{'G' if row.path == 'geo_ht' else 'R'}-{row.epoch}" for row in main.itertuples()]
    x = np.arange(len(main))
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(x, main.oracle_gap_captured)
    axis.axhline(0, color="black", linewidth=1); axis.axhline(1, color="black", linestyle="--")
    axis.set(xticks=x, xticklabels=labels, ylabel="Oracle gap captured by SW pair policy")
    axis.grid(axis="y", alpha=0.25); fig.tight_layout()
    fig.savefig(output_dir / "pairwise_oracle_gap_captured.png", dpi=200); plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(x - 0.18, bound.actual_regret, 0.36, label="Actual SW regret")
    axis.bar(x + 0.18, bound.theorem_bound_2epsilon, 0.36, label="2 epsilon bound")
    axis.set(xticks=x, xticklabels=labels, ylabel="Variance regret / bound")
    axis.grid(axis="y", alpha=0.25); axis.legend(); fig.tight_layout()
    fig.savefig(output_dir / "pairwise_theorem_bound.png", dpi=200); plt.close(fig)

    pivot = controls.pivot(index=["path", "epoch"], columns="policy", values="variance").loc[
        [(row.path, row.epoch) for row in main.itertuples()]
    ]
    fig, axis = plt.subplots(figsize=(11, 5))
    width = 0.14
    for index, policy in enumerate(pivot.columns):
        axis.bar(x + (index - (len(pivot.columns)-1)/2) * width, pivot[policy], width, label=policy)
    axis.set(xticks=x, xticklabels=labels, ylabel="Exact conditional variance")
    axis.grid(axis="y", alpha=0.25); axis.legend(fontsize=8); fig.tight_layout()
    fig.savefig(output_dir / "pairwise_control_variances.png", dpi=200); plt.close(fig)


def run_pairwise_surrogate_regret(
    quick_root: str | Path,
    output_dir: str | Path,
    ht_root: str | Path | None = None,
    shuffle_draws: int = SHUFFLE_DRAWS,
) -> dict:
    quick_root, output_dir = Path(quick_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(shuffle_draws) < 100:
        raise ValueError("Use at least 100 shuffled-SW controls")
    pair_geometry = pd.read_csv(quick_root / "quick_pair_structure.csv")
    flops = _resource_flops(quick_root, ht_root)
    width_scores = np.asarray([
        (INTERIOR_WIDTHS[i] - INTERIOR_WIDTHS[j]) ** 2 for i, j in PAIR_INDICES
    ])
    resource_scores = np.asarray([(flops[i] - flops[j]) ** 2 for i, j in PAIR_INDICES])
    q_uniform = np.full(NUM_PAIRS, 1.0 / NUM_PAIRS)
    q_width = solve_pair_lp(width_scores, maximize=True)
    q_resource = solve_pair_lp(resource_scores, maximize=True)
    rng = np.random.default_rng(SHUFFLE_SEED)
    main_rows, theorem_rows, control_rows, shuffle_rows, assignment_rows = [], [], [], [], []
    for path in PATHS:
        for epoch in EPOCHS:
            grams = np.load(
                quick_root / "worker_paths" / path / f"interior_grams_epoch_{epoch:03d}.npy"
            ).astype(np.float64)
            if grams.shape[1:] != (NUM_WIDTHS, NUM_WIDTHS):
                raise RuntimeError("Invalid saved trajectory Gram tensor")
            gram = grams.mean(axis=0)
            gradient_distance_sq, estimator_second_scores = gram_pair_scores(gram)
            state_pairs = pair_geometry.loc[
                pair_geometry.path.eq(path) & pair_geometry.epoch.eq(epoch)
            ].sort_values(["width_i", "width_j"])
            expected_pairs = [(INTERIOR_WIDTHS[i], INTERIOR_WIDTHS[j]) for i, j in PAIR_INDICES]
            observed_pairs = list(zip(state_pairs.width_i.round(2), state_pairs.width_j.round(2)))
            if observed_pairs != expected_pairs:
                raise RuntimeError(f"Pairwise SW rows are incomplete for {path} epoch {epoch}")
            sw_sq = np.square(state_pairs.representation_sw.to_numpy(float))
            q_oracle = solve_pair_lp(gradient_distance_sq, maximize=True)
            q_direct_variance = solve_pair_lp(estimator_second_scores, maximize=False)
            q_sw = solve_pair_lp(sw_sq, maximize=True)
            variances = {
                "uniform_pair": _variance(gram, q_uniform),
                "width_distance": _variance(gram, q_width),
                "resource_distance": _variance(gram, q_resource),
                "wasserstein": _variance(gram, q_sw),
                "gradient_oracle": _variance(gram, q_oracle),
                "direct_variance_lp": _variance(gram, q_direct_variance),
            }
            theorem1_error = abs(variances["gradient_oracle"] - variances["direct_variance_lp"])
            theorem1_tolerance = 1e-7 * max(1.0, abs(variances["gradient_oracle"]))
            if theorem1_error > theorem1_tolerance:
                raise RuntimeError(f"Theorem-1 LP identity failed at {path}, epoch {epoch}")
            alpha, epsilon = minimax_positive_scale(gradient_distance_sq, sw_sq)
            regret = variances["wasserstein"] - variances["gradient_oracle"]
            if regret < -theorem1_tolerance or regret > 2 * epsilon + theorem1_tolerance:
                raise RuntimeError(f"Theorem-2 regret bound failed at {path}, epoch {epoch}")
            oracle_gap = variances["uniform_pair"] - variances["gradient_oracle"]
            captured = (
                (variances["uniform_pair"] - variances["wasserstein"]) / oracle_gap
                if oracle_gap > theorem1_tolerance else np.nan
            )
            shuffled_values = []
            for draw in range(int(shuffle_draws)):
                shuffled = rng.permutation(sw_sq)
                q_shuffle = solve_pair_lp(shuffled, maximize=True)
                value = _variance(gram, q_shuffle)
                shuffled_values.append(value)
                shuffle_rows.append({
                    "path": path, "epoch": epoch, "draw": draw,
                    "variance_shuffled_sw": value,
                    "wasserstein_better": bool(variances["wasserstein"] < value),
                })
            shuffled_values = np.asarray(shuffled_values)
            main_rows.append({
                "state": f"{'G' if path == 'geo_ht' else 'R'}-{epoch}",
                "path": path, "epoch": epoch,
                "V_U": variances["uniform_pair"],
                "V_R": variances["resource_distance"],
                "V_W": variances["wasserstein"],
                "V_oracle": variances["gradient_oracle"],
                "oracle_gap_captured": captured,
                "V_shuffled_mean": float(shuffled_values.mean()),
                "V_shuffled_median": float(np.median(shuffled_values)),
                "wasserstein_better_than_shuffle_fraction": float(
                    np.mean(variances["wasserstein"] < shuffled_values)
                ),
            })
            theorem_rows.append({
                "path": path, "epoch": epoch,
                "theorem1_oracle_variance": variances["gradient_oracle"],
                "theorem1_direct_lp_variance": variances["direct_variance_lp"],
                "theorem1_absolute_error": theorem1_error,
                "theorem1_pass": bool(theorem1_error <= theorem1_tolerance),
                "alpha_star": alpha, "epsilon_star": epsilon,
                "actual_regret": regret, "theorem_bound_2epsilon": 2 * epsilon,
                "regret_over_bound": regret / (2 * epsilon) if epsilon > 0 else np.nan,
                "theorem2_pass": bool(regret <= 2 * epsilon + theorem1_tolerance),
            })
            for policy, variance in variances.items():
                control_rows.append({
                    "path": path, "epoch": epoch, "policy": policy, "variance": variance,
                    "ratio_vs_uniform": variance / variances["uniform_pair"],
                })
            for policy, q in (
                ("uniform_pair", q_uniform), ("width_distance", q_width),
                ("resource_distance", q_resource), ("wasserstein", q_sw),
                ("gradient_oracle", q_oracle),
            ):
                for pair_index, probability in enumerate(q):
                    i, j = PAIR_INDICES[pair_index]
                    assignment_rows.append({
                        "path": path, "epoch": epoch, "policy": policy,
                        "pair_index": pair_index, "width_i": INTERIOR_WIDTHS[i],
                        "width_j": INTERIOR_WIDTHS[j], "probability": probability,
                    })
    main = pd.DataFrame(main_rows)
    theorem = pd.DataFrame(theorem_rows)
    controls = pd.DataFrame(control_rows)
    shuffles = pd.DataFrame(shuffle_rows)
    assignments = pd.DataFrame(assignment_rows)
    main.to_csv(output_dir / "pairwise_surrogate_regret.csv", index=False)
    theorem.to_csv(output_dir / "pairwise_theorem_checks.csv", index=False)
    controls.to_csv(output_dir / "pairwise_control_variances.csv", index=False)
    shuffles.to_csv(output_dir / "pairwise_shuffled_sw_controls.csv", index=False)
    assignments.to_csv(output_dir / "pairwise_policy_assignments.csv", index=False)
    _plots(output_dir, main, theorem, controls)
    metadata = {
        "status": "PAIRWISE_SURROGATE_REGRET_COMPLETE",
        "states": main.state.tolist(), "num_states": len(main),
        "uniform_marginal": 1.0 / 7.0, "num_pairs": NUM_PAIRS,
        "theorem1_all_pass": bool(theorem.theorem1_pass.all()),
        "theorem2_all_pass": bool(theorem.theorem2_pass.all()),
        "shuffle_draws_per_state": int(shuffle_draws), "shuffle_seed": SHUFFLE_SEED,
        "mean_oracle_gap_captured": float(main.oracle_gap_captured.mean()),
        "median_oracle_gap_captured": float(main.oracle_gap_captured.median()),
        "wasserstein_better_than_shuffle_all_states": bool(
            (main.wasserstein_better_than_shuffle_fraction > 0.5).all()
        ),
        "training_performed": False, "gpu_required": False,
        "accuracy_used": False, "test_used": False, "policy_updated": False,
        "interpretation": "CPU-only surrogate diagnostic; does not authorize training",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
