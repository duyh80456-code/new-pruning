"""CPU-only theory probe for geometry-driven probabilistic width support.

This module never reads accuracy and never authorizes model training. It checks
the chain geometry -> functional cell mass -> waiting cost -> analytic marginal
allocation -> maximum-entropy pair realization.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import xlogy

from rq2_anchor_placement import GRID, PRIMARY_REPRESENTATION, _sha256
from rq2_probabilistic_support import (
    ENDPOINTS,
    INTERIOR_WIDTHS,
    maximum_entropy_pairs,
    pair_marginals,
)


P_VALUES = (0.25, 0.5, 1.0, 2.0, 4.0)
LARGE_P_VALUES = (10.0, 100.0, 1000.0)
INTERIOR_BUDGET = 2.0
NUMERIC_EPS = 1e-10
WAITING_SIMULATION_STEPS = 2_000_000
REGIONS = {
    "low": (0.30, 0.35, 0.40, 0.45),
    "mid": (0.50, 0.55, 0.60, 0.65, 0.70, 0.75),
    "high": (0.80, 0.85, 0.90, 0.95),
}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _p_tag(p: float) -> str:
    return f"{int(round(100 * p)):03d}"


def load_frozen_inputs(development_root: str | Path):
    """Read only frozen development geometry and FLOPs metadata."""
    root = Path(development_root)
    geometry_path = root / "rq2_geometry_all.csv"
    coordinate_path = root / "protocol" / "geometry_trajectory_coordinates.csv"
    metrics_path = root / "rq2_dense_metrics_all.csv"
    required = (geometry_path, coordinate_path, metrics_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete frozen RQ2 development evidence: {missing}")

    geometry = pd.read_csv(
        geometry_path,
        usecols=["method", "seed", "representation", "budget_start", "budget_end", "G"],
    )
    seed_12 = geometry.loc[
        geometry["method"].eq("uniform")
        & geometry["representation"].eq(PRIMARY_REPRESENTATION)
        & geometry["seed"].astype(int).isin([1, 2]),
        ["seed", "budget_start", "budget_end", "G"],
    ].copy()

    coordinates = pd.read_csv(
        coordinate_path, usecols=["width", "incoming_edge_length"]
    ).sort_values("width")
    if tuple(coordinates["width"].round(2)) != GRID:
        raise RuntimeError("Seed-0 frozen geometry does not cover the registered dense grid")
    seed_zero = pd.DataFrame({
        "seed": 0,
        "budget_start": GRID[:-1],
        "budget_end": GRID[1:],
        "G": (
            coordinates["incoming_edge_length"].iloc[1:].to_numpy(float)
            / np.diff(np.asarray(GRID, float))
        ),
    })
    per_seed_edges = pd.concat([seed_zero, seed_12], ignore_index=True)
    per_seed_edges["budget_start"] = per_seed_edges["budget_start"].round(2)
    per_seed_edges["budget_end"] = per_seed_edges["budget_end"].round(2)
    expected_edges = set(zip(GRID[:-1], GRID[1:]))
    observed_edges = set(zip(per_seed_edges["budget_start"], per_seed_edges["budget_end"]))
    counts = per_seed_edges.groupby(["budget_start", "budget_end"])["seed"].nunique()
    if observed_edges != expected_edges or len(per_seed_edges) != 45 or not (counts == 3).all():
        raise RuntimeError("Expected all 15 adjacent edges for Uniform seeds 0,1,2")
    per_seed_edges["delta_c"] = (
        per_seed_edges["budget_end"] - per_seed_edges["budget_start"]
    )
    per_seed_edges["edge_length"] = per_seed_edges["G"] * per_seed_edges["delta_c"]
    edge = (
        per_seed_edges.groupby(["budget_start", "budget_end"], as_index=False)["edge_length"]
        .median().sort_values("budget_start").reset_index(drop=True)
        .rename(columns={"budget_start": "left", "budget_end": "right", "edge_length": "e_i"})
    )
    edge["delta_c"] = edge["right"] - edge["left"]
    edge["functional_speed_g"] = edge["e_i"] / edge["delta_c"]
    if len(edge) != 15 or np.any(~np.isfinite(edge["e_i"])) or np.any(edge["e_i"] <= 0):
        raise RuntimeError("Median edge geometry must contain 15 strictly positive finite edges")

    metrics = pd.read_csv(metrics_path, usecols=["method", "budget", "flops"])
    uniform = metrics.loc[metrics["method"].eq("uniform")]
    flops = {
        round(float(width), 2): float(value)
        for width, value in uniform.groupby("budget")["flops"].first().items()
    }
    if set(flops) != set(GRID) or any(value <= 0 for value in flops.values()):
        raise RuntimeError("A positive Uniform FLOPs value is required at every width")
    hashes = {path.name: _sha256(path) for path in required}
    return edge, flops, hashes


def integrate_piecewise_cell(edge: pd.DataFrame, lower: float, upper: float) -> float:
    """Exactly integrate the piecewise-constant functional speed over one cell."""
    total = 0.0
    for row in edge.itertuples():
        overlap = max(0.0, min(upper, float(row.right)) - max(lower, float(row.left)))
        total += overlap * float(row.functional_speed_g)
    return float(total)


def build_functional_cells(edge: pd.DataFrame, flops: dict[float, float]):
    edge = edge.sort_values("left").reset_index(drop=True)
    rows = []
    for index, width in enumerate(INTERIOR_WIDTHS):
        left_edge = float(edge.iloc[index]["e_i"])
        right_edge = float(edge.iloc[index + 1]["e_i"])
        mass = 0.5 * (left_edge + right_edge)
        left_bound = 0.5 * (GRID[index] + GRID[index + 1])
        right_bound = 0.5 * (GRID[index + 1] + GRID[index + 2])
        integrated = integrate_piecewise_cell(edge, left_bound, right_bound)
        rows.append({
            "width": width,
            "voronoi_left": left_bound,
            "voronoi_right": right_bound,
            "left_edge_length": left_edge,
            "right_edge_length": right_edge,
            "functional_cell_mass": mass,
            "piecewise_integral_mass": integrated,
            "absolute_mass_difference": abs(mass - integrated),
            "flops": flops[width],
        })
    cells = pd.DataFrame(rows)
    if np.any(cells["functional_cell_mass"].to_numpy(float) <= 0):
        raise RuntimeError("Every interior functional cell must have positive mass")
    if not np.allclose(
        cells["functional_cell_mass"], cells["piecewise_integral_mass"],
        rtol=1e-10, atol=1e-12,
    ):
        raise RuntimeError("Continuous piecewise integral does not reproduce discrete cell mass")
    return cells


def analytic_allocation(a: np.ndarray, p: float, budget: float = INTERIOR_BUDGET) -> np.ndarray:
    a = np.asarray(a, float)
    if p <= 0 or np.any(~np.isfinite(a)) or np.any(a <= 0):
        raise ValueError("p and all functional cell masses must be positive")
    weights = np.power(a, 1.0 / (p + 1.0))
    pi = budget * weights / weights.sum()
    if np.any(pi <= 0) or abs(pi.sum() - budget) >= 1e-12:
        raise RuntimeError("Invalid analytic allocation")
    return pi


def objective_jp(pi: np.ndarray, a: np.ndarray, p: float) -> float:
    return float(np.sum(np.asarray(a, float) / np.power(np.asarray(pi, float), p)))


def numerical_allocation(
    a: np.ndarray,
    p: float,
    flops: np.ndarray | None = None,
    compute_cap: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Independently verify the analytic solution, adding the cap only if requested."""
    a = np.asarray(a, float)
    weights = a / float(a.mean())
    x0 = np.full(len(a), INTERIOR_BUDGET / len(a))
    constraints = [{"type": "eq", "fun": lambda x: x.sum() - INTERIOR_BUDGET}]
    if compute_cap is not None:
        if flops is None:
            raise ValueError("flops are required with a compute cap")
        scale = float(compute_cap)
        scaled = np.asarray(flops, float) / scale
        constraints.append({"type": "ineq", "fun": lambda x: 1.0 - scaled @ x})
    result = minimize(
        lambda x: float(np.sum(weights / np.power(x, p))), x0,
        jac=lambda x: -p * weights / np.power(x, p + 1.0), method="SLSQP",
        bounds=[(NUMERIC_EPS, 1.0)] * len(a), constraints=constraints,
        options={"ftol": 1e-13, "maxiter": 10000, "disp": False},
    )
    pi = np.asarray(result.x, float)
    sum_error = abs(pi.sum() - INTERIOR_BUDGET)
    cap_ok = compute_cap is None or float(np.asarray(flops) @ pi) <= compute_cap * (1 + 1e-8)
    if not result.success or sum_error > 1e-8 or np.any(pi <= 0) or np.any(pi > 1 + 1e-8) or not cap_ok:
        raise RuntimeError(
            f"Numerical J_p verification failed for p={p}: {result.message}; sum_error={sum_error}"
        )
    return pi, {
        "solver": "scipy.optimize.minimize/SLSQP",
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "iterations": int(result.nit),
        "numeric_objective": objective_jp(pi, a, p),
    }


def pair_expectation_checks(pair_table: pd.DataFrame, pi: np.ndarray, seed: int = 123) -> dict:
    """Check scalar-loss and vector-gradient expectations implied by pair marginals."""
    rng = np.random.default_rng(seed)
    losses = rng.normal(size=len(pi))
    gradients = rng.normal(size=(len(pi), 128))
    width_index = {width: index for index, width in enumerate(INTERIOR_WIDTHS)}
    pair_loss = 0.0
    pair_gradient = np.zeros(128)
    for row in pair_table.itertuples():
        i = width_index[round(float(row.width_i), 2)]
        j = width_index[round(float(row.width_j), 2)]
        probability = float(row.probability)
        pair_loss += probability * (losses[i] + losses[j])
        pair_gradient += probability * (gradients[i] + gradients[j])
    marginal_loss = float(pi @ losses)
    marginal_gradient = pi @ gradients
    return {
        "loss_identity_absolute_error": float(abs(pair_loss - marginal_loss)),
        "gradient_identity_max_absolute_error": float(
            np.max(np.abs(pair_gradient - marginal_gradient))
        ),
    }


def simulate_pair_waiting_times(
    pair_table: pd.DataFrame,
    pi: np.ndarray,
    p: float,
    n_steps: int = WAITING_SIMULATION_STEPS,
    seed: int = 123,
) -> pd.DataFrame:
    """Validate E[T_i]=1/pi_i using draws from the actual pair distribution."""
    rng = np.random.default_rng(seed)
    probabilities = pair_table["probability"].to_numpy(float)
    pair_i = pair_table["width_i"].to_numpy(float)
    pair_j = pair_table["width_j"].to_numpy(float)
    draws = rng.choice(len(pair_table), size=int(n_steps), p=probabilities / probabilities.sum())
    rows = []
    for index, width in enumerate(INTERIOR_WIDTHS):
        hits = np.flatnonzero((pair_i[draws] == width) | (pair_j[draws] == width))
        gaps = np.diff(hits)
        if len(gaps) == 0:
            raise RuntimeError(f"No repeated inclusion observed for width {width}")
        empirical = float(gaps.mean())
        theoretical = float(1.0 / pi[index])
        rows.append({
            "p": p,
            "width": width,
            "pi": float(pi[index]),
            "theoretical_mean_wait": theoretical,
            "empirical_mean_wait": empirical,
            "relative_error": abs(empirical - theoretical) / theoretical,
            "observed_intervals": int(len(gaps)),
            "simulation_batches": int(n_steps),
            "simulation_seed": int(seed),
        })
    return pd.DataFrame(rows)


def _plot_outputs(
    output_dir: Path,
    edge: pd.DataFrame,
    marginals: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    dense = np.linspace(GRID[0], GRID[-1], 5000)
    right_edges = edge["right"].to_numpy(float)
    speeds = edge["functional_speed_g"].to_numpy(float)
    indices = np.clip(np.searchsorted(right_edges, dense, side="right"), 0, len(speeds) - 1)
    g_dense = speeds[indices]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(dense, g_dense, where="post")
    ax.set(xlabel="Width c", ylabel="Functional speed g(c)")
    ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(output_dir / "functional_speed.png", dpi=200)
    fig.savefig(output_dir / "functional_speed_g.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for p in P_VALUES:
        density = np.power(g_dense, 1.0 / (p + 1.0))
        density /= np.trapz(density, dense)
        ax.plot(dense, density, label=f"p={p:g}")
    ax.set(xlabel="Width c", ylabel="Normalized optimal density rho_p(c)")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / "continuous_optimal_density_by_p.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for p, group in marginals.groupby("p"):
        ax.plot(group["width"], group["pi"], marker="o", label=f"p={p:g}")
    ax.set(xlabel="Interior width", ylabel="Marginal inclusion probability")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / "allocation_family.png", dpi=200); plt.close(fig)

    for filename, columns, ylabel in (
        ("entropy_vs_p.png", ["entropy"], "Entropy (nats)"),
        ("compute_vs_p.png", ["expected_total_compute_ratio"], "Compute / Uniform"),
        ("region_mass_vs_p.png", ["low_mass", "mid_mass", "high_mass"], "Marginal mass"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for column in columns:
            ax.plot(summary["p"], summary[column], marker="o", label=column)
        ax.set(xlabel="p", ylabel=ylabel, xscale="log")
        ax.grid(alpha=0.25)
        if len(columns) > 1:
            ax.legend()
        fig.tight_layout(); fig.savefig(output_dir / filename, dpi=200); plt.close(fig)


def run_theory_probe(
    development_root: str | Path,
    output_dir: str | Path,
    waiting_steps: int = WAITING_SIMULATION_STEPS,
) -> dict:
    """Run the complete no-accuracy, no-training theory validation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    edge, flops_by_width, source_hashes = load_frozen_inputs(development_root)
    cells = build_functional_cells(edge, flops_by_width)
    a = cells["functional_cell_mass"].to_numpy(float)
    flops = cells["flops"].to_numpy(float)
    endpoint_compute = float(sum(flops_by_width[width] for width in ENDPOINTS))
    uniform_interior_compute = float(flops_by_width[0.50] + flops_by_width[0.75])
    uniform_total_compute = endpoint_compute + uniform_interior_compute

    marginal_frames, summary_rows, validation_rows = [], [], []
    waiting_frames, pair_checks = [], {}
    all_closed, all_numeric, all_pairs, all_identities = True, True, True, True
    for p_index, p in enumerate(P_VALUES):
        closed = analytic_allocation(a, p)
        unconstrained_compute = float(flops @ closed)
        cap_violated = unconstrained_compute > uniform_interior_compute * (1 + 1e-10)
        bound_violated = bool(np.any(closed > 1.0 + 1e-10))
        if cap_violated or bound_violated:
            policy, solver = numerical_allocation(
                a, p, flops=flops, compute_cap=uniform_interior_compute
            )
            analytic_is_policy = False
            difference = np.nan
        else:
            policy = closed
            numeric, solver = numerical_allocation(a, p)
            difference = float(np.max(np.abs(closed - numeric)))
            analytic_is_policy = True
            all_numeric &= bool(difference < 1e-6)
        all_numeric &= bool(solver["solver_success"])
        all_closed &= bool(
            abs(closed.sum() - INTERIOR_BUDGET) < 1e-12
            and np.all(closed > 0) and np.all(closed <= 1 + 1e-10)
        )
        expected_interior_compute = float(flops @ policy)
        total_compute = endpoint_compute + expected_interior_compute
        validation_rows.append({
            "p": p,
            "analytic_unconstrained_is_final_policy": analytic_is_policy,
            "compute_cap_violated_by_unconstrained": cap_violated,
            "upper_bound_violated_by_unconstrained": bound_violated,
            "closed_form_objective": objective_jp(closed, a, p),
            "numeric_objective": solver["numeric_objective"],
            "max_abs_pi_difference": difference,
            "solver_success": solver["solver_success"],
            "solver_message": solver["solver_message"],
        })

        pair_table, pair_solver = maximum_entropy_pairs(policy)
        marginal_error = float(np.max(np.abs(pair_marginals(pair_table) - policy)))
        identities = pair_expectation_checks(pair_table, policy, seed=123 + p_index)
        pair_ok = bool(
            pair_solver["solver_success"] and abs(pair_table["probability"].sum() - 1) < 1e-8
            and marginal_error < 1e-8
        )
        identity_ok = bool(
            identities["loss_identity_absolute_error"] < 1e-10
            and identities["gradient_identity_max_absolute_error"] < 1e-10
        )
        all_pairs &= pair_ok
        all_identities &= identity_ok
        pair_checks[str(p)] = {
            **pair_solver,
            **identities,
            "maximum_marginal_absolute_error_recomputed": marginal_error,
        }
        pair_table.insert(0, "p", p)
        pair_table.to_csv(output_dir / f"pair_distribution_p{_p_tag(p)}.csv", index=False)
        waiting_frames.append(simulate_pair_waiting_times(
            pair_table, policy, p, n_steps=waiting_steps, seed=123 + p_index
        ))

        marginal_frames.append(pd.DataFrame({
            "p": p,
            "exponent": 1.0 / (p + 1.0),
            "width": INTERIOR_WIDTHS,
            "functional_cell_mass": a,
            "flops": flops,
            "pi": policy,
            "theoretical_wait": 1.0 / policy,
        }))
        normalized = policy / INTERIOR_BUDGET
        summary_rows.append({
            "p": p,
            "exponent": 1.0 / (p + 1.0),
            "objective_Jp": objective_jp(policy, a, p),
            "J1_waiting_cost": objective_jp(policy, a, 1.0),
            "min_pi": float(policy.min()),
            "max_pi": float(policy.max()),
            "pi_ratio_max_min": float(policy.max() / policy.min()),
            "entropy": float(-np.sum(xlogy(normalized, normalized))),
            "expected_total_compute_ratio": total_compute / uniform_total_compute,
            "low_mass": float(sum(policy[list(INTERIOR_WIDTHS).index(w)] for w in REGIONS["low"])),
            "mid_mass": float(sum(policy[list(INTERIOR_WIDTHS).index(w)] for w in REGIONS["mid"])),
            "high_mass": float(sum(policy[list(INTERIOR_WIDTHS).index(w)] for w in REGIONS["high"])),
            "effective_num_widths": float(np.exp(-np.sum(xlogy(normalized, normalized)))),
            "compute_cap_active": bool(abs(expected_interior_compute / uniform_interior_compute - 1) < 1e-6),
        })

    marginals = pd.concat(marginal_frames, ignore_index=True)
    family_summary = pd.DataFrame(summary_rows)
    numeric_validation = pd.DataFrame(validation_rows)
    waiting = pd.concat(waiting_frames, ignore_index=True)
    waiting_ok = bool(waiting["relative_error"].max() < 0.02)

    large_rows = []
    uniform_pi = INTERIOR_BUDGET / len(INTERIOR_WIDTHS)
    previous_spread = np.inf
    for p in LARGE_P_VALUES:
        pi = analytic_allocation(a, p)
        spread = float(pi.max() - pi.min())
        large_rows.append({
            "p": p, "max_pi_minus_min_pi": spread,
            "max_abs_difference_from_uniform": float(np.max(np.abs(pi - uniform_pi))),
        })
        if spread >= previous_spread:
            raise RuntimeError("Large-p allocation did not move monotonically toward uniform")
        previous_spread = spread
    large_p = pd.DataFrame(large_rows)

    consistency = cells[[
        "width", "functional_cell_mass", "piecewise_integral_mass", "absolute_mass_difference"
    ]].copy()
    continuous_ok = bool(np.allclose(
        consistency["functional_cell_mass"], consistency["piecewise_integral_mass"],
        rtol=1e-10, atol=1e-12,
    ))

    edge.to_csv(output_dir / "edge_geometry.csv", index=False)
    cells.to_csv(output_dir / "functional_cells.csv", index=False)
    marginals.to_csv(output_dir / "policy_family_marginals.csv", index=False)
    family_summary.to_csv(output_dir / "policy_family_summary.csv", index=False)
    waiting.to_csv(output_dir / "waiting_time_validation.csv", index=False)
    numeric_validation.to_csv(output_dir / "analytic_vs_numeric_validation.csv", index=False)
    consistency.to_csv(output_dir / "continuous_discrete_consistency.csv", index=False)
    large_p.to_csv(output_dir / "large_p_uniform_limit.csv", index=False)

    frozen_inputs = {
        "development_root": str(Path(development_root)),
        "geometry_source_hashes": source_hashes,
        "accuracy_used": False,
        "test_used": False,
        "development_geometry_seeds": [0, 1, 2],
        "grid": list(GRID),
        "interior_widths": list(INTERIOR_WIDTHS),
        "fixed_endpoints": list(ENDPOINTS),
        "interior_budget": INTERIOR_BUDGET,
        "uniform_interior_compute": uniform_interior_compute,
        "uniform_total_compute": uniform_total_compute,
    }
    (output_dir / "frozen_inputs.json").write_text(
        json.dumps(frozen_inputs, indent=2, default=_json_default) + "\n"
    )
    summary = {
        "status": "THEORY_CPU_PROBE_ONLY",
        "accuracy_used": False,
        "test_used": False,
        "development_geometry_seeds": [0, 1, 2],
        "p_values": list(P_VALUES),
        "primary_theory_policy": {
            "p": 1.0,
            "reason": "linear expected geometry-weighted waiting-time debt",
        },
        "all_closed_form_checks_pass": bool(all_closed),
        "all_numeric_solver_checks_pass": bool(all_numeric and numeric_validation.solver_success.all()),
        "continuous_discrete_mass_check_pass": continuous_ok,
        "all_pair_marginals_pass": bool(all_pairs),
        "all_pair_expectation_identities_pass": bool(all_identities),
        "waiting_time_sampler_check_pass": waiting_ok,
        "maximum_waiting_time_relative_error": float(waiting["relative_error"].max()),
        "large_p_uniform_limit_check_pass": bool(
            np.all(np.diff(large_p["max_pi_minus_min_pi"].to_numpy(float)) < 0)
        ),
        "pair_checks": pair_checks,
        "training_authorized": False,
    }
    required_checks = [
        summary["all_closed_form_checks_pass"],
        summary["all_numeric_solver_checks_pass"],
        summary["continuous_discrete_mass_check_pass"],
        summary["all_pair_marginals_pass"],
        summary["all_pair_expectation_identities_pass"],
        summary["waiting_time_sampler_check_pass"],
        summary["large_p_uniform_limit_check_pass"],
    ]
    if not all(required_checks):
        raise RuntimeError(f"Theory probe failed one or more checks: {summary}")
    (output_dir / "theory_probe_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n"
    )
    _plot_outputs(output_dir, edge, marginals, family_summary)
    return summary
