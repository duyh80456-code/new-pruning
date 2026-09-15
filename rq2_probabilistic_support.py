"""CPU-only expected-K=4 support-allocation preview for RQ2."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, linprog, minimize
from scipy.special import logsumexp, xlogy

from rq2_anchor_placement import GRID
from rq2_finalgeo_selector import EXPECTED_PUREGEO
from rq2_parameter_exposure import _coordinates


INTERIOR_WIDTHS = tuple(GRID[1:-1])
ENDPOINTS = (GRID[0], GRID[-1])
UNIFORM_INTERIORS = (0.50, 0.75)
NUM_INTERIOR_SLOTS = 2.0
NUMERICAL_LOWER_BOUND = 1e-8


def _json_default(value):
    """Convert NumPy scalar diagnostics without masking unsupported objects."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def functional_mass(geometry_coordinate: dict[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Return frozen edge lengths and centered interior functional masses."""
    coordinate = np.asarray([geometry_coordinate[width] for width in GRID], float)
    edges = np.diff(coordinate)
    if len(edges) != len(GRID) - 1 or not np.all(np.isfinite(edges)) or np.any(edges < 0):
        raise ValueError("Frozen functional coordinate must have finite nonnegative edge lengths")
    mass = 0.5 * (edges[:-1] + edges[1:])
    if not np.any(mass > 0):
        raise ValueError("Functional mass is degenerate")
    return edges, mass


def solve_geometry_closed_form(
    mass: np.ndarray,
    flops: np.ndarray,
    compute_cap: float,
    lower: float = NUMERICAL_LOWER_BOUND,
) -> tuple[np.ndarray, dict]:
    """Solve sum(a_i / pi_i) with two slots, using the analytic solution when feasible.

    Without an active upper bound or compute cap, Cauchy--Schwarz gives
    pi_i = 2 sqrt(a_i) / sum_j sqrt(a_j). A numerical constrained solver is
    used only if that candidate violates an actual constraint.
    """
    mass, flops = np.asarray(mass, float), np.asarray(flops, float)
    if mass.shape != flops.shape or mass.ndim != 1:
        raise ValueError("mass and flops must be aligned one-dimensional arrays")
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("Closed-form geometry allocation requires strictly positive finite mass")
    root_mass = np.sqrt(mass)
    candidate = NUM_INTERIOR_SLOTS * root_mass / root_mass.sum()
    candidate_compute = float(flops @ candidate)
    reasons = []
    if np.any(candidate < lower):
        reasons.append("numerical_lower_bound")
    if np.any(candidate > 1.0 + 1e-10):
        reasons.append("upper_probability_bound")
    if candidate_compute > compute_cap * (1.0 + 1e-10):
        reasons.append("uniform_compute_cap")

    if reasons:
        pi, diagnostics = solve_marginals(
            mass, flops, compute_cap, "geometry", lower=lower, compute_constraint="cap"
        )
        diagnostics.update({
            "analytic_candidate_used": False,
            "analytic_fallback_reasons": reasons,
            "analytic_candidate_expected_interior_flops": candidate_compute,
        })
        return pi, diagnostics

    pi = candidate
    stationarity = mass / np.square(pi)
    stationarity_residual = float(
        np.max(np.abs(stationarity - stationarity.mean()))
        / max(float(np.max(np.abs(stationarity))), np.finfo(float).tiny)
    )
    compute_ratio = candidate_compute / float(compute_cap)
    diagnostics = {
        "mode": "geometry",
        "compute_constraint": "cap",
        "solver": "closed_form/Cauchy-Schwarz",
        "solver_success": True,
        "solver_message": "Analytic candidate satisfies probability bounds and compute cap",
        "iterations": 0,
        "analytic_candidate_used": True,
        "analytic_fallback_reasons": [],
        "objective_definition": "sum_i a_i / pi_i",
        "objective_value": float(np.sum(mass / pi)),
        "sum_pi": float(pi.sum()),
        "sum_constraint_absolute_residual": float(abs(pi.sum() - NUM_INTERIOR_SLOTS)),
        "expected_interior_flops": candidate_compute,
        "expected_compute_ratio_vs_uniform_interiors": compute_ratio,
        "compute_cap_slack_fraction": float(1.0 - compute_ratio),
        "compute_cap_active": bool(abs(compute_ratio - 1.0) <= 1e-6),
        "compute_constraint_relative_residual": float(max(0.0, compute_ratio - 1.0)),
        "minimum_pi": float(pi.min()),
        "maximum_pi": float(pi.max()),
        "upper_bound_active_count": int(np.sum(pi >= 1.0 - 1e-7)),
        "pi_below_0_01_count": int(np.sum(pi < 0.01)),
        "pi_below_0_05_count": int(np.sum(pi < 0.05)),
        "KKT_free_variable_count": int(len(pi)),
        "KKT_max_relative_residual": stationarity_residual,
    }
    return pi, diagnostics


def _feasible_start(
    F_scaled: np.ndarray, compute_scaled: float, lower: float, compute_constraint: str
) -> np.ndarray:
    # Project the uniform marginal vector onto both equality constraints. For
    # this protocol the projection is strictly interior and is a much better
    # numerical start than a vertex returned by a feasibility LP.
    constraints = np.vstack([np.ones(len(F_scaled)), F_scaled])
    target = np.asarray([NUM_INTERIOR_SLOTS, compute_scaled])
    uniform = np.full(len(F_scaled), NUM_INTERIOR_SLOTS / len(F_scaled))
    if compute_constraint == "cap" and F_scaled @ uniform <= compute_scaled + 1e-12:
        return uniform
    if compute_constraint == "equal":
        projected = uniform - constraints.T @ np.linalg.solve(
            constraints @ constraints.T, constraints @ uniform - target
        )
        if np.all(projected > lower) and np.all(projected < 1.0):
            return projected
    result = linprog(
        np.zeros(len(F_scaled)),
        A_eq=(np.vstack([np.ones(len(F_scaled)), F_scaled])
              if compute_constraint == "equal" else np.ones((1, len(F_scaled)))),
        b_eq=(np.asarray([NUM_INTERIOR_SLOTS, compute_scaled])
              if compute_constraint == "equal" else np.asarray([NUM_INTERIOR_SLOTS])),
        A_ub=(None if compute_constraint == "equal" else F_scaled[None, :]),
        b_ub=(None if compute_constraint == "equal" else np.asarray([compute_scaled])),
        bounds=[(lower, 1.0)] * len(F_scaled),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Marginal allocation constraints are infeasible: {result.message}")
    return result.x


def solve_marginals(
    mass: np.ndarray,
    flops: np.ndarray,
    compute_target: float,
    mode: str,
    lower: float = NUMERICAL_LOWER_BOUND,
    compute_constraint: str = "equal",
) -> tuple[np.ndarray, dict]:
    """Solve geometry or resource-only marginals under identical affine constraints."""
    mass, flops = np.asarray(mass, float), np.asarray(flops, float)
    if mass.shape != flops.shape or mass.ndim != 1:
        raise ValueError("mass and flops must be aligned one-dimensional arrays")
    if mode not in {"geometry", "resource"}:
        raise ValueError("mode must be 'geometry' or 'resource'")
    if compute_constraint not in {"equal", "cap"}:
        raise ValueError("compute_constraint must be 'equal' or 'cap'")
    scale = float(compute_target)
    F_scaled = flops / scale
    x0 = _feasible_start(F_scaled, 1.0, lower, compute_constraint)
    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x) - NUM_INTERIOR_SLOTS,
         "jac": lambda x: np.ones_like(x)},
    ]
    if compute_constraint == "equal":
        constraints.append({
            "type": "eq", "fun": lambda x: F_scaled @ x - 1.0,
            "jac": lambda x: F_scaled,
        })
    else:
        constraints.append({
            "type": "ineq", "fun": lambda x: 1.0 - F_scaled @ x,
            "jac": lambda x: -F_scaled,
        })
    if mode == "geometry":
        positive_scale = max(float(np.mean(mass[mass > 0])), np.finfo(float).tiny)
        weights = np.maximum(mass / positive_scale, 1e-12)
        objective = lambda x: float(np.sum(weights / x))
        gradient = lambda x: -weights / np.square(x)
        objective_definition = "sum_i a_i / pi_i (a_i rescaled by its positive mean)"
    else:
        objective = lambda x: float(np.sum(xlogy(x, x)))
        gradient = lambda x: np.log(x) + 1.0
        objective_definition = "minimize sum_i pi_i log(pi_i), equivalent to maximum allocation entropy"
    result = minimize(
        objective, x0, jac=gradient, method="SLSQP",
        bounds=[(lower, 1.0)] * len(flops), constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 5000, "disp": False},
    )
    pi = np.asarray(result.x, float)
    sum_residual = float(abs(pi.sum() - NUM_INTERIOR_SLOTS))
    compute_residual = float(abs(flops @ pi - compute_target) / compute_target)
    compute_ratio = float(flops @ pi / compute_target)
    bounds_ok = bool(np.all(pi > 0) and np.all(pi <= 1.0 + 1e-8))
    compute_ok = (
        compute_residual <= 1e-7 if compute_constraint == "equal"
        else compute_ratio <= 1.0 + 1e-7
    )
    if not result.success or sum_residual > 1e-7 or not bounds_ok or not compute_ok:
        raise RuntimeError(
            f"{mode} marginal solve failed: {result.message}; "
            f"sum residual={sum_residual}, compute residual={compute_residual}"
        )
    diagnostics = {
        "mode": mode,
        "compute_constraint": compute_constraint,
        "solver": "scipy.optimize.minimize/SLSQP",
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "iterations": int(result.nit),
        "objective_definition": objective_definition,
        "objective_value": float(objective(pi)),
        "sum_pi": float(pi.sum()),
        "sum_constraint_absolute_residual": sum_residual,
        "expected_interior_flops": float(flops @ pi),
        "expected_compute_ratio_vs_uniform_interiors": compute_ratio,
        "compute_cap_slack_fraction": float(1.0 - compute_ratio),
        "compute_cap_active": bool(abs(compute_ratio - 1.0) <= 1e-6),
        "compute_constraint_relative_residual": compute_residual,
        "minimum_pi": float(pi.min()),
        "maximum_pi": float(pi.max()),
        "upper_bound_active_count": int(np.sum(pi >= 1.0 - 1e-7)),
        "pi_below_0_01_count": int(np.sum(pi < 0.01)),
        "pi_below_0_05_count": int(np.sum(pi < 0.05)),
    }
    if mode == "geometry":
        free = (pi > lower * 100) & (pi < 1.0 - 1e-7)
        lhs = weights[free] / np.square(pi[free])
        design = np.column_stack([np.ones(free.sum()), flops[free]])
        coefficients, *_ = np.linalg.lstsq(design, lhs, rcond=None)
        fitted = design @ coefficients
        diagnostics.update({
            "KKT_free_variable_count": int(free.sum()),
            "KKT_lambda": float(coefficients[0]),
            "KKT_eta": float(coefficients[1]),
            "KKT_max_relative_residual": float(
                np.max(np.abs(lhs - fitted)) / max(np.max(np.abs(lhs)), 1e-12)
            ),
        })
    return pi, diagnostics


def maximum_entropy_pairs(pi: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Find the maximum-entropy distribution over distinct interior-width pairs."""
    pi = np.asarray(pi, float)
    if abs(pi.sum() - 2.0) > 1e-6 or np.any(pi <= 0) or np.any(pi > 1.0 + 1e-8):
        raise ValueError("Pair marginals require positive pi <= 1 summing to two")
    pairs = [(i, j) for i in range(len(pi)) for j in range(i + 1, len(pi))]

    def values(theta_free):
        theta = np.concatenate([theta_free, [0.0]])
        scores = np.asarray([theta[i] + theta[j] for i, j in pairs])
        log_z = logsumexp(scores)
        q = np.exp(scores - log_z)
        marginals = np.zeros(len(pi))
        for probability, (i, j) in zip(q, pairs):
            marginals[i] += probability
            marginals[j] += probability
        objective = float(log_z - theta @ pi)
        return objective, marginals[:-1] - pi[:-1], q, marginals

    result = least_squares(
        lambda theta: values(theta)[1], np.zeros(len(pi) - 1),
        xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=10000,
    )
    _, _, q, marginals = values(result.x)
    marginal_residual = float(np.max(np.abs(marginals - pi)))
    if (not result.success or abs(q.sum() - 1.0) > 1e-10
            or marginal_residual > 1e-7 or np.any(q < 0)):
        raise RuntimeError(
            f"Maximum-entropy pair solve failed: {result.message}; residual={marginal_residual}"
        )
    table = pd.DataFrame([
        {
            "pair_index": index,
            "width_i": INTERIOR_WIDTHS[i],
            "width_j": INTERIOR_WIDTHS[j],
            "probability": probability,
        }
        for index, (probability, (i, j)) in enumerate(zip(q, pairs))
    ])
    diagnostics = {
        "solver": "maximum-entropy exponential-family dual/scipy least_squares",
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "function_evaluations": int(result.nfev),
        "pair_count": len(table),
        "sum_q": float(q.sum()),
        "maximum_marginal_absolute_residual": marginal_residual,
        "pair_entropy_nats": float(-np.sum(xlogy(q, q))),
        "minimum_pair_probability": float(q.min()),
        "maximum_pair_probability": float(q.max()),
    }
    return table, diagnostics


def pair_marginals(pair_table: pd.DataFrame) -> np.ndarray:
    result = np.zeros(len(INTERIOR_WIDTHS))
    index = {width: i for i, width in enumerate(INTERIOR_WIDTHS)}
    for row in pair_table.itertuples():
        result[index[round(float(row.width_i), 2)]] += float(row.probability)
        result[index[round(float(row.width_j), 2)]] += float(row.probability)
    return result


def sample_pair(pair_table: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float]:
    probabilities = pair_table["probability"].to_numpy(float)
    row = pair_table.iloc[int(rng.choice(len(pair_table), p=probabilities / probabilities.sum()))]
    return float(row["width_i"]), float(row["width_j"])


def build_support_policy(
    development_root: str | Path,
    output_dir: str | Path,
    pi_min: float = NUMERICAL_LOWER_BOUND,
) -> dict:
    """Build and audit both policies without consulting accuracy or training a model."""
    development_root, output_dir = Path(development_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coordinate, _, flops_by_width, puregeo, source_hashes = _coordinates(development_root)
    if tuple(puregeo) != EXPECTED_PUREGEO:
        raise RuntimeError(f"Unexpected frozen PureGeo anchors: {puregeo}")
    edges, mass = functional_mass(coordinate)
    F = np.asarray([flops_by_width[width] for width in INTERIOR_WIDTHS], float)
    compute_target = float(flops_by_width[0.50] + flops_by_width[0.75])
    endpoint_compute = float(flops_by_width[ENDPOINTS[0]] + flops_by_width[ENDPOINTS[1]])
    pi_geometry, geometry_solver = solve_geometry_closed_form(
        mass, F, compute_target, lower=pi_min
    )
    geometry_compute = float(F @ pi_geometry)
    # Fair dynamic control: maximize allocation entropy at exactly the compute
    # naturally selected by geometry, rather than forcing geometry to spend the
    # Uniform budget or comparing against a differently priced random policy.
    pi_resource, resource_solver = solve_marginals(
        mass, F, geometry_compute, "resource", lower=pi_min, compute_constraint="equal"
    )
    # This is retained only as a diagnostic of what the old Uniform-cap
    # resource policy would do; it is not the primary comparison.
    pi_resource_cap, resource_cap_solver = solve_marginals(
        mass, F, compute_target, "resource", lower=pi_min, compute_constraint="cap"
    )
    q_geometry, geometry_pairs = maximum_entropy_pairs(pi_geometry)
    q_resource, resource_pairs = maximum_entropy_pairs(pi_resource)

    debug = pd.DataFrame({
        "width": INTERIOR_WIDTHS,
        "functional_mass_a_i": mass,
        "flops": F.astype(np.int64),
        "pi_geometry": pi_geometry,
        "pi_resource_matched_compute": pi_resource,
        "pi_resource_uniform_cap_diagnostic": pi_resource_cap,
        "geometry_minus_matched_resource_pi": pi_geometry - pi_resource,
    })
    debug["geometry_expected_batches_between_inclusions"] = 1.0 / debug["pi_geometry"]
    debug["resource_matched_expected_batches_between_inclusions"] = (
        1.0 / debug["pi_resource_matched_compute"]
    )
    edge_table = pd.DataFrame({
        "budget_start": GRID[:-1], "budget_end": GRID[1:], "edge_length": edges,
        "G": edges / np.diff(np.asarray(GRID, float)),
    })
    q_geometry.insert(0, "policy", "geometry_closed_form")
    q_resource.insert(0, "policy", "resource_matched_compute")
    debug.to_csv(output_dir / "support_allocation_marginals.csv", index=False)
    edge_table.to_csv(output_dir / "frozen_functional_edges.csv", index=False)
    q_geometry.to_csv(output_dir / "geometry_pair_distribution.csv", index=False)
    q_resource.to_csv(output_dir / "resource_pair_distribution.csv", index=False)

    checks = {
        "geometry_solver_success": bool(geometry_solver["solver_success"]),
        "resource_matched_solver_success": bool(resource_solver["solver_success"]),
        "resource_cap_diagnostic_solver_success": bool(resource_cap_solver["solver_success"]),
        "geometry_pair_solver_success": bool(geometry_pairs["solver_success"]),
        "resource_pair_solver_success": bool(resource_pairs["solver_success"]),
        "geometry_sum_pi": bool(abs(pi_geometry.sum() - 2.0) < 1e-6),
        "geometry_compute_at_or_below_cap": bool(F @ pi_geometry <= compute_target * (1 + 1e-6)),
        "geometry_bounds": bool(np.all(pi_geometry > 0) and np.all(pi_geometry <= 1 + 1e-8)),
        "resource_sum_pi": bool(abs(pi_resource.sum() - 2.0) < 1e-6),
        "resource_compute_matches_geometry": bool(
            abs(F @ pi_resource - geometry_compute) / geometry_compute < 1e-6
        ),
        "resource_bounds": bool(np.all(pi_resource > 0) and np.all(pi_resource <= 1 + 1e-8)),
        "geometry_sum_q": bool(abs(q_geometry["probability"].sum() - 1.0) < 1e-6),
        "resource_sum_q": bool(abs(q_resource["probability"].sum() - 1.0) < 1e-6),
        "geometry_pair_marginals": bool(np.max(np.abs(pair_marginals(q_geometry) - pi_geometry)) < 1e-6),
        "resource_pair_marginals": bool(np.max(np.abs(pair_marginals(q_resource) - pi_resource)) < 1e-6),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Support-allocation invariant failed: {checks}")
    diagnostics = {
        "status": "POLICY_PREVIEW_ONLY",
        "training_authorized": False,
        "accuracy_used": False,
        "development_geometry_seeds": [0, 1, 2],
        "interior_widths": list(INTERIOR_WIDTHS),
        "fixed_endpoints": list(ENDPOINTS),
        "interior_slots_per_batch": 2,
        "total_subnets_per_batch": 4,
        "uniform_interior_compute_target": compute_target,
        "fixed_endpoint_compute": endpoint_compute,
        "geometry_expected_interior_compute": geometry_compute,
        "geometry_interior_compute_ratio_vs_uniform": geometry_compute / compute_target,
        "geometry_total_compute_ratio_vs_uniform": (
            endpoint_compute + geometry_compute
        ) / (endpoint_compute + compute_target),
        "resource_matched_expected_interior_compute": float(F @ pi_resource),
        "resource_matched_total_compute_ratio_vs_uniform": (
            endpoint_compute + float(F @ pi_resource)
        ) / (endpoint_compute + compute_target),
        "pi_min": float(pi_min),
        "pi_min_role": ("numerical_only" if pi_min <= NUMERICAL_LOWER_BOUND
                        else "explicit_scientific_coverage_floor"),
        "compute_rule": (
            "geometry uses no more than Uniform interior FLOPs; primary resource-dynamic "
            "control exactly matches geometry's expected interior FLOPs"
        ),
        "geometry_source_hashes": source_hashes,
        "geometry_solver": geometry_solver,
        "resource_solver": resource_solver,
        "resource_uniform_cap_diagnostic_solver": resource_cap_solver,
        "geometry_pair_solver": geometry_pairs,
        "resource_pair_solver": resource_pairs,
        "assertions": checks,
        "starvation_review": {
            "geometry_widths_below_pi_0_01": debug.loc[debug.pi_geometry < 0.01, "width"].tolist(),
            "geometry_widths_below_pi_0_05": debug.loc[debug.pi_geometry < 0.05, "width"].tolist(),
            "resource_widths_below_pi_0_01": debug.loc[
                debug.pi_resource_matched_compute < 0.01, "width"
            ].tolist(),
            "resource_widths_below_pi_0_05": debug.loc[
                debug.pi_resource_matched_compute < 0.05, "width"
            ].tolist(),
        },
        "surrogate_note": "sum a_i/pi_i is an experimental convex surrogate, not an accuracy theorem",
    }
    (output_dir / "support_allocation_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, default=_json_default) + "\n"
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(debug.width, debug.pi_geometry, marker="o", label="Geometry allocation (closed form)")
    ax1.plot(debug.width, debug.pi_resource_matched_compute, marker="o",
             label="Resource dynamic (matched compute)")
    ax1.plot(debug.width, debug.pi_resource_uniform_cap_diagnostic,
             marker=".", linestyle=":", color="grey", label="Resource dynamic (Uniform cap diagnostic)")
    ax1.set(xlabel="Interior width", ylabel="Inclusion probability per batch", ylim=(0, 1.02))
    ax1.grid(alpha=0.25); ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(debug.width, debug.functional_mass_a_i, color="grey", linestyle="--", alpha=0.65,
             label="Functional mass")
    ax2.set_ylabel("Frozen functional mass")
    fig.tight_layout()
    fig.savefig(output_dir / "support_allocation_preview.png", dpi=200)
    plt.close(fig)

    report = [
        "# Expected-K=4 support-allocation preview", "",
        "No accuracy was read and no model was trained. Training remains unauthorized until the "
        "marginals and pair distributions are reviewed.", "",
        f"Geometry uses {geometry_compute / compute_target:.3%} of Uniform interior compute and "
        f"{(endpoint_compute + geometry_compute) / (endpoint_compute + compute_target):.3%} of "
        "Uniform total per-batch compute. The resource-dynamic control is matched to this cost.", "",
        "```", debug.to_string(index=False), "```", "",
        "The geometry objective is an experimental surrogate, not a theorem about accuracy.",
    ]
    (output_dir / "support_allocation_preview.md").write_text("\n".join(report) + "\n")
    return diagnostics
