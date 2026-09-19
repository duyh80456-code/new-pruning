"""Fixed-marginal pair policies for the RQ2 end-to-end pilot."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from rq2_pairwise_surrogate_regret import (
    INTERIOR_WIDTHS,
    NUM_PAIRS,
    PAIR_INDICES,
    UNIFORM_PI,
    incidence_matrix,
    solve_pair_lp,
)


def validate_pair_probabilities(probabilities: np.ndarray) -> np.ndarray:
    q = np.asarray(probabilities, dtype=float)
    if q.shape != (NUM_PAIRS,) or not np.isfinite(q).all():
        raise ValueError("Pair probabilities must be one finite 91-vector")
    if q.min() < -1e-9:
        raise ValueError("Pair probabilities contain a negative entry")
    q = np.clip(q, 0.0, None)
    if abs(q.sum() - 1.0) >= 1e-7:
        raise ValueError("Pair probabilities do not sum to one")
    marginals = incidence_matrix() @ q
    if np.max(np.abs(marginals - UNIFORM_PI)) >= 1e-6:
        raise ValueError("Pair probabilities violate fixed marginals pi_i=1/7")
    return q


def solve_fixed_marginal_lp(score_matrix: np.ndarray) -> np.ndarray:
    score_matrix = np.asarray(score_matrix, dtype=float)
    if score_matrix.shape != (14, 14) or not np.isfinite(score_matrix).all():
        raise ValueError("Pair score matrix must be finite and 14x14")
    if not np.allclose(score_matrix, score_matrix.T, atol=1e-12, rtol=1e-12):
        raise ValueError("Pair score matrix must be symmetric")
    scores = np.asarray([score_matrix[i, j] for i, j in PAIR_INDICES])
    return validate_pair_probabilities(solve_pair_lp(scores, maximize=True))


def resource_score_vector(flops_by_width: dict[float, float]) -> np.ndarray:
    values = np.asarray([float(flops_by_width[width]) for width in INTERIOR_WIDTHS])
    if np.any(values <= 0) or not np.all(np.diff(values) > 0):
        raise ValueError("Resource policy requires strictly increasing positive FLOPs")
    logs = np.log(values)
    return np.asarray([
        float((logs[i] - logs[j]) ** 2) for i, j in PAIR_INDICES
    ])


def anti_monotone_resource_probabilities() -> np.ndarray:
    """Closed-form Resource optimum for strictly increasing log-FLOPs."""
    pair_to_index = {pair: index for index, pair in enumerate(PAIR_INDICES)}
    q = np.zeros(NUM_PAIRS, dtype=float)
    for i in range(7):
        q[pair_to_index[(i, 13 - i)]] = 1.0 / 7.0
    return validate_pair_probabilities(q)


def solve_resource_preserving_geo(
    sw_matrix: np.ndarray,
    flops_by_width: dict[float, float],
    resource_retention: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Maximize squared sliced-W1 on a Resource-optimal (or retained) face."""
    retention = float(resource_retention)
    if not 0.0 < retention <= 1.0:
        raise ValueError("resource_retention must lie in (0, 1]")
    sw = np.asarray(sw_matrix, dtype=float)
    if (
        sw.shape != (14, 14) or np.any(sw < 0) or not np.isfinite(sw).all()
        or not np.allclose(sw, sw.T, atol=1e-12, rtol=1e-12)
    ):
        raise ValueError("RP-Geo requires one finite nonnegative symmetric 14x14 SW matrix")
    resource = resource_score_vector(flops_by_width)
    geometry = np.asarray([float(sw[i, j] ** 2) for i, j in PAIR_INDICES])
    q_resource = validate_pair_probabilities(solve_pair_lp(resource, maximize=True))
    resource_star = float(resource @ q_resource)
    resource_scale = max(float(np.max(np.abs(resource))), abs(resource_star), 1.0)
    geometry_scale = max(float(np.max(np.abs(geometry))), 1.0)
    if retention == 1.0:
        # This must be a mathematical face, not an epsilon-thickened near-optimal set.
        a_eq = np.vstack([incidence_matrix(), resource[None, :] / resource_scale])
        b_eq = np.concatenate([UNIFORM_PI, [resource_star / resource_scale]])
        result = linprog(
            -geometry / geometry_scale,
            A_eq=a_eq, b_eq=b_eq,
            bounds=[(0.0, None)] * NUM_PAIRS, method="highs",
        )
        resource_minimum = resource_star
        constraint_type = "exact_resource_equality"
    else:
        resource_minimum = retention * resource_star
        result = linprog(
            -geometry / geometry_scale,
            A_ub=-resource[None, :] / resource_scale,
            b_ub=np.asarray([-resource_minimum / resource_scale]),
            A_eq=incidence_matrix(), b_eq=UNIFORM_PI,
            bounds=[(0.0, None)] * NUM_PAIRS, method="highs",
        )
        constraint_type = "near_optimal_resource_inequality"
    if not result.success:
        raise RuntimeError(f"RP-Geo LP failed: {result.message}")
    q = validate_pair_probabilities(result.x)
    resource_rg = float(resource @ q)
    tolerance = 1e-8 * max(1.0, abs(resource_star))
    if retention == 1.0 and abs(resource_rg - resource_star) > tolerance:
        raise RuntimeError("Exact-face RP-Geo solution does not equal the Resource optimum")
    if retention < 1.0 and resource_rg < resource_minimum - tolerance:
        raise RuntimeError("RP-Geo solution violates the Resource-retention constraint")
    positive = q[q > 0]
    diagnostics = {
        "resource_star": resource_star,
        "resource_rg": resource_rg,
        "resource_retention_target": retention,
        "resource_retention_achieved": (
            resource_rg / resource_star if resource_star > 0 else 1.0
        ),
        "resource_constraint_type": constraint_type,
        "resource_constraint_tolerance": tolerance,
        "geo_resource": float(geometry @ q_resource),
        "geo_rg": float(geometry @ q),
        "l1_vs_resource": float(np.abs(q - q_resource).sum()),
        "entropy": float(-(positive * np.log(positive)).sum()),
        "support_size": int(np.count_nonzero(q > 1e-8)),
        "marginal_error": float(np.max(np.abs(incidence_matrix() @ q - UNIFORM_PI))),
    }
    return q, diagnostics


def pair_table(probabilities: np.ndarray, policy: str) -> pd.DataFrame:
    q = validate_pair_probabilities(probabilities)
    return pd.DataFrame([
        {
            "policy": policy,
            "pair_index": index,
            "width_i": INTERIOR_WIDTHS[i],
            "width_j": INTERIOR_WIDTHS[j],
            "probability": float(q[index]),
        }
        for index, (i, j) in enumerate(PAIR_INDICES)
    ])


def sample_pair(probabilities: np.ndarray, uniform_draw: float) -> tuple[float, float, int]:
    q = validate_pair_probabilities(probabilities)
    value = float(uniform_draw)
    if not 0.0 <= value < 1.0:
        raise ValueError("Pair draw must be in [0, 1)")
    index = min(int(np.searchsorted(np.cumsum(q), value, side="right")), NUM_PAIRS - 1)
    i, j = PAIR_INDICES[index]
    return INTERIOR_WIDTHS[i], INTERIOR_WIDTHS[j], index


@dataclass(frozen=True)
class PairPolicy:
    name: str
    probabilities: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "probabilities", validate_pair_probabilities(self.probabilities))

    def sample(self, uniform_draw: float) -> tuple[float, float, int]:
        return sample_pair(self.probabilities, uniform_draw)

    def frame(self) -> pd.DataFrame:
        return pair_table(self.probabilities, self.name)


class UniformPairPolicy(PairPolicy):
    def __init__(self):
        super().__init__("uniform", np.full(NUM_PAIRS, 1.0 / NUM_PAIRS))


class ResourcePairPolicy(PairPolicy):
    def __init__(self, flops_by_width: dict[float, float]):
        scores = resource_score_vector(flops_by_width)
        probabilities = validate_pair_probabilities(solve_pair_lp(scores, maximize=True))
        closed_form = anti_monotone_resource_probabilities()
        if np.max(np.abs(probabilities - closed_form)) > 1e-7:
            raise RuntimeError("Resource LP disagrees with the anti-monotone closed-form optimum")
        super().__init__("resource", probabilities)


class SWPairPolicy(PairPolicy):
    def __init__(self, sw_matrix: np.ndarray):
        sw = np.asarray(sw_matrix, dtype=float)
        if sw.shape != (14, 14) or np.any(sw < 0) or not np.isfinite(sw).all():
            raise ValueError("SW policy requires one finite nonnegative 14x14 matrix")
        super().__init__("pure_sw", solve_fixed_marginal_lp(np.square(sw)))


class ResourceGeoPairPolicy(PairPolicy):
    def __init__(
        self,
        flops_by_width: dict[float, float],
        sw_matrix: np.ndarray,
        resource_retention: float = 1.0,
    ):
        probabilities, diagnostics = solve_resource_preserving_geo(
            sw_matrix, flops_by_width, resource_retention
        )
        super().__init__("resource_geo", probabilities)
        object.__setattr__(self, "diagnostics", diagnostics)


def solve_temporal_sw_policy(
    sw_matrix: np.ndarray,
    previous_probabilities: np.ndarray,
    continuity_strength_in_score_std: float = 1.0,
    uniform_mixture: float = 0.0,
    use_continuity: bool = True,
) -> tuple[np.ndarray, dict]:
    """Build a fixed-marginal SW policy with temporal continuity and annealing.

    Continuity uses KL(q || q_previous).  Its coefficient is expressed in
    units of the standard deviation of the current raw SW^2 pair scores, so
    rescaling representations cannot silently change the regularization.
    Uniform annealing is an explicit convex mixture after the geometry solve;
    both operands have the same fixed marginals, hence the mixture does too.
    """
    sw = np.asarray(sw_matrix, dtype=np.float64)
    if (
        sw.shape != (14, 14) or np.any(sw < 0) or not np.isfinite(sw).all()
        or not np.allclose(sw, sw.T, atol=1e-12, rtol=1e-12)
    ):
        raise ValueError("Temporal SW requires one finite nonnegative symmetric 14x14 matrix")
    previous = validate_pair_probabilities(previous_probabilities)
    alpha = float(uniform_mixture)
    strength = float(continuity_strength_in_score_std)
    if not 0.0 <= alpha < 1.0:
        raise ValueError("uniform_mixture must lie in [0, 1)")
    if strength <= 0:
        raise ValueError("continuity_strength_in_score_std must be positive")

    scores = np.asarray([sw[i, j] ** 2 for i, j in PAIR_INDICES], dtype=np.float64)
    score_std = float(np.std(scores))
    q_uniform = UniformPairPolicy().probabilities
    if score_std <= 1e-15:
        geometry_solution = previous.copy() if use_continuity else q_uniform.copy()
        solver_status = "flat_scores"
        iterations = 0
        kl_coefficient = 0.0
    elif use_continuity:
        if np.any(previous <= 0):
            raise ValueError("KL continuity requires a strictly positive previous policy")
        kl_coefficient = strength * score_std
        incidence = incidence_matrix()

        def objective(q):
            divergence = np.sum(q * np.log(q / previous) - q + previous)
            return float(-scores @ q + kl_coefficient * divergence)

        def gradient(q):
            return -scores + kl_coefficient * np.log(q / previous)

        result = minimize(
            objective,
            previous,
            jac=gradient,
            method="SLSQP",
            bounds=[(1e-12, 1.0)] * NUM_PAIRS,
            constraints={
                "type": "eq",
                "fun": lambda q: incidence @ q - UNIFORM_PI,
                "jac": lambda q: incidence,
            },
            options={"maxiter": 2000, "ftol": 1e-12, "disp": False},
        )
        if not result.success:
            raise RuntimeError(f"Temporal KL policy solve failed: {result.message}")
        geometry_solution = validate_pair_probabilities(result.x)
        solver_status = str(result.message)
        iterations = int(result.nit)
    else:
        geometry_solution = SWPairPolicy(sw).probabilities
        solver_status = "pure_sw_lp"
        iterations = 0
        kl_coefficient = 0.0

    final = validate_pair_probabilities(
        (1.0 - alpha) * geometry_solution + alpha * q_uniform
    )
    positive = final[final > 0]
    def kl_divergence(q, reference):
        positive = q > 0
        if np.any(reference[positive] <= 0):
            return float("inf")
        return float(np.sum(q[positive] * np.log(q[positive] / reference[positive])))

    diagnostics = {
        "use_continuity": bool(use_continuity),
        "continuity_strength_in_score_std": strength if use_continuity else 0.0,
        "score_std": score_std,
        "kl_coefficient": kl_coefficient,
        "uniform_mixture": alpha,
        "solver_status": solver_status,
        "solver_iterations": iterations,
        "geometry_score_previous": float(scores @ previous),
        "geometry_score_solution": float(scores @ geometry_solution),
        "geometry_score_final": float(scores @ final),
        "geometry_score_uniform": float(scores @ q_uniform),
        "kl_final_to_previous": kl_divergence(final, previous),
        "l1_final_to_previous": float(np.abs(final - previous).sum()),
        "l1_final_to_uniform": float(np.abs(final - q_uniform).sum()),
        "entropy": float(-(positive * np.log(positive)).sum()),
        "effective_support": float(1.0 / np.square(final).sum()),
        "support_size": int(np.count_nonzero(final > 1e-8)),
        "maximum_probability": float(final.max()),
        "marginal_error": float(np.max(np.abs(incidence_matrix() @ final - UNIFORM_PI))),
    }
    return final, diagnostics


class TemporalSWPairPolicy(PairPolicy):
    def __init__(
        self,
        name: str,
        sw_matrix: np.ndarray,
        previous_probabilities: np.ndarray,
        continuity_strength_in_score_std: float = 1.0,
        uniform_mixture: float = 0.0,
        use_continuity: bool = True,
    ):
        probabilities, diagnostics = solve_temporal_sw_policy(
            sw_matrix,
            previous_probabilities,
            continuity_strength_in_score_std,
            uniform_mixture,
            use_continuity,
        )
        super().__init__(name, probabilities)
        object.__setattr__(self, "diagnostics", diagnostics)


__all__ = [
    "PairPolicy", "UniformPairPolicy", "ResourcePairPolicy", "SWPairPolicy",
    "ResourceGeoPairPolicy", "TemporalSWPairPolicy", "resource_score_vector",
    "solve_resource_preserving_geo", "solve_temporal_sw_policy",
    "anti_monotone_resource_probabilities",
    "solve_fixed_marginal_lp", "sample_pair", "pair_table",
    "validate_pair_probabilities",
]
