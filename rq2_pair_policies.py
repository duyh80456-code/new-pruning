"""Fixed-marginal pair policies for the RQ2 end-to-end pilot."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import linprog

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
    eps_num = max(1e-10, 1e-8 * abs(resource_star))
    resource_minimum = retention * resource_star - eps_num
    resource_scale = max(float(np.max(np.abs(resource))), abs(resource_star), 1.0)
    geometry_scale = max(float(np.max(np.abs(geometry))), 1.0)
    result = linprog(
        -geometry / geometry_scale,
        A_ub=-resource[None, :] / resource_scale,
        b_ub=np.asarray([-resource_minimum / resource_scale]),
        A_eq=incidence_matrix(),
        b_eq=UNIFORM_PI,
        bounds=[(0.0, None)] * NUM_PAIRS,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"RP-Geo LP failed: {result.message}")
    q = validate_pair_probabilities(result.x)
    resource_rg = float(resource @ q)
    tolerance = 10.0 * eps_num
    if resource_rg < resource_minimum - tolerance:
        raise RuntimeError("RP-Geo solution violates the Resource-retention constraint")
    positive = q[q > 0]
    diagnostics = {
        "resource_star": resource_star,
        "resource_rg": resource_rg,
        "resource_retention_target": retention,
        "resource_retention_achieved": (
            resource_rg / resource_star if resource_star > 0 else 1.0
        ),
        "resource_numerical_epsilon": eps_num,
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
        super().__init__("resource", validate_pair_probabilities(solve_pair_lp(scores, maximize=True)))


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


__all__ = [
    "PairPolicy", "UniformPairPolicy", "ResourcePairPolicy", "SWPairPolicy",
    "ResourceGeoPairPolicy", "resource_score_vector", "solve_resource_preserving_geo",
    "solve_fixed_marginal_lp", "sample_pair", "pair_table",
    "validate_pair_probabilities",
]
