"""Fixed-marginal pair policies for the RQ2 end-to-end pilot."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

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
        values = np.asarray([float(flops_by_width[width]) for width in INTERIOR_WIDTHS])
        if np.any(values <= 0) or not np.all(np.diff(values) > 0):
            raise ValueError("Resource policy requires strictly increasing positive FLOPs")
        logs = np.log(values)
        scores = np.square(logs[:, None] - logs[None, :])
        super().__init__("resource", solve_fixed_marginal_lp(scores))


class SWPairPolicy(PairPolicy):
    def __init__(self, sw_matrix: np.ndarray):
        sw = np.asarray(sw_matrix, dtype=float)
        if sw.shape != (14, 14) or np.any(sw < 0) or not np.isfinite(sw).all():
            raise ValueError("SW policy requires one finite nonnegative 14x14 matrix")
        super().__init__("pure_sw", solve_fixed_marginal_lp(np.square(sw)))


__all__ = [
    "PairPolicy", "UniformPairPolicy", "ResourcePairPolicy", "SWPairPolicy",
    "solve_fixed_marginal_lp", "sample_pair", "pair_table",
    "validate_pair_probabilities",
]
