import numpy as np

from rq2_pair_policies import UniformPairPolicy
from rq2_pairwise_surrogate_regret import INTERIOR_WIDTHS, incidence_matrix
from rq2_temporal_pair_training import (
    FINAL_UNIFORM_MIXTURE,
    TEMPORAL_METHODS,
    temporal_policy,
    uniform_mixture_at_epoch,
)


def test_cosine_annealing_is_frozen_and_monotone():
    epochs = list(range(10, 100, 10))
    values = [uniform_mixture_at_epoch(epoch) for epoch in epochs]
    assert values[0] == 0.0
    assert np.isclose(values[-1], FINAL_UNIFORM_MIXTURE)
    assert np.all(np.diff(values) > 0)


def test_three_ablation_policies_differ_but_keep_fixed_marginals():
    coordinates = np.asarray(INTERIOR_WIDTHS)
    sw = np.abs(np.cos(9 * coordinates[:, None]) - np.cos(9 * coordinates[None, :]))
    previous = UniformPairPolicy().probabilities
    policies = {
        method: temporal_policy(method, sw, previous, 50)
        for method in TEMPORAL_METHODS
    }
    for policy in policies.values():
        assert np.max(np.abs(incidence_matrix() @ policy.probabilities - 1 / 7)) < 1e-8
    assert not np.allclose(
        policies["sw_continuity"].probabilities,
        policies["sw_anneal"].probabilities,
    )
    assert not np.allclose(
        policies["sw_continuity"].probabilities,
        policies["sw_continuity_anneal"].probabilities,
    )

