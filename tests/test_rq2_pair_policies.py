import numpy as np

from rq2_pair_policies import (
    ResourcePairPolicy,
    SWPairPolicy,
    UniformPairPolicy,
    sample_pair,
    validate_pair_probabilities,
)
from rq2_pairwise_surrogate_regret import INTERIOR_WIDTHS, incidence_matrix


def test_all_pair_policies_have_exact_uniform_marginals():
    flops = {width: 1e8 * (0.1 + width ** 2) for width in INTERIOR_WIDTHS}
    coordinates = np.asarray(INTERIOR_WIDTHS)
    sw = np.abs(coordinates[:, None] - coordinates[None, :])
    policies = [UniformPairPolicy(), ResourcePairPolicy(flops), SWPairPolicy(sw)]
    for policy in policies:
        q = validate_pair_probabilities(policy.probabilities)
        assert abs(q.sum() - 1.0) < 1e-9
        assert np.max(np.abs(incidence_matrix() @ q - 1 / 7)) < 1e-8
        wi, wj, index = sample_pair(q, 0.314159)
        assert wi < wj and 0 <= index < 91

