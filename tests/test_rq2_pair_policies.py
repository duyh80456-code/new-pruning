import numpy as np

from rq2_pair_policies import (
    ResourceGeoPairPolicy,
    ResourcePairPolicy,
    SWPairPolicy,
    TemporalSWPairPolicy,
    UniformPairPolicy,
    sample_pair,
    validate_pair_probabilities,
)
from rq2_pairwise_surrogate_regret import INTERIOR_WIDTHS, incidence_matrix


def test_all_pair_policies_have_exact_uniform_marginals():
    flops = {width: 1e8 * (0.1 + width ** 2) for width in INTERIOR_WIDTHS}
    coordinates = np.asarray(INTERIOR_WIDTHS)
    sw = np.abs(coordinates[:, None] - coordinates[None, :])
    policies = [
        UniformPairPolicy(), ResourcePairPolicy(flops), SWPairPolicy(sw),
        ResourceGeoPairPolicy(flops, sw),
        TemporalSWPairPolicy("temporal", sw, UniformPairPolicy().probabilities),
    ]
    for policy in policies:
        q = validate_pair_probabilities(policy.probabilities)
        assert abs(q.sum() - 1.0) < 1e-9
        assert np.max(np.abs(incidence_matrix() @ q - 1 / 7)) < 1e-8
        wi, wj, index = sample_pair(q, 0.314159)
        assert wi < wj and 0 <= index < 91


def test_rpgeo_preserves_resource_optimum_and_reports_refinement():
    flops = {width: 1e8 * (0.1 + width ** 2) for width in INTERIOR_WIDTHS}
    coordinates = np.asarray(INTERIOR_WIDTHS)
    sw = np.abs(np.sin(7 * coordinates[:, None]) - np.sin(7 * coordinates[None, :]))
    resource = ResourcePairPolicy(flops)
    rpgeo = ResourceGeoPairPolicy(flops, sw, resource_retention=1.0)
    q = validate_pair_probabilities(rpgeo.probabilities)
    assert abs(q.sum() - 1.0) < 1e-9
    assert np.max(np.abs(incidence_matrix() @ q - 1 / 7)) < 1e-8
    assert rpgeo.diagnostics["resource_retention_achieved"] >= 1.0 - 1e-7
    assert rpgeo.diagnostics["geo_rg"] >= rpgeo.diagnostics["geo_resource"] - 1e-10
    assert rpgeo.diagnostics["resource_constraint_type"] == "exact_resource_equality"
    assert rpgeo.diagnostics["l1_vs_resource"] < 1e-8
    assert np.isclose(
        rpgeo.diagnostics["l1_vs_resource"],
        np.abs(rpgeo.probabilities - resource.probabilities).sum(),
    )


def test_temporal_sw_continuity_and_uniform_mixture_keep_fixed_marginals():
    coordinates = np.asarray(INTERIOR_WIDTHS)
    sw = np.abs(np.sin(11 * coordinates[:, None]) - np.sin(11 * coordinates[None, :]))
    uniform = UniformPairPolicy().probabilities
    continuity = TemporalSWPairPolicy(
        "continuity", sw, uniform,
        continuity_strength_in_score_std=1.0,
        uniform_mixture=0.0,
        use_continuity=True,
    )
    combined = TemporalSWPairPolicy(
        "combined", sw, continuity.probabilities,
        continuity_strength_in_score_std=1.0,
        uniform_mixture=0.8,
        use_continuity=True,
    )
    for policy in (continuity, combined):
        assert np.max(np.abs(incidence_matrix() @ policy.probabilities - 1 / 7)) < 1e-8
        assert policy.diagnostics["support_size"] == 91
    assert combined.diagnostics["l1_final_to_uniform"] < continuity.diagnostics["l1_final_to_uniform"]
