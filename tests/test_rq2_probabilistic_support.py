import json

import numpy as np

from rq2_anchor_placement import GRID
from rq2_probabilistic_support import (
    INTERIOR_WIDTHS,
    functional_mass,
    maximum_entropy_pairs,
    pair_marginals,
    sample_pair,
    solve_geometry_closed_form,
    solve_marginals,
)


def _problem():
    coordinate = {width: value for width, value in zip(
        GRID, np.cumsum([0.0, 0.01, 0.04, 0.08, 0.05, 0.02, 0.01, 0.01,
                         0.01, 0.01, 0.015, 0.012, 0.011, 0.01, 0.01, 0.01])
    )}
    _, mass = functional_mass(coordinate)
    flops = np.linspace(1.0, 8.0, len(INTERIOR_WIDTHS)) ** 2
    target = float(flops[4] + flops[9])
    return mass, flops, target


def test_marginal_solvers_match_slots_compute_and_bounds():
    mass, flops, target = _problem()
    for mode in ("geometry", "resource"):
        pi, diagnostics = solve_marginals(mass, flops, target, mode)
        assert abs(pi.sum() - 2.0) < 1e-6
        assert abs(flops @ pi - target) / target < 1e-6
        assert np.all(pi > 0)
        assert np.all(pi <= 1.0 + 1e-8)
        assert diagnostics["solver_success"]
    geometry, _ = solve_marginals(mass, flops, target, "geometry")
    assert geometry[np.argmax(mass)] > geometry[np.argmin(mass)]


def test_compute_cap_does_not_force_uniform_compute_spending():
    mass, flops, target = _problem()
    geometry, diagnostics = solve_geometry_closed_form(mass, flops, target)
    resource, resource_diagnostics = solve_marginals(
        mass, flops, float(flops @ geometry), "resource", compute_constraint="equal"
    )
    expected = 2.0 * np.sqrt(mass) / np.sqrt(mass).sum()
    np.testing.assert_allclose(geometry, expected, atol=1e-12)
    np.testing.assert_allclose(geometry.sum(), 2.0, atol=1e-7)
    assert flops @ geometry <= target * (1 + 1e-7)
    assert abs(flops @ resource - flops @ geometry) / (flops @ geometry) < 1e-7
    assert diagnostics["compute_constraint"] == "cap"
    assert diagnostics["analytic_candidate_used"]
    assert diagnostics["solver_success"]
    assert resource_diagnostics["compute_constraint"] == "equal"
    assert resource_diagnostics["solver_success"]


def test_maximum_entropy_pairs_reproduce_marginals_and_sample_distinct_widths():
    mass, flops, target = _problem()
    pi, _ = solve_marginals(mass, flops, target, "geometry")
    pairs, diagnostics = maximum_entropy_pairs(pi)
    assert abs(pairs["probability"].sum() - 1.0) < 1e-8
    assert np.max(np.abs(pair_marginals(pairs) - pi)) < 1e-6
    assert diagnostics["solver_success"]
    assert diagnostics["maximum_marginal_absolute_residual"] < 1e-6
    for _ in range(100):
        first, second = sample_pair(pairs, np.random.default_rng(_))
        assert first != second


def test_numpy_diagnostic_scalars_are_json_serializable():
    from rq2_probabilistic_support import _json_default

    payload = {"check": np.bool_(True), "value": np.float64(0.25), "count": np.int64(2)}
    assert json.loads(json.dumps(payload, default=_json_default)) == {
        "check": True, "value": 0.25, "count": 2,
    }
