import numpy as np

from rq2_anchor_placement import GRID
from rq2_probabilistic_support import (
    INTERIOR_WIDTHS,
    functional_mass,
    maximum_entropy_pairs,
    pair_marginals,
    sample_pair,
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


def test_maximum_entropy_pairs_reproduce_marginals_and_sample_distinct_widths():
    mass, flops, target = _problem()
    pi, _ = solve_marginals(mass, flops, target, "geometry")
    pairs, diagnostics = maximum_entropy_pairs(pi)
    assert abs(pairs["probability"].sum() - 1.0) < 1e-8
    assert np.max(np.abs(pair_marginals(pairs) - pi)) < 1e-6
    assert diagnostics["maximum_marginal_absolute_residual"] < 1e-6
    for _ in range(100):
        first, second = sample_pair(pairs, np.random.default_rng(_))
        assert first != second

