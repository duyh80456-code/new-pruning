import numpy as np
import pandas as pd

from rq2_anchor_placement import GRID
from rq2_probabilistic_support import maximum_entropy_pairs, pair_marginals
from rq2_theory_allocation_probe import (
    INTERIOR_BUDGET,
    INTERIOR_WIDTHS,
    analytic_allocation,
    build_functional_cells,
    numerical_allocation,
    objective_jp,
    pair_expectation_checks,
    simulate_pair_waiting_times,
)


def _synthetic_problem():
    edge_lengths = np.asarray([
        0.01, 0.02, 0.04, 0.08, 0.05, 0.03, 0.02, 0.015,
        0.014, 0.013, 0.012, 0.011, 0.010, 0.009, 0.008,
    ])
    edge = pd.DataFrame({
        "left": GRID[:-1],
        "right": GRID[1:],
        "e_i": edge_lengths,
        "delta_c": np.diff(np.asarray(GRID)),
        "functional_speed_g": edge_lengths / np.diff(np.asarray(GRID)),
    })
    flops = {width: float((10 + index) ** 2) for index, width in enumerate(GRID)}
    cells = build_functional_cells(edge, flops)
    return edge, cells


def test_functional_cell_mass_equals_piecewise_voronoi_integral():
    _, cells = _synthetic_problem()
    expected = 0.5 * np.asarray([
        0.01 + 0.02, 0.02 + 0.04, 0.04 + 0.08, 0.08 + 0.05,
        0.05 + 0.03, 0.03 + 0.02, 0.02 + 0.015, 0.015 + 0.014,
        0.014 + 0.013, 0.013 + 0.012, 0.012 + 0.011, 0.011 + 0.010,
        0.010 + 0.009, 0.009 + 0.008,
    ])
    np.testing.assert_allclose(cells["functional_cell_mass"], expected, atol=1e-14)
    np.testing.assert_allclose(
        cells["functional_cell_mass"], cells["piecewise_integral_mass"], atol=1e-14
    )


def test_jp_closed_form_matches_independent_numeric_solver_for_fixed_family():
    _, cells = _synthetic_problem()
    mass = cells["functional_cell_mass"].to_numpy(float)
    for p in (0.25, 0.5, 1.0, 2.0, 4.0):
        closed = analytic_allocation(mass, p)
        numeric, diagnostics = numerical_allocation(mass, p)
        assert diagnostics["solver_success"]
        assert abs(closed.sum() - INTERIOR_BUDGET) < 1e-12
        assert np.max(np.abs(closed - numeric)) < 1e-6
        assert objective_jp(closed, mass, p) <= objective_jp(numeric, mass, p) + 1e-10


def test_large_p_allocations_converge_toward_uniform():
    _, cells = _synthetic_problem()
    mass = cells["functional_cell_mass"].to_numpy(float)
    spreads = [np.ptp(analytic_allocation(mass, p)) for p in (10.0, 100.0, 1000.0)]
    assert spreads[0] > spreads[1] > spreads[2]
    uniform = INTERIOR_BUDGET / len(INTERIOR_WIDTHS)
    assert np.max(np.abs(analytic_allocation(mass, 1000.0) - uniform)) < 1e-3


def test_pair_realization_preserves_marginals_loss_and_gradient_expectations():
    _, cells = _synthetic_problem()
    pi = analytic_allocation(cells["functional_cell_mass"].to_numpy(float), 1.0)
    pairs, solver = maximum_entropy_pairs(pi)
    checks = pair_expectation_checks(pairs, pi)
    assert solver["solver_success"]
    assert np.max(np.abs(pair_marginals(pairs) - pi)) < 1e-8
    assert checks["loss_identity_absolute_error"] < 1e-10
    assert checks["gradient_identity_max_absolute_error"] < 1e-10


def test_waiting_time_simulation_uses_pair_sampler_and_matches_geometric_mean():
    _, cells = _synthetic_problem()
    pi = analytic_allocation(cells["functional_cell_mass"].to_numpy(float), 1.0)
    pairs, _ = maximum_entropy_pairs(pi)
    waiting = simulate_pair_waiting_times(pairs, pi, 1.0, n_steps=300_000, seed=321)
    assert len(waiting) == len(INTERIOR_WIDTHS)
    assert waiting["relative_error"].max() < 0.04
