import json

import numpy as np
import pandas as pd

from rq2_anchor_placement import GRID
from rq2_dynamic_seed3_pilot import (
    GEOMETRY_CONTINUOUS_METHOD,
    METHODS,
    finalize_pilot,
    freeze_pilot_policies,
    load_frozen_policy,
    simulate_sampler_sanity,
)
from rq2_probabilistic_support import INTERIOR_WIDTHS, maximum_entropy_pairs


def _make_frozen_sources(tmp_path):
    theory, preview, development = (
        tmp_path / "theory", tmp_path / "preview", tmp_path / "development"
    )
    theory.mkdir(); preview.mkdir(); development.mkdir()
    pi_geometry = np.linspace(0.21, 0.08, len(INTERIOR_WIDTHS))
    pi_geometry *= 2.0 / pi_geometry.sum()
    flops_values = np.linspace(1e8, 1e9, len(GRID))
    F = flops_values[1:-1]
    target = float(F @ pi_geometry)

    # Find the matched-compute maximum-entropy resource marginal using the
    # production solver so the fixture is mathematically feasible.
    from rq2_probabilistic_support import solve_marginals
    pi_resource, _ = solve_marginals(
        np.ones(len(F)), F, target, "resource", compute_constraint="equal"
    )
    geometry_pairs, _ = maximum_entropy_pairs(pi_geometry)
    resource_pairs, _ = maximum_entropy_pairs(pi_resource)
    geometry_pairs.to_csv(theory / "pair_distribution_p100.csv", index=False)
    resource_pairs.to_csv(preview / "resource_pair_distribution.csv", index=False)
    pd.DataFrame({
        "p": 1.0, "width": INTERIOR_WIDTHS, "pi": pi_geometry,
    }).to_csv(theory / "policy_family_marginals.csv", index=False)
    (theory / "theory_probe_summary.json").write_text(json.dumps({
        "accuracy_used": False, "test_used": False, "training_authorized": False,
        "development_geometry_seeds": [0, 1, 2],
        "primary_theory_policy": {"p": 1.0},
        "all_closed_form_checks_pass": True,
        "all_numeric_solver_checks_pass": True,
    }))
    pd.DataFrame({
        "width": INTERIOR_WIDTHS,
        "pi_resource_matched_compute": pi_resource,
    }).to_csv(preview / "support_allocation_marginals.csv", index=False)
    (preview / "support_allocation_diagnostics.json").write_text(json.dumps({
        "accuracy_used": False, "training_authorized": False,
        "development_geometry_seeds": [0, 1, 2], "assertions": {"all": True},
    }))
    pd.DataFrame([
        {"method": "uniform", "budget": width, "flops": flops}
        for width, flops in zip(GRID, flops_values)
    ]).to_csv(development / "rq2_dense_metrics_all.csv", index=False)
    return theory, preview, development


def test_frozen_pilot_policies_are_matched_hashed_and_sampler_valid(tmp_path):
    theory, preview, development = _make_frozen_sources(tmp_path)
    protocol_dir = tmp_path / "protocol"
    protocol = freeze_pilot_policies(
        theory, preview, development, protocol_dir, pilot_seed=4
    )
    assert protocol["accuracy_used_to_build_policy"] is False
    assert protocol["test_used"] is False
    assert protocol["status"] == "FROZEN_BEFORE_DECLARED_DYNAMIC_PILOT_SEED"
    assert protocol["seed"] == 4
    expected_compute = protocol["expected_total_compute"]
    for method in METHODS:
        pairs, pi, flops, loaded = load_frozen_policy(protocol_dir, method)
        sanity = simulate_sampler_sanity(
            pairs, pi, flops, loaded["fixed_endpoint_compute"], method,
            seed=4, draws=100_000,
        )
        assert sanity["passed"]
        assert sanity["sampler_seed"] in (100004, 200004)
        assert abs(sanity["expected_total_flops"] - expected_compute) / expected_compute < 1e-8
    _, continuous_pi, _, _ = load_frozen_policy(protocol_dir, GEOMETRY_CONTINUOUS_METHOD)
    _, geometry_pi, _, _ = load_frozen_policy(protocol_dir, "geometry_dynamic")
    np.testing.assert_allclose(continuous_pi, geometry_pi)


def test_finalize_pilot_produces_validation_only_width_delta_and_pattern_gate(tmp_path):
    root = tmp_path
    seed = 4
    widths = np.asarray(GRID)
    resource = 0.55 + 0.12 * widths
    geometry = resource.copy()
    geometry[(widths >= 0.30) & (widths <= 0.45)] += 0.01
    geometry[(widths >= 0.50) & (widths <= 0.55)] -= 0.001
    continuous = geometry + 0.002
    methods = ("geometry_dynamic", "resource_dynamic", GEOMETRY_CONTINUOUS_METHOD)
    for method, accuracy in (
        ("resource_dynamic", resource), ("geometry_dynamic", geometry),
        (GEOMETRY_CONTINUOUS_METHOD, continuous),
    ):
        output = root / "evaluation" / method / f"seed_{seed}"
        output.mkdir(parents=True)
        pd.DataFrame({
            "method": method, "seed": seed, "split": "validation_5k",
            "width": GRID, "accuracy": accuracy, "loss": 1.0,
        }).to_csv(output / "dense_validation_accuracy.csv", index=False)
    decision = finalize_pilot(root, seed=seed, methods=methods)
    assert decision["test_used"] is False
    assert decision["verdict"] == "DEVELOPMENT GO"
    assert decision["seed"] == 4
    assert decision["continuous_minus_reset_geometry"]["dense_mean_accuracy"] > 0
    comparison = pd.read_csv(root / "dynamic_pilot_width_comparison.csv")
    assert len(comparison) == 16
    assert comparison.loc[comparison.width.eq(0.40), "geometry_minus_resource_accuracy"].iloc[0] > 0
