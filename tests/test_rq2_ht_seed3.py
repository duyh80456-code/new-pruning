import numpy as np
import pandas as pd
import torch

from rq2_anchor_placement import GRID
from rq2_ht_seed3 import (
    METHODS,
    TARGET_INTERIOR_WEIGHT,
    finalize_ht_development,
    ht_coefficients,
    ht_weighted_objective,
    simulate_ht_sanity,
    validate_ht_policy,
)
from rq2_probabilistic_support import INTERIOR_WIDTHS, maximum_entropy_pairs


def test_ht_coefficients_are_unbiased_without_clipping_or_renormalization():
    pi = np.linspace(0.04, 0.24, len(INTERIOR_WIDTHS))
    pi *= 2 / pi.sum()
    alpha = ht_coefficients(pi)
    np.testing.assert_allclose(pi * alpha, TARGET_INTERIOR_WEIGHT)
    assert alpha.max() > 1.0
    assert not np.isclose(alpha[0] + alpha[-1], 2.0)


def test_ht_weighted_objective_preserves_historical_divide_by_four_scale():
    losses = [torch.tensor(value, requires_grad=True) for value in (1.0, 2.0, 3.0, 4.0)]
    alpha = {0.30: 0.5, 0.90: 1.5}
    total = ht_weighted_objective(
        losses[0], losses[1], (losses[2], losses[3]), (0.30, 0.90), alpha
    )
    assert float(total) == (1 + 2 + 0.5 * 3 + 1.5 * 4) / 4
    total.backward()
    np.testing.assert_allclose([value.grad for value in losses], [0.25, 0.25, 0.125, 0.375])


def test_pair_policy_and_100k_sampler_reproduce_ht_target_weight():
    pi = np.linspace(0.08, 0.20, len(INTERIOR_WIDTHS))
    pi *= 2 / pi.sum()
    pairs, _ = maximum_entropy_pairs(pi)
    flops = dict(zip(INTERIOR_WIDTHS, np.linspace(1e8, 9e8, len(INTERIOR_WIDTHS))))
    checks = validate_ht_policy(pairs, pi, flops)
    assert checks["max_expected_target_weight_error"] < 1e-12
    sanity = simulate_ht_sanity(pairs, pi, "geo_ht", draws=100_000)
    assert sanity.effective_weight_error.abs().max() < 0.01


def test_finalizer_exports_checkpoint_and_final_width_comparisons(tmp_path):
    rows = []
    for method_index, method in enumerate(METHODS):
        output = tmp_path / "evaluation" / method / "seed_3"
        output.mkdir(parents=True)
        method_rows = []
        for epoch in range(10, 101, 10):
            for width in GRID:
                method_rows.append({
                    "method": method, "seed": 3, "epoch": epoch,
                    "split": "validation_5k", "width": width,
                    "cumulative_realized_flops": epoch * 1e12 * (1 + 0.001 * method_index),
                    "accuracy": 0.4 + 0.002 * epoch + 0.01 * width + 0.001 * (method == "geo_ht"),
                    "loss": 1.0,
                })
        pd.DataFrame(method_rows).to_csv(output / "dense_validation_by_checkpoint.csv", index=False)
        rows.extend(method_rows)
    decision = finalize_ht_development(tmp_path)
    assert decision["test_used"] is False
    assert decision["final_dense_accuracy_delta"] > 0
    assert (tmp_path / "ht_convergence_common_compute.csv").is_file()
    final = pd.read_csv(tmp_path / "final_width_comparison.csv")
    assert len(final) == 16
    assert (final.geo_minus_resource_accuracy > 0).all()
