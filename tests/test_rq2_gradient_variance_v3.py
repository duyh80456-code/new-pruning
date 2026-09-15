import json

import numpy as np
import pandas as pd

from rq2_anchor_placement import GRID
from rq2_gradient_variance_v3 import (
    capped_neyman_allocation,
    importance_corrected_variance,
    run_gradient_variance_v3,
)
from rq2_probabilistic_support import INTERIOR_WIDTHS, maximum_entropy_pairs


def test_capped_neyman_allocation_matches_uncapped_formula_when_interior():
    moments = np.linspace(1.0, 4.0, len(INTERIOR_WIDTHS))
    weights = np.full(len(INTERIOR_WIDTHS), 1 / len(INTERIOR_WIDTHS))
    pi = capped_neyman_allocation(moments, weights)
    expected = 2 * np.sqrt(moments) / np.sqrt(moments).sum()
    np.testing.assert_allclose(pi, expected)
    assert abs(pi.sum() - 2) < 1e-10
    assert np.all((pi > 0) & (pi <= 1))


def test_importance_corrected_variance_matches_pair_enumeration():
    rng = np.random.default_rng(31)
    gradients = rng.normal(size=(len(INTERIOR_WIDTHS), 7))
    dot = gradients @ gradients.T
    weights = np.full(len(INTERIOR_WIDTHS), 1 / len(INTERIOR_WIDTHS))
    pi = np.full(len(INTERIOR_WIDTHS), 2 / len(INTERIOR_WIDTHS))
    pairs, _ = maximum_entropy_pairs(pi)
    result = importance_corrected_variance(dot, weights, pi, pairs)

    target = weights @ gradients
    brute = 0.0
    index = {width: i for i, width in enumerate(INTERIOR_WIDTHS)}
    for row in pairs.itertuples():
        i = index[round(float(row.width_i), 2)]
        j = index[round(float(row.width_j), 2)]
        estimate = weights[i] * gradients[i] / pi[i] + weights[j] * gradients[j] / pi[j]
        brute += float(row.probability) * float(np.square(estimate - target).sum())
    np.testing.assert_allclose(result["importance_corrected_variance"], brute, rtol=1e-10)


def test_neyman_marginals_reduce_diagonal_surrogate_against_uniform():
    moments = np.geomspace(1.0, 100.0, len(INTERIOR_WIDTHS))
    weights = np.full(len(INTERIOR_WIDTHS), 1 / len(INTERIOR_WIDTHS))
    oracle = capped_neyman_allocation(moments, weights)
    uniform = np.full(len(INTERIOR_WIDTHS), 2 / len(INTERIOR_WIDTHS))
    oracle_surrogate = np.sum(np.square(weights) * moments / oracle)
    uniform_surrogate = np.sum(np.square(weights) * moments / uniform)
    assert oracle_surrogate < uniform_surrogate


def test_complete_cpu_diagnostic_writes_expected_outputs(tmp_path):
    root = tmp_path / "interaction"
    theory = root / "frozen_policy" / "theory-allocation-probe"
    preview = root / "frozen_policy" / "probabilistic-support-preview"
    theory.mkdir(parents=True)
    preview.mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps({
        "training_performed": False, "test_used": False, "weights_unchanged": True,
        "num_fixed_training_batches": 2,
    }))
    moments = np.linspace(1.0, 3.0, len(GRID))
    dot = np.diag(moments)
    matrix = pd.DataFrame(dot, columns=[f"{width:.2f}" for width in GRID])
    matrix.insert(0, "width", GRID)
    matrix.to_csv(root / "gradient_dot_matrix.csv", index=False)
    pd.DataFrame([
        {"batch": batch, "width": width, "gradient_norm": np.sqrt(moments[index])}
        for batch in range(2) for index, width in enumerate(GRID)
    ]).to_csv(root / "gradient_norms_by_batch.csv", index=False)

    interior_moments = moments[1:-1]
    mass = np.linspace(0.5, 1.5, len(INTERIOR_WIDTHS))
    geometry_pi = capped_neyman_allocation(mass, np.ones(len(mass)))
    resource_pi = np.full(len(INTERIOR_WIDTHS), 2 / len(INTERIOR_WIDTHS))
    geometry_pairs, _ = maximum_entropy_pairs(geometry_pi)
    resource_pairs, _ = maximum_entropy_pairs(resource_pi)
    pd.DataFrame({
        "p": 1.0, "width": INTERIOR_WIDTHS, "functional_cell_mass": mass,
        "flops": np.linspace(10, 100, len(INTERIOR_WIDTHS)), "pi": geometry_pi,
    }).to_csv(theory / "policy_family_marginals.csv", index=False)
    geometry_pairs.assign(p=1.0).to_csv(theory / "pair_distribution_p100.csv", index=False)
    pd.DataFrame({
        "width": INTERIOR_WIDTHS, "pi_resource_matched_compute": resource_pi,
    }).to_csv(preview / "support_allocation_marginals.csv", index=False)
    resource_pairs.to_csv(preview / "resource_pair_distribution.csv", index=False)

    output = tmp_path / "output"
    summary = run_gradient_variance_v3(root, output)
    assert summary["training_performed"] is False
    assert summary["gpu_used"] is False
    assert (output / "importance_corrected_estimator_variance.csv").is_file()
    table = pd.read_csv(output / "gradient_second_moment_by_width.csv")
    np.testing.assert_allclose(table.gradient_second_moment, interior_moments)
