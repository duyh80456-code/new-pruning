import json

import numpy as np
import pandas as pd

from rq2_cross_batch_variance_4fold import run_four_fold_cross_batch_validation
from rq2_gradient_variance_v3 import capped_neyman_allocation
from rq2_probabilistic_support import INTERIOR_WIDTHS, maximum_entropy_pairs


def test_four_fold_oracle_never_uses_heldout_batches(tmp_path):
    root = tmp_path / "interaction"
    theory = root / "frozen_policy" / "theory-allocation-probe"
    preview = root / "frozen_policy" / "probabilistic-support-preview"
    theory.mkdir(parents=True)
    preview.mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps({
        "training_performed": False, "test_used": False,
        "per_batch_interior_grams_saved": True,
    }))
    diagonal = np.asarray([
        np.linspace(1.0 + 0.1 * batch, 2.0 + 0.3 * batch, len(INTERIOR_WIDTHS))
        for batch in range(16)
    ])
    grams = np.asarray([np.diag(values) for values in diagonal])
    np.save(root / "gram_matrices.npy", grams)

    geometry_pi = capped_neyman_allocation(
        np.linspace(0.5, 1.5, len(INTERIOR_WIDTHS)), np.ones(len(INTERIOR_WIDTHS))
    )
    resource_pi = np.full(len(INTERIOR_WIDTHS), 2 / len(INTERIOR_WIDTHS))
    geometry_pairs, _ = maximum_entropy_pairs(geometry_pi)
    resource_pairs, _ = maximum_entropy_pairs(resource_pi)
    pd.DataFrame({
        "p": 1.0, "width": INTERIOR_WIDTHS,
        "functional_cell_mass": np.linspace(0.5, 1.5, len(INTERIOR_WIDTHS)),
        "flops": np.linspace(1e8, 9e8, len(INTERIOR_WIDTHS)), "pi": geometry_pi,
    }).to_csv(theory / "policy_family_marginals.csv", index=False)
    geometry_pairs.to_csv(theory / "pair_distribution_p100.csv", index=False)
    pd.DataFrame({
        "width": INTERIOR_WIDTHS, "pi_resource_matched_compute": resource_pi,
    }).to_csv(preview / "support_allocation_marginals.csv", index=False)
    resource_pairs.to_csv(preview / "resource_pair_distribution.csv", index=False)

    output = tmp_path / "output"
    result = run_four_fold_cross_batch_validation(root, output, bootstrap_draws=1000)
    assert result["num_folds"] == 4
    assert result["oracle_fit_only_on_training_folds"] is True
    assignment = pd.read_csv(output / "fold_assignment.csv")
    assert np.array_equal(assignment.fold, assignment.batch_id % 4)
    fold_zero = pd.read_csv(output / "oracle_policy_fold0.csv")
    train_m = diagonal[np.arange(16) % 4 != 0].mean(axis=0)
    expected = capped_neyman_allocation(
        train_m, np.full(len(INTERIOR_WIDTHS), 1 / len(INTERIOR_WIDTHS))
    )
    np.testing.assert_allclose(fold_zero.pi, expected)
    batch = pd.read_csv(output / "batch_variance.csv")
    assert len(batch) == 16 and batch.batch_id.nunique() == 16
    assert np.load(output / "bootstrap_geo_minus_resource.npy").shape == (1000,)
    assert all((output / name).is_file() for name in (
        "policy_uniform.csv", "policy_resource.csv", "policy_geometry.csv"
    ))
