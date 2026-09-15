import json

import numpy as np
import pandas as pd

from rq2_anchor_placement import GRID
from rq2_cross_batch_variance import run_cross_batch_variance_validation
from rq2_gradient_variance_v3 import capped_neyman_allocation
from rq2_probabilistic_support import INTERIOR_WIDTHS, maximum_entropy_pairs


def _matrix_frame(matrix):
    frame = pd.DataFrame(matrix, columns=[f"{width:.2f}" for width in GRID])
    frame.insert(0, "width", GRID)
    return frame


def test_cross_batch_oracle_is_fit_only_on_opposite_shard(tmp_path):
    root = tmp_path / "interaction"
    workers = root / "worker_shards"
    for worker, (start, moments) in enumerate((
        (0, np.linspace(1.0, 2.0, len(GRID))),
        (8, np.linspace(2.0, 5.0, len(GRID))),
    )):
        shard = workers / f"gpu_{worker}"
        shard.mkdir(parents=True)
        (shard / "metadata.json").write_text(json.dumps({
            "training_performed": False, "test_used": False,
            "weights_unchanged": True, "bn_buffers_unchanged_during_probe": True,
            "global_batch_indices": list(range(start, start + 8)),
        }))
        _matrix_frame(np.diag(moments * 1.0001)).to_csv(
            shard / "gradient_dot_matrix.csv", index=False
        )
        pd.DataFrame([
            {"batch": batch, "width": width, "gradient_norm": np.sqrt(moments[index])}
            for batch in range(start, start + 8) for index, width in enumerate(GRID)
        ]).to_csv(shard / "gradient_norms_by_batch.csv", index=False)

    theory = root / "frozen_policy" / "theory-allocation-probe"
    preview = root / "frozen_policy" / "probabilistic-support-preview"
    theory.mkdir(parents=True)
    preview.mkdir(parents=True)
    mass = np.linspace(0.5, 1.5, len(INTERIOR_WIDTHS))
    geometry_pi = capped_neyman_allocation(mass, np.ones(len(mass)))
    resource_pi = np.full(len(INTERIOR_WIDTHS), 2 / len(INTERIOR_WIDTHS))
    geometry_pairs, _ = maximum_entropy_pairs(geometry_pi)
    resource_pairs, _ = maximum_entropy_pairs(resource_pi)
    pd.DataFrame({
        "p": 1.0, "width": INTERIOR_WIDTHS, "pi": geometry_pi,
    }).to_csv(theory / "policy_family_marginals.csv", index=False)
    geometry_pairs.to_csv(theory / "pair_distribution_p100.csv", index=False)
    pd.DataFrame({
        "width": INTERIOR_WIDTHS, "pi_resource_matched_compute": resource_pi,
    }).to_csv(preview / "support_allocation_marginals.csv", index=False)
    resource_pairs.to_csv(preview / "resource_pair_distribution.csv", index=False)
    (root / "metadata.json").write_text("{}")

    output = tmp_path / "output"
    summary = run_cross_batch_variance_validation(root, output)
    assert summary["gradient_oracle_cross_fitted"] is True
    assert summary["full_q_aware_variance_used"] is True
    by_fold = pd.read_csv(output / "cross_batch_variance_by_fold.csv")
    assert set(by_fold.fold) == {0, 1}
    assert set(by_fold.policy) == {
        "uniform_dynamic", "resource", "geometry", "gradient_oracle_crossfit"
    }
    oracle = pd.read_csv(output / "cross_batch_oracle_marginals.csv")
    first = oracle.loc[oracle.fold.eq(0)].sort_values("width")
    expected = capped_neyman_allocation(
        np.linspace(1.0, 2.0, len(GRID))[1:-1],
        np.full(len(INTERIOR_WIDTHS), 1 / len(INTERIOR_WIDTHS)),
    )
    np.testing.assert_allclose(first.crossfit_oracle_pi, expected)
