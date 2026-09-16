import json

import numpy as np
import pandas as pd

from rq2_pairwise_incremental_value import (
    build_pairwise_state_table,
    run_pairwise_incremental_value,
)
from rq2_pairwise_surrogate_regret import EPOCHS, INTERIOR_WIDTHS, PAIR_INDICES


def _make_quick_fixture(root):
    (root / "worker_paths").mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps({
        "status": "RQ2_V3_QUICK_TRAJECTORY_DIAGNOSTIC_COMPLETE"
    }))
    rng = np.random.default_rng(44)
    base = rng.normal(size=(len(INTERIOR_WIDTHS), 6))
    pair_rows, geometry_rows = [], []
    for path_index, path in enumerate(("geo_ht", "resource_ht")):
        worker = root / "worker_paths" / path
        worker.mkdir(parents=True)
        for epoch_index, epoch in enumerate(EPOCHS):
            gradients = base + 0.03 * (path_index + epoch_index) * rng.normal(size=base.shape)
            gram = gradients @ gradients.T
            np.save(worker / f"interior_grams_epoch_{epoch:03d}.npy", np.repeat(
                gram[None], 8, axis=0
            ))
            for i, j in PAIR_INDICES:
                distance_sq = gram[i, i] + gram[j, j] - 2 * gram[i, j]
                pair_rows.append({
                    "path": path, "epoch": epoch,
                    "width_i": INTERIOR_WIDTHS[i], "width_j": INTERIOR_WIDTHS[j],
                    "representation_sw": np.sqrt(max(distance_sq, 0.0)),
                    "gradient_euclidean_distance": np.sqrt(max(distance_sq, 0.0)),
                })
            for width in INTERIOR_WIDTHS:
                geometry_rows.append({
                    "path": path, "epoch": epoch, "width": width,
                    "flops": 1e8 * width * width,
                })
    pd.DataFrame(pair_rows).to_csv(root / "quick_pair_structure.csv", index=False)
    pd.DataFrame(geometry_rows).to_csv(root / "quick_dynamic_geometry_by_width.csv", index=False)


def test_pairwise_state_table_has_six_normalized_states(tmp_path):
    root = tmp_path / "quick"; _make_quick_fixture(root)
    table = build_pairwise_state_table(root)
    assert len(table) == 6 * 91 and table.state.nunique() == 6
    means = table.groupby("state")[[
        "gradient_distance_sq_normalized", "sw_sq_normalized",
        "width_distance_sq_normalized", "log_flops_distance_sq_normalized",
    ]].mean()
    np.testing.assert_allclose(means, 1.0)


def test_full_loso_incremental_value_recovers_exact_sw_signal(tmp_path):
    root = tmp_path / "quick"; _make_quick_fixture(root)
    output = tmp_path / "output"
    result = run_pairwise_incremental_value(root, output)
    assert result["preprocessing_fit_train_states_only"]
    assert result["prediction_mae_improved_states"] == 6
    metrics = pd.read_csv(output / "pairwise_loso_metrics.csv")
    hybrid = metrics.loc[metrics.model.eq("Resource+SW")]
    assert hybrid.mae.max() < 1e-10
    policy = pd.read_csv(output / "pairwise_loso_policy_variance.csv")
    np.testing.assert_allclose(
        policy.V_hybrid_resource_plus_sw_prediction,
        policy.V_gradient_oracle,
        rtol=1e-8, atol=1e-8,
    )
    for name in (
        "pairwise_loso_predictions.csv", "pairwise_loso_summary.csv",
        "loso_pair_distance_mae.png", "loso_hybrid_policy_variance_delta.png",
        "loso_hybrid_predicted_vs_observed.png", "metadata.json",
    ):
        assert (output / name).is_file()
