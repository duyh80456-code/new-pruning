import json
from pathlib import Path

import numpy as np
import pandas as pd

from rq2_fresh_seed6_policy_decomposition import (
    EPOCHS,
    find_fresh_seed6_inputs,
    run_fresh_seed6_policy_decomposition,
)
from rq2_pairwise_surrogate_regret import INTERIOR_WIDTHS, PAIR_INDICES


def _fixture(input_root: Path) -> tuple[Path, Path]:
    fresh = input_root / "checkpoint_6" / "results" / "fresh_seed_6"
    (fresh / "gradient_grams").mkdir(parents=True)
    rows = []
    flops = {width: 1e8 * (0.1 + width ** 2) for width in INTERIOR_WIDTHS}
    for epoch in EPOCHS:
        rng = np.random.default_rng(6000 + epoch)
        gradients = rng.normal(size=(len(INTERIOR_WIDTHS), 8))
        gram = gradients @ gradients.T + np.eye(len(INTERIOR_WIDTHS)) * 0.01
        np.save(fresh / "gradient_grams" / f"epoch_{epoch:03d}.npy", gram)
        for i, j in PAIR_INDICES:
            distance2 = gram[i, i] + gram[j, j] - 2 * gram[i, j]
            wi, wj = INTERIOR_WIDTHS[i], INTERIOR_WIDTHS[j]
            rows.append({
                "state": f"F{epoch}", "seed": 6, "epoch": epoch,
                "width_i": wi, "width_j": wj,
                "width_distance": abs(wi - wj),
                "log_flops_distance": abs(np.log(flops[wi]) - np.log(flops[wj])),
                "representation_sw": abs(wi - wj) * (1 + epoch / 1000),
                "gradient_euclidean_distance": np.sqrt(max(distance2, 0.0)),
                "flops_i": flops[wi], "flops_j": flops[wj],
            })
    pd.DataFrame(rows).to_csv(fresh / "pair_structure.csv", index=False)
    predictor = input_root / "frozen-development-gates" / "frozen_pairwise_predictors.json"
    predictor.parent.mkdir(parents=True)
    predictor.write_text(json.dumps({
        "status": "FROZEN_BEFORE_FRESH_STATES",
        "development_seeds": [3],
        "models": {
            "Resource": {
                "features": ["width_distance_sq_normalized", "log_flops_distance_sq_normalized"],
                "scaler_mean": [0.0, 0.0], "scaler_scale": [1.0, 1.0],
                "linear_intercept": 0.0, "linear_coefficients": [0.5, 0.5],
            },
            "Resource+SW": {
                "features": ["width_distance_sq_normalized", "log_flops_distance_sq_normalized", "sw_sq_normalized"],
                "scaler_mean": [0.0, 0.0, 0.0], "scaler_scale": [1.0, 1.0, 1.0],
                "linear_intercept": 0.0, "linear_coefficients": [0.3, 0.3, 0.4],
            },
        },
    }))
    return fresh, predictor


def test_fresh_seed6_policy_decomposition_is_cpu_artifact_only(tmp_path):
    fresh, predictor = _fixture(tmp_path / "input")
    resolved_fresh, resolved_predictor = find_fresh_seed6_inputs(
        tmp_path / "input", tmp_path / "materialized"
    )
    assert resolved_fresh == fresh
    assert resolved_predictor == predictor

    output = tmp_path / "output"
    summary = run_fresh_seed6_policy_decomposition(fresh, predictor, output)
    assert summary["status"] == "FRESH_SEED6_POLICY_DECOMPOSITION_COMPLETE"
    assert summary["checkpoint_loaded"] is False
    assert summary["predictors_refit"] is False
    assert summary["gpu_required"] is False
    assert summary["gate_c_authorized"] is False

    wide = pd.read_csv(output / "fresh_seed6_policy_decomposition.csv")
    long = pd.read_csv(output / "fresh_seed6_policy_variance_long.csv")
    assignments = pd.read_csv(output / "fresh_seed6_pair_policy_assignments.csv")
    assert len(wide) == 3
    assert len(long) == 3 * 6
    assert len(assignments) == 3 * 6 * 91
    assert np.all(
        wide.V_gradient_oracle.to_numpy()
        <= wide[[column for column in wide if column.startswith("V_")]].min(axis=1).to_numpy() + 1e-8
    )
    for name in (
        "fresh_seed6_policy_l1_to_oracle.csv",
        "fresh_seed6_sw_gradient_correlations.csv",
        "fresh_seed6_policy_decomposition_summary.json",
        "fresh_seed6_policy_decomposition_summary.md",
        "fresh_seed6_policy_variance.png",
        "fresh_seed6_policy_l1_to_oracle.png",
    ):
        assert (output / name).is_file()

