import json

import numpy as np
import pandas as pd

from rq2_fresh_seed_gate_b1 import CHECKPOINT_EPOCHS, merge_and_evaluate_fresh_states
from rq2_pairwise_incremental_value import MODEL_FEATURES
from rq2_pairwise_surrogate_regret import INTERIOR_WIDTHS, PAIR_INDICES


def _model_spec(features):
    return {
        "features": features,
        "scaler_mean": [0.0] * len(features), "scaler_scale": [1.0] * len(features),
        "linear_intercept": 0.0, "linear_coefficients": [1.0] * len(features),
    }


def test_merge_fresh_states_and_apply_frozen_predictor(tmp_path):
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({
        "status": "FROZEN_BEFORE_FRESH_STATES", "development_seeds": [3],
        "models": {name: _model_spec(features) for name, features in MODEL_FEATURES.items()},
    }))
    workers = []
    widths = np.asarray(INTERIOR_WIDTHS)
    for epoch in CHECKPOINT_EPOCHS:
        worker = tmp_path / f"worker_{epoch}"; worker.mkdir(); workers.append(worker)
        rng = np.random.default_rng(epoch)
        vectors = rng.normal(size=(14, 8)); gram = vectors @ vectors.T + np.eye(14)
        rows = []
        for i, j in PAIR_INDICES:
            distance2 = gram[i, i] + gram[j, j] - 2 * gram[i, j]
            rows.append({
                "state": f"F{epoch}", "seed": 6, "epoch": epoch,
                "width_i": widths[i], "width_j": widths[j],
                "width_distance": abs(widths[i] - widths[j]),
                "log_flops_distance": abs(np.log(widths[i] ** 2 + .1) - np.log(widths[j] ** 2 + .1)),
                "representation_sw": abs(widths[i] - widths[j]),
                "sw2": (widths[i] - widths[j]) ** 2,
                "gradient_euclidean_distance": np.sqrt(distance2),
                "gradient_distance2": distance2,
                "flops_i": 1e8 * (widths[i] ** 2 + .1),
                "flops_j": 1e8 * (widths[j] ** 2 + .1),
            })
        pd.DataFrame(rows).to_csv(worker / f"pair_structure_epoch_{epoch:03d}.csv", index=False)
        np.save(worker / f"gradient_gram_epoch_{epoch:03d}.npy", gram[None])
        np.save(worker / f"sw_matrix_epoch_{epoch:03d}.npy", np.zeros((14, 14)))
        (worker / "metadata.json").write_text(json.dumps({
            "status": "FRESH_PAIRWISE_STATE_COMPLETE", "seed": 6, "epoch": epoch,
            "gradient_probe_ids": list(range(8)), "geometry_probe_ids": list(range(10)),
        }))
    root = tmp_path / "fresh"; root.mkdir()
    result = merge_and_evaluate_fresh_states(root, workers, frozen)
    assert result["one_seed_screening_only"] is True
    table = pd.read_csv(root / "evaluation" / "resource_vs_hybrid.csv")
    assert list(table.State) == ["F10", "F50", "F100"]
    assert {"mae_R", "mae_R_plus_SW", "spearman_rho_R", "spearman_rho_R_plus_SW"}.issubset(table)
    assert (root / "state_metadata.json").is_file()
