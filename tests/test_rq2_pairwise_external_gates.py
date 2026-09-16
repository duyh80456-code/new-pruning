import json
from pathlib import Path

import numpy as np
import pandas as pd

from rq2_pairwise_external_gates import run_all_development_gates, run_gate_b1_fresh
from rq2_pairwise_surrogate_regret import EPOCHS, INTERIOR_WIDTHS, PAIR_INDICES


def _fixture(root: Path):
    (root / "worker_paths").mkdir(parents=True)
    geometry, pairs = [], []
    flops = {w: 1e8 * (w ** 2 + 0.1) for w in INTERIOR_WIDTHS}
    for path_index, path in enumerate(("geo_ht", "resource_ht")):
        folder = root / "worker_paths" / path
        folder.mkdir()
        for epoch in EPOCHS:
            rng = np.random.default_rng(1000 * path_index + epoch)
            matrix = rng.normal(size=(14, 9))
            gram = matrix @ matrix.T + np.eye(14)
            np.save(folder / f"interior_grams_epoch_{epoch:03d}.npy", gram[None])
            for width in INTERIOR_WIDTHS:
                geometry.append({"path": path, "epoch": epoch, "width": width, "flops": flops[width]})
            for i, j in PAIR_INDICES:
                distance = np.sqrt(max(0, gram[i, i] + gram[j, j] - 2 * gram[i, j]))
                sw = abs(INTERIOR_WIDTHS[i] - INTERIOR_WIDTHS[j]) * (1 + 0.01 * epoch)
                pairs.append({
                    "path": path, "epoch": epoch,
                    "width_i": INTERIOR_WIDTHS[i], "width_j": INTERIOR_WIDTHS[j],
                    "gradient_euclidean_distance": distance,
                    "representation_sw": sw,
                })
    pd.DataFrame(geometry).to_csv(root / "quick_dynamic_geometry_by_width.csv", index=False)
    pd.DataFrame(pairs).to_csv(root / "quick_pair_structure.csv", index=False)


def test_external_gates_and_freeze(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir(); _fixture(source)
    result = run_all_development_gates(source, output)
    assert result["status"] == "DEVELOPMENT_GATES_COMPLETE_FRESH_GATE_PENDING"
    for name in (
        "gate_a_resource_pair_scores.csv", "gate_a_resource_pair_policies.csv",
        "gate_a_resource_exact_variance.csv", "gate_a_summary.json",
        "gate_b0_heldout_predictions.csv", "gate_b0_heldout_metrics.csv",
        "gate_b0_heldout_exact_variance.csv", "gate_b0_summary.csv",
        "frozen_pairwise_predictors.json", "gate_a_b0_complete.json",
    ):
        assert (output / name).is_file()
    gate_a = json.loads((output / "gate_a_summary.json").read_text())
    assert gate_a["flops_are_model_profiled_not_width_proxy"] is True
    policies = pd.read_csv(output / "gate_b0_heldout_exact_variance.csv")
    assert set(policies.protocol) == {"LOEO", "LPO"}
    assert len(policies.loc[policies.protocol.eq("LOEO")]) == 6
    assert len(policies.loc[policies.protocol.eq("LPO")]) == 6
    frozen = json.loads((output / "frozen_pairwise_predictors.json").read_text())
    assert frozen["status"] == "FROZEN_BEFORE_FRESH_STATES"
    assert frozen["no_refit_on_fresh_states"] is True


def test_fresh_gate_uses_frozen_predictors_without_refit(tmp_path):
    source, development, fresh, output = (
        tmp_path / "source", tmp_path / "development", tmp_path / "fresh", tmp_path / "fresh_out"
    )
    source.mkdir(); _fixture(source)
    run_all_development_gates(source, development)
    (fresh / "grams").mkdir(parents=True)
    original = pd.read_csv(source / "quick_pair_structure.csv")
    rows = []
    for seed in (4, 5):
        for epoch in EPOCHS:
            state = original.loc[(original.path == "geo_ht") & (original.epoch == epoch)].copy()
            state["seed"] = seed
            state["flops_i"] = 1e8 * (state.width_i ** 2 + 0.1)
            state["flops_j"] = 1e8 * (state.width_j ** 2 + 0.1)
            rows.append(state)
            gram = np.load(source / "worker_paths" / "geo_ht" / f"interior_grams_epoch_{epoch:03d}.npy")
            np.save(fresh / "grams" / f"seed_{seed}_epoch_{epoch:03d}.npy", gram)
    pd.concat(rows, ignore_index=True).to_csv(fresh / "fresh_pair_structure.csv", index=False)
    (fresh / "fresh_state_metadata.json").write_text(json.dumps({
        "status": "FRESH_PAIRWISE_STATES_COMPLETE", "seeds": [4, 5],
        "predictor_used_during_extraction": False,
    }))
    result = run_gate_b1_fresh(
        fresh, development / "frozen_pairwise_predictors.json", output, bootstrap_draws=100
    )
    assert result["fresh_states"] == 6
    assert result["predictors_refit_on_fresh_states"] is False
    assert (output / "gate_b1_fresh_exact_variance.csv").is_file()
