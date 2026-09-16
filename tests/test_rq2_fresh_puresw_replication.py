import json
from pathlib import Path

import numpy as np
import pandas as pd

from rq2_fresh_puresw_replication import CHECKPOINT_EPOCHS, merge_and_evaluate
from rq2_pairwise_surrogate_regret import INTERIOR_WIDTHS, PAIR_INDICES


def test_merge_and_evaluate_fresh_puresw_replication(tmp_path):
    root = tmp_path / "fresh_seed_7"
    workers = []
    flops = {width: 1e8 * (0.1 + width ** 2) for width in INTERIOR_WIDTHS}
    for epoch in CHECKPOINT_EPOCHS:
        worker = tmp_path / f"worker_{epoch}"
        worker.mkdir()
        workers.append(worker)
        rng = np.random.default_rng(7000 + epoch)
        grams = []
        for _ in range(8):
            matrix = rng.normal(size=(14, 7))
            grams.append(matrix @ matrix.T + np.eye(14) * 0.01)
        grams = np.asarray(grams)
        np.save(worker / f"gradient_gram_epoch_{epoch:03d}.npy", grams)
        sw = np.zeros((14, 14))
        rows = []
        mean_gram = grams.mean(axis=0)
        for i, j in PAIR_INDICES:
            wi, wj = INTERIOR_WIDTHS[i], INTERIOR_WIDTHS[j]
            representation_sw = abs(wi - wj) * (1 + epoch / 1000)
            sw[i, j] = sw[j, i] = representation_sw
            distance2 = mean_gram[i, i] + mean_gram[j, j] - 2 * mean_gram[i, j]
            rows.append({
                "state": f"F{epoch}", "seed": 7, "epoch": epoch,
                "width_i": wi, "width_j": wj,
                "representation_sw": representation_sw,
                "gradient_euclidean_distance": np.sqrt(max(distance2, 0.0)),
                "flops_i": flops[wi], "flops_j": flops[wj],
            })
        np.save(worker / f"sw_matrix_epoch_{epoch:03d}.npy", sw)
        pd.DataFrame(rows).to_csv(worker / f"pair_structure_epoch_{epoch:03d}.csv", index=False)
        (worker / "metadata.json").write_text(json.dumps({
            "status": "FRESH_PAIRWISE_STATE_COMPLETE", "seed": 7, "epoch": epoch,
            "gradient_probe_ids": list(range(1024)),
            "geometry_probe_ids": list(range(2000)),
        }))

    summary = merge_and_evaluate(root, workers, bootstrap_draws=100)
    assert summary["status"] == "FRESH_SEED7_PURE_SW_REPLICATION_COMPLETE"
    assert summary["accuracy_used_for_gate"] is False
    assert summary["fixed_uniform_marginals"] is True
    assert summary["predictor_fitted_or_used"] is False
    assert summary["gate_c_end_to_end_authorized"] is False
    results = pd.read_csv(root / "evaluation" / "fresh_seed7_puresw_replication.csv")
    policies = pd.read_csv(root / "evaluation" / "fresh_seed7_pair_policies.csv")
    bootstrap = pd.read_csv(root / "evaluation" / "fresh_seed7_bootstrap_contrasts.csv")
    assert len(results) == 3
    assert len(policies) == 3 * 4 * 91
    assert len(bootstrap) == 6
    assert np.all(results.V_oracle <= results[["V_uniform", "V_resource", "V_SW"]].min(axis=1) + 1e-8)
    assert (root / "evaluation" / "fresh_seed7_puresw_variance.png").is_file()
    assert (root / "evaluation" / "fresh_seed7_puresw_replication_summary.md").is_file()
