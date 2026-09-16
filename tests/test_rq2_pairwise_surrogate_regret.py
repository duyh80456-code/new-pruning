import json

import numpy as np
import pandas as pd

from rq2_pairwise_surrogate_regret import (
    EPOCHS,
    INTERIOR_WIDTHS,
    PAIR_INDICES,
    UNIFORM_PI,
    gram_pair_scores,
    minimax_positive_scale,
    pair_table,
    run_pairwise_surrogate_regret,
    solve_pair_lp,
)
from rq2_probabilistic_support import pair_marginals


def test_pair_lp_respects_uniform_marginals_and_improves_score():
    scores = np.linspace(-1.0, 2.0, len(PAIR_INDICES))
    q = solve_pair_lp(scores)
    uniform = np.full(len(PAIR_INDICES), 1 / len(PAIR_INDICES))
    np.testing.assert_allclose(pair_marginals(pair_table(q)), UNIFORM_PI, atol=1e-9)
    assert scores @ q >= scores @ uniform


def test_theorem1_distance_lp_matches_direct_second_moment_lp():
    rng = np.random.default_rng(18)
    vectors = rng.normal(size=(len(INTERIOR_WIDTHS), 11))
    gram = vectors @ vectors.T
    distance, second = gram_pair_scores(gram)
    q_distance = solve_pair_lp(distance, maximize=True)
    q_second = solve_pair_lp(second, maximize=False)
    # The two LPs can choose different points in a degenerate optimum face, but
    # each must attain the other's optimal objective under fixed marginals.
    np.testing.assert_allclose(distance @ q_distance, distance @ q_second, rtol=1e-8)
    np.testing.assert_allclose(second @ q_distance, second @ q_second, rtol=1e-8)


def test_minimax_scale_recovers_exact_positive_proportionality():
    sw_sq = np.linspace(0.01, 2.0, len(PAIR_INDICES))
    alpha, epsilon = minimax_positive_scale(3.25 * sw_sq, sw_sq)
    np.testing.assert_allclose(alpha, 3.25, rtol=1e-8)
    assert epsilon < 1e-8


def test_full_six_state_regret_probe_writes_outputs(tmp_path):
    root = tmp_path / "quick"
    (root / "worker_paths").mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps({
        "status": "RQ2_V3_QUICK_TRAJECTORY_DIAGNOSTIC_COMPLETE"
    }))
    widths = np.asarray(INTERIOR_WIDTHS)
    # One-dimensional gradients whose exact distances are proportional to SW.
    gram = np.outer(widths, widths)
    pair_rows, geometry_rows = [], []
    for path in ("geo_ht", "resource_ht"):
        worker = root / "worker_paths" / path
        worker.mkdir(parents=True)
        for epoch in EPOCHS:
            np.save(worker / f"interior_grams_epoch_{epoch:03d}.npy", np.repeat(
                gram[None], 8, axis=0
            ))
            for i, j in PAIR_INDICES:
                pair_rows.append({
                    "path": path, "epoch": epoch,
                    "width_i": widths[i], "width_j": widths[j],
                    "representation_sw": abs(widths[i] - widths[j]),
                })
            for width in widths:
                geometry_rows.append({
                    "path": path, "epoch": epoch, "width": width,
                    "flops": float(1e6 * width * width),
                })
    pd.DataFrame(pair_rows).to_csv(root / "quick_pair_structure.csv", index=False)
    pd.DataFrame(geometry_rows).to_csv(root / "quick_dynamic_geometry_by_width.csv", index=False)
    output = tmp_path / "output"
    result = run_pairwise_surrogate_regret(root, output, shuffle_draws=100)
    assert result["theorem1_all_pass"] and result["theorem2_all_pass"]
    table = pd.read_csv(output / "pairwise_surrogate_regret.csv")
    assert len(table) == 6
    np.testing.assert_allclose(table.oracle_gap_captured, 1.0, atol=1e-7)
    for name in (
        "pairwise_theorem_checks.csv", "pairwise_control_variances.csv",
        "pairwise_shuffled_sw_controls.csv", "pairwise_policy_assignments.csv",
        "pairwise_oracle_gap_captured.png", "pairwise_theorem_bound.png",
        "pairwise_control_variances.png", "metadata.json",
    ):
        assert (output / name).is_file()
