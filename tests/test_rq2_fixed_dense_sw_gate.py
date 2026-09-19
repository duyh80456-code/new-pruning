import json

import numpy as np

from rq2_fixed_dense_sw_gate import (
    KAPPAS, decorrelated_control, freeze_policy, load_frozen_policy,
    policy_summary, run_tau_probe, sha256, soft_sw_policy,
)
from rq2_pairwise_surrogate_regret import INTERIOR_WIDTHS, incidence_matrix


def _synthetic_sw():
    widths = np.asarray(INTERIOR_WIDTHS)
    values = np.sin(8 * widths) + widths
    return np.abs(values[:, None] - values[None, :])


def test_tau_regimes_and_fixed_marginals():
    sw = _synthetic_sw()
    summaries = [policy_summary(soft_sw_policy(sw, kappa), sw) for kappa in KAPPAS]
    assert summaries[0]["entropy"] < summaries[1]["entropy"] < summaries[2]["entropy"]
    assert summaries[0]["effective_support"] < summaries[2]["effective_support"]
    assert max(item["marginal_error"] for item in summaries) < 1e-6


def test_shuffled_control_matches_shape_exactly():
    sw = _synthetic_sw()
    q = soft_sw_policy(sw, 1.0)
    shuffled, permutation, diagnostics = decorrelated_control(q, sw)
    a, b = policy_summary(q, sw), policy_summary(shuffled, sw)
    assert sorted(permutation.tolist()) == list(range(14))
    assert np.max(np.abs(incidence_matrix() @ shuffled - 1 / 7)) < 1e-6
    assert np.allclose(np.sort(q), np.sort(shuffled))
    for key in ("entropy", "effective_support", "support_size", "max_probability"):
        assert np.isclose(a[key], b[key])
    assert abs(diagnostics["spearman_sw2_vs_control_q"]) < 0.2


def test_probe_then_freeze_binds_common_checkpoint_and_uniform(tmp_path):
    root = tmp_path / "source"
    probe = tmp_path / "probe"
    (root / "common_warmup").mkdir(parents=True)
    (root / "pure_sw/sw_policies").mkdir(parents=True)
    (root / "uniform/checkpoints").mkdir(parents=True)
    (root / "common_warmup/epoch_010.pt").write_bytes(b"common-e10")
    (root / "uniform/checkpoints/epoch_100.pt").write_bytes(b"uniform-e100")
    (root / "pure_sw/training_provenance.json").write_text(json.dumps({
        "common_epoch10_sha256": sha256(root / "common_warmup/epoch_010.pt")
    }))
    (root / "uniform/training_provenance.json").write_text(json.dumps({
        "common_epoch10_sha256": sha256(root / "common_warmup/epoch_010.pt")
    }))
    np.savez(root / "pure_sw/sw_policies/epoch_010.npz",
             sw_matrix=_synthetic_sw(), widths=np.asarray(INTERIOR_WIDTHS))
    table = run_tau_probe(root, probe)
    assert list(table.kappa) == list(KAPPAS)
    assert table.entropy.is_monotonic_increasing
    frozen = freeze_policy(root, probe, 1.0)
    assert frozen == load_frozen_policy(root)
    assert frozen["accuracy_used_to_choose_kappa"] is False
