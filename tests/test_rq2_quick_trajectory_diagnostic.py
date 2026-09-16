import json

import numpy as np
import pandas as pd

from rq2_quick_trajectory_diagnostic import (
    EPOCHS,
    INTERIOR_WIDTHS,
    _gram_checks,
    find_ht_development_root,
    merge_trajectory_paths,
)


def test_gram_checks_accept_psd_batch_tensor():
    rng = np.random.default_rng(9)
    vectors = rng.normal(size=(8, len(INTERIOR_WIDTHS), 17))
    grams = vectors @ vectors.transpose(0, 2, 1)
    result = _gram_checks(grams)
    assert result["max_relative_symmetry_error"] < 1e-12
    assert result["minimum_relative_eigenvalue"] > -1e-10


def test_find_complete_ht_development_root(tmp_path):
    root = tmp_path / "ht"
    (root / "protocol").mkdir(parents=True)
    (root / "ht_development_decision.json").write_text(json.dumps({
        "status": "RQ2_V3_HT_SEED3_DEVELOPMENT_COMPLETE"
    }))
    (root / "resolved_config.yaml").write_text("experiment: {}\n")
    (root / "protocol" / "frozen_dynamic_marginals.csv").write_text("width\n")
    for method in ("geo_ht", "resource_ht"):
        directory = root / method / "seed_3"
        directory.mkdir(parents=True)
        for epoch in EPOCHS:
            (directory / f"epoch_{epoch:03d}.pt").write_bytes(b"checkpoint")
    assert find_ht_development_root(tmp_path, tmp_path / "materialized") == root


def _worker_fixture(root, path, offset):
    root.mkdir(parents=True)
    ids = list(range(1024))
    metadata = {
        "status": "QUICK_TRAJECTORY_PATH_COMPLETE", "path": path,
        "probe_batch_ids": ids, "training_performed": False,
        "optimizer_steps": 0, "test_used": False,
        "weights_unchanged": True, "bn_buffers_unchanged": True,
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    trajectory, total, distance, drift = [], [], [], []
    for epoch in EPOCHS:
        trajectory.append({
            "path": path, "epoch": epoch, "V_geo": 1.0 + offset,
            "V_resource": 2.0 + offset, "delta_geo_resource": -1.0,
            "ratio_geo_resource": (1.0 + offset) / (2.0 + offset),
        })
        total.append({
            "path": path, "epoch": epoch, "V_data": 8.0,
            "V_subnet_geo": 1.0, "V_subnet_resource": 2.0,
            "V_total_geo": 9.0, "V_total_resource": 10.0,
            "subnet_fraction_geo": 1 / 9, "subnet_fraction_resource": 0.2,
            "total_relative_gain_geo": 0.1,
        })
        distance.append({
            "path": path, "epoch": epoch,
            "tv_geo_oracle": 0.2, "tv_resource_oracle": 0.3,
        })
        for width in INTERIOR_WIDTHS:
            drift.append({
                "path": path, "epoch": epoch, "width": width,
                "pi_geo": 1 / 7, "pi_resource": 1 / 7,
                "pi_oracle": 1 / 7, "gradient_second_moment": 1.0,
            })
    pd.DataFrame(trajectory).to_csv(root / "quick_trajectory_variance.csv", index=False)
    pd.DataFrame(total).to_csv(root / "quick_total_variance.csv", index=False)
    pd.DataFrame(distance).to_csv(root / "quick_policy_distance.csv", index=False)
    pd.DataFrame(drift).to_csv(root / "quick_oracle_drift.csv", index=False)


def test_merge_two_paths_writes_all_primary_outputs(tmp_path):
    geo, resource = tmp_path / "geo", tmp_path / "resource"
    _worker_fixture(geo, "geo_ht", 0.0)
    _worker_fixture(resource, "resource_ht", 0.2)
    output = tmp_path / "merged"
    result = merge_trajectory_paths([geo, resource], output)
    assert result["status"] == "RQ2_V3_QUICK_TRAJECTORY_DIAGNOSTIC_COMPLETE"
    assert result["geo_lower_conditional_variance_count"] == 6
    assert result["same_fixed_batch_ids_and_order"]
    for name in (
        "quick_trajectory_variance.csv", "quick_total_variance.csv",
        "quick_oracle_drift.csv", "quick_policy_distance.csv",
        "quick_trajectory_variance.png", "quick_total_variance.png",
        "quick_oracle_drift.png", "metadata.json",
    ):
        assert (output / name).is_file()
