import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from rq2_quick_trajectory_diagnostic import (
    EPOCHS,
    INTERIOR_WIDTHS,
    PROTOCOL_VERSION,
    _gram_checks,
    _sw_from_projected,
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


def test_projected_sw_is_mean_absolute_sorted_difference():
    left = np.asarray([[0.0, 2.0], [1.0, 4.0], [3.0, 8.0]])
    right = np.asarray([[1.0, 1.0], [2.0, 5.0], [5.0, 9.0]])
    assert _sw_from_projected(left, right) == np.abs(left - right).mean()


def test_find_complete_ht_development_root(tmp_path):
    root = tmp_path / "ht"
    (root / "protocol").mkdir(parents=True)
    (root / "ht_development_decision.json").write_text(json.dumps({
        "status": "RQ2_V3_HT_SEED3_DEVELOPMENT_COMPLETE"
    }))
    (root / "resolved_config.yaml").write_text("experiment: {}\n")
    for name in (
        "frozen_dynamic_marginals.csv", "geometry_pair_distribution.csv",
        "resource_pair_distribution.csv",
    ):
        (root / "protocol" / name).write_text("width\n")
    for method in ("geo_ht", "resource_ht"):
        directory = root / method / "seed_3"
        directory.mkdir(parents=True)
        for epoch in EPOCHS:
            (directory / f"epoch_{epoch:03d}.pt").write_bytes(b"checkpoint")
    assert find_ht_development_root(tmp_path, tmp_path / "materialized") == root


def test_finder_accepts_renamed_zip_and_missing_finalizer_decision(tmp_path):
    source = tmp_path / "source"
    (source / "protocol").mkdir(parents=True)
    (source / "resolved_config.yaml").write_text("experiment: {}\n")
    for name in (
        "frozen_dynamic_marginals.csv", "geometry_pair_distribution.csv",
        "resource_pair_distribution.csv",
    ):
        (source / "protocol" / name).write_text("width\n")
    for method in ("geo_ht", "resource_ht"):
        directory = source / method / "seed_3"
        directory.mkdir(parents=True)
        for epoch in EPOCHS:
            (directory / f"epoch_{epoch:03d}.pt").write_bytes(b"checkpoint")
    attached = tmp_path / "attached"
    attached.mkdir()
    archive = attached / "notebook-output-renamed.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in source.rglob("*"):
            if path.is_file():
                bundle.write(path, Path("nested-export") / path.relative_to(source))
    result = find_ht_development_root(attached, tmp_path / "materialized")
    assert result.name == "nested-export"
    assert _is_complete_ht_root_for_test(result)


def _is_complete_ht_root_for_test(root):
    return all(
        (root / method / "seed_3" / f"epoch_{epoch:03d}.pt").is_file()
        for method in ("geo_ht", "resource_ht") for epoch in EPOCHS
    )


def _worker_fixture(root, path, offset):
    root.mkdir(parents=True)
    ids = list(range(1024))
    metadata = {
        "status": "QUICK_TRAJECTORY_PATH_COMPLETE", "path": path,
        "protocol_version": PROTOCOL_VERSION,
        "probe_batch_ids": ids, "geometry_subset_ids": list(range(2000)),
        "training_performed": False,
        "optimizer_steps": 0, "test_used": False,
        "weights_unchanged": True, "bn_buffers_unchanged": True,
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    trajectory, total, distance, drift = [], [], [], []
    geometry, edges, pairs, correlations, pair_correlations = [], [], [], [], []
    for epoch in EPOCHS:
        trajectory.append({
            "path": path, "epoch": epoch, "V_geo": 1.0 + offset,
            "V_resource": 2.0 + offset, "delta_geo_resource": -1.0,
            "V_current_geometry": 0.9, "V_gradient_oracle_marginal": 0.8,
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
                "pi_current_geometry": 1 / 7,
                "pi_oracle": 1 / 7, "gradient_second_moment": 1.0,
            })
            geometry.append({
                "path": path, "epoch": epoch, "width": width,
                "current_functional_mass": 1.0,
                "current_functional_mass_normalized": 1 / 14,
                "frozen_functional_mass_normalized": 1 / 14,
                "gradient_second_moment": 1.0 + width,
                "pi_frozen_geo": 1 / 7, "pi_resource": 1 / 7,
                "pi_current_geometry": 1 / 7, "pi_oracle": 1 / 7,
            })
        for left, right in zip((0.25, *INTERIOR_WIDTHS), (*INTERIOR_WIDTHS, 1.0)):
            edges.append({
                "path": path, "epoch": epoch, "budget_start": left,
                "budget_end": right, "wasserstein_jump": 0.1, "G": 2.0,
            })
        for i, left in enumerate(INTERIOR_WIDTHS):
            for right in INTERIOR_WIDTHS[i + 1:]:
                pairs.append({
                    "path": path, "epoch": epoch, "width_i": left, "width_j": right,
                    "representation_sw": right - left, "gradient_dot": 1.0,
                    "gradient_euclidean_distance": right - left,
                    "gradient_cosine_dissimilarity": right - left,
                })
        for analysis, coefficient in (("current_a_vs_m", 0.8), ("frozen_a_vs_m", 0.6)):
            correlations.append({
                "path": path, "epoch": epoch, "analysis": analysis,
                "metric": "Spearman", "coefficient": coefficient,
                "pvalue": 0.01, "n": 14,
            })
        for analysis in (
            "representation_SW_vs_gradient_euclidean_distance",
            "representation_SW_vs_gradient_cosine_dissimilarity",
        ):
            pair_correlations.append({
                "path": path, "epoch": epoch, "analysis": analysis,
                "metric": "Spearman", "coefficient": 0.9,
                "pvalue": 0.01, "n_pairs": 91,
            })
    pd.DataFrame(trajectory).to_csv(root / "quick_trajectory_variance.csv", index=False)
    pd.DataFrame(total).to_csv(root / "quick_total_variance.csv", index=False)
    pd.DataFrame(distance).to_csv(root / "quick_policy_distance.csv", index=False)
    pd.DataFrame(drift).to_csv(root / "quick_oracle_drift.csv", index=False)
    pd.DataFrame(geometry).to_csv(root / "quick_dynamic_geometry_by_width.csv", index=False)
    pd.DataFrame(edges).to_csv(root / "quick_dynamic_geometry_edges.csv", index=False)
    pd.DataFrame(correlations).to_csv(root / "quick_geometry_gradient_correlations.csv", index=False)
    pd.DataFrame(pairs).to_csv(root / "quick_pair_structure.csv", index=False)
    pd.DataFrame(pair_correlations).to_csv(root / "quick_pair_structure_correlations.csv", index=False)


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
        "quick_dynamic_geometry_by_width.csv", "quick_dynamic_geometry_edges.csv",
        "quick_geometry_gradient_correlations.csv", "quick_pair_structure.csv",
        "quick_pair_structure_correlations.csv",
        "quick_trajectory_variance.png", "quick_total_variance.png",
        "quick_oracle_drift.png", "metadata.json",
        "quick_current_geometry_vs_m.png", "quick_geometry_gradient_tracking.png",
        "quick_pair_structure_tracking.png",
    ):
        assert (output / name).is_file()
