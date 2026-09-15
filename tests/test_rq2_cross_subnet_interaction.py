import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from rq2_anchor_placement import GRID
from rq2_cross_subnet_interaction import (
    find_uniform_seed3_checkpoint,
    interaction_parameters,
    merge_cross_subnet_interaction_workers,
    policy_expected_benefits,
)
from rq2_probabilistic_support import INTERIOR_WIDTHS


def test_policy_expected_benefits_match_explicit_weighted_gradient_sum():
    rng = np.random.default_rng(17)
    gradients = rng.normal(size=(len(GRID), 9))
    dot = gradients @ gradients.T
    pi_geometry = np.linspace(0.05, 0.30, len(INTERIOR_WIDTHS))
    pi_geometry *= 2.0 / pi_geometry.sum()
    pi_resource = np.full(len(INTERIOR_WIDTHS), 2.0 / len(INTERIOR_WIDTHS))

    geometry, resource, delta = policy_expected_benefits(
        dot, pi_geometry, pi_resource
    )
    geometry_bar = gradients[0] + gradients[-1] + pi_geometry @ gradients[1:-1]
    resource_bar = gradients[0] + gradients[-1] + pi_resource @ gradients[1:-1]

    np.testing.assert_allclose(geometry, gradients @ geometry_bar)
    np.testing.assert_allclose(resource, gradients @ resource_bar)
    np.testing.assert_allclose(delta, geometry - resource)


def test_policy_expected_benefits_reject_wrong_shapes():
    with pytest.raises(ValueError):
        policy_expected_benefits(np.eye(3), np.ones(14), np.ones(14))


def test_checkpoint_finder_requires_uniform_seed3_path(tmp_path):
    checkpoint = tmp_path / "run" / "uniform" / "seed_3" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    assert find_uniform_seed3_checkpoint(tmp_path, tmp_path / "materialized") == checkpoint


def test_interaction_parameter_scope_excludes_bn_and_biases():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3)
            self.bn = nn.BatchNorm2d(4)
            self.head = nn.Linear(4, 2)

    selected = interaction_parameters(Tiny())
    names = [name for name, _ in selected]
    assert names == ["head.weight"]
    assert all("bn" not in name and not name.endswith("bias") for name in names)


def test_metadata_json_booleans_are_native(tmp_path):
    payload = {"training_performed": False, "weights_unchanged": True}
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(payload))
    assert json.loads(path.read_text()) == payload


def test_two_worker_merge_covers_disjoint_batches(tmp_path):
    worker_dirs = []
    for worker, offset in enumerate((0, 1)):
        root = tmp_path / f"worker_{worker}"
        root.mkdir()
        worker_dirs.append(root)
        metadata = {
            "training_performed": False, "optimizer_steps": 0, "test_used": False,
            "weights_unchanged": True, "bn_buffers_unchanged_during_probe": True,
            "checkpoint_sha256": "checkpoint", "geometry_marginals_sha256": "geo",
            "resource_marginals_sha256": "resource", "global_batch_indices": [offset],
            "num_fixed_training_batches": 1, "checkpoint_role": "common",
            "checkpoint": "checkpoint.pt", "loss": {"full_width": "CE"},
            "gradient_scope": "weights", "gradient_parameter_tensors": 2,
            "gradient_parameter_scalars": 10,
        }
        (root / "metadata.json").write_text(json.dumps(metadata))
        matrix = pd.DataFrame(np.eye(len(GRID)) * (worker + 1))
        matrix.columns = [f"{width:.2f}" for width in GRID]
        matrix.insert(0, "width", GRID)
        matrix.to_csv(root / "gradient_dot_matrix.csv", index=False)
        matrix.to_csv(root / "gradient_cosine_matrix.csv", index=False)
        pd.DataFrame({
            "batch": offset, "width": GRID, "gradient_norm": worker + 1.0,
        }).to_csv(root / "gradient_norms_by_batch.csv", index=False)
        pd.DataFrame({
            "batch": offset, "width": GRID,
            "benefit_geometry": worker + np.arange(len(GRID)),
            "benefit_resource": np.arange(len(GRID)),
            "delta_benefit": worker,
        }).to_csv(root / "policy_expected_effect_by_batch.csv", index=False)
        pd.DataFrame({"order": range(2), "sample_id": [2 * worker, 2 * worker + 1]}).to_csv(
            root / "fixed_training_subset_ids.csv", index=False
        )
        np.save(
            root / "gradient_gram_matrices.npy",
            np.eye(len(INTERIOR_WIDTHS), dtype=np.float64)[None] * (worker + 1),
        )
        np.save(root / "gradient_gram_batch_ids.npy", np.asarray([offset], dtype=np.int64))
    pd.DataFrame({
        "source_width": [0.4], "target_width": [0.25], "epsilon": [1e-5],
        "predicted_loss_change": [-1e-4], "observed_loss_change": [-9e-5],
        "sign_match": [True],
    }).to_csv(worker_dirs[0] / "one_step_transfer.csv", index=False)

    output = tmp_path / "merged"
    result = merge_cross_subnet_interaction_workers(
        worker_dirs, output, expected_batches=2
    )
    merged_dot = pd.read_csv(output / "gradient_dot_matrix.csv").drop(columns="width")
    np.testing.assert_allclose(np.diag(merged_dot), 1.5)
    assert result["gpu_workers"] == 2
    assert result["num_fixed_training_batches"] == 2
    assert result["weights_unchanged"]
    assert np.load(output / "gram_matrices.npy").shape == (2, 14, 14)
