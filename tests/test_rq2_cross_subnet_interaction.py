import json

import numpy as np
import pytest
import torch
from torch import nn

from rq2_anchor_placement import GRID
from rq2_cross_subnet_interaction import (
    find_uniform_seed3_checkpoint,
    interaction_parameters,
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
