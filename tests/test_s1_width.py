import json

import pytest
import torch
import yaml

from models import slimmable_resnet18
from s1_width import fixed_random_projection, read_s0_selection
from scripts.run_s1_width import validate_protocol


def test_s0_selection_requires_pass_and_positive_horizon(tmp_path):
    valid = tmp_path / "s0_selection.json"
    valid.write_text(json.dumps({"s0_pass": True, "selected_horizon": 80}))
    assert read_s0_selection(valid)["selected_horizon"] == 80

    failed = tmp_path / "failed.json"
    failed.write_text(json.dumps({"s0_pass": False, "selected_horizon": 80}))
    with pytest.raises(RuntimeError, match="does not record"):
        read_s0_selection(failed)

    missing_horizon = tmp_path / "missing.json"
    missing_horizon.write_text(json.dumps({"s0_pass": True}))
    with pytest.raises(ValueError, match="selected_horizon"):
        read_s0_selection(missing_horizon)


def test_fixed_random_projection_is_frozen_and_column_normalized():
    first = fixed_random_projection(512, 128, 424242)
    second = fixed_random_projection(512, 128, 424242)
    assert torch.equal(first, second)
    assert first.shape == (512, 128)
    assert torch.allclose(torch.linalg.vector_norm(first, dim=0), torch.ones(128), atol=1e-6)


def test_backbone_preprojection_uses_nested_active_prefix():
    widths = [round(0.25 + 0.05 * index, 2) for index in range(16)]
    model = slimmable_resnet18(num_classes=10, supported_widths=widths, projection_dim=128)
    inputs = torch.randn(2, 3, 32, 32)
    model.set_width(0.25)
    assert model.forward_backbone_features(inputs).shape == (2, 128)
    model.set_width(1.0)
    assert model.forward_backbone_features(inputs).shape == (2, 512)
    assert model.forward_features(inputs).shape == (2, 128)


def test_s1_config_is_locked_and_has_no_default_horizon():
    config = yaml.safe_load(open("configs/kaggle_s1_width.yaml"))
    validate_protocol(config)
    assert config["training"]["epochs"] is None
    assert config["dataset"]["num_workers"] == 0
    assert config["specialization"]["initialization"] == "scratch"
