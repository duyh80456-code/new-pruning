import json
import zipfile

import pytest
import torch
import yaml

from models import slimmable_resnet18
from s1_width import fixed_random_projection, read_s0_selection
from scripts.run_s1_width import validate_protocol
from scripts.run_s1_extension import materialize_source


def test_s0_selection_requires_pass_and_positive_horizon(tmp_path):
    valid = tmp_path / "s0_selection.json"
    valid.write_text(json.dumps({"s0_pass": True, "selected_horizon": 80}))
    assert read_s0_selection(valid)["selected_horizon"] == 80

    failed = tmp_path / "failed.json"
    failed.write_text(json.dumps({"s0_pass": False, "selected_horizon": 80}))
    with pytest.raises(RuntimeError, match="require s0_pass"):
        read_s0_selection(failed)

    authorized = tmp_path / "authorized.json"
    authorized.write_text(json.dumps({
        "s0_pass": False, "s1_user_authorized": True, "selected_horizon": 50
    }))
    result = read_s0_selection(authorized)
    assert result["selected_horizon"] == 50
    assert result["selection_basis"] == "user_fixed"

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


def test_extension_materializes_complete_exported_zip(tmp_path):
    source = tmp_path / "source"
    (source / "protocol").mkdir(parents=True)
    (source / "resolved_config.yaml").write_text("training:\n  epochs: 50\n")
    (source / "specialization_table.csv").write_text("seed,width\n")
    (source / "shared_dense_metrics_all_seeds.csv").write_text("seed,budget\n")
    (source / "s1_complete.json").write_text("{}")
    (source / "protocol" / "fixed_random_projection.pt").write_bytes(b"matrix")
    for seed in (0, 1, 2):
        path = source / "shared" / f"seed_{seed}" / "checkpoint.pt"
        path.parent.mkdir(parents=True); path.write_bytes(b"checkpoint")
        for width in (30, 40, 60, 80):
            path = source / "specialized" / f"seed_{seed}" / f"width_{width:03d}" / "best_checkpoint.pt"
            path.parent.mkdir(parents=True); path.write_bytes(b"checkpoint")
    input_root = tmp_path / "input"; input_root.mkdir()
    archive = input_root / "kaggle-s1-width-test.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in source.rglob("*"):
            if path.is_file(): bundle.write(path, path.relative_to(source))
    materialized = materialize_source(input_root, tmp_path / "materialized")
    assert (materialized / "shared" / "seed_2" / "checkpoint.pt").is_file()
