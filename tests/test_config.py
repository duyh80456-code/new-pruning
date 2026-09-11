from pathlib import Path

import pytest
import yaml

from research_utils import load_config


def test_smoke_config_keeps_unseen_widths_out_of_training():
    config = load_config(Path("configs/smoke.yaml"))
    train = set(config["compression"]["train_widths"])
    unseen = set(config["compression"]["eval_widths"]) - train
    assert train == {0.25, 0.5, 0.75, 1.0}
    assert 0.3 in unseen and 0.8 in unseen


def test_non_anchor_training_config_is_rejected(tmp_path):
    config = yaml.safe_load(Path("configs/smoke.yaml").read_text())
    config["compression"]["train_widths"].append(0.3)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="Anchor-only"):
        load_config(path)
