import json
import zipfile

import pandas as pd
import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rq2_anchor_placement import (
    GRID,
    UNIFORM_ANCHORS,
    find_uniform_100_root,
    finalize_rq2,
    select_geometry_anchors,
    select_hybrid_anchors,
    validate_rq2_config,
)
from s1_width import train_shared_reference


def _rq2_config():
    return yaml.safe_load(open("configs/kaggle_rq2_anchor.yaml"))


def _uniform_root(root):
    (root / "protocol").mkdir(parents=True)
    config = _rq2_config()
    config["compression"]["train_widths"] = list(UNIFORM_ANCHORS)
    config["training"]["epochs"] = 100
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    (root / "protocol" / "fixed_random_projection.pt").write_bytes(b"projection")
    geometry = []
    for representation in ("learned_projection", "backbone_padded", "fixed_random_projection"):
        for seed in (0, 1, 2):
            for start, end in zip(GRID[:-1], GRID[1:]):
                geometry.append({
                    "seed": seed, "representation": representation,
                    "budget_start": start, "budget_end": end,
                    "G": 5.0 if start in (0.35, 0.40) else 1.0,
                })
    pd.DataFrame(geometry).to_csv(root / "representation_local_geometry_all_seeds.csv", index=False)
    pd.DataFrame([
        {"seed": seed, "budget": width, "flops": int(1e8 * width * width), "accuracy": 0.5}
        for seed in (0, 1, 2) for width in GRID
    ]).to_csv(root / "shared_dense_metrics_all_seeds.csv", index=False)
    for seed in (0, 1, 2):
        path = root / "shared" / f"seed_{seed}" / "checkpoint.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"checkpoint {seed}".encode())
    return root


def test_geometry_selection_is_deterministic_locked_and_reports_compute(tmp_path):
    source = _uniform_root(tmp_path / "uniform")
    output = tmp_path / "selection"
    first = select_geometry_anchors(source, output)
    second = select_geometry_anchors(source, output)
    assert first == second
    assert first["selected_anchors"][0] == 0.25
    assert first["selected_anchors"][-1] == 1.0
    assert len(first["selected_anchors"]) == 4
    assert not set(first["common_holdout"]) & set(first["selected_anchors"])
    assert not set(first["common_holdout"]) & set(UNIFORM_ANCHORS)
    assert len(pd.read_csv(output / "geometry_anchor_selection.csv")) == 91
    compute = pd.read_csv(output / "anchor_training_compute.csv")
    assert set(compute["method"]) == {"Uniform-4", "Geometry-4"}


def test_find_uniform_root_direct_and_archive(tmp_path):
    source = _uniform_root(tmp_path / "direct" / "run")
    assert find_uniform_100_root(tmp_path / "direct", tmp_path / "unused") == source
    archive_root = tmp_path / "archive_input"
    archive_root.mkdir()
    archive = archive_root / "kaggle-s1-extension-100-test.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in source.rglob("*"):
            if path.is_file():
                bundle.write(path, path.relative_to(source))
    extracted = find_uniform_100_root(archive_root, tmp_path / "extracted")
    assert (extracted / "shared" / "seed_2" / "checkpoint.pt").is_file()


def test_hybrid_selector_enumerates_pareto_and_freezes_new_seeds(tmp_path):
    source = _uniform_root(tmp_path / "uniform")
    output = tmp_path / "hybrid"
    result = select_hybrid_anchors(source, output)
    candidates = pd.read_csv(output / "hybrid_anchor_candidates.csv")
    selected = candidates.loc[candidates["selected"]].iloc[0]
    assert len(candidates) == 91
    assert int(candidates["selected"].sum()) == 1
    assert bool(selected["pareto_optimal"])
    assert selected["hybrid_objective"] <= 1.0 + 1e-12
    assert result["development_seeds"] == [0, 1, 2]
    assert result["confirmatory_seeds"] == [3, 4, 5]
    assert result["accuracy_or_specialization_gap_used_for_selection"] is False
    gates = json.loads((output / "hybrid_v2_preregistered_gates.json").read_text())
    assert gates["run_all_confirmatory_seeds_regardless_of_seed_3_result"] is True
    assert gates["methods_required_per_seed"] == ["Uniform-4", "PureGeo-4", "Hybrid-4"]
    assert (output / "hybrid_anchor_pareto.png").stat().st_size > 0


def test_rq2_config_lock():
    config = _rq2_config()
    validate_rq2_config(config)
    config["experiment"]["confirmatory_seeds"] = [0, 1]
    with pytest.raises(ValueError, match="confirmatory"):
        validate_rq2_config(config)


class _TinySlimmable(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)
        self.width = 1.0

    def set_width(self, width):
        self.width = float(width)

    def forward(self, images):
        return self.linear(images.flatten(1)) * self.width


def test_shared_training_reads_dynamic_anchors(tmp_path):
    config = _rq2_config()
    config["compression"]["train_widths"] = [0.25, 0.4, 0.6, 1.0]
    config["training"]["epochs"] = 1
    images = torch.randn(8, 1, 2, 2)
    labels = torch.randint(0, 3, (8,))
    ids = torch.arange(8)
    loader = DataLoader(
        TensorDataset(images, labels, ids), batch_size=4, shuffle=True,
        generator=torch.Generator().manual_seed(1),
    )
    train_shared_reference(
        _TinySlimmable(), loader, config, torch.device("cpu"), tmp_path / "training"
    )
    metrics = pd.read_csv(tmp_path / "training" / "training_metrics.csv")
    assert set(metrics["width"].round(2)) == {0.25, 0.4, 0.6, 1.0}
    assert len(metrics) == 4


def test_finalize_rq2_writes_common_holdout_decision(tmp_path):
    config = _rq2_config()
    config["evaluation"]["bootstrap_replicates"] = 20
    root = tmp_path / "run"
    selection = {
        "selected_anchors": [0.25, 0.4, 0.6, 1.0],
        "common_holdout": sorted(set(GRID) - set(UNIFORM_ANCHORS) - {0.25, 0.4, 0.6, 1.0}),
        "subnet_compute_ratio_geo_to_uniform": 0.8,
    }
    for method, accuracy, correct_count in (("uniform", 0.6, 12), ("geo", 0.7, 14)):
        for seed in (1, 2):
            evaluation = root / "evaluation" / method / f"seed_{seed}"
            evaluation.mkdir(parents=True)
            pd.DataFrame([
                {
                    "seed": seed, "budget": width, "accuracy": accuracy,
                    "loss": 1.0, "flops": 1e8 * width * width,
                    "params": 1e6 * width, "is_train_anchor": False,
                }
                for width in GRID
            ]).to_csv(evaluation / "budget_metrics.csv", index=False)
            pd.DataFrame([
                {
                    "seed": seed, "representation": "learned_projection",
                    "budget_start": start, "budget_end": end,
                    "wasserstein_jump": 0.01, "G": 0.2 if method == "uniform" else 0.15,
                }
                for start, end in zip(GRID[:-1], GRID[1:])
            ]).to_csv(evaluation / "representation_local_geometry.csv", index=False)
            prediction_dir = root / "predictions" / method / f"seed_{seed}"
            prediction_dir.mkdir(parents=True)
            pd.DataFrame([
                {
                    "seed": seed, "budget": width, "sample_id": sample_id,
                    "label": 0, "prediction": 0 if sample_id < correct_count else 1,
                    "correct": int(sample_id < correct_count),
                }
                for width in GRID for sample_id in range(20)
            ]).to_csv(prediction_dir / "predictions_all_widths.csv", index=False)
    decision = finalize_rq2(config, root, selection)
    assert decision["rq2_screening_pass"] is True
    assert decision["mean_delta_accuracy_H"] == pytest.approx(0.1)
    assert (root / "rq2_seed_comparison.csv").is_file()
    assert (root / "rq2_dense_accuracy_curves.png").stat().st_size > 0
