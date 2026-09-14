import itertools
import json

import pandas as pd

from data import build_interim_validation_loaders
from finalgeo_confirmatory import (
    METHOD_ANCHORS,
    finalize_confirmatory,
    method_config,
    validate_confirmatory_base_config,
    validate_frozen_protocol,
)
from rq2_anchor_placement import GRID, _sha256
from rq2_finalgeo_selector import _canonical_hash


def _frozen(root, freeze):
    protocol_dir = root / "protocol"
    protocol_dir.mkdir(parents=True)
    sources = {
        "rq2_geometry_all.csv": root / "rq2_geometry_all.csv",
        "geometry_trajectory_coordinates.csv": protocol_dir / "geometry_trajectory_coordinates.csv",
        "rq2_dense_metrics_all.csv": root / "rq2_dense_metrics_all.csv",
        "selected_anchors.json": protocol_dir / "selected_anchors.json",
    }
    for index, path in enumerate(sources.values()):
        path.write_text(f"source {index}")
    hashes = {name: _sha256(path) for name, path in sources.items()}
    common = sorted(set(GRID) - {0.25, 0.40, 0.50, 0.60, 0.70, 0.75, 1.00})
    core = {
        "status": "FROZEN_BEFORE_CONFIRMATORY_SEEDS",
        "K": 4, "development_seeds": [0, 1, 2], "seeds_confirmatory": [3, 4, 5],
        "uniform_anchors": [0.25, 0.50, 0.75, 1.00],
        "puregeo_anchors": [0.25, 0.40, 0.60, 1.00],
        "finalgeo_anchors": [0.25, 0.40, 0.70, 1.00],
        "dense_eval_grid": list(GRID),
        "selection_rule": {
            "objective": "minimize_R_G", "objective_is_geometry_only": True,
            "constraints": {
                "minimum_compute_ratio_vs_uniform": 0.90,
                "maximum_compute_ratio_vs_uniform": 1.00,
            },
            "accuracy_used": False, "parameter_exposure_used": False,
            "gradient_interference_used": False,
        },
        "compute_constraint": {"selected_ratio_vs_uniform": 0.908},
        "geometry_source": {"source_artifact_sha256": hashes},
        "common_holdout": common,
        "candidate_grid": list(GRID), "fixed_endpoints": [0.25, 1.0],
        "candidate_count": 91,
        "methods_confirmatory": ["Uniform", "PureGeo", "FinalGeo"],
        "success_gates": {
            "minimum_pooled_mean_common_holdout_effect": 0.002,
            "worst_common_holdout_accuracy_margin_vs_uniform": -0.005,
            "full_width_accuracy_margin_vs_uniform": -0.005,
            "high_region_accuracy_margin_vs_uniform": -0.002,
            "minimum_low_region_puregeo_gain_retention": 0.60,
        },
    }
    frozen_id = _canonical_hash(core)
    protocol = {**core, "freeze_id_sha256": frozen_id}
    freeze.mkdir()
    (freeze / "finalgeo_frozen_protocol.json").write_text(json.dumps(protocol))
    (freeze / "finalgeo_selected_anchors.json").write_text(json.dumps({
        "freeze_id_sha256": frozen_id, "finalgeo_anchors": [0.25, 0.40, 0.70, 1.00]
    }))
    rows = []
    for a1, a2 in itertools.combinations(GRID[1:-1], 2):
        rows.append({
            "a1": a1, "a2": a2, "anchors": f"0.25,{a1:.2f},{a2:.2f},1.00",
            "selected": a1 == 0.40 and a2 == 0.70,
        })
    pd.DataFrame(rows).to_csv(freeze / "finalgeo_selector_candidates.csv", index=False)
    return protocol


def test_validate_frozen_protocol_and_method_configs(tmp_path):
    root, freeze = tmp_path / "rq2", tmp_path / "freeze"
    protocol = _frozen(root, freeze)
    assert validate_frozen_protocol(root, freeze, verify_git=False) == protocol
    base = {"compression": {"train_widths": []}}
    for method, anchors in METHOD_ANCHORS.items():
        assert method_config(base, method)["compression"]["train_widths"] == list(anchors)


def test_validate_confirmatory_base_config():
    config = {
        "dataset": {"name": "cifar100", "num_workers": 0},
        "model": {"backbone": "slimmable_resnet18"},
        "compression": {"eval_widths": list(GRID)},
        "training": {
            "phase_1_epochs": 50, "total_epochs": 100,
            "anchors_per_batch": 4, "extension_learning_rate": 0.01,
        },
    }
    validate_confirmatory_base_config(config)


def test_interim_validation_loader_never_constructs_test_split(monkeypatch, tmp_path):
    calls = []

    class FakeCifar:
        def __init__(self, root, train, transform, download):
            calls.append(bool(train))
            self.targets = list(range(100)) * 2

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            raise AssertionError("Loader construction should not fetch an image")

    monkeypatch.setattr("data.datasets.CIFAR100", FakeCifar)
    config = {
        "dataset": {
            "name": "cifar100", "root": str(tmp_path), "download": False,
            "split_seed": 7, "validation_size": 100, "bn_calibration_size": 20,
            "feature_subset_size": 20, "num_workers": 0,
        },
        "training": {"batch_size": 8},
        "evaluation": {"batch_size": 16},
    }
    loaders = build_interim_validation_loaders(config)
    assert calls == [True, True]
    assert len(loaders.validation.dataset) == 100


def test_finalize_applies_frozen_success_gates(tmp_path):
    root, freeze = tmp_path / "rq2", tmp_path / "freeze"
    protocol = _frozen(root, freeze)
    output = tmp_path / "run"
    for method in METHOD_ANCHORS:
        for seed in (3, 4, 5):
            evaluation = output / "evaluation" / method / f"seed_{seed}"
            evaluation.mkdir(parents=True)
            metrics = []
            for width in GRID:
                accuracy = 0.65
                if method == "puregeo" and 0.30 <= width <= 0.45:
                    accuracy += 0.020
                if method == "finalgeo":
                    accuracy += 0.005
                    if 0.30 <= width <= 0.45:
                        accuracy += 0.010
                metrics.append({"seed": seed, "budget": width, "accuracy": accuracy})
            pd.DataFrame(metrics).to_csv(evaluation / "budget_metrics.csv", index=False)
            pd.DataFrame([
                {
                    "seed": seed, "representation": "learned_projection",
                    "budget_start": start, "budget_end": end, "G": 0.1,
                }
                for start, end in zip(GRID[:-1], GRID[1:])
            ]).to_csv(evaluation / "representation_local_geometry.csv", index=False)
    decision = finalize_confirmatory({}, output, protocol)
    assert decision["verdict"] == "FINALGEO CONFIRMATORY PASS"
    assert decision["all_preregistered_gates_pass"] is True
    assert len(pd.read_csv(output / "finalgeo_method_summary.csv")) == 9
