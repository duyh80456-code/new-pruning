import json

import numpy as np
import pandas as pd
import yaml

from rq2_anchor_placement import GRID
from rq2_geometry_exposure_decomposition import (
    build_per_width_table,
    fit_decomposition,
    run_geometry_exposure_decomposition,
)


def _rq2_root(root):
    protocol = root / "protocol"
    protocol.mkdir(parents=True)
    config = yaml.safe_load(open("configs/kaggle_rq2_anchor.yaml"))
    config["compression"]["train_widths"] = [0.25, 0.40, 0.60, 1.00]
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    edges = [0.03, 0.04, 0.10, 0.12, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
    coordinate = np.concatenate([[0.0], np.cumsum(edges)])
    pd.DataFrame({
        "width": GRID,
        "geometry_coordinate": coordinate,
        "incoming_edge_length": [np.nan, *edges],
    }).to_csv(protocol / "geometry_trajectory_coordinates.csv", index=False)
    (protocol / "selected_anchors.json").write_text(json.dumps({
        "selected_anchors": [0.25, 0.40, 0.60, 1.00]
    }))
    rows = []
    for seed in (1, 2):
        for width in GRID:
            base = 0.55 + 0.10 * width + 0.001 * seed
            delta = 0.008 if width in (0.30, 0.35, 0.45) else -0.004 * width
            for method, accuracy in (("uniform", base), ("geo", base + delta)):
                rows.append({
                    "method": method, "seed": seed, "budget": width,
                    "accuracy": accuracy, "flops": 1e8 * width * width,
                    "params": 1e6 * width,
                })
    pd.DataFrame(rows).to_csv(root / "rq2_dense_metrics_all.csv", index=False)
    return root


def test_per_width_decomposition_uses_only_active_parameters(tmp_path):
    table, detail, _, anchors, holdout = build_per_width_table(_rq2_root(tmp_path / "rq2"))
    unique = table.drop_duplicates("width").set_index("width")
    assert anchors == (0.25, 0.40, 0.60, 1.00)
    assert len(holdout) == 10
    assert unique.loc[0.25, "D_E"] == 0.0
    assert unique.loc[0.45, "D_E"] > 0.0
    assert unique.loc[0.65, "lost_exposure_updates"] > unique.loc[0.45, "lost_exposure_updates"]
    assert len(table) == 2 * len(GRID)
    assert set(detail["parameter_group"]) == {
        "stem", "layer1", "layer2", "layer3", "layer4", "projection", "classifier"
    }


def test_two_factor_fit_recovers_requested_coefficient_signs():
    rows = []
    for seed in (1, 2):
        for index, width in enumerate(np.linspace(0.3, 0.9, 10)):
            geometry = (index % 4) * 0.03 + 0.01 * index
            exposure = (index % 3) * 0.05 + 0.015 * index
            rows.append({
                "seed": seed, "width": width, "is_common_holdout": True,
                "B_G": geometry, "D_E": exposure,
                "delta_accuracy": 0.01 + 1.5 * geometry - 0.8 * exposure,
            })
    regression, _ = fit_decomposition(pd.DataFrame(rows))
    two = regression.loc[regression["model"].eq("B_G+D_E")]
    assert (two["beta_G"] > 0).all()
    assert (two["beta_E"] < 0).all()
    assert np.allclose(two["in_sample_r2"], 1.0)


def test_complete_cpu_decomposition_exports_requested_files(tmp_path):
    output = tmp_path / "output"
    result = run_geometry_exposure_decomposition(_rq2_root(tmp_path / "rq2"), output)
    required = [
        "per_width_geometry_exposure_accuracy.csv",
        "decomposition_regression.csv",
        "exposure_by_width_and_band.csv",
        "geometry_gain_vs_accuracy.png",
        "exposure_deficit_vs_accuracy.png",
        "two_factor_fit.png",
    ]
    assert all((output / name).is_file() and (output / name).stat().st_size > 0 for name in required)
    assert result["training_or_checkpoint_forward_used"] is False
    assert len(pd.read_csv(output / "decomposition_regression.csv")) == 6

