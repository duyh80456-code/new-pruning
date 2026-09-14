import json

import numpy as np
import pandas as pd
import pytest

from rq2_anchor_placement import GRID
from rq2_finalgeo_selector import (
    EXPECTED_PUREGEO,
    enumerate_finalgeo_candidates,
    freeze_finalgeo_selector,
)


def _development_root(root):
    protocol = root / "protocol"
    protocol.mkdir(parents=True)
    # A low/mid trajectory cliff makes the geometry optimum cheaper than Uniform.
    edge_lengths = np.array([
        0.03, 0.04, 0.11, 0.13, 0.05, 0.04, 0.03, 0.025,
        0.022, 0.020, 0.018, 0.017, 0.016, 0.015, 0.014,
    ])
    coordinates = np.concatenate([[0.0], np.cumsum(edge_lengths)])
    pd.DataFrame({
        "width": GRID,
        "geometry_coordinate": coordinates,
        "incoming_edge_length": [np.nan, *edge_lengths],
    }).to_csv(protocol / "geometry_trajectory_coordinates.csv", index=False)
    geometry = []
    for seed, scale in ((1, 0.98), (2, 1.02)):
        for index, (start, end) in enumerate(zip(GRID[:-1], GRID[1:])):
            geometry.append({
                "method": "uniform", "seed": seed,
                "representation": "learned_projection",
                "budget_start": start, "budget_end": end,
                "G": edge_lengths[index] * scale / 0.05,
            })
    pd.DataFrame(geometry).to_csv(root / "rq2_geometry_all.csv", index=False)
    pd.DataFrame([
        {
            "method": "uniform", "seed": seed, "budget": width,
            "flops": 1e8 * width * width,
            # Accuracy must not be read or used by the selector.
            "accuracy": 0.1 + seed * width,
        }
        for seed in (1, 2) for width in GRID
    ]).to_csv(root / "rq2_dense_metrics_all.csv", index=False)
    (protocol / "selected_anchors.json").write_text(json.dumps({
        "selected_anchors": list(EXPECTED_PUREGEO)
    }))
    return root


def test_finalgeo_enumerates_and_applies_compute_as_constraint_only(tmp_path):
    candidates, metadata = enumerate_finalgeo_candidates(_development_root(tmp_path / "rq2"))
    selected = candidates.loc[candidates["selected"]].iloc[0]
    feasible = candidates.loc[candidates["compute_feasible"]]
    assert len(candidates) == 91
    assert len(candidates.loc[candidates["selected"]]) == 1
    assert 0.90 - 1e-12 <= selected["compute_ratio_vs_uniform"] <= 1.0 + 1e-12
    assert np.isclose(selected["R_G"], feasible["R_G"].min())
    assert tuple(metadata["puregeo_anchors"]) == EXPECTED_PUREGEO


def test_finalgeo_freezes_complete_preconfirmatory_protocol(tmp_path):
    root = _development_root(tmp_path / "rq2")
    output = tmp_path / "freeze"
    protocol = freeze_finalgeo_selector(root, output, repo_root=tmp_path / "not-a-repo")
    assert protocol["status"] == "FROZEN_BEFORE_CONFIRMATORY_SEEDS"
    assert protocol["seeds_confirmatory"] == [3, 4, 5]
    assert protocol["K"] == 4
    assert protocol["selection_rule"]["objective"] == "minimize_R_G"
    assert protocol["selection_rule"]["accuracy_used"] is False
    assert protocol["selection_rule"]["parameter_exposure_used"] is False
    assert protocol["compute_constraint"]["selected_ratio_vs_uniform"] >= 0.90
    assert protocol["compute_constraint"]["selected_ratio_vs_uniform"] <= 1.00
    assert set(path.name for path in output.iterdir()) == {
        "finalgeo_selector_candidates.csv",
        "finalgeo_selected_anchors.json",
        "finalgeo_frozen_protocol.json",
    }
    # An identical rerun validates rather than changing the frozen decision.
    assert freeze_finalgeo_selector(root, output, repo_root=tmp_path / "not-a-repo") == protocol


def test_finalgeo_refuses_source_change_after_freeze(tmp_path):
    root = _development_root(tmp_path / "rq2")
    output = tmp_path / "freeze"
    freeze_finalgeo_selector(root, output, repo_root=tmp_path / "not-a-repo")
    coordinate_path = root / "protocol" / "geometry_trajectory_coordinates.csv"
    coordinate = pd.read_csv(coordinate_path)
    coordinate.loc[3, "incoming_edge_length"] *= 1.5
    coordinate.to_csv(coordinate_path, index=False)
    with pytest.raises(RuntimeError, match="already frozen"):
        freeze_finalgeo_selector(root, output, repo_root=tmp_path / "not-a-repo")

