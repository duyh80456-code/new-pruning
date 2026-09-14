import json

import pandas as pd
import yaml

from rq2_anchor_placement import GRID
from rq2_parameter_exposure import (
    activation_bands,
    exact_active_parameter_counts,
    exposure_count,
    run_parameter_exposure_preview,
)


def _development_root(root):
    protocol = root / "protocol"
    protocol.mkdir(parents=True)
    config = yaml.safe_load(open("configs/kaggle_rq2_anchor.yaml"))
    config["compression"]["train_widths"] = [0.25, 0.40, 0.60, 1.00]
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    rows = []
    for seed in (1, 2):
        for start, end in zip(GRID[:-1], GRID[1:]):
            rows.append({
                "method": "uniform", "seed": seed,
                "representation": "learned_projection",
                "budget_start": start, "budget_end": end,
                "G": 4.0 if 0.35 <= start <= 0.45 else 1.0,
            })
    pd.DataFrame(rows).to_csv(root / "rq2_geometry_all.csv", index=False)
    pd.DataFrame([
        {"method": "uniform", "seed": seed, "budget": width, "flops": 1e8 * width * width}
        for seed in (1, 2) for width in GRID
    ]).to_csv(root / "rq2_dense_metrics_all.csv", index=False)
    pd.DataFrame({
        "width": GRID,
        "incoming_edge_length": [float("nan")] + [
            0.2 if 0.35 <= start <= 0.45 else 0.05 for start in GRID[:-1]
        ],
    }).to_csv(protocol / "geometry_trajectory_coordinates.csv", index=False)
    (protocol / "selected_anchors.json").write_text(json.dumps({
        "selected_anchors": [0.25, 0.40, 0.60, 1.00]
    }))
    return root


def test_exact_activation_bands_match_nested_model_counts():
    config = yaml.safe_load(open("configs/kaggle_rq2_anchor.yaml"))
    active = exact_active_parameter_counts(config)
    bands = activation_bands(active)
    assert set(active["parameter_group"]) == {
        "stem", "layer1", "layer2", "layer3", "layer4", "projection", "classifier"
    }
    for group, subset in bands.groupby("parameter_group"):
        expected = active.loc[
            active["parameter_group"].eq(group) & active["width"].eq(1.0),
            "active_parameters",
        ].iloc[0]
        assert subset["newly_active_parameters"].sum() == expected
    classifier = active.loc[active["parameter_group"].eq("classifier")]
    assert classifier["active_parameters"].nunique() == 1


def test_exposure_profile_has_the_expected_uniform_to_puregeo_drops():
    uniform = (0.25, 0.50, 0.75, 1.00)
    puregeo = (0.25, 0.40, 0.60, 1.00)
    assert exposure_count(uniform, 0.45) == 3
    assert exposure_count(puregeo, 0.45) == 2
    assert exposure_count(uniform, 0.65) == 2
    assert exposure_count(puregeo, 0.65) == 1


def test_cpu_preview_enumerates_91_sets_and_does_not_freeze_a_new_one(tmp_path):
    root = _development_root(tmp_path / "rq2")
    output = tmp_path / "exposure"
    result = run_parameter_exposure_preview(root, output)
    candidates = pd.read_csv(output / "parameter_exposure_all_anchor_sets.csv")
    uniform = candidates.loc[candidates["is_uniform"]].iloc[0]
    puregeo = candidates.loc[candidates["is_puregeo_v1"]].iloc[0]
    assert len(candidates) == 91
    assert uniform["D_E"] == 0.0
    assert puregeo["D_E"] > 0.0
    assert puregeo["fraction_parameters_underexposed"] > 0.0
    assert candidates["pareto_RG_DE"].any()
    assert result["new_anchor_set_selected"] is False
    assert result["accuracy_or_test_data_used"] is False
    assert (output / "parameter_exposure_pareto.png").stat().st_size > 0

