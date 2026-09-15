import pandas as pd

from rq2_anchor_placement import GRID
from rq2_resource_coverage_audit import enumerate_anchor_sets, widthwise_audit


def _coordinates():
    increments = [0.01, 0.04, 0.08, 0.05, 0.01, 0.01, 0.01, 0.01,
                  0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    values = [0.0]
    for increment in increments:
        values.append(values[-1] + increment)
    return dict(zip(GRID, values))


def test_endpoint_locked_enumeration_and_forced_040_resource_limit():
    coordinates = _coordinates()
    flops = {width: 1.0 + width for width in GRID}
    k4 = enumerate_anchor_sets(4, coordinates, flops)
    k5 = enumerate_anchor_sets(5, coordinates, flops)
    assert len(k4) == 91
    assert len(k5) == 364
    assert not k4.loc[k4["has_040"], "R_res_no_worse_than_uniform"].any()
    assert set(k4.loc[k4["reference_set"].ne(""), "reference_set"]) == {
        "Uniform", "PureGeo", "FinalGeo"
    }


def test_widthwise_audit_preserves_seed_specific_accuracy_deltas():
    rows = []
    for seed in (3, 4):
        for method in ("uniform", "finalgeo"):
            for width in GRID:
                rows.append({
                    "seed": seed, "method": method, "budget": width,
                    "split": "validation_5k",
                    "accuracy": 0.60 + (0.01 if method == "finalgeo" else 0.0),
                })
    result = widthwise_audit(pd.DataFrame(rows), _coordinates())
    assert len(result) == 16
    assert (result["delta_accuracy_seed_3"].round(8) == 0.01).all()
    assert (result["delta_accuracy_seed_4"].round(8) == 0.01).all()
    assert result.loc[result["width"].eq(0.40), "functional_coverage_better_in_finalgeo"].item()
