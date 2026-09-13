import numpy as np
import pandas as pd

from rq1_analysis import (
    REPRESENTATIONS,
    block_bootstrap_correlations,
    loso_predictors,
    matched_pair_table,
    resource_residuals,
    run_rq1_analysis,
)


def _frame():
    rows = []
    geometry = {0.3: 0.05, 0.4: 0.08, 0.6: 0.03, 0.8: 0.01}
    gaps = {0.3: 0.03, 0.4: 0.06, 0.6: 0.015, 0.8: 0.005}
    for representation_index, representation in enumerate(REPRESENTATIONS):
        for seed in (0, 1, 2):
            for width in (0.3, 0.4, 0.6, 0.8):
                rows.append({
                    "seed": seed,
                    "width": width,
                    "representation": representation,
                    "G": geometry[width] * (1 + 0.01 * representation_index) + seed * 1e-3,
                    "specialization_gap": gaps[width] + seed * 1e-3,
                    "coverage": min(abs(width - anchor) for anchor in (0.25, 0.5, 0.75, 1.0)),
                    "log_flops": np.log(1e8 * width * width),
                    "flops": 1e8 * width * width,
                    "params": 1e6 * width,
                })
    return pd.DataFrame(rows)


def test_matched_pair_has_exact_requested_columns_and_positive_deltas():
    result = matched_pair_table(_frame())
    assert result.columns.tolist() == [
        "seed", "G_040", "G_060", "delta_G", "gap_040", "gap_060", "delta_gap"
    ]
    assert len(result) == 3
    assert (result[["delta_G", "delta_gap"]].to_numpy() > 0).all()


def test_loso_residual_and_seed_block_bootstrap_outputs():
    frame = _frame()
    summary, folds = loso_predictors(frame)
    assert len(summary) == len(REPRESENTATIONS) * 6
    assert len(folds) == len(REPRESENTATIONS) * 6 * 3
    assert np.isfinite(summary[["loso_mae", "loso_r2"]]).all().all()
    residuals = resource_residuals(frame)
    assert len(residuals) == len(frame)
    assert np.isfinite(residuals["resource_residual"]).all()
    bootstrap = block_bootstrap_correlations(frame, replicates=25, seed=7)
    assert len(bootstrap) == len(REPRESENTATIONS) * 2 * 2
    assert set(bootstrap["bootstrap_unit"]) == {"seed_block"}
    assert np.isfinite(bootstrap[["estimate", "ci_low", "ci_high"]]).all().all()


def test_end_to_end_writes_required_outputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    frame = _frame()
    frame.loc[frame.representation.eq("learned_projection"), [
        "seed", "width", "specialization_gap", "flops", "params"
    ]].to_csv(source / "specialization_table.csv", index=False)
    geometry = frame[["seed", "representation", "width", "G"]].rename(
        columns={"width": "budget_start"}
    )
    geometry["budget_end"] = geometry["budget_start"] + 0.05
    geometry.to_csv(source / "representation_local_geometry_all_seeds.csv", index=False)
    pd.DataFrame([
        {"seed": seed, "budget": width, "accuracy": 0.5 + 0.1 * width + 0.01 * seed}
        for seed in (0, 1, 2) for width in (0.3, 0.4, 0.6, 0.8)
    ]).to_csv(source / "shared_dense_metrics_all_seeds.csv", index=False)
    output = tmp_path / "output"
    run_rq1_analysis(source, output, bootstrap_replicates=50)
    required = {
        "rq1_matched_pair_040_060.csv", "rq1_loso_predictors.csv",
        "rq1_resource_residuals.csv", "rq1_bootstrap_correlations.csv",
        "rq1_summary.md", "rq1_dense_geometry_vs_accuracy.png",
        "rq1_geometry_vs_resource_residual.png",
    }
    assert all((output / name).stat().st_size > 0 for name in required)
