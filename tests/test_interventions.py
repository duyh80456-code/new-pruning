import numpy as np
import pandas as pd
import torch

from baseline_artifacts import compute_anchor_geometry_prior, find_confirmatory_root
from cfm_mechanism import (
    evaluate_holdout,
    fixed_sample_split,
    train_mechanism_models,
)
from scripts.run_combined_interventions import compare_cfm, compare_geoweighting


def test_baseline_resolver_and_geometry_prior(tmp_path):
    run = tmp_path / "nested" / "confirmatory"
    (run / "confirmatory_specialization_table.csv").parent.mkdir(parents=True)
    (run / "confirmatory_specialization_table.csv").write_text("seed,width\n")
    for seed in range(3):
        result_dir = run / f"seed_{seed}" / "results"
        result_dir.mkdir(parents=True)
        (run / f"seed_{seed}" / "features").mkdir()
        pd.DataFrame(
            {
                "budget_start": [0.25, 0.50, 0.75, 0.95],
                "G_width": np.asarray([1.0, 2.0, 3.0, 4.0]) + seed * 0.1,
            }
        ).to_csv(result_dir / "local_sensitivity.csv", index=False)
    (run / "central_analysis_all_seeds.csv").write_text("seed,budget\n")
    assert find_confirmatory_root(tmp_path) == run
    prior, weights = compute_anchor_geometry_prior(
        run, [0.25, 0.50, 0.75, 1.0], alpha=1.0, beta=0.5
    )
    assert len(prior) == 4
    assert np.isclose(np.mean(list(weights.values())), 1.0)
    assert all(weight > 0 for weight in weights.values())


def test_cfm_mechanism_smoke_produces_all_comparators():
    generator = torch.Generator().manual_seed(7)
    bank = {
        budget: torch.nn.functional.normalize(torch.randn(24, 6, generator=generator), dim=1)
        for budget in [0.3, 0.4, 0.6, 0.8]
    }
    train_indices, validation_indices, test_indices = fixed_sample_split(24, 1, 0.5, 0.25)
    config = {
        "cfm": {
            "hidden_dim": 12,
            "learning_rate": 0.001,
            "batch_size": 6,
            "epochs": 1,
            "training_seed": 3,
            "integration_steps": 2,
        },
        "geometry": {"num_projections": 4, "projection_seed": 0},
    }
    vector_field, mlp, history, best_cfm, best_mlp = train_mechanism_models(
        bank, [0.3, 0.6, 0.8], train_indices, validation_indices, config, torch.device("cpu")
    )
    result = evaluate_holdout(
        vector_field,
        mlp,
        bank,
        [0.3, 0.6, 0.8],
        0.4,
        test_indices,
        config,
        torch.device("cpu"),
    )
    assert len(history) == 1 and best_cfm["state"] and best_mlp["state"]
    assert set(result["method"]) == {
        "shared_nearest",
        "linear_interpolation",
        "conditional_mlp",
        "cfm",
    }
    assert np.isfinite(result[["sliced_wasserstein", "paired_mse", "paired_cosine"]]).all().all()


def test_combined_comparisons_write_independent_decisions(tmp_path):
    baseline_root = tmp_path / "baseline"
    a1_root = tmp_path / "a1"
    cfm_root = tmp_path / "cfm"
    baseline_root.mkdir(); a1_root.mkdir(); cfm_root.mkdir()
    budgets = np.round(np.arange(0.25, 1.001, 0.05), 2)

    def dense_frame(accuracy_shift, geometry_scale):
        rows = []
        for seed in range(3):
            for index, budget in enumerate(budgets):
                rows.append(
                    {
                        "seed": seed,
                        "budget": budget,
                        "accuracy": 0.50 + 0.1 * budget + accuracy_shift,
                        "flops": 1_000_000 * (1 + index),
                        "local_wasserstein_sensitivity": (
                            geometry_scale * (1.0 - budget) if index < len(budgets) - 1 else np.nan
                        ),
                    }
                )
        return pd.DataFrame(rows)

    dense_frame(0.0, 1.0).to_csv(
        baseline_root / "central_analysis_all_seeds.csv", index=False
    )
    dense_frame(0.01, 0.8).to_csv(a1_root / "central_analysis_all_seeds.csv", index=False)
    _, _, _, a1_comparison, a1_checks = compare_geoweighting(baseline_root, a1_root)
    assert len(a1_comparison) == 3 and a1_checks["go"]
    assert (a1_root / "a1_decision.json").is_file()

    records = []
    distances = {
        "shared_nearest": 0.40,
        "linear_interpolation": 0.30,
        "conditional_mlp": 0.25,
        "cfm": 0.20,
    }
    for seed in range(3):
        for holdout in (0.4, 0.6):
            for method, distance in distances.items():
                records.append(
                    {
                        "seed": seed,
                        "holdout_width": holdout,
                        "method": method,
                        "sliced_wasserstein": distance,
                        "paired_mse": distance,
                        "paired_cosine": 1.0 - distance,
                    }
                )
    cfm_comparison, _, cfm_checks = compare_cfm(pd.DataFrame(records), cfm_root)
    assert len(cfm_comparison) == 6 and cfm_checks["go"] and cfm_checks["beats_mlp_all"]
    assert (cfm_root / "cfm_decision.json").is_file()
