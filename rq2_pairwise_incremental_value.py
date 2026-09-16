"""CPU-only LOSO test of pairwise SW information beyond resource geometry."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rq2_pairwise_surrogate_regret import (
    EPOCHS,
    INTERIOR_WIDTHS,
    NUM_PAIRS,
    PAIR_INDICES,
    PATHS,
    _resource_flops,
    _variance,
    find_quick_trajectory_root,
    gram_pair_scores,
    solve_pair_lp,
)


MODEL_FEATURES = {
    "Resource": ["width_distance_sq_normalized", "log_flops_distance_sq_normalized"],
    "Resource+SW": [
        "width_distance_sq_normalized", "log_flops_distance_sq_normalized",
        "sw_sq_normalized",
    ],
}


def build_pairwise_state_table(quick_root: str | Path, ht_root: str | Path | None = None):
    quick_root = Path(quick_root)
    pairs = pd.read_csv(quick_root / "quick_pair_structure.csv")
    flops = _resource_flops(quick_root, ht_root)
    flops_by_width = dict(zip(INTERIOR_WIDTHS, flops))
    rows = []
    for path in PATHS:
        for epoch in EPOCHS:
            state = pairs.loc[pairs.path.eq(path) & pairs.epoch.eq(epoch)].copy()
            state = state.sort_values(["width_i", "width_j"]).reset_index(drop=True)
            if len(state) != NUM_PAIRS:
                raise RuntimeError(f"Incomplete pair table for {path}, epoch {epoch}")
            state["state"] = f"{'G' if path == 'geo_ht' else 'R'}-{epoch}"
            state["gradient_distance_sq"] = np.square(state.gradient_euclidean_distance)
            state["sw_sq"] = np.square(state.representation_sw)
            state["width_distance_sq"] = np.square(state.width_i - state.width_j)
            state["log_flops_distance_sq"] = np.square(
                state.width_i.map(lambda width: np.log(flops_by_width[round(float(width), 2)]))
                - state.width_j.map(lambda width: np.log(flops_by_width[round(float(width), 2)]))
            )
            for raw, normalized in (
                ("gradient_distance_sq", "gradient_distance_sq_normalized"),
                ("sw_sq", "sw_sq_normalized"),
                ("width_distance_sq", "width_distance_sq_normalized"),
                ("log_flops_distance_sq", "log_flops_distance_sq_normalized"),
            ):
                mean = float(state[raw].mean())
                if not np.isfinite(mean) or mean <= 0:
                    raise RuntimeError(f"Degenerate {raw} for {path}, epoch {epoch}")
                state[normalized] = state[raw] / mean
            rows.append(state)
    result = pd.concat(rows, ignore_index=True)
    if result.state.nunique() != 6 or len(result) != 6 * NUM_PAIRS:
        raise RuntimeError("Expected exactly 91 pairs in each of six states")
    return result


def _plots(output_dir: Path, metrics: pd.DataFrame, policy: pd.DataFrame,
           predictions: pd.DataFrame) -> None:
    state_order = [f"G-{epoch}" for epoch in EPOCHS] + [f"R-{epoch}" for epoch in EPOCHS]
    pivot = metrics.pivot(index="heldout_state", columns="model", values="mae").reindex(state_order)
    fig, axis = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(pivot)); width = 0.35
    axis.bar(x - width/2, pivot["Resource"], width, label="Resource")
    axis.bar(x + width/2, pivot["Resource+SW"], width, label="Resource+SW")
    axis.set(xticks=x, xticklabels=state_order, ylabel="Held-out normalized pair-distance MAE")
    axis.grid(axis="y", alpha=0.25); axis.legend(); fig.tight_layout()
    fig.savefig(output_dir / "loso_pair_distance_mae.png", dpi=200); plt.close(fig)

    ordered = policy.set_index("heldout_state").reindex(state_order)
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(x, ordered.hybrid_minus_resource_variance)
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xticks=x, xticklabels=state_order,
             ylabel="V(hybrid Resource+SW) - V(Resource)")
    axis.grid(axis="y", alpha=0.25); fig.tight_layout()
    fig.savefig(output_dir / "loso_hybrid_policy_variance_delta.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for axis, state_name in zip(axes.flat, state_order):
        frame = predictions.loc[
            predictions.heldout_state.eq(state_name)
            & predictions.model.eq("Resource+SW")
        ]
        axis.scatter(frame.observed, frame.predicted, s=18, alpha=0.8)
        low = min(frame.observed.min(), frame.predicted.min())
        high = max(frame.observed.max(), frame.predicted.max())
        axis.plot([low, high], [low, high], color="black", linewidth=1)
        axis.set_title(state_name); axis.grid(alpha=0.2)
    fig.supxlabel("Observed normalized gradient pair distance")
    fig.supylabel("LOSO Resource+SW prediction")
    fig.tight_layout(); fig.savefig(output_dir / "loso_hybrid_predicted_vs_observed.png", dpi=200)
    plt.close(fig)


def run_pairwise_incremental_value(
    quick_root: str | Path,
    output_dir: str | Path,
    ht_root: str | Path | None = None,
) -> dict:
    quick_root, output_dir = Path(quick_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = build_pairwise_state_table(quick_root, ht_root)
    states = [f"G-{epoch}" for epoch in EPOCHS] + [f"R-{epoch}" for epoch in EPOCHS]
    prediction_rows, metric_rows, policy_rows, coefficient_rows = [], [], [], []
    for heldout in states:
        train, test = table.loc[table.state.ne(heldout)], table.loc[table.state.eq(heldout)]
        if train.state.nunique() != 5 or len(test) != NUM_PAIRS:
            raise RuntimeError("LOSO split does not contain five train states and one test state")
        predictions = {}
        for model_name, features in MODEL_FEATURES.items():
            model = make_pipeline(StandardScaler(), LinearRegression())
            model.fit(train[features], train.gradient_distance_sq_normalized)
            predicted = model.predict(test[features])
            predictions[model_name] = predicted
            observed = test.gradient_distance_sq_normalized.to_numpy(float)
            rho = spearmanr(observed, predicted)
            metric_rows.append({
                "heldout_state": heldout, "model": model_name,
                "mae": mean_absolute_error(observed, predicted),
                "rmse": np.sqrt(mean_squared_error(observed, predicted)),
                "r2": r2_score(observed, predicted),
                "spearman_rho": float(rho.statistic), "spearman_p": float(rho.pvalue),
            })
            linear = model.named_steps["linearregression"]
            for feature, coefficient in zip(features, linear.coef_):
                coefficient_rows.append({
                    "heldout_state": heldout, "model": model_name,
                    "feature": feature, "standardized_coefficient": coefficient,
                })
            for pair_index, (row, estimate) in enumerate(zip(test.itertuples(), predicted)):
                prediction_rows.append({
                    "heldout_state": heldout, "path": row.path, "epoch": row.epoch,
                    "pair_index": pair_index, "width_i": row.width_i, "width_j": row.width_j,
                    "model": model_name, "observed": row.gradient_distance_sq_normalized,
                    "predicted": estimate, "error": estimate - row.gradient_distance_sq_normalized,
                })
        path = str(test.path.iloc[0]); epoch = int(test.epoch.iloc[0])
        grams = np.load(
            quick_root / "worker_paths" / path / f"interior_grams_epoch_{epoch:03d}.npy"
        ).astype(np.float64)
        gram = grams.mean(axis=0)
        oracle_scores, _ = gram_pair_scores(gram)
        q_resource = solve_pair_lp(predictions["Resource"], maximize=True)
        q_hybrid = solve_pair_lp(predictions["Resource+SW"], maximize=True)
        q_oracle = solve_pair_lp(oracle_scores, maximize=True)
        q_sw = solve_pair_lp(test.sw_sq_normalized.to_numpy(float), maximize=True)
        v_resource = _variance(gram, q_resource)
        v_hybrid = _variance(gram, q_hybrid)
        v_oracle = _variance(gram, q_oracle)
        v_sw = _variance(gram, q_sw)
        denominator = v_resource - v_oracle
        policy_rows.append({
            "heldout_state": heldout, "path": path, "epoch": epoch,
            "V_resource_prediction": v_resource,
            "V_hybrid_resource_plus_sw_prediction": v_hybrid,
            "V_direct_sw": v_sw, "V_gradient_oracle": v_oracle,
            "hybrid_minus_resource_variance": v_hybrid - v_resource,
            "hybrid_better_than_resource": bool(v_hybrid < v_resource),
            "resource_oracle_gap_captured_by_hybrid": (
                (v_resource - v_hybrid) / denominator if denominator > 1e-12 else np.nan
            ),
        })
    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    policy = pd.DataFrame(policy_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    wide = metrics.pivot(index="heldout_state", columns="model")
    summary = pd.DataFrame([{
        "comparison": "Resource+SW vs Resource",
        "pooled_resource_mae": float(
            predictions.loc[predictions.model.eq("Resource")].error.abs().mean()
        ),
        "pooled_resource_plus_sw_mae": float(
            predictions.loc[predictions.model.eq("Resource+SW")].error.abs().mean()
        ),
        "mean_delta_mae_hybrid_minus_resource": float(
            (wide["mae"]["Resource+SW"] - wide["mae"]["Resource"]).mean()
        ),
        "prediction_mae_improved_states": int(
            (wide["mae"]["Resource+SW"] < wide["mae"]["Resource"]).sum()
        ),
        "hybrid_policy_variance_win_states": int(policy.hybrid_better_than_resource.sum()),
        "num_states": len(states),
        "mean_hybrid_oracle_gap_captured": float(
            policy.resource_oracle_gap_captured_by_hybrid.mean()
        ),
    }])
    table.to_csv(output_dir / "pairwise_state_design_matrix.csv", index=False)
    predictions.to_csv(output_dir / "pairwise_loso_predictions.csv", index=False)
    metrics.to_csv(output_dir / "pairwise_loso_metrics.csv", index=False)
    coefficients.to_csv(output_dir / "pairwise_loso_coefficients.csv", index=False)
    policy.to_csv(output_dir / "pairwise_loso_policy_variance.csv", index=False)
    summary.to_csv(output_dir / "pairwise_loso_summary.csv", index=False)
    _plots(output_dir, metrics, policy, predictions)
    row = summary.iloc[0]
    metadata = {
        "status": "PAIRWISE_SW_INCREMENTAL_VALUE_COMPLETE",
        "cross_validation": "leave-one-state-out (5 train states -> 1 test state)",
        "states": states, "pairs_per_state": NUM_PAIRS,
        "primary_resource_features": MODEL_FEATURES["Resource"],
        "added_feature": "state-normalized current SW_ij^2",
        "state_normalization": "each pair-distance family divided by its within-state mean",
        "preprocessing_fit_train_states_only": True,
        "prediction_mae_improved_states": int(row.prediction_mae_improved_states),
        "hybrid_policy_variance_win_states": int(row.hybrid_policy_variance_win_states),
        "all_six_prediction_and_policy_wins": bool(
            row.prediction_mae_improved_states == 6 and row.hybrid_policy_variance_win_states == 6
        ),
        "training_performed": False, "gpu_required": False,
        "accuracy_used": False, "test_used": False, "policy_updated": False,
        "interpretation": "held-out diagnostic only; does not authorize training",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


__all__ = [
    "find_quick_trajectory_root", "build_pairwise_state_table",
    "run_pairwise_incremental_value",
]
