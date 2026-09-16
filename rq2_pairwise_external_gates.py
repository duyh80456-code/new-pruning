"""CPU gates for resource baselines and externally valid pairwise SW value.

Gate A audits width-distance and log-FLOPs-distance pair policies. Gate B0 runs
leave-one-epoch-out (LOEO) and leave-one-path-out (LPO) validation on the six
development states, then freezes predictors for a separate fresh-state Gate B1.
No model training, accuracy data, or policy tuning is performed here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rq2_pairwise_incremental_value import MODEL_FEATURES, build_pairwise_state_table
from rq2_pairwise_surrogate_regret import (
    EPOCHS,
    INTERIOR_WIDTHS,
    NUM_PAIRS,
    PATHS,
    _resource_flops,
    _variance,
    gram_pair_scores,
    solve_pair_lp,
)


STATE_ORDER = tuple(
    [f"G-{epoch}" for epoch in EPOCHS] + [f"R-{epoch}" for epoch in EPOCHS]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_parts(state: str) -> tuple[str, int]:
    prefix, epoch = state.split("-")
    return ("geo_ht" if prefix == "G" else "resource_ht"), int(epoch)


def _load_gram(quick_root: Path, state: str) -> np.ndarray:
    path, epoch = _state_parts(state)
    grams = np.load(
        quick_root / "worker_paths" / path / f"interior_grams_epoch_{epoch:03d}.npy"
    ).astype(np.float64)
    if grams.ndim != 3 or grams.shape[1:] != (14, 14):
        raise RuntimeError(f"Invalid Gram stack for {state}: {grams.shape}")
    return grams.mean(axis=0)


def _q_support(q: np.ndarray, threshold: float = 1e-9) -> str:
    pairs = []
    from rq2_pairwise_surrogate_regret import PAIR_INDICES

    for probability, (i, j) in zip(q, PAIR_INDICES):
        if probability > threshold:
            pairs.append(f"{INTERIOR_WIDTHS[i]:.2f}-{INTERIOR_WIDTHS[j]:.2f}:{probability:.8f}")
    return ";".join(pairs)


def run_gate_a(
    quick_root: str | Path,
    output_dir: str | Path,
    ht_root: str | Path | None = None,
) -> dict:
    """Audit width and model-profiled FLOPs resource controls."""
    quick_root, output_dir = Path(quick_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flops = _resource_flops(quick_root, ht_root)
    if flops.shape != (14,) or not np.all(np.diff(flops) > 0):
        raise RuntimeError("Expected 14 strictly increasing model-profiled FLOPs values")
    from rq2_pairwise_surrogate_regret import PAIR_INDICES

    log_flops = np.log(flops)
    width_scores = np.asarray([
        (INTERIOR_WIDTHS[i] - INTERIOR_WIDTHS[j]) ** 2 for i, j in PAIR_INDICES
    ])
    flops_scores = np.asarray([
        (log_flops[i] - log_flops[j]) ** 2 for i, j in PAIR_INDICES
    ])
    q_width = solve_pair_lp(width_scores, maximize=True)
    q_flops = solve_pair_lp(flops_scores, maximize=True)
    score_pearson = pearsonr(width_scores, flops_scores)
    score_spearman = spearmanr(width_scores, flops_scores)
    normalized_width = width_scores / width_scores.mean()
    normalized_flops = flops_scores / flops_scores.mean()
    score_rows = []
    policy_rows = []
    for index, (i, j) in enumerate(PAIR_INDICES):
        score_rows.append({
            "pair_index": index,
            "width_i": INTERIOR_WIDTHS[i], "width_j": INTERIOR_WIDTHS[j],
            "width_distance_sq": width_scores[index],
            "log_flops_distance_sq": flops_scores[index],
            "width_distance_sq_normalized": normalized_width[index],
            "log_flops_distance_sq_normalized": normalized_flops[index],
        })
        policy_rows.extend([
            {"policy": "width_distance", "pair_index": index,
             "width_i": INTERIOR_WIDTHS[i], "width_j": INTERIOR_WIDTHS[j],
             "probability": q_width[index]},
            {"policy": "log_flops_distance", "pair_index": index,
             "width_i": INTERIOR_WIDTHS[i], "width_j": INTERIOR_WIDTHS[j],
             "probability": q_flops[index]},
        ])
    variance_rows = []
    for state in STATE_ORDER:
        gram = _load_gram(quick_root, state)
        variance_rows.extend([
            {"state": state, "policy": "width_distance", "exact_variance": _variance(gram, q_width)},
            {"state": state, "policy": "log_flops_distance", "exact_variance": _variance(gram, q_flops)},
        ])
    pd.DataFrame(score_rows).to_csv(output_dir / "gate_a_resource_pair_scores.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(output_dir / "gate_a_resource_pair_policies.csv", index=False)
    pd.DataFrame(variance_rows).to_csv(output_dir / "gate_a_resource_exact_variance.csv", index=False)
    source_file = quick_root / "quick_dynamic_geometry_by_width.csv"
    source_frame = pd.read_csv(source_file)
    flops_from_quick_artifact = "flops" in source_frame.columns
    source_kind = "quick_dynamic_geometry_by_width.csv:model_profiled_flops"
    if not flops_from_quick_artifact:
        source_kind = "HT protocol/frozen_dynamic_marginals.csv:model_profiled_flops"
    summary = {
        "status": "GATE_A_RESOURCE_AUDIT_COMPLETE",
        "flops_source": source_kind,
        "flops_are_model_profiled_not_width_proxy": True,
        "flops": {f"{w:.2f}": float(f) for w, f in zip(INTERIOR_WIDTHS, flops)},
        "score_pearson": float(score_pearson.statistic),
        "score_spearman": float(score_spearman.statistic),
        "score_frobenius_raw": float(np.linalg.norm(width_scores - flops_scores)),
        "score_frobenius_mean_normalized": float(np.linalg.norm(normalized_width - normalized_flops)),
        "policy_l1_difference": float(np.abs(q_width - q_flops).sum()),
        "policy_max_abs_difference": float(np.abs(q_width - q_flops).max()),
        "width_policy_support": _q_support(q_width),
        "flops_policy_support": _q_support(q_flops),
        "policy_equivalent_at_1e_8": bool(np.max(np.abs(q_width - q_flops)) <= 1e-8),
        "interpretation": (
            "PASS_EQUIVALENT_OPTIMUM" if np.max(np.abs(q_width - q_flops)) <= 1e-8
            else "DISTINCT_RESOURCE_CONTROLS_RETAIN_BOTH"
        ),
        "training_performed": False,
    }
    (output_dir / "gate_a_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _folds() -> list[dict]:
    folds = []
    for epoch in EPOCHS:
        test = [f"G-{epoch}", f"R-{epoch}"]
        folds.append({
            "protocol": "LOEO", "fold": f"holdout_epoch_{epoch}",
            "train_states": [state for state in STATE_ORDER if state not in test],
            "test_states": test,
        })
    folds.extend([
        {"protocol": "LPO", "fold": "train_G_test_R",
         "train_states": [f"G-{epoch}" for epoch in EPOCHS],
         "test_states": [f"R-{epoch}" for epoch in EPOCHS]},
        {"protocol": "LPO", "fold": "train_R_test_G",
         "train_states": [f"R-{epoch}" for epoch in EPOCHS],
         "test_states": [f"G-{epoch}" for epoch in EPOCHS]},
    ])
    return folds


def _serialize_pipeline(model, features: list[str]) -> dict:
    scaler = model.named_steps["standardscaler"]
    linear = model.named_steps["linearregression"]
    return {
        "features": list(features),
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_scale": [float(value) for value in scaler.scale_],
        "linear_intercept": float(linear.intercept_),
        "linear_coefficients": [float(value) for value in linear.coef_],
    }


def frozen_predict(payload: dict, model_name: str, frame: pd.DataFrame) -> np.ndarray:
    spec = payload["models"][model_name]
    x = frame[spec["features"]].to_numpy(float)
    scaled = (x - np.asarray(spec["scaler_mean"])) / np.asarray(spec["scaler_scale"])
    return float(spec["linear_intercept"]) + scaled @ np.asarray(spec["linear_coefficients"])


def run_gate_b0_and_freeze(
    quick_root: str | Path,
    output_dir: str | Path,
    ht_root: str | Path | None = None,
) -> dict:
    """Run LOEO/LPO and freeze full-development predictors for fresh Gate B1."""
    quick_root, output_dir = Path(quick_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = build_pairwise_state_table(quick_root, ht_root)
    prediction_rows, metric_rows, policy_rows, coefficient_rows = [], [], [], []
    for split in _folds():
        train = table.loc[table.state.isin(split["train_states"])].copy()
        test_all = table.loc[table.state.isin(split["test_states"])].copy()
        fitted = {}
        for model_name, features in MODEL_FEATURES.items():
            model = make_pipeline(StandardScaler(), LinearRegression())
            model.fit(train[features], train.gradient_distance_sq_normalized)
            fitted[model_name] = model
            serialized = _serialize_pipeline(model, features)
            for feature, coefficient in zip(features, serialized["linear_coefficients"]):
                coefficient_rows.append({
                    "protocol": split["protocol"], "fold": split["fold"],
                    "model": model_name, "feature": feature,
                    "standardized_coefficient": coefficient,
                })
        for state in split["test_states"]:
            test = test_all.loc[test_all.state.eq(state)].sort_values(
                ["width_i", "width_j"]
            ).reset_index(drop=True)
            observed = test.gradient_distance_sq_normalized.to_numpy(float)
            predicted = {}
            for model_name, features in MODEL_FEATURES.items():
                values = fitted[model_name].predict(test[features])
                predicted[model_name] = values
                rho = spearmanr(observed, values)
                metric_rows.append({
                    "protocol": split["protocol"], "fold": split["fold"],
                    "heldout_state": state, "model": model_name,
                    "mae": mean_absolute_error(observed, values),
                    "rmse": np.sqrt(mean_squared_error(observed, values)),
                    "r2": r2_score(observed, values),
                    "spearman_rho": float(rho.statistic),
                })
                for row, estimate in zip(test.itertuples(), values):
                    prediction_rows.append({
                        "protocol": split["protocol"], "fold": split["fold"],
                        "heldout_state": state, "model": model_name,
                        "width_i": row.width_i, "width_j": row.width_j,
                        "observed": row.gradient_distance_sq_normalized,
                        "predicted": estimate,
                    })
            gram = _load_gram(quick_root, state)
            oracle_scores, _ = gram_pair_scores(gram)
            q_resource = solve_pair_lp(predicted["Resource"], maximize=True)
            q_hybrid = solve_pair_lp(predicted["Resource+SW"], maximize=True)
            q_oracle = solve_pair_lp(oracle_scores, maximize=True)
            v_resource, v_hybrid, v_oracle = (
                _variance(gram, q_resource), _variance(gram, q_hybrid), _variance(gram, q_oracle)
            )
            denominator = v_resource - v_oracle
            policy_rows.append({
                "protocol": split["protocol"], "fold": split["fold"],
                "heldout_state": state,
                "V_resource": v_resource, "V_resource_plus_sw": v_hybrid,
                "V_oracle": v_oracle,
                "delta_V_hybrid_minus_resource": v_hybrid - v_resource,
                "hybrid_variance_win": bool(v_hybrid < v_resource),
                "resource_oracle_gap_captured": (
                    (v_resource - v_hybrid) / denominator if denominator > 1e-12 else np.nan
                ),
            })
    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    policies = pd.DataFrame(policy_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    predictions.to_csv(output_dir / "gate_b0_heldout_predictions.csv", index=False)
    metrics.to_csv(output_dir / "gate_b0_heldout_metrics.csv", index=False)
    policies.to_csv(output_dir / "gate_b0_heldout_exact_variance.csv", index=False)
    coefficients.to_csv(output_dir / "gate_b0_fold_coefficients.csv", index=False)
    aggregate_rows = []
    for protocol in ("LOEO", "LPO"):
        m = metrics.loc[metrics.protocol.eq(protocol)]
        p = policies.loc[policies.protocol.eq(protocol)]
        pivot = m.pivot_table(index=["fold", "heldout_state"], columns="model", values="mae")
        aggregate_rows.append({
            "protocol": protocol,
            "heldout_state_evaluations": len(p),
            "resource_mae": float(m.loc[m.model.eq("Resource"), "mae"].mean()),
            "resource_plus_sw_mae": float(m.loc[m.model.eq("Resource+SW"), "mae"].mean()),
            "mae_win_states": int((pivot["Resource+SW"] < pivot["Resource"]).sum()),
            "variance_win_states": int(p.hybrid_variance_win.sum()),
            "mean_delta_V": float(p.delta_V_hybrid_minus_resource.mean()),
            "mean_resource_oracle_gap_captured": float(p.resource_oracle_gap_captured.mean()),
        })
    aggregates = pd.DataFrame(aggregate_rows)
    aggregates.to_csv(output_dir / "gate_b0_summary.csv", index=False)

    frozen_models = {}
    for model_name, features in MODEL_FEATURES.items():
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(table[features], table.gradient_distance_sq_normalized)
        frozen_models[model_name] = _serialize_pipeline(model, features)
    predictor_payload = {
        "status": "FROZEN_BEFORE_FRESH_STATES",
        "development_states": list(STATE_ORDER),
        "development_seeds": [3],
        "pairs_per_state": NUM_PAIRS,
        "target": "gradient_distance_sq / within_state_mean",
        "feature_normalization": "each raw pair feature divided by its within-state mean",
        "models": frozen_models,
        "no_refit_on_fresh_states": True,
        "source_hashes": {
            "quick_pair_structure.csv": _sha256(quick_root / "quick_pair_structure.csv"),
            "quick_dynamic_geometry_by_width.csv": _sha256(
                quick_root / "quick_dynamic_geometry_by_width.csv"
            ),
        },
    }
    predictor_path = output_dir / "frozen_pairwise_predictors.json"
    predictor_path.write_text(json.dumps(predictor_payload, indent=2) + "\n")
    summary = {
        "status": "GATE_B0_COMPLETE_PREDICTORS_FROZEN",
        "primary_metric": "exact held-out pair-policy variance",
        "secondary_metrics": ["pair-distance MAE", "Spearman", "oracle-gap captured"],
        "protocols": ["LOEO: 4 states train, paired epoch held out", "LPO: one path train, other path test"],
        "frozen_predictor": str(predictor_path),
        "fresh_state_refit_allowed": False,
        "training_performed": False,
        "gate_c_authorized": False,
    }
    (output_dir / "gate_b0_metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot_gate_b0(output_dir, policies, metrics)
    return summary


def _plot_gate_b0(output_dir: Path, policies: pd.DataFrame, metrics: pd.DataFrame) -> None:
    labels = [f"{row.protocol}:{row.fold}:{row.heldout_state}" for row in policies.itertuples()]
    x = np.arange(len(policies))
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.bar(x, policies.delta_V_hybrid_minus_resource)
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xticks=x, xticklabels=labels, ylabel="V(Resource+SW) - V(Resource)")
    axis.tick_params(axis="x", rotation=55); axis.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(output_dir / "gate_b0_exact_variance_delta.png", dpi=200)
    plt.close(fig)


def run_all_development_gates(
    quick_root: str | Path,
    output_dir: str | Path,
    ht_root: str | Path | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_a = run_gate_a(quick_root, output_dir, ht_root)
    gate_b0 = run_gate_b0_and_freeze(quick_root, output_dir, ht_root)
    combined = {
        "status": "DEVELOPMENT_GATES_COMPLETE_FRESH_GATE_PENDING",
        "gate_a": gate_a,
        "gate_b0": gate_b0,
        "next_required": "Evaluate frozen_pairwise_predictors.json on disjoint fresh checkpoint states",
        "gate_c_authorized": False,
        "gpu_required": False,
    }
    (output_dir / "gate_a_b0_complete.json").write_text(json.dumps(combined, indent=2) + "\n")
    return combined


def _prepare_fresh_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "seed", "epoch", "width_i", "width_j", "representation_sw",
        "gradient_euclidean_distance", "flops_i", "flops_j",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Fresh pair table is missing columns: {sorted(missing)}")
    result = frame.copy()
    result["state"] = result.apply(
        lambda row: f"seed_{int(row.seed)}_epoch_{int(row.epoch):03d}", axis=1
    )
    result["gradient_distance_sq"] = np.square(result.gradient_euclidean_distance)
    result["sw_sq"] = np.square(result.representation_sw)
    result["width_distance_sq"] = np.square(result.width_i - result.width_j)
    result["log_flops_distance_sq"] = np.square(
        np.log(result.flops_i) - np.log(result.flops_j)
    )
    for raw, normalized in (
        ("gradient_distance_sq", "gradient_distance_sq_normalized"),
        ("sw_sq", "sw_sq_normalized"),
        ("width_distance_sq", "width_distance_sq_normalized"),
        ("log_flops_distance_sq", "log_flops_distance_sq_normalized"),
    ):
        means = result.groupby("state")[raw].transform("mean")
        if np.any(~np.isfinite(means)) or np.any(means <= 0):
            raise RuntimeError(f"Degenerate fresh-state feature: {raw}")
        result[normalized] = result[raw] / means
    counts = result.groupby("state").size()
    if not (counts == NUM_PAIRS).all():
        raise RuntimeError(f"Each fresh state must contain 91 pairs; counts={counts.to_dict()}")
    return result


def _paired_state_bootstrap(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, float)
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.asarray([
        rng.choice(values, size=len(values), replace=True).mean() for _ in range(int(draws))
    ])
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def run_gate_b1_fresh(
    fresh_root: str | Path,
    frozen_predictor_path: str | Path,
    output_dir: str | Path,
    bootstrap_draws: int = 10000,
    bootstrap_seed: int = 20260918,
) -> dict:
    """Evaluate frozen development predictors on disjoint fresh checkpoint states.

    Required fresh-root files:
      fresh_pair_structure.csv
      fresh_state_metadata.json
      grams/<state>.npy (a 14x14 Gram or a stack whose mean is 14x14)
    """
    fresh_root, output_dir = Path(fresh_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor_payload = json.loads(Path(frozen_predictor_path).read_text())
    if predictor_payload.get("status") != "FROZEN_BEFORE_FRESH_STATES":
        raise RuntimeError("Predictors were not frozen before fresh-state evaluation")
    metadata = json.loads((fresh_root / "fresh_state_metadata.json").read_text())
    if metadata.get("status") != "FRESH_PAIRWISE_STATES_COMPLETE":
        raise RuntimeError("Fresh-state artifact is incomplete")
    if metadata.get("predictor_used_during_extraction", True) is not False:
        raise RuntimeError("Fresh states must be extracted independently of the frozen predictor")
    development_seeds = set(map(int, predictor_payload.get("development_seeds", [])))
    declared_seeds = set(map(int, metadata.get("seeds", [])))
    if not declared_seeds or development_seeds & declared_seeds:
        raise RuntimeError("Fresh seeds must be nonempty and disjoint from development seeds")
    table = _prepare_fresh_table(pd.read_csv(fresh_root / "fresh_pair_structure.csv"))
    table_seeds = set(map(int, table.seed.unique()))
    if table_seeds != declared_seeds:
        raise RuntimeError("Fresh table seeds disagree with fresh-state metadata")
    prediction_rows, metric_rows, policy_rows = [], [], []
    for state, test in table.groupby("state", sort=True):
        test = test.sort_values(["width_i", "width_j"]).reset_index(drop=True)
        observed = test.gradient_distance_sq_normalized.to_numpy(float)
        predicted = {}
        for model_name in MODEL_FEATURES:
            values = frozen_predict(predictor_payload, model_name, test)
            predicted[model_name] = values
            rho = spearmanr(observed, values)
            metric_rows.append({
                "state": state, "seed": int(test.seed.iloc[0]),
                "epoch": int(test.epoch.iloc[0]), "model": model_name,
                "mae": mean_absolute_error(observed, values),
                "rmse": np.sqrt(mean_squared_error(observed, values)),
                "r2": r2_score(observed, values), "spearman_rho": float(rho.statistic),
            })
            for row, estimate in zip(test.itertuples(), values):
                prediction_rows.append({
                    "state": state, "seed": int(row.seed), "epoch": int(row.epoch),
                    "width_i": row.width_i, "width_j": row.width_j,
                    "model": model_name, "observed": row.gradient_distance_sq_normalized,
                    "predicted": estimate,
                })
        gram_path = fresh_root / "grams" / f"{state}.npy"
        grams = np.load(gram_path).astype(np.float64)
        gram = grams.mean(axis=0) if grams.ndim == 3 else grams
        if gram.shape != (14, 14):
            raise RuntimeError(f"Invalid fresh Gram shape for {state}: {grams.shape}")
        oracle_scores, _ = gram_pair_scores(gram)
        q_resource = solve_pair_lp(predicted["Resource"], maximize=True)
        q_hybrid = solve_pair_lp(predicted["Resource+SW"], maximize=True)
        q_oracle = solve_pair_lp(oracle_scores, maximize=True)
        v_resource, v_hybrid, v_oracle = (
            _variance(gram, q_resource), _variance(gram, q_hybrid), _variance(gram, q_oracle)
        )
        denominator = v_resource - v_oracle
        policy_rows.append({
            "state": state, "seed": int(test.seed.iloc[0]), "epoch": int(test.epoch.iloc[0]),
            "V_resource": v_resource, "V_resource_plus_sw": v_hybrid, "V_oracle": v_oracle,
            "delta_V_hybrid_minus_resource": v_hybrid - v_resource,
            "relative_delta_V": (v_hybrid - v_resource) / max(abs(v_resource), 1e-30),
            "hybrid_variance_win": bool(v_hybrid < v_resource),
            "resource_oracle_gap_captured": (
                (v_resource - v_hybrid) / denominator if denominator > 1e-12 else np.nan
            ),
        })
    predictions, metrics, policies = map(pd.DataFrame, (prediction_rows, metric_rows, policy_rows))
    predictions.to_csv(output_dir / "gate_b1_fresh_predictions.csv", index=False)
    metrics.to_csv(output_dir / "gate_b1_fresh_metrics.csv", index=False)
    policies.to_csv(output_dir / "gate_b1_fresh_exact_variance.csv", index=False)
    lower, upper = _paired_state_bootstrap(
        policies.delta_V_hybrid_minus_resource.to_numpy(float), bootstrap_draws, bootstrap_seed
    )
    improvements = -policies.delta_V_hybrid_minus_resource
    positive_total = float(improvements.clip(lower=0).sum())
    max_share = (
        float(improvements.clip(lower=0).max() / positive_total) if positive_total > 0 else np.nan
    )
    mean_relative_gain = float((-policies.relative_delta_V).mean())
    summary = {
        "status": "GATE_B1_FRESH_EVALUATION_COMPLETE",
        "fresh_seeds": sorted(declared_seeds),
        "fresh_states": int(len(policies)),
        "predictors_refit_on_fresh_states": False,
        "mean_delta_V_hybrid_minus_resource": float(policies.delta_V_hybrid_minus_resource.mean()),
        "bootstrap_95_ci_delta_V": [float(lower), float(upper)],
        "variance_win_states": int(policies.hybrid_variance_win.sum()),
        "majority_variance_win": bool(policies.hybrid_variance_win.mean() > 0.5),
        "mean_relative_variance_reduction": mean_relative_gain,
        "largest_positive_improvement_share": max_share,
        "no_single_state_dominates_at_50pct": bool(np.isfinite(max_share) and max_share < 0.5),
        "effect_size_class": (
            "STRONG_5_TO_10PCT_OR_MORE" if mean_relative_gain >= 0.05
            else "SMALL_BELOW_1PCT" if mean_relative_gain < 0.01
            else "MODERATE_1_TO_5PCT"
        ),
        "gate_b_pass": bool(
            policies.delta_V_hybrid_minus_resource.mean() < 0
            and policies.hybrid_variance_win.mean() > 0.5
            and upper < 0
            and np.isfinite(max_share) and max_share < 0.5
            and mean_relative_gain >= 0.01
        ),
        "gate_c_authorized": False,
        "training_performed": False,
    }
    # Gate C remains a separate explicit decision even when Gate B passes.
    (output_dir / "gate_b1_fresh_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


__all__ = [
    "run_gate_a", "run_gate_b0_and_freeze", "run_all_development_gates",
    "frozen_predict", "run_gate_b1_fresh",
]
