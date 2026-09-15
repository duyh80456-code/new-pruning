"""Four-fold held-out variance validation using saved per-batch gradient Grams."""

from __future__ import annotations

import json
import shutil
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rq2_anchor_placement import _sha256
from rq2_gradient_variance_v3 import (
    INTERIOR_SLOTS,
    NUM_INTERIORS,
    capped_neyman_allocation,
    importance_corrected_variance,
)
from rq2_probabilistic_support import INTERIOR_WIDTHS, maximum_entropy_pairs, pair_marginals


NUM_BATCHES = 16
NUM_FOLDS = 4
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260916


def _policy_inputs(interaction_root: Path):
    theory_root = interaction_root / "frozen_policy" / "theory-allocation-probe"
    preview_root = interaction_root / "frozen_policy" / "probabilistic-support-preview"
    theory = pd.read_csv(theory_root / "policy_family_marginals.csv")
    geometry = theory.loc[np.isclose(theory.p, 1.0)].sort_values("width")
    resource = pd.read_csv(preview_root / "support_allocation_marginals.csv").sort_values("width")
    if tuple(geometry.width.round(2)) != INTERIOR_WIDTHS or tuple(
        resource.width.round(2)
    ) != INTERIOR_WIDTHS:
        raise RuntimeError("Frozen policies do not cover the registered interior grid")
    uniform_pi = np.full(NUM_INTERIORS, INTERIOR_SLOTS / NUM_INTERIORS)
    uniform_pairs = pd.DataFrame([
        {
            "pair_index": index,
            "width_i": INTERIOR_WIDTHS[i],
            "width_j": INTERIOR_WIDTHS[j],
            "probability": 1.0 / len(tuple(combinations(range(NUM_INTERIORS), 2))),
        }
        for index, (i, j) in enumerate(combinations(range(NUM_INTERIORS), 2))
    ])
    policies = {
        "uniform_dynamic": (uniform_pi, uniform_pairs),
        "resource_dynamic": (
            resource.pi_resource_matched_compute.to_numpy(float),
            pd.read_csv(preview_root / "resource_pair_distribution.csv"),
        ),
        "geometry_dynamic": (
            geometry.pi.to_numpy(float),
            pd.read_csv(theory_root / "pair_distribution_p100.csv"),
        ),
    }
    for name, (pi, pairs) in policies.items():
        if abs(pi.sum() - 2) > 1e-8 or np.max(np.abs(pair_marginals(pairs) - pi)) > 1e-6:
            raise RuntimeError(f"Invalid frozen {name} policy")
    return policies, geometry, resource


def _save_policy(output_dir: Path, name: str, pi: np.ndarray, pairs: pd.DataFrame) -> None:
    pd.DataFrame({"width": INTERIOR_WIDTHS, "pi": pi}).to_csv(
        output_dir / f"policy_{name}.csv", index=False
    )
    pairs.to_csv(output_dir / f"pair_policy_{name}.csv", index=False)


def _bridge_diagnostic(
    grams: np.ndarray, folds: np.ndarray, geometry: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    a = geometry.functional_cell_mass.to_numpy(float)
    flops = geometry.flops.to_numpy(float)
    for fold in range(NUM_FOLDS):
        train_m = np.diagonal(grams[folds != fold], axis1=1, axis2=2).mean(axis=0)
        test_m = np.diagonal(grams[folds == fold], axis1=1, axis2=2).mean(axis=0)
        pearson = pearsonr(a, test_m)
        spearman = spearmanr(a, test_m)
        rows.extend([
            {"fold": fold, "analysis": "a_i vs heldout m_i", "metric": "Pearson",
             "value": pearson.statistic, "pvalue": pearson.pvalue},
            {"fold": fold, "analysis": "a_i vs heldout m_i", "metric": "Spearman",
             "value": spearman.statistic, "pvalue": spearman.pvalue},
        ])
        for name, X in (
            ("FLOPs", flops[:, None]),
            ("FLOPs + geometry", np.column_stack([flops, a])),
        ):
            model = make_pipeline(StandardScaler(), LinearRegression()).fit(X, train_m)
            prediction = model.predict(X)
            rows.extend([
                {"fold": fold, "analysis": f"train-m fit / heldout-m prediction: {name}",
                 "metric": "MAE", "value": mean_absolute_error(test_m, prediction), "pvalue": np.nan},
                {"fold": fold, "analysis": f"train-m fit / heldout-m prediction: {name}",
                 "metric": "R2", "value": r2_score(test_m, prediction), "pvalue": np.nan},
            ])
    return pd.DataFrame(rows)


def run_four_fold_cross_batch_validation(
    interaction_root: str | Path,
    output_dir: str | Path,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict:
    interaction_root, output_dir = Path(interaction_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_metadata = json.loads((interaction_root / "metadata.json").read_text())
    if (
        source_metadata.get("training_performed") is not False
        or source_metadata.get("test_used") is not False
        or source_metadata.get("per_batch_interior_grams_saved") is not True
    ):
        raise RuntimeError("Input must be a read-only probe containing per-batch Gram matrices")
    grams_path = interaction_root / "gram_matrices.npy"
    grams = np.asarray(np.load(grams_path), dtype=np.float64)
    if grams.shape != (NUM_BATCHES, NUM_INTERIORS, NUM_INTERIORS):
        raise RuntimeError(f"Expected Gram tensor (16,14,14), got {grams.shape}")
    symmetry_error = float(np.max(np.abs(grams - grams.transpose(0, 2, 1))))
    scale = max(float(np.max(np.abs(grams))), 1e-30)
    symmetry_relative_error = symmetry_error / scale
    if not np.isfinite(grams).all() or symmetry_relative_error > 1e-4:
        raise RuntimeError("Per-batch Gram tensor is non-finite or asymmetric")
    # The source GEMM was FP32, so C_ij and C_ji can differ at reduction noise
    # scale. All downstream exact quadratic forms use the symmetric float64 view.
    grams = 0.5 * (grams + grams.transpose(0, 2, 1))
    if np.any(np.diagonal(grams, axis1=1, axis2=2) <= 0):
        raise RuntimeError("Per-batch Gram diagonal must be strictly positive")
    folds = np.arange(NUM_BATCHES, dtype=int) % NUM_FOLDS
    assignment = pd.DataFrame({"batch_id": range(NUM_BATCHES), "fold": folds})
    assignment.to_csv(output_dir / "fold_assignment.csv", index=False)
    shutil.copy2(grams_path, output_dir / "gram_matrices.npy")

    frozen, geometry, _ = _policy_inputs(interaction_root)
    policy_file_labels = {
        "uniform_dynamic": "uniform",
        "resource_dynamic": "resource",
        "geometry_dynamic": "geometry",
    }
    for name, (pi, pairs) in frozen.items():
        _save_policy(output_dir, policy_file_labels[name], pi, pairs)
    weights = np.full(NUM_INTERIORS, 1.0 / NUM_INTERIORS)
    batch_rows, fold_rows, oracle_frames = [], [], []
    for fold in range(NUM_FOLDS):
        train_mask, heldout_mask = folds != fold, folds == fold
        train_m = np.diagonal(grams[train_mask], axis1=1, axis2=2).mean(axis=0)
        oracle_pi = capped_neyman_allocation(train_m, weights)
        oracle_pairs = maximum_entropy_pairs(oracle_pi)[0]
        oracle_frame = pd.DataFrame({
            "fold": fold, "width": INTERIOR_WIDTHS,
            "train_gradient_second_moment": train_m,
            "pi": oracle_pi,
        })
        oracle_frame.to_csv(output_dir / f"oracle_policy_fold{fold}.csv", index=False)
        oracle_pairs.to_csv(output_dir / f"oracle_pair_policy_fold{fold}.csv", index=False)
        oracle_frames.append(oracle_frame)
        policies = {**frozen, "gradient_oracle_marginal": (oracle_pi, oracle_pairs)}
        heldout_batch_rows = []
        for batch_id in np.flatnonzero(heldout_mask):
            values = {}
            for name, (pi, pairs) in policies.items():
                values[name] = importance_corrected_variance(
                    grams[batch_id], weights, pi, pairs
                )["importance_corrected_variance"]
            row = {
                "batch_id": int(batch_id), "fold": fold,
                "V_uniform": values["uniform_dynamic"],
                "V_resource": values["resource_dynamic"],
                "V_geometry": values["geometry_dynamic"],
                "V_oracle_crossfit": values["gradient_oracle_marginal"],
                "delta_geo_resource": values["geometry_dynamic"] - values["resource_dynamic"],
            }
            batch_rows.append(row); heldout_batch_rows.append(row)
        heldout = pd.DataFrame(heldout_batch_rows)
        fold_rows.append({
            "fold": fold,
            "train_probe_batches": ",".join(map(str, np.flatnonzero(train_mask))),
            "heldout_batches": ",".join(map(str, np.flatnonzero(heldout_mask))),
            "Uniform": heldout.V_uniform.mean(),
            "Resource": heldout.V_resource.mean(),
            "Geometry": heldout.V_geometry.mean(),
            "Oracle": heldout.V_oracle_crossfit.mean(),
            "r_G_over_U": heldout.V_geometry.mean() / heldout.V_uniform.mean(),
            "r_R_over_U": heldout.V_resource.mean() / heldout.V_uniform.mean(),
            "r_G_over_R": heldout.V_geometry.mean() / heldout.V_resource.mean(),
            "r_O_over_U": heldout.V_oracle_crossfit.mean() / heldout.V_uniform.mean(),
            "delta_V_G_minus_R": heldout.delta_geo_resource.mean(),
            "G_lt_R_lt_U": bool(
                heldout.V_geometry.mean() < heldout.V_resource.mean() < heldout.V_uniform.mean()
            ),
        })
    batch = pd.DataFrame(batch_rows).sort_values("batch_id")
    fold_table = pd.DataFrame(fold_rows)
    batch.to_csv(output_dir / "batch_variance.csv", index=False)
    fold_table.to_csv(output_dir / "fold_variance.csv", index=False)
    oracle_all = pd.concat(oracle_frames, ignore_index=True)
    stability = oracle_all.groupby("width", as_index=False).agg(
        oracle_pi_mean=("pi", "mean"), oracle_pi_std=("pi", "std"),
        oracle_pi_min=("pi", "min"), oracle_pi_max=("pi", "max"),
    )
    stability["oracle_pi_cv"] = stability.oracle_pi_std / stability.oracle_pi_mean
    stability.to_csv(output_dir / "oracle_marginal_stability.csv", index=False)

    policy_columns = {
        "Uniform-Dynamic": "V_uniform", "Resource-Dynamic": "V_resource",
        "Geometry-Dynamic": "V_geometry", "Gradient-Oracle-Marginal": "V_oracle_crossfit",
    }
    summary_rows = []
    for policy, column in policy_columns.items():
        values = batch[column].to_numpy(float)
        summary_rows.append({
            "policy": policy, "mean_variance": values.mean(), "std": values.std(ddof=1),
            "se": values.std(ddof=1) / np.sqrt(len(values)),
            "ratio_vs_uniform": values.mean() / batch.V_uniform.mean(),
        })
    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(output_dir / "summary.csv", index=False)

    delta = batch.delta_geo_resource.to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = int(bootstrap_draws)
    if draws < 1000:
        raise ValueError("bootstrap_draws must be at least 1000")
    bootstrap = delta[rng.integers(0, len(delta), size=(draws, len(delta)))].mean(axis=1)
    np.save(output_dir / "bootstrap_geo_minus_resource.npy", bootstrap)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
    uncertainty = pd.DataFrame([{
        "comparison": "Geometry minus Resource", "mean_delta": delta.mean(),
        "std": delta.std(ddof=1), "se": delta.std(ddof=1) / np.sqrt(len(delta)),
        "geo_below_resource_batches": int((delta < 0).sum()), "num_batches": len(delta),
        "bootstrap_draws": draws, "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_ci_2_5": ci_low, "bootstrap_ci_97_5": ci_high,
    }])
    uncertainty.to_csv(output_dir / "geo_resource_paired_uncertainty.csv", index=False)
    bridge = _bridge_diagnostic(grams, folds, geometry)
    bridge.to_csv(output_dir / "heldout_geometry_moment_bridge.csv", index=False)

    fig, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(NUM_FOLDS); bar_width = 0.2
    for index, column in enumerate(("Uniform", "Resource", "Geometry", "Oracle")):
        axis.bar(x + (index - 1.5) * bar_width, fold_table[column], bar_width, label=column)
    axis.set(xticks=x, xlabel="Held-out fold", ylabel="Mean exact HT variance")
    axis.grid(axis="y", alpha=0.25); axis.legend(); fig.tight_layout()
    fig.savefig(output_dir / "variance_by_fold.png", dpi=200); plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4.5))
    colors = np.where(delta < 0, "tab:green", "tab:red")
    axis.bar(batch.batch_id, delta, color=colors); axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="Batch ID", ylabel="V_G - V_R")
    axis.grid(axis="y", alpha=0.25); fig.tight_layout()
    fig.savefig(output_dir / "geo_vs_resource_batch_delta.png", dpi=200); plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4.8))
    for fold, group in oracle_all.groupby("fold"):
        axis.plot(group.width, group.pi, marker="o", label=f"fold {fold}")
    axis.set(xlabel="Interior width", ylabel="Cross-fit oracle marginal pi")
    axis.grid(alpha=0.25); axis.legend(); fig.tight_layout()
    fig.savefig(output_dir / "oracle_marginal_stability.png", dpi=200); plt.close(fig)

    ordering_count = int(fold_table.G_lt_R_lt_U.sum())
    strong_go = bool(ordering_count >= 3 and ci_high < 0)
    weak_go = bool(not strong_go and delta.mean() < 0)
    decision = "STRONG_GO" if strong_go else ("WEAK_GO" if weak_go else "NO_GO")
    metadata = {
        "experiment": "v3_cross_batch_heldout_variance",
        "status": "COMPLETE",
        "decision": decision,
        "checkpoint": "uniform_seed3_epoch100",
        "network_training": False, "accuracy_used": False, "test_used": False,
        "num_batches": NUM_BATCHES, "batch_size": 128, "num_folds": NUM_FOLDS,
        "fold_rule": "fold_id = batch_idx % 4",
        "interior_widths": list(INTERIOR_WIDTHS), "slots": 2,
        "target_weights": "uniform_dense_interior (w_i=1/14)",
        "estimator": "Horvitz-Thompson exact enumeration over 91 pairs",
        "geometry_policy_frozen": True, "resource_policy_frozen": True,
        "oracle_fit_only_on_training_folds": True,
        "pair_policy": "maximum_entropy_given_marginals",
        "gradient_oracle_scope": "marginal oracle; q is not optimized on held-out Grams",
        "bootstrap": {"paired_unit": "batch", "draws": draws, "seed": BOOTSTRAP_SEED,
                      "ci_95": [float(ci_low), float(ci_high)]},
        "mean_delta_V_geometry_minus_resource": float(delta.mean()),
        "geo_below_resource_batch_count": int((delta < 0).sum()),
        "G_lt_R_lt_U_fold_count": ordering_count,
        "strong_go_rule": "G<R<U in >=3/4 folds and paired bootstrap CI upper bound < 0",
        "weak_go_rule": "mean(V_G-V_R)<0 but strong rule not met",
        "gram_matrices_sha256": _sha256(grams_path),
        "gram_symmetry_max_relative_error_before_symmetrization": symmetry_relative_error,
        "source_metadata_sha256": _sha256(interaction_root / "metadata.json"),
        "interpretation": "held-out mechanistic variance gate; not an accuracy or training result",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
