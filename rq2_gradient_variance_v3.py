"""CPU-only RQ2-v3 bridge: geometry, gradient moments, and sampler variance."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rq2_anchor_placement import GRID, _sha256
from rq2_probabilistic_support import INTERIOR_WIDTHS, maximum_entropy_pairs, pair_marginals


NUM_INTERIORS = len(INTERIOR_WIDTHS)
INTERIOR_SLOTS = 2.0


def _safe_extract(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    resolved = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != resolved and resolved not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)
    return destination


def find_interaction_probe_root(
    input_root: str | Path, materialized_root: str | Path
) -> Path:
    """Find exactly one completed merged interaction probe, extracting its ZIP if needed."""
    input_root, materialized_root = Path(input_root), Path(materialized_root)

    def candidates(root: Path) -> list[Path]:
        found = []
        for metadata_path in root.rglob("metadata.json"):
            try:
                metadata = json.loads(metadata_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            candidate = metadata_path.parent
            required = (
                candidate / "gradient_dot_matrix.csv",
                candidate / "gradient_norms_by_batch.csv",
                candidate / "policy_expected_effect.csv",
                candidate / "frozen_policy" / "theory-allocation-probe" / "policy_family_marginals.csv",
                candidate / "frozen_policy" / "probabilistic-support-preview" / "support_allocation_marginals.csv",
            )
            if (
                metadata.get("status") == "POST_HOC_MECHANISTIC_DIAGNOSTIC_COMPLETE"
                and metadata.get("execution") == "two_gpu_disjoint_batch_shards"
                and all(path.is_file() for path in required)
            ):
                found.append(candidate)
        return sorted(set(found))

    found = candidates(input_root)
    if not found:
        archives = sorted(input_root.rglob("rq2-cross-subnet-interaction.zip"))
        if len(archives) == 1:
            found = candidates(_safe_extract(archives[0], materialized_root))
    if len(found) != 1:
        raise FileNotFoundError(f"Expected exactly one completed interaction probe, found: {found}")
    return found[0]


def _read_square_matrix(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    if tuple(frame["width"].round(2)) != GRID:
        raise RuntimeError(f"Unexpected width order in {path}")
    matrix = frame.drop(columns="width").to_numpy(float)
    if matrix.shape != (len(GRID), len(GRID)) or not np.isfinite(matrix).all():
        raise RuntimeError(f"Invalid interaction matrix in {path}")
    return matrix


def capped_neyman_allocation(
    second_moment: np.ndarray,
    target_weights: np.ndarray,
    slots: float = INTERIOR_SLOTS,
) -> np.ndarray:
    """Solve pi_i proportional to w_i sqrt(m_i), respecting pi_i <= 1."""
    second_moment = np.asarray(second_moment, float)
    target_weights = np.asarray(target_weights, float)
    if (
        second_moment.shape != (NUM_INTERIORS,)
        or target_weights.shape != (NUM_INTERIORS,)
        or np.any(second_moment <= 0)
        or np.any(target_weights <= 0)
    ):
        raise ValueError("Neyman allocation requires aligned positive moments and weights")
    score = target_weights * np.sqrt(second_moment)
    remaining = np.ones(NUM_INTERIORS, dtype=bool)
    pi = np.zeros(NUM_INTERIORS)
    budget = float(slots)
    while remaining.any():
        proposal = budget * score[remaining] / score[remaining].sum()
        active_indices = np.flatnonzero(remaining)
        saturated = proposal >= 1.0
        if not saturated.any():
            pi[active_indices] = proposal
            break
        pi[active_indices[saturated]] = 1.0
        budget -= float(saturated.sum())
        remaining[active_indices[saturated]] = False
        if budget <= 0:
            break
    if abs(pi.sum() - slots) > 1e-9 or np.any(pi <= 0) or np.any(pi > 1 + 1e-10):
        raise RuntimeError("Invalid capped Neyman allocation")
    return pi


def importance_corrected_variance(
    dot_matrix: np.ndarray,
    target_weights: np.ndarray,
    pi: np.ndarray,
    pair_table: pd.DataFrame,
) -> dict:
    """Empirical E_b Var_pair[sum_{i in pair} w_i g_i/pi_i]."""
    dot_matrix = np.asarray(dot_matrix, float)
    target_weights = np.asarray(target_weights, float)
    pi = np.asarray(pi, float)
    if dot_matrix.shape != (NUM_INTERIORS, NUM_INTERIORS):
        raise ValueError("dot_matrix must cover the 14 interior widths")
    if target_weights.shape != pi.shape or np.any(pi <= 0):
        raise ValueError("Positive marginals and aligned target weights are required")
    reconstructed = pair_marginals(pair_table)
    if np.max(np.abs(reconstructed - pi)) > 1e-6:
        raise RuntimeError("Pair distribution does not reproduce supplied marginals")
    scaled = target_weights / pi
    first = float(np.sum(np.square(target_weights) * np.diag(dot_matrix) / pi))
    cross = 0.0
    width_index = {width: index for index, width in enumerate(INTERIOR_WIDTHS)}
    for row in pair_table.itertuples():
        i = width_index[round(float(row.width_i), 2)]
        j = width_index[round(float(row.width_j), 2)]
        cross += 2.0 * float(row.probability) * scaled[i] * scaled[j] * dot_matrix[i, j]
    estimator_second_moment = first + cross
    target_second_moment = float(target_weights @ dot_matrix @ target_weights)
    variance = estimator_second_moment - target_second_moment
    tolerance = 1e-8 * max(abs(estimator_second_moment), 1.0)
    if variance < -tolerance:
        raise FloatingPointError(f"Negative estimator variance beyond tolerance: {variance}")
    return {
        "estimator_second_moment": estimator_second_moment,
        "target_gradient_second_moment": target_second_moment,
        "importance_corrected_variance": max(0.0, variance),
    }


def _regression_rows(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = table["gradient_second_moment"].to_numpy(float)
    definitions = {
        "FLOPs": ["flops"],
        "FLOPs + geometry": ["flops", "functional_mass_a_i"],
    }
    rows, prediction_rows = [], []
    loo = LeaveOneOut()
    for name, columns in definitions.items():
        X = table[columns].to_numpy(float)
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(X, y)
        fitted = model.predict(X)
        predicted = cross_val_predict(model, X, y, cv=loo)
        linear = model.named_steps["linearregression"]
        rows.append({
            "model": name,
            "predictors": " + ".join(columns),
            "in_sample_r2": r2_score(y, fitted),
            "loow_mae": mean_absolute_error(y, predicted),
            "loow_r2": r2_score(y, predicted),
            "standardized_flops_coefficient": float(linear.coef_[0]),
            "standardized_geometry_coefficient": (
                float(linear.coef_[1]) if len(columns) == 2 else np.nan
            ),
        })
        for width, observed, estimate in zip(table["width"], y, predicted):
            prediction_rows.append({
                "model": name, "width": width,
                "observed_gradient_second_moment": observed,
                "loow_prediction": estimate,
                "loow_residual": observed - estimate,
            })
    regression = pd.DataFrame(rows)
    regression["delta_in_sample_r2_vs_flops"] = (
        regression["in_sample_r2"] - regression.loc[regression.model.eq("FLOPs"), "in_sample_r2"].iloc[0]
    )
    regression["delta_loow_mae_vs_flops"] = (
        regression["loow_mae"] - regression.loc[regression.model.eq("FLOPs"), "loow_mae"].iloc[0]
    )

    # Classical nested-model F test is descriptive here (n=14); LOOW remains the robustness check.
    reduced = LinearRegression().fit(table[["flops"]], y)
    full = LinearRegression().fit(table[["flops", "functional_mass_a_i"]], y)
    rss_reduced = float(np.square(y - reduced.predict(table[["flops"]])).sum())
    rss_full = float(np.square(y - full.predict(table[["flops", "functional_mass_a_i"]])).sum())
    df_denominator = len(y) - 3
    statistic = ((rss_reduced - rss_full) / 1) / (rss_full / df_denominator)
    regression["nested_f_statistic"] = np.nan
    regression["nested_f_pvalue"] = np.nan
    full_index = regression.index[regression.model.eq("FLOPs + geometry")][0]
    regression.loc[full_index, "nested_f_statistic"] = max(0.0, statistic)
    regression.loc[full_index, "nested_f_pvalue"] = float(
        f_distribution.sf(max(0.0, statistic), 1, df_denominator)
    )
    return regression, pd.DataFrame(prediction_rows)


def _plots(output_dir: Path, table: pd.DataFrame, variance: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(table["functional_mass_a_i"], table["gradient_second_moment"], s=50)
    axes[0].set(xlabel="Frozen functional mass a_i", ylabel="m_i = E||g_i||^2")
    axes[1].scatter(table["flops"], table["gradient_second_moment"], s=50)
    axes[1].set(xlabel="FLOPs", ylabel="m_i = E||g_i||^2")
    for axis in axes:
        axis.grid(alpha=0.25)
    for row in table.itertuples():
        axes[0].annotate(f"{row.width:.2f}", (row.functional_mass_a_i, row.gradient_second_moment))
        axes[1].annotate(f"{row.width:.2f}", (row.flops, row.gradient_second_moment))
    fig.tight_layout(); fig.savefig(output_dir / "geometry_flops_vs_gradient_moment.png", dpi=200); plt.close(fig)

    finite = variance.loc[np.isfinite(variance["importance_corrected_variance"])].copy()
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(finite["policy"], finite["variance_ratio_vs_uniform_dynamic"])
    axis.axhline(1, color="black", linewidth=1)
    axis.set(ylabel="Importance-corrected variance / Uniform-dynamic", xlabel="Policy")
    axis.tick_params(axis="x", rotation=20); axis.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(output_dir / "importance_corrected_variance.png", dpi=200); plt.close(fig)


def run_gradient_variance_v3(
    interaction_root: str | Path,
    output_dir: str | Path,
) -> dict:
    """Run the full CPU-only diagnostic without accuracy, model loading, or training."""
    interaction_root, output_dir = Path(interaction_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = interaction_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("training_performed") is not False
        or metadata.get("test_used") is not False
        or metadata.get("weights_unchanged") is not True
    ):
        raise RuntimeError("Interaction input does not satisfy the read-only protocol")

    theory_root = interaction_root / "frozen_policy" / "theory-allocation-probe"
    preview_root = interaction_root / "frozen_policy" / "probabilistic-support-preview"
    theory = pd.read_csv(theory_root / "policy_family_marginals.csv")
    geometry = theory.loc[np.isclose(theory["p"], 1.0)].sort_values("width")
    preview = pd.read_csv(preview_root / "support_allocation_marginals.csv").sort_values("width")
    if tuple(geometry["width"].round(2)) != INTERIOR_WIDTHS or tuple(
        preview["width"].round(2)
    ) != INTERIOR_WIDTHS:
        raise RuntimeError("Frozen policy tables do not cover the 14 interior widths")

    dot_full = _read_square_matrix(interaction_root / "gradient_dot_matrix.csv")
    interior_indices = [GRID.index(width) for width in INTERIOR_WIDTHS]
    dot = dot_full[np.ix_(interior_indices, interior_indices)]
    moments = np.diag(dot).copy()
    norm_rows = pd.read_csv(interaction_root / "gradient_norms_by_batch.csv")
    norm_rows["width"] = norm_rows["width"].round(2)
    direct = (
        norm_rows.loc[norm_rows.width.isin(INTERIOR_WIDTHS)]
        .assign(squared_norm=lambda frame: np.square(frame.gradient_norm))
        .groupby("width")["squared_norm"].mean().reindex(INTERIOR_WIDTHS).to_numpy(float)
    )
    diagonal_error = float(np.max(np.abs(moments - direct) / np.maximum(moments, 1e-30)))
    if diagonal_error > 1e-5 or np.any(moments <= 0):
        raise RuntimeError("Gradient norm second moments disagree with dot-matrix diagonal")

    table = pd.DataFrame({
        "width": INTERIOR_WIDTHS,
        "functional_mass_a_i": geometry["functional_cell_mass"].to_numpy(float),
        "flops": geometry["flops"].to_numpy(float),
        "gradient_second_moment": moments,
        "rms_gradient_norm": np.sqrt(direct),
        "pi_geometry": geometry["pi"].to_numpy(float),
        "pi_resource": preview["pi_resource_matched_compute"].to_numpy(float),
        "pi_uniform_dynamic": np.full(NUM_INTERIORS, INTERIOR_SLOTS / NUM_INTERIORS),
    })
    target_weights = np.full(NUM_INTERIORS, 1.0 / NUM_INTERIORS)
    table["target_weight_w_i"] = target_weights
    table["pi_gradient_oracle"] = capped_neyman_allocation(moments, target_weights)
    table.to_csv(output_dir / "gradient_second_moment_by_width.csv", index=False)

    pearson = pearsonr(table.functional_mass_a_i, table.gradient_second_moment)
    spearman = spearmanr(table.functional_mass_a_i, table.gradient_second_moment)
    flops_pearson = pearsonr(table.flops, table.gradient_second_moment)
    flops_spearman = spearmanr(table.flops, table.gradient_second_moment)
    flops_model = LinearRegression().fit(table[["flops"]], table.gradient_second_moment)
    resource_residual = table.gradient_second_moment - flops_model.predict(table[["flops"]])
    partial_pearson = pearsonr(table.functional_mass_a_i, resource_residual)
    partial_spearman = spearmanr(table.functional_mass_a_i, resource_residual)
    correlations = pd.DataFrame([
        {"analysis": "a_i vs m_i", "metric": "Pearson", "coefficient": pearson.statistic, "pvalue": pearson.pvalue},
        {"analysis": "a_i vs m_i", "metric": "Spearman", "coefficient": spearman.statistic, "pvalue": spearman.pvalue},
        {"analysis": "FLOPs_i vs m_i", "metric": "Pearson", "coefficient": flops_pearson.statistic, "pvalue": flops_pearson.pvalue},
        {"analysis": "FLOPs_i vs m_i", "metric": "Spearman", "coefficient": flops_spearman.statistic, "pvalue": flops_spearman.pvalue},
        {"analysis": "a_i vs residual(m_i ~ FLOPs)", "metric": "Pearson", "coefficient": partial_pearson.statistic, "pvalue": partial_pearson.pvalue},
        {"analysis": "a_i vs residual(m_i ~ FLOPs)", "metric": "Spearman", "coefficient": partial_spearman.statistic, "pvalue": partial_spearman.pvalue},
    ])
    correlations.to_csv(output_dir / "gradient_second_moment_correlations.csv", index=False)
    regression, predictions = _regression_rows(table)
    regression.to_csv(output_dir / "gradient_second_moment_regression.csv", index=False)
    predictions.to_csv(output_dir / "gradient_second_moment_loow_predictions.csv", index=False)

    policies = {}
    uniform_pi = table.pi_uniform_dynamic.to_numpy(float)
    policies["uniform_dynamic"] = (uniform_pi, maximum_entropy_pairs(uniform_pi)[0])
    geometry_pi = table.pi_geometry.to_numpy(float)
    policies["geometry"] = (
        geometry_pi, pd.read_csv(theory_root / "pair_distribution_p100.csv")
    )
    resource_pi = table.pi_resource.to_numpy(float)
    policies["resource"] = (
        resource_pi, pd.read_csv(preview_root / "resource_pair_distribution.csv")
    )
    oracle_pi = table.pi_gradient_oracle.to_numpy(float)
    policies["gradient_oracle_marginal"] = (
        oracle_pi, maximum_entropy_pairs(oracle_pi)[0]
    )
    variance_rows = []
    for name, (pi, pairs) in policies.items():
        values = importance_corrected_variance(dot, target_weights, pi, pairs)
        variance_rows.append({
            "policy": name,
            **values,
            "minimum_pi": float(pi.min()),
            "maximum_pi": float(pi.max()),
            "effective_pair_marginal_slots": float(pi.sum()),
            "pair_marginal_max_abs_error": float(np.max(np.abs(pair_marginals(pairs) - pi))),
        })
    # Fixed {.50,.75} cannot unbiasedly estimate the uniform 14-width target.
    variance_rows.append({
        "policy": "uniform_fixed_050_075",
        "estimator_second_moment": np.inf,
        "target_gradient_second_moment": float(target_weights @ dot @ target_weights),
        "importance_corrected_variance": np.inf,
        "minimum_pi": 0.0, "maximum_pi": 1.0,
        "effective_pair_marginal_slots": 2.0,
        "pair_marginal_max_abs_error": 0.0,
    })
    variance = pd.DataFrame(variance_rows)
    uniform_variance = variance.loc[
        variance.policy.eq("uniform_dynamic"), "importance_corrected_variance"
    ].iloc[0]
    oracle_variance = variance.loc[
        variance.policy.eq("gradient_oracle_marginal"), "importance_corrected_variance"
    ].iloc[0]
    variance["variance_ratio_vs_uniform_dynamic"] = (
        variance.importance_corrected_variance / uniform_variance
    )
    variance["excess_variance_over_gradient_oracle"] = (
        variance.importance_corrected_variance - oracle_variance
    )
    variance.to_csv(output_dir / "importance_corrected_estimator_variance.csv", index=False)
    _plots(output_dir, table, variance)

    flops_row = regression.loc[regression.model.eq("FLOPs")].iloc[0]
    combined_row = regression.loc[regression.model.eq("FLOPs + geometry")].iloc[0]
    geometry_variance = variance.loc[
        variance.policy.eq("geometry"), "importance_corrected_variance"
    ].iloc[0]
    resource_variance = variance.loc[
        variance.policy.eq("resource"), "importance_corrected_variance"
    ].iloc[0]
    summary = {
        "status": "RQ2_V3_CPU_DIAGNOSTIC_COMPLETE",
        "training_performed": False,
        "model_or_checkpoint_loaded": False,
        "gpu_used": False,
        "test_used": False,
        "accuracy_used": False,
        "interaction_root": str(interaction_root),
        "input_metadata_sha256": _sha256(metadata_path),
        "num_gradient_batches": int(metadata["num_fixed_training_batches"]),
        "target_definition": "uniform mean gradient over the 14 interior dense-grid widths",
        "target_weights": "w_i = 1/14 for every interior width",
        "gradient_oracle_definition": "pi_i proportional to w_i * sqrt(m_i), capped at one, sum pi=2",
        "pair_realization": "maximum entropy subject to each policy's marginals",
        "fixed_uniform_note": "Static {.50,.75} has zero inclusion elsewhere, so its unbiased importance-corrected variance is infinite/undefined",
        "dot_diagonal_vs_squared_norm_max_relative_error": diagonal_error,
        "geometry_predicts_beyond_flops_in_sample": bool(
            combined_row.in_sample_r2 > flops_row.in_sample_r2
        ),
        "geometry_improves_loow_mae_beyond_flops": bool(
            combined_row.loow_mae < flops_row.loow_mae
        ),
        "geometry_variance_below_resource": bool(geometry_variance < resource_variance),
        "geometry_variance_ratio_vs_resource": float(geometry_variance / resource_variance),
        "geometry_variance_ratio_vs_uniform_dynamic": float(geometry_variance / uniform_variance),
        "gradient_oracle_variance_ratio_vs_uniform_dynamic": float(oracle_variance / uniform_variance),
        "interpretation": "descriptive offline diagnostic; not a training result or causal proof",
    }
    (output_dir / "rq2_v3_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary
