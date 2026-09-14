"""CPU-only geometry/exposure decomposition of the existing RQ2-v1 result."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut

from rq2_anchor_placement import GRID, UNIFORM_ANCHORS
from rq2_parameter_exposure import (
    ALL_GROUPS,
    ELASTIC_GROUPS,
    activation_bands,
    exact_active_parameter_counts,
    exposure_count,
)


def _anchor_text(anchors) -> str:
    return ",".join(f"{float(width):.2f}" for width in anchors)


def _load_geometry_coordinate(root: Path) -> dict[float, float]:
    path = root / "protocol" / "geometry_trajectory_coordinates.csv"
    frame = pd.read_csv(path, usecols=["width", "geometry_coordinate"]).sort_values("width")
    if tuple(frame["width"].round(2)) != GRID:
        raise RuntimeError("Frozen seed-0 geometry coordinate does not cover the registered grid")
    values = frame["geometry_coordinate"].to_numpy(float)
    if not np.isfinite(values).all() or np.any(np.diff(values) < -1e-12):
        raise RuntimeError("Frozen geometry coordinate is non-finite or non-monotone")
    return dict(zip(GRID, values))


def _geometry_distance(width: float, anchors, coordinate: dict[float, float]) -> float:
    return float(min(abs(coordinate[width] - coordinate[float(anchor)]) for anchor in anchors))


def exposure_deficit_by_width(
    bands: pd.DataFrame,
    uniform_anchors,
    geo_anchors,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute D_E(w) exactly over parameters active in each subnet w."""
    summary_rows, detail_rows = [], []
    for width in GRID:
        active = bands.loc[bands["activation_width"].le(width + 1e-12)].copy()
        if active.empty:
            raise RuntimeError(f"No active parameters found at width {width}")
        for row in active.itertuples():
            count = int(row.newly_active_parameters)
            uniform_exposure = exposure_count(uniform_anchors, row.activation_width)
            geo_exposure = exposure_count(geo_anchors, row.activation_width)
            deficit_per_parameter = max(0, uniform_exposure - geo_exposure)
            detail_rows.append({
                "width": width,
                "parameter_group": row.parameter_group,
                "activation_width": float(row.activation_width),
                "active_band_parameters": count,
                "uniform_exposure_count": uniform_exposure,
                "puregeo_exposure_count": geo_exposure,
                "exposure_count_delta_geo_minus_uniform": geo_exposure - uniform_exposure,
                "deficit_per_parameter": deficit_per_parameter,
                "deficit_contribution": count * deficit_per_parameter,
                "is_underexposed_band": bool(deficit_per_parameter > 0),
            })
        total_parameters = int(active["newly_active_parameters"].sum())
        elastic = active.loc[active["parameter_group"].isin(ELASTIC_GROUPS)]
        elastic_parameters = int(elastic["newly_active_parameters"].sum())
        detail = pd.DataFrame(detail_rows)
        current = detail.loc[np.isclose(detail["width"], width)]
        deficit = int(current["deficit_contribution"].sum())
        elastic_deficit = int(current.loc[
            current["parameter_group"].isin(ELASTIC_GROUPS), "deficit_contribution"
        ].sum())
        affected = int(current.loc[current["is_underexposed_band"], "active_band_parameters"].sum())
        summary_rows.append({
            "width": width,
            "active_parameters": total_parameters,
            "active_elastic_parameters": elastic_parameters,
            "underexposed_active_parameters": affected,
            "lost_exposure_updates": deficit,
            "D_E": deficit / total_parameters,
            "elastic_D_E": elastic_deficit / elastic_parameters,
            "fraction_active_parameters_underexposed": affected / total_parameters,
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def build_per_width_table(root: str | Path):
    root = Path(root)
    config = yaml.safe_load((root / "resolved_config.yaml").read_text())
    geo_anchors = tuple(map(float, json.loads(
        (root / "protocol" / "selected_anchors.json").read_text()
    )["selected_anchors"]))
    if len(geo_anchors) != 4 or geo_anchors[0] != 0.25 or geo_anchors[-1] != 1.0:
        raise RuntimeError(f"Invalid frozen PureGeo anchors: {geo_anchors}")
    coordinate = _load_geometry_coordinate(root)
    active_counts = exact_active_parameter_counts(config)
    bands = activation_bands(active_counts)
    exposure, detail = exposure_deficit_by_width(bands, UNIFORM_ANCHORS, geo_anchors)

    dense_path = root / "rq2_dense_metrics_all.csv"
    dense = pd.read_csv(
        dense_path, usecols=["method", "seed", "budget", "accuracy", "flops", "params"]
    )
    dense = dense.loc[
        dense["method"].isin(["uniform", "geo"])
        & dense["seed"].astype(int).isin([1, 2])
    ].copy()
    counts = dense.groupby(["seed", "method"])["budget"].nunique()
    if len(counts) != 4 or not (counts == len(GRID)).all():
        raise RuntimeError("Expected complete 16-width Uniform/PureGeo metrics for seeds 1 and 2")
    accuracy = dense.pivot(index=["seed", "budget"], columns="method", values="accuracy").reset_index()
    resources = dense.loc[dense["method"].eq("uniform"), ["seed", "budget", "flops", "params"]]
    table = accuracy.merge(resources, on=["seed", "budget"], validate="one_to_one")
    table = table.rename(columns={
        "budget": "width", "uniform": "uniform_accuracy", "geo": "puregeo_accuracy"
    })
    table["delta_accuracy"] = table["puregeo_accuracy"] - table["uniform_accuracy"]
    table["log_flops"] = np.log(table["flops"].to_numpy(float))
    table["normalized_log_flops"] = (
        table["log_flops"] - table["log_flops"].min()
    ) / (table["log_flops"].max() - table["log_flops"].min())
    table["d_G_uniform"] = table["width"].map(
        lambda width: _geometry_distance(round(float(width), 2), UNIFORM_ANCHORS, coordinate)
    )
    table["d_G_puregeo"] = table["width"].map(
        lambda width: _geometry_distance(round(float(width), 2), geo_anchors, coordinate)
    )
    table["B_G"] = table["d_G_uniform"] - table["d_G_puregeo"]
    table = table.merge(exposure, on="width", validate="many_to_one")
    uniform_set, geo_set = set(UNIFORM_ANCHORS), set(geo_anchors)
    common_holdout = sorted(set(GRID) - uniform_set - geo_set)
    table["is_uniform_anchor"] = table["width"].round(2).isin(uniform_set)
    table["is_puregeo_anchor"] = table["width"].round(2).isin(geo_set)
    table["is_common_holdout"] = table["width"].round(2).isin(common_holdout)
    table = table.sort_values(["seed", "width"]).reset_index(drop=True)
    return table, detail, bands, geo_anchors, common_holdout


def fit_decomposition(table: pd.DataFrame):
    feature_sets = {"B_G": ["B_G"], "D_E": ["D_E"], "B_G+D_E": ["B_G", "D_E"]}
    rows, prediction_rows = [], []
    for seed in (1, 2):
        subset = table.loc[table["seed"].eq(seed) & table["is_common_holdout"]].copy()
        y = subset["delta_accuracy"].to_numpy(float)
        for name, features in feature_sets.items():
            X = subset[features].to_numpy(float)
            model = LinearRegression().fit(X, y)
            fitted = model.predict(X)
            loow = np.empty_like(y)
            for train, test in LeaveOneOut().split(X):
                fold_model = LinearRegression().fit(X[train], y[train])
                loow[test] = fold_model.predict(X[test])
            y_std = float(np.std(y, ddof=0))
            standardized = {
                feature: float(coef * np.std(X[:, index], ddof=0) / y_std) if y_std > 0 else np.nan
                for index, (feature, coef) in enumerate(zip(features, model.coef_))
            }
            singular = np.linalg.svd(
                np.column_stack([np.ones(len(X)), (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)]),
                compute_uv=False,
            )
            condition_number = float("inf") if singular.min() <= 1e-12 else float(
                singular.max() / singular.min()
            )
            rows.append({
                "seed": seed, "model": name, "predictors": "+".join(features),
                "n_widths": len(subset), "intercept": float(model.intercept_),
                "beta_G": float(model.coef_[features.index("B_G")]) if "B_G" in features else np.nan,
                "beta_E": float(model.coef_[features.index("D_E")]) if "D_E" in features else np.nan,
                "standardized_beta_G": standardized.get("B_G", np.nan),
                "standardized_beta_E": standardized.get("D_E", np.nan),
                "in_sample_mae": float(mean_absolute_error(y, fitted)),
                "in_sample_r2": float(r2_score(y, fitted)),
                "loow_mae": float(mean_absolute_error(y, loow)),
                "loow_r2": float(r2_score(y, loow)),
                "design_condition_number": condition_number,
                "beta_G_positive": bool(model.coef_[features.index("B_G")] > 0) if "B_G" in features else False,
                "beta_E_negative": bool(model.coef_[features.index("D_E")] < 0) if "D_E" in features else False,
            })
            for index, source in enumerate(subset.itertuples()):
                prediction_rows.append({
                    "seed": seed, "width": float(source.width), "model": name,
                    "observed_delta_accuracy": float(y[index]),
                    "fitted_delta_accuracy": float(fitted[index]),
                    "loow_predicted_delta_accuracy": float(loow[index]),
                    "residual": float(y[index] - fitted[index]),
                })
    return pd.DataFrame(rows), pd.DataFrame(prediction_rows)


def exposure_proxy_correlations(table: pd.DataFrame) -> pd.DataFrame:
    unique = table.drop_duplicates("width").sort_values("width")
    proxies = ["width", "flops", "log_flops", "normalized_log_flops", "params", "active_parameters"]
    rows = []
    for population, subset in (
        ("dense_grid", unique),
        ("common_holdout", unique.loc[unique["is_common_holdout"]]),
    ):
        for proxy in proxies:
            pearson = pearsonr(subset["D_E"], subset[proxy])
            spearman = spearmanr(subset["D_E"], subset[proxy])
            rows.append({
                "population": population, "proxy": proxy, "n_widths": len(subset),
                "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
            })
    return pd.DataFrame(rows)


def _scatter(table, x, x_label, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True)
    for axis, seed in zip(axes, (1, 2)):
        subset = table.loc[table["seed"].eq(seed) & table["is_common_holdout"]].copy()
        axis.scatter(subset[x], subset["delta_accuracy"], s=48, color="tab:blue")
        model = LinearRegression().fit(subset[[x]].to_numpy(float), subset["delta_accuracy"])
        positions = np.linspace(subset[x].min(), subset[x].max(), 100)
        axis.plot(positions, model.predict(positions[:, None]), color="black", linewidth=1.5)
        for row in subset.itertuples():
            axis.annotate(f"{row.width:.2f}", (getattr(row, x), row.delta_accuracy),
                          xytext=(4, 3), textcoords="offset points", fontsize=8)
        axis.axhline(0, color="grey", linewidth=0.8)
        axis.set_title(f"Seed {seed}")
        axis.set_xlabel(x_label)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("PureGeo − Uniform accuracy")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _two_factor_plot(predictions: pd.DataFrame, output: Path) -> None:
    data = predictions.loc[predictions["model"].eq("B_G+D_E")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), sharex=True, sharey=True)
    low = min(data["observed_delta_accuracy"].min(), data["fitted_delta_accuracy"].min())
    high = max(data["observed_delta_accuracy"].max(), data["fitted_delta_accuracy"].max())
    margin = max((high - low) * 0.08, 1e-4)
    for axis, seed in zip(axes, (1, 2)):
        subset = data.loc[data["seed"].eq(seed)]
        axis.scatter(subset["observed_delta_accuracy"], subset["fitted_delta_accuracy"], s=50)
        axis.plot([low - margin, high + margin], [low - margin, high + margin], "k--", linewidth=1)
        for row in subset.itertuples():
            axis.annotate(f"{row.width:.2f}",
                          (row.observed_delta_accuracy, row.fitted_delta_accuracy),
                          xytext=(4, 3), textcoords="offset points", fontsize=8)
        axis.set_title(f"Seed {seed}")
        axis.set_xlabel("Observed accuracy delta")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Fitted accuracy delta")
    fig.suptitle("Two-factor fit on common holdout")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _band_plot(detail: pd.DataFrame, output: Path) -> None:
    values = detail.groupby(["width", "activation_width"], as_index=False)["deficit_contribution"].sum()
    pivot = values.pivot(index="width", columns="activation_width", values="deficit_contribution").fillna(0)
    fig, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(np.log10(pivot.to_numpy(float) + 1), aspect="auto", cmap="magma")
    axis.set_yticks(range(len(pivot.index)), [f"{value:.2f}" for value in pivot.index])
    axis.set_xticks(range(len(pivot.columns)), [f"{value:.2f}" for value in pivot.columns], rotation=45)
    axis.set_ylabel("Evaluated subnet width")
    axis.set_xlabel("Parameter activation band")
    axis.set_title("Under-exposure contribution by subnet and parameter band")
    fig.colorbar(image, ax=axis, label="log10(lost exposure incidences + 1)")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    clean = frame.copy()
    for column in clean:
        clean[column] = clean[column].map(
            lambda value: f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)
        )
    header = "| " + " | ".join(clean.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(clean.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in clean.to_numpy()]
    return "\n".join([header, separator, *rows])


def run_geometry_exposure_decomposition(root: str | Path, output_dir: str | Path) -> dict:
    root, output_dir = Path(root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table, detail, bands, geo_anchors, common_holdout = build_per_width_table(root)
    regressions, predictions = fit_decomposition(table)
    correlations = exposure_proxy_correlations(table)

    table.to_csv(output_dir / "per_width_geometry_exposure_accuracy.csv", index=False)
    regressions.to_csv(output_dir / "decomposition_regression.csv", index=False)
    detail.to_csv(output_dir / "exposure_by_width_and_band.csv", index=False)
    predictions.to_csv(output_dir / "decomposition_predictions.csv", index=False)
    correlations.to_csv(output_dir / "exposure_proxy_correlations.csv", index=False)
    bands.to_csv(output_dir / "structural_parameter_activation_bands.csv", index=False)

    _scatter(table, "B_G", "Functional coverage benefit B_G", output_dir / "geometry_gain_vs_accuracy.png")
    _scatter(table, "D_E", "Active-parameter exposure deficit D_E", output_dir / "exposure_deficit_vs_accuracy.png")
    _two_factor_plot(predictions, output_dir / "two_factor_fit.png")
    _band_plot(detail, output_dir / "exposure_by_parameter_band.png")

    two_factor = regressions.loc[regressions["model"].eq("B_G+D_E")].copy()
    two_factor["requested_sign_pattern"] = two_factor["beta_G_positive"] & two_factor["beta_E_negative"]
    signs_pass = bool(two_factor["requested_sign_pattern"].all())
    key_columns = [
        "seed", "beta_G", "beta_E", "standardized_beta_G", "standardized_beta_E",
        "in_sample_r2", "loow_r2", "loow_mae", "design_condition_number",
        "requested_sign_pattern",
    ]
    report = f"""# RQ2-v1 geometry/exposure decomposition

The fit uses only the common holdout widths. `B_G` uses the frozen seed-0
pre-intervention geometry coordinate that selected PureGeo-v1. `D_E(w)` averages
the Uniform-to-PureGeo loss-gradient exposure deficit only over parameters active
in subnet `w`. Accuracy deltas come from existing RQ2-v1 seed 1/2 CSVs.

## Two-factor diagnostic

{_markdown_table(two_factor[key_columns])}

The requested coefficient sign pattern (`beta_G > 0`, `beta_E < 0`) holds for
both seeds: **{signs_pass}**.

These are small-sample diagnostic fits ({len(common_holdout)} widths per seed),
not confirmatory causal estimates. Inspect leave-one-width-out performance and
the design condition number before using coefficient signs to freeze Hybrid-v3.
"""
    (output_dir / "geometry_exposure_decomposition_report.md").write_text(report)
    result = {
        "status": "complete",
        "uniform_anchors": list(UNIFORM_ANCHORS),
        "puregeo_anchors": list(geo_anchors),
        "common_holdout": common_holdout,
        "fit_seeds": [1, 2],
        "widths_per_seed": len(common_holdout),
        "both_seeds_beta_G_positive_beta_E_negative": signs_pass,
        "training_or_checkpoint_forward_used": False,
        "test_predictions_recomputed": False,
    }
    (output_dir / "geometry_exposure_decomposition_complete.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
