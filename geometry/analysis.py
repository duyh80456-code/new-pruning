from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/compression_geometry_matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research_utils import budget_tag
from .distances import compute_distribution_distance


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _load_features(feature_dir: Path, budgets: list[float]) -> list[np.ndarray]:
    arrays, reference_ids = [], None
    for budget in budgets:
        payload = torch.load(
            feature_dir / f"features_budget_{budget_tag(budget)}.pt",
            map_location="cpu",
            weights_only=False,
        )
        ids = payload["sample_ids"]
        if reference_ids is None:
            reference_ids = ids
        elif not torch.equal(reference_ids, ids):
            raise RuntimeError("Feature files do not have identical sample IDs/order")
        arrays.append(payload["features"].numpy())
    return arrays


def pairwise_geometry(
    features: list[np.ndarray], budgets: list[float], config: dict, result_dir: Path
) -> np.ndarray:
    geometry_cfg = config["geometry"]
    matrix = np.zeros((len(budgets), len(budgets)), dtype=np.float64)
    for i in range(len(budgets)):
        for j in range(i + 1, len(budgets)):
            value = compute_distribution_distance(
                features[i],
                features[j],
                method=geometry_cfg["method"],
                num_projections=int(geometry_cfg["num_projections"]),
                seed=int(geometry_cfg["projection_seed"]),
            )
            matrix[i, j] = matrix[j, i] = value
    result_dir.mkdir(parents=True, exist_ok=True)
    np.save(result_dir / "wasserstein_matrix.npy", matrix)
    pd.DataFrame(matrix, index=budgets, columns=budgets).to_csv(result_dir / "wasserstein_matrix.csv")
    budget_matrix = np.abs(np.subtract.outer(budgets, budgets))
    np.save(result_dir / "budget_distance_matrix.npy", budget_matrix)
    pd.DataFrame(budget_matrix, index=budgets, columns=budgets).to_csv(
        result_dir / "budget_distance_matrix.csv"
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="magma", origin="lower")
    ax.set_xticks(range(len(budgets)), [f"{b:.2f}" for b in budgets], rotation=90)
    ax.set_yticks(range(len(budgets)), [f"{b:.2f}" for b in budgets])
    ax.set_xlabel("Width multiplier")
    ax.set_ylabel("Width multiplier")
    ax.set_title(f"Pairwise {geometry_cfg['method']} distance")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(result_dir / "wasserstein_heatmap.png", dpi=180)
    plt.close(fig)
    return matrix


def _safe_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(pearsonr(x, y).statistic if method == "pearson" else spearmanr(x, y).statistic)


def _bootstrap_correlations(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    results = {"pearson": [], "spearman": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        for method in results:
            value = _safe_correlation(x[idx], y[idx], method)
            if np.isfinite(value):
                results[method].append(value)
    summary = {}
    for method, values in results.items():
        summary[method] = {
            "estimate": _safe_correlation(x, y, method),
            "ci_low": float(np.percentile(values, 2.5)) if values else float("nan"),
            "ci_high": float(np.percentile(values, 97.5)) if values else float("nan"),
        }
    return summary


def _robust_cliffs(local_geometry: np.ndarray, threshold: float) -> np.ndarray:
    median = np.median(local_geometry)
    mad = np.median(np.abs(local_geometry - median))
    if mad <= 1e-12:
        cutoff = np.percentile(local_geometry, 95)
        return local_geometry >= cutoff if np.ptp(local_geometry) > 0 else np.zeros_like(local_geometry, bool)
    robust_z = 0.6745 * (local_geometry - median) / mad
    return robust_z >= threshold


def _regression_table(frame: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "width only": ["budget"],
        "FLOPs only": ["flops_ratio"],
        "parameters only": ["params"],
        "coverage only": ["nearest_anchor_distance"],
        "geometry only": ["local_wasserstein_sensitivity"],
        "fit only": ["ID_fit_proxy"],
        "oracle representation distance": ["baseline_to_oracle_wasserstein"],
        "coverage + geometry": ["nearest_anchor_distance", "local_wasserstein_sensitivity"],
        "coverage + fit": ["nearest_anchor_distance", "ID_fit_proxy"],
        "geometry + fit": ["local_wasserstein_sensitivity", "ID_fit_proxy"],
        "coverage + geometry + fit": [
            "nearest_anchor_distance",
            "local_wasserstein_sensitivity",
            "ID_fit_proxy",
        ],
    }
    records = []
    for target in ["accuracy_degradation", "oracle_gap_if_available"]:
        for name, columns in groups.items():
            clean = frame[columns + [target]].dropna()
            if len(clean) <= len(columns) or clean[target].std() <= 1e-12:
                score = float("nan")
            else:
                estimator = make_pipeline(StandardScaler(), LinearRegression())
                estimator.fit(clean[columns], clean[target])
                score = float(estimator.score(clean[columns], clean[target]))
            records.append(
                {"target": target, "predictors": name, "n": len(clean), "in_sample_r2": score}
            )
    return pd.DataFrame(records)


def analyze_seed(output_dir: Path, config: dict, seed: int) -> tuple[pd.DataFrame, dict]:
    result_dir = output_dir / "results"
    metrics = pd.read_csv(result_dir / "budget_metrics.csv").sort_values("budget").reset_index(drop=True)
    budgets = metrics["budget"].astype(float).tolist()
    features = _load_features(output_dir / "features", budgets)
    matrix = pairwise_geometry(features, budgets, config, result_dir)
    flops_ratio = metrics["flops"].to_numpy(float) / float(metrics["flops"].iloc[-1])
    flops_matrix = np.abs(np.subtract.outer(flops_ratio, flops_ratio))
    np.save(result_dir / "flops_distance_matrix.npy", flops_matrix)
    pd.DataFrame(flops_matrix, index=budgets, columns=budgets).to_csv(
        result_dir / "flops_distance_matrix.csv"
    )

    delta_width = np.diff(budgets)
    jumps = np.diag(matrix, k=1)
    delta_flops = np.diff(flops_ratio)
    delta_accuracy = np.abs(np.diff(metrics["accuracy"].to_numpy(float)))
    local_geometry = jumps / delta_width
    local_flops_geometry = jumps / np.maximum(np.abs(delta_flops), 1e-12)
    local_performance = delta_accuracy / delta_width
    corr = _bootstrap_correlations(
        local_geometry,
        local_performance,
        int(config["geometry"]["bootstrap_samples"]),
        seed,
    )
    control_records = []
    for method in config["geometry"].get("control_methods", []):
        control_jumps = np.asarray(
            [
                compute_distribution_distance(
                    features[i],
                    features[i + 1],
                    method=method,
                    num_projections=int(config["geometry"]["num_projections"]),
                    seed=int(config["geometry"]["projection_seed"]),
                )
                for i in range(len(features) - 1)
            ]
        )
        control_sensitivity = control_jumps / delta_width
        control_records.append(
            {
                "method": method,
                "pearson_with_accuracy_sensitivity": _safe_correlation(
                    control_sensitivity, local_performance, "pearson"
                ),
                "spearman_with_accuracy_sensitivity": _safe_correlation(
                    control_sensitivity, local_performance, "spearman"
                ),
            }
        )
    pd.DataFrame(control_records).to_csv(result_dir / "control_metric_correlations.csv", index=False)

    cliff_mask = _robust_cliffs(local_geometry, float(config["geometry"]["cliff_robust_z"]))
    cliffs = pd.DataFrame(
        {
            "budget_start": budgets[:-1],
            "budget_end": budgets[1:],
            "wasserstein_jump": jumps,
            "accuracy_change": np.diff(metrics["accuracy"].to_numpy(float)),
            "flops_change": np.diff(metrics["flops"].to_numpy(float)),
            "geometric_sensitivity": local_geometry,
            "is_cliff": cliff_mask,
        }
    )
    cliffs[cliffs["is_cliff"]].to_csv(result_dir / "compression_cliffs.csv", index=False)
    pd.DataFrame(
        {
            "budget_start": budgets[:-1],
            "budget_end": budgets[1:],
            "G_width": local_geometry,
            "G_flops": local_flops_geometry,
            "P": local_performance,
        }
    ).to_csv(result_dir / "local_sensitivity.csv", index=False)

    anchors = np.asarray(config["compression"]["train_widths"], dtype=float)
    anchor_flops = np.interp(anchors, budgets, flops_ratio)
    nearest_width = np.min(np.abs(np.asarray(budgets)[:, None] - anchors[None, :]), axis=1)
    nearest_flops = np.min(np.abs(flops_ratio[:, None] - anchor_flops[None, :]), axis=1)
    width_scale = float(np.dot(delta_width, jumps) / max(np.dot(delta_width, delta_width), 1e-12))
    flops_scale = float(np.dot(delta_flops, jumps) / max(np.dot(delta_flops, delta_flops), 1e-12))
    distortion_width = np.r_[np.abs(jumps - width_scale * delta_width), np.nan]
    distortion_flops = np.r_[np.abs(jumps - flops_scale * delta_flops), np.nan]
    neighbor_scale = np.empty_like(local_geometry)
    for i in range(len(local_geometry)):
        neighbors = local_geometry[max(0, i - 1) : min(len(local_geometry), i + 2)]
        neighbor_scale[i] = np.median(neighbors)
    distortion_local = np.r_[np.abs(jumps - neighbor_scale * delta_width), np.nan]
    local_padded = np.r_[local_geometry, np.nan]
    full_idx = len(budgets) - 1
    fit_proxy = matrix[:, full_idx]

    central = metrics.rename(
        columns={
            "is_train_anchor": "is_seen",
            "accuracy": "accuracy",
        }
    ).copy()
    central["oracle_accuracy_if_available"] = np.nan
    central["oracle_gap_if_available"] = np.nan
    central["baseline_to_oracle_wasserstein"] = np.nan
    oracle_path = result_dir / "oracle_metrics.csv"
    if oracle_path.exists():
        oracle = pd.read_csv(oracle_path)
        for _, row in oracle.iterrows():
            mask = np.isclose(central["budget"], row["budget"])
            central.loc[mask, "oracle_accuracy_if_available"] = row["oracle_accuracy"]
            central.loc[mask, "oracle_gap_if_available"] = row["oracle_gap"]
            central.loc[mask, "baseline_to_oracle_wasserstein"] = row[
                "baseline_to_oracle_wasserstein"
            ]
    central["flops_ratio"] = flops_ratio
    central["nearest_anchor_distance"] = nearest_width
    central["nearest_anchor_flops_distance"] = nearest_flops
    central["local_wasserstein_sensitivity"] = local_padded
    central["local_accuracy_sensitivity"] = np.r_[local_performance, np.nan]
    central["local_flops_wasserstein_sensitivity"] = np.r_[local_flops_geometry, np.nan]
    central["geometric_distortion"] = distortion_local
    central["geometric_distortion_width"] = distortion_width
    central["geometric_distortion_flops"] = distortion_flops
    central["ID_fit_proxy"] = fit_proxy
    central["ID_fit_is_proxy"] = True
    # Full-width is the baseline, rather than whichever noisy budget happened to
    # obtain the maximum validation score.
    central["accuracy_degradation"] = central.loc[full_idx, "accuracy"] - central["accuracy"]
    central.to_csv(result_dir / "central_analysis.csv", index=False)
    regression = _regression_table(central)
    regression.to_csv(result_dir / "regression_comparison.csv", index=False)
    systematic = _safe_correlation(np.abs(1 - np.asarray(budgets)), fit_proxy, "spearman")
    summary = {
        "seed": seed,
        "systematic_geometry_spearman": systematic,
        "sensitivity_accuracy_correlation": corr,
        "compression_cliff_count": int(cliff_mask.sum()),
        "regressions": regression.replace({np.nan: None}).to_dict(orient="records"),
    }
    with (result_dir / "analysis_summary.json").open("w") as handle:
        json.dump(_json_safe(summary), handle, indent=2, allow_nan=False)
    return central, summary


def correlation_summary(x, y, bootstrap_samples: int, seed: int) -> dict:
    """Public wrapper used for the pooled multi-seed statistic."""
    return _bootstrap_correlations(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float), bootstrap_samples, seed
    )
