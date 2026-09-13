from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PRIMARY_REPRESENTATION = "learned_projection"
REPRESENTATIONS = ("learned_projection", "backbone_padded", "fixed_random_projection")
FEATURE_SETS = {
    "coverage": ["coverage"],
    "log_flops": ["log_flops"],
    "resource": ["coverage", "log_flops"],
    "G": ["G"],
    "coverage_G": ["coverage", "G"],
    "resource_G": ["coverage", "log_flops", "G"],
}


def load_rq1_frame(source_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(source_root)
    spec = pd.read_csv(root / "specialization_table.csv")
    geometry = pd.read_csv(root / "representation_local_geometry_all_seeds.csv")
    dense = pd.read_csv(root / "shared_dense_metrics_all_seeds.csv")
    required_spec = {"seed", "width", "specialization_gap", "flops", "params"}
    required_geo = {"seed", "representation", "budget_start", "budget_end", "G"}
    if not required_spec.issubset(spec.columns):
        raise ValueError(f"specialization_table.csv missing {sorted(required_spec-set(spec.columns))}")
    if not required_geo.issubset(geometry.columns):
        raise ValueError(
            f"representation_local_geometry_all_seeds.csv missing "
            f"{sorted(required_geo-set(geometry.columns))}"
        )
    geometry = geometry.rename(columns={"budget_start": "width"})
    df = spec.merge(
        geometry[["seed", "representation", "width", "budget_end", "G"]],
        on=["seed", "width"], how="inner", validate="one_to_many",
    )
    expected = len(spec) * len(REPRESENTATIONS)
    if len(df) != expected or set(df["representation"]) != set(REPRESENTATIONS):
        raise RuntimeError(f"Expected {expected} aligned specialization/geometry rows, got {len(df)}")
    anchors = np.asarray([0.25, 0.50, 0.75, 1.00])
    df["coverage"] = [float(np.min(np.abs(anchors - width))) for width in df["width"]]
    df["log_flops"] = np.log(df["flops"].astype(float))
    return df.sort_values(["representation", "seed", "width"]), geometry, dense


def matched_pair_table(df: pd.DataFrame, representation=PRIMARY_REPRESENTATION) -> pd.DataFrame:
    selected = df.loc[
        (df["representation"] == representation) & df["width"].isin([0.4, 0.6])
    ]
    g = selected.pivot(index="seed", columns="width", values="G")
    gap = selected.pivot(index="seed", columns="width", values="specialization_gap")
    if list(g.columns) != [0.4, 0.6] or list(gap.columns) != [0.4, 0.6]:
        raise RuntimeError("Matched pair requires exactly widths 0.40 and 0.60 for every seed")
    result = pd.DataFrame({
        "seed": g.index.astype(int), "G_040": g[0.4].to_numpy(),
        "G_060": g[0.6].to_numpy(), "delta_G": (g[0.4] - g[0.6]).to_numpy(),
        "gap_040": gap[0.4].to_numpy(), "gap_060": gap[0.6].to_numpy(),
        "delta_gap": (gap[0.4] - gap[0.6]).to_numpy(),
    })
    return result.reset_index(drop=True)


def loso_predictors(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, folds = [], []
    logo = LeaveOneGroupOut()
    for representation in REPRESENTATIONS:
        frame = df.loc[df["representation"] == representation].reset_index(drop=True)
        y = frame["specialization_gap"].to_numpy(float)
        groups = frame["seed"].to_numpy(int)
        for name, features in FEATURE_SETS.items():
            predictions = np.empty_like(y)
            for train_index, test_index in logo.split(frame, y, groups):
                estimator = make_pipeline(StandardScaler(), LinearRegression())
                estimator.fit(frame.loc[train_index, features], y[train_index])
                predictions[test_index] = estimator.predict(frame.loc[test_index, features])
                folds.append({
                    "representation": representation, "model": name,
                    "heldout_seed": int(groups[test_index][0]),
                    "mae": mean_absolute_error(y[test_index], predictions[test_index]),
                    "r2": r2_score(y[test_index], predictions[test_index]),
                })
            summaries.append({
                "representation": representation, "model": name,
                "predictors": " + ".join(features), "n": len(y),
                "loso_mae": mean_absolute_error(y, predictions),
                "loso_r2": r2_score(y, predictions),
            })
    return pd.DataFrame(summaries), pd.DataFrame(folds)


def resource_residuals(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for representation in REPRESENTATIONS:
        frame = df.loc[df["representation"] == representation].copy().reset_index(drop=True)
        estimator = make_pipeline(StandardScaler(), LinearRegression())
        features = ["coverage", "log_flops"]
        estimator.fit(frame[features], frame["specialization_gap"])
        frame["resource_prediction"] = estimator.predict(frame[features])
        frame["resource_residual"] = (
            frame["specialization_gap"] - frame["resource_prediction"]
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _correlation(x, y, method):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    result = pearsonr(x, y) if method == "pearson" else spearmanr(x, y)
    return float(result.statistic)


def _seed_block_counts(frame: pd.DataFrame, rng: np.random.Generator) -> tuple[int, ...]:
    seeds = np.asarray(sorted(frame["seed"].unique()))
    sampled = rng.choice(seeds, size=len(seeds), replace=True)
    return tuple(int(np.sum(sampled == seed)) for seed in seeds)


def _indices_from_block_counts(frame: pd.DataFrame, counts: tuple[int, ...]) -> np.ndarray:
    seeds = np.asarray(sorted(frame["seed"].unique()))
    blocks = []
    for seed, count in zip(seeds, counts):
        block = np.flatnonzero(frame["seed"].to_numpy() == seed)
        blocks.extend([block] * count)
    return np.concatenate(blocks)


def block_bootstrap_correlations(
    df: pd.DataFrame, replicates: int = 5000, seed: int = 20260913
) -> pd.DataFrame:
    rows = []
    for view_index, representation in enumerate(REPRESENTATIONS):
        frame = df.loc[df["representation"] == representation].reset_index(drop=True)
        rng = np.random.default_rng(seed + view_index)
        resource = make_pipeline(StandardScaler(), LinearRegression())
        resource.fit(frame[["coverage", "log_flops"]], frame["specialization_gap"])
        residual = frame["specialization_gap"].to_numpy() - resource.predict(
            frame[["coverage", "log_flops"]]
        )
        targets = {
            "specialization_gap": frame["specialization_gap"].to_numpy(float),
            "resource_residual": residual,
        }
        for target_name, target in targets.items():
            estimates = {
                method: _correlation(frame["G"], target, method)
                for method in ("pearson", "spearman")
            }
            bootstrap = {"pearson": [], "spearman": []}
            # With three seed clusters there are only ten distinct bootstrap
            # multisets. Cache them instead of refitting the resource model
            # thousands of times; replicate frequencies still define the CI.
            cache = {}
            for _ in range(int(replicates)):
                counts = _seed_block_counts(frame, rng)
                if counts not in cache:
                    indices = _indices_from_block_counts(frame, counts)
                    sampled = frame.iloc[indices].reset_index(drop=True)
                    if target_name == "resource_residual":
                        fitted = make_pipeline(StandardScaler(), LinearRegression())
                        fitted.fit(sampled[["coverage", "log_flops"]], sampled["specialization_gap"])
                        sampled_target = sampled["specialization_gap"].to_numpy() - fitted.predict(
                            sampled[["coverage", "log_flops"]]
                        )
                    else:
                        sampled_target = sampled["specialization_gap"].to_numpy(float)
                    cache[counts] = {
                        method: _correlation(sampled["G"], sampled_target, method)
                        for method in bootstrap
                    }
                for method in bootstrap:
                    value = cache[counts][method]
                    if np.isfinite(value):
                        bootstrap[method].append(value)
            for method, values in bootstrap.items():
                rows.append({
                    "representation": representation, "target": target_name,
                    "method": method, "estimate": estimates[method],
                    "ci_low": float(np.percentile(values, 2.5)),
                    "ci_high": float(np.percentile(values, 97.5)),
                    "bootstrap_unit": "seed_block", "bootstrap_replicates": int(replicates),
                    "unique_bootstrap_multisets": len(cache),
                    "n_seeds": frame["seed"].nunique(), "n_rows": len(frame),
                })
    return pd.DataFrame(rows)


def _plot_dense(geometry, dense, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    accuracy = dense.groupby("budget")["accuracy"].agg(["mean", "min", "max"])
    axes[0].plot(accuracy.index, accuracy["mean"], marker="o", color="black")
    axes[0].fill_between(accuracy.index, accuracy["min"], accuracy["max"], alpha=0.18,
                         color="black", label="seed range")
    axes[0].set_ylabel("Shared test accuracy"); axes[0].legend(); axes[0].grid(alpha=0.25)
    for representation, group in geometry.groupby("representation"):
        summary = group.groupby("width")["G"].agg(["mean", "min", "max"])
        axes[1].plot(summary.index, summary["mean"], marker="o", label=representation)
        axes[1].fill_between(summary.index, summary["min"], summary["max"], alpha=0.12)
    axes[1].axvline(0.40, color="crimson", linestyle="--", alpha=0.7)
    axes[1].axvline(0.60, color="steelblue", linestyle="--", alpha=0.7)
    axes[1].set(xlabel="Width / interval start", ylabel="G(c)")
    axes[1].legend(); axes[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(output_path, dpi=200); plt.close(fig)


def _plot_residual(residuals, output_path):
    frame = residuals.loc[residuals["representation"] == PRIMARY_REPRESENTATION]
    fig, ax = plt.subplots(figsize=(8, 6))
    for seed, group in frame.groupby("seed"):
        ax.scatter(group["G"], group["resource_residual"], s=65, label=f"seed {seed}")
        for row in group.itertuples():
            ax.annotate(f"{row.width:.2f}", (row.G, row.resource_residual),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
    line = LinearRegression().fit(frame[["G"]], frame["resource_residual"])
    grid = np.linspace(frame["G"].min(), frame["G"].max(), 100)
    ax.plot(grid, line.predict(pd.DataFrame({"G": grid})), color="black", linestyle="--")
    ax.axhline(0, color="grey", linewidth=1)
    ax.set(xlabel="G(c), learned projection", ylabel="Resource-model residual gap",
           title="Geometry versus specialization gap unexplained by coverage/FLOPs")
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(output_path, dpi=200); plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown renderer for the exported report."""
    values = frame.reset_index(drop=True).copy()
    for column in values:
        values[column] = values[column].map(
            lambda value: f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)
        )
    header = "| " + " | ".join(map(str, values.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(values.columns)) + " |"
    rows = [
        "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"
        for row in values.astype(str).to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def run_rq1_analysis(source_root, output_dir, bootstrap_replicates=5000) -> dict:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    df, geometry, dense = load_rq1_frame(source_root)
    matched = matched_pair_table(df)
    matched.to_csv(output_dir / "rq1_matched_pair_040_060.csv", index=False)
    all_pairs = pd.concat([
        matched_pair_table(df, representation).assign(representation=representation)
        for representation in REPRESENTATIONS
    ], ignore_index=True)
    all_pairs.to_csv(output_dir / "rq1_matched_pair_all_representations.csv", index=False)
    loso, folds = loso_predictors(df)
    loso.to_csv(output_dir / "rq1_loso_predictors.csv", index=False)
    folds.to_csv(output_dir / "rq1_loso_fold_metrics.csv", index=False)
    residuals = resource_residuals(df)
    residuals.to_csv(output_dir / "rq1_resource_residuals.csv", index=False)
    bootstrap = block_bootstrap_correlations(df, bootstrap_replicates)
    bootstrap.to_csv(output_dir / "rq1_bootstrap_correlations.csv", index=False)
    _plot_dense(geometry, dense, output_dir / "rq1_dense_geometry_vs_accuracy.png")
    _plot_residual(residuals, output_dir / "rq1_geometry_vs_resource_residual.png")

    primary_loso = loso.loc[loso["representation"] == PRIMARY_REPRESENTATION].set_index("model")
    primary_boot = bootstrap.loc[
        (bootstrap["representation"] == PRIMARY_REPRESENTATION)
        & (bootstrap["target"] == "resource_residual")
    ]
    matched_pass = bool(((matched["delta_G"] > 0) & (matched["delta_gap"] > 0)).all())
    mae_pass = bool(primary_loso.loc["resource_G", "loso_mae"] < primary_loso.loc["resource", "loso_mae"])
    r2_pass = bool(primary_loso.loc["resource_G", "loso_r2"] > primary_loso.loc["resource", "loso_r2"])
    residual_pearson = float(primary_boot.loc[primary_boot["method"] == "pearson", "estimate"].iloc[0])
    decision = {
        "primary_representation": PRIMARY_REPRESENTATION,
        "matched_pair_3_of_3": matched_pass,
        "resource_G_mae_better": mae_pass,
        "resource_G_r2_better": r2_pass,
        "residual_pearson_positive": residual_pearson > 0,
        "rq1_all_requested_checks_pass": bool(
            matched_pass and mae_pass and residual_pearson > 0
        ),
    }
    lines = [
        "# RQ1 post-hoc analysis", "", f"Decision: **{'PASS' if decision['rq1_all_requested_checks_pass'] else 'NOT ALL CHECKS PASS'}**.", "",
        "Bootstrap unit: complete seed block; the four widths within a sampled seed are kept together.", "",
        "## Matched 0.40 versus 0.60", "", _markdown_table(matched), "",
        "## LOSO predictors — learned projection", "",
        _markdown_table(primary_loso.reset_index().sort_values("loso_mae")), "",
        "## Block-bootstrap correlations — learned projection", "",
        _markdown_table(bootstrap.loc[bootstrap["representation"] == PRIMARY_REPRESENTATION]), "",
        "## Decision checks", "", _markdown_table(pd.DataFrame([decision])), "",
        "Pooled estimates contain only three independent seed clusters; intervals should be interpreted as sensitivity summaries, not definitive population-level inference.",
    ]
    (output_dir / "rq1_summary.md").write_text("\n".join(lines))
    (output_dir / "rq1_decision.json").write_text(pd.Series(decision).to_json(indent=2) + "\n")
    return {"decision": decision, "matched": matched, "loso": loso, "bootstrap": bootstrap}
