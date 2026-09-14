"""CPU-only parameter-exposure analysis for RQ2 anchor placement.

The analysis is architecture exact for the repository's nested-prefix modules.
It does not load checkpoints, datasets, predictions, or accuracy columns.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from models.slimmable_ops import SharedProjection, SlimmableBatchNorm2d, SlimmableConv2d
from rq2_anchor_placement import GRID, PRIMARY_REPRESENTATION, UNIFORM_ANCHORS, _sha256
from s1_width import make_model


ELASTIC_GROUPS = ("stem", "layer1", "layer2", "layer3", "layer4", "projection")
ALL_GROUPS = (*ELASTIC_GROUPS, "classifier")


def _group_name(module_name: str) -> str:
    head = module_name.split(".", 1)[0]
    if head in {"conv1", "bn1"}:
        return "stem"
    if head in {"layer1", "layer2", "layer3", "layer4", "projection"}:
        return head
    raise ValueError(f"Unexpected width-dependent module outside registered groups: {module_name}")


def exact_active_parameter_counts(config: dict) -> pd.DataFrame:
    """Count actual active scalar parameters at every registered width and stage."""
    model = make_model(config, device="cpu")
    rows = []
    classifier_count = sum(parameter.numel() for parameter in model.classifier.parameters())
    for width in GRID:
        model.set_width(width)
        counts = {group: 0 for group in ALL_GROUPS}
        for name, module in model.named_modules():
            if isinstance(module, (SlimmableConv2d, SlimmableBatchNorm2d, SharedProjection)):
                counts[_group_name(name)] += int(module.active_parameter_count())
        counts["classifier"] = classifier_count
        total = sum(counts.values())
        if total != int(model.active_parameter_count(width)):
            raise RuntimeError(
                f"Exact group count {total} disagrees with model count "
                f"{model.active_parameter_count(width)} at width {width}"
            )
        for group in ALL_GROUPS:
            rows.append({
                "width": width,
                "parameter_group": group,
                "active_parameters": counts[group],
                "is_width_dependent_group": group in ELASTIC_GROUPS,
            })
    frame = pd.DataFrame(rows)
    for group, subset in frame.groupby("parameter_group"):
        if np.any(np.diff(subset.sort_values("width")["active_parameters"]) < 0):
            raise RuntimeError(f"Non-nested active-parameter counts in {group}")
    return frame


def activation_bands(active_counts: pd.DataFrame) -> pd.DataFrame:
    """Convert cumulative active counts to exact grid-relevant activation bands."""
    rows = []
    for group in ALL_GROUPS:
        subset = active_counts.loc[active_counts["parameter_group"].eq(group)].sort_values("width")
        cumulative = subset["active_parameters"].to_numpy(np.int64)
        increments = np.diff(np.concatenate([[0], cumulative]))
        if np.any(increments < 0) or int(increments.sum()) != int(cumulative[-1]):
            raise RuntimeError(f"Invalid activation bands for {group}")
        for width, count, active in zip(subset["width"], increments, cumulative):
            rows.append({
                "parameter_group": group,
                "activation_width": float(width),
                "newly_active_parameters": int(count),
                "cumulative_active_parameters": int(active),
                "group_parameters_at_full_width": int(cumulative[-1]),
                "is_width_dependent_group": group in ELASTIC_GROUPS,
            })
    return pd.DataFrame(rows)


def exposure_count(anchors, activation_width: float) -> int:
    return sum(float(anchor) + 1e-12 >= float(activation_width) for anchor in anchors)


def _radius(anchors, coordinate: dict[float, float]) -> tuple[float, float]:
    distances = [
        min(abs(coordinate[width] - coordinate[anchor]) for anchor in anchors)
        for width in GRID
    ]
    return float(max(distances)), float(np.mean(distances))


def _pareto_mask(first: np.ndarray, second: np.ndarray, tolerance=1e-12) -> np.ndarray:
    result = np.ones(len(first), dtype=bool)
    for index in range(len(first)):
        weakly_better = (first <= first[index] + tolerance) & (second <= second[index] + tolerance)
        strictly_better = (first < first[index] - tolerance) | (second < second[index] - tolerance)
        if np.any(weakly_better & strictly_better):
            result[index] = False
    return result


def _coordinates(development_root: Path):
    geometry_path = development_root / "rq2_geometry_all.csv"
    coordinate_path = development_root / "protocol" / "geometry_trajectory_coordinates.csv"
    metrics_path = development_root / "rq2_dense_metrics_all.csv"
    selection_path = development_root / "protocol" / "selected_anchors.json"
    required = [geometry_path, coordinate_path, metrics_path, selection_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete RQ2-v1 development artifact: {missing}")

    geometry = pd.read_csv(
        geometry_path,
        usecols=["method", "seed", "representation", "budget_start", "budget_end", "G"],
    )
    learned = geometry.loc[
        geometry["method"].eq("uniform")
        & geometry["representation"].eq(PRIMARY_REPRESENTATION)
        & geometry["seed"].astype(int).isin([1, 2]),
        ["seed", "budget_start", "budget_end", "G"],
    ]
    seed_zero_coordinates = pd.read_csv(coordinate_path).sort_values("width")
    if tuple(seed_zero_coordinates["width"].round(2)) != GRID:
        raise RuntimeError("Seed-0 geometry coordinate does not cover the registered grid")
    seed_zero = pd.DataFrame({
        "seed": 0,
        "budget_start": GRID[:-1],
        "budget_end": GRID[1:],
        "G": seed_zero_coordinates["incoming_edge_length"].iloc[1:].to_numpy(float) / 0.05,
    })
    learned = pd.concat([seed_zero, learned], ignore_index=True)
    counts = learned.groupby(["budget_start", "budget_end"])["seed"].nunique()
    if len(counts) != 15 or not (counts == 3).all():
        raise RuntimeError("Exposure preview requires Uniform geometry for development seeds 0,1,2")
    median_edges = learned.groupby(["budget_start", "budget_end"], as_index=False)["G"].median()
    median_edges = median_edges.sort_values("budget_start")
    geometry_coordinate = dict(zip(
        GRID,
        np.concatenate([[0.0], np.cumsum(0.05 * median_edges["G"].to_numpy(float))]),
    ))

    metrics = pd.read_csv(metrics_path, usecols=["method", "budget", "flops"])
    uniform_metrics = metrics.loc[metrics["method"].eq("uniform")]
    flops = {
        round(float(width), 2): float(value)
        for width, value in uniform_metrics.groupby("budget")["flops"].first().items()
    }
    if set(flops) != set(GRID) or any(value <= 0 for value in flops.values()):
        raise RuntimeError("A positive Uniform FLOPs value is required at every grid width")
    values = np.log([flops[width] for width in GRID])
    resource_coordinate = dict(zip(GRID, (values - values.min()) / (values.max() - values.min())))
    pure_geo = tuple(map(float, json.loads(selection_path.read_text())["selected_anchors"]))
    hashes = {path.name: _sha256(path) for path in required}
    return geometry_coordinate, resource_coordinate, flops, pure_geo, hashes


def candidate_exposure_metrics(anchors, bands: pd.DataFrame):
    detail_rows = []
    for group in ALL_GROUPS:
        subset = bands.loc[bands["parameter_group"].eq(group)]
        group_total = int(subset["newly_active_parameters"].sum())
        deficit = gain = underexposed = exposure_updates = uniform_updates = exposure_once = 0
        for row in subset.itertuples():
            count = int(row.newly_active_parameters)
            uniform = exposure_count(UNIFORM_ANCHORS, row.activation_width)
            candidate = exposure_count(anchors, row.activation_width)
            deficit += count * max(0, uniform - candidate)
            gain += count * max(0, candidate - uniform)
            underexposed += count * int(candidate < uniform)
            exposure_updates += count * candidate
            uniform_updates += count * uniform
            exposure_once += count * int(candidate == 1)
        detail_rows.append({
            "parameter_group": group,
            "group_parameters": group_total,
            "uniform_exposure_updates": uniform_updates,
            "candidate_exposure_updates": exposure_updates,
            "lost_exposure_updates": deficit,
            "gained_exposure_updates": gain,
            "net_exposure_updates": exposure_updates - uniform_updates,
            "D_E_group": deficit / group_total,
            "fraction_group_parameters_underexposed": underexposed / group_total,
            "fraction_group_parameters_exposed_once": exposure_once / group_total,
            "mean_exposure_count": exposure_updates / group_total,
        })
    detail = pd.DataFrame(detail_rows)
    total_parameters = int(detail["group_parameters"].sum())
    elastic = detail.loc[detail["parameter_group"].isin(ELASTIC_GROUPS)]
    elastic_parameters = int(elastic["group_parameters"].sum())
    deficit = int(detail["lost_exposure_updates"].sum())
    gain = int(detail["gained_exposure_updates"].sum())
    underexposed_parameters = sum(
        row.group_parameters * row.fraction_group_parameters_underexposed
        for row in detail.itertuples()
    )
    uniform_updates = int(detail["uniform_exposure_updates"].sum())
    candidate_updates = int(detail["candidate_exposure_updates"].sum())
    metrics = {
        # Exact definition requested: average number of lost anchor-update paths
        # per scalar trainable parameter, relative to Uniform-4.
        "D_E": deficit / total_parameters,
        "lost_exposure_updates": deficit,
        "gained_exposure_updates": gain,
        "net_exposure_updates": candidate_updates - uniform_updates,
        "relative_lost_uniform_exposure": deficit / uniform_updates,
        "relative_gained_uniform_exposure": gain / uniform_updates,
        "mean_exposure_count": candidate_updates / total_parameters,
        "fraction_parameters_underexposed": underexposed_parameters / total_parameters,
        "elastic_D_E": int(elastic["lost_exposure_updates"].sum()) / elastic_parameters,
        "worst_group_D_E": float(elastic["D_E_group"].max()),
        "worst_group": str(elastic.sort_values(
            ["D_E_group", "parameter_group"], ascending=[False, True]
        ).iloc[0]["parameter_group"]),
    }
    return metrics, detail


def _plot_frontier(candidates: pd.DataFrame, pure_geo, hybrid_v2, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 6.2))
    scatter = axis.scatter(
        candidates["D_E"], candidates["R_G"], c=candidates["R_R"],
        cmap="viridis", alpha=0.45, s=28, label="All 91 candidates",
    )
    frontier = candidates.loc[candidates["pareto_RG_DE"]].sort_values("D_E")
    axis.plot(frontier["D_E"], frontier["R_G"], color="black", marker="o", label="Pareto frontier")
    references = {
        "Uniform": UNIFORM_ANCHORS,
        "PureGeo-v1": pure_geo,
        "Hybrid-v2": hybrid_v2,
    }
    colors = {"Uniform": "tab:blue", "PureGeo-v1": "tab:red", "Hybrid-v2": "tab:orange"}
    for label, anchors in references.items():
        anchor_text = ",".join(f"{value:.2f}" for value in anchors)
        row = candidates.loc[candidates["anchors"].eq(anchor_text)].iloc[0]
        axis.scatter(row["D_E"], row["R_G"], s=110, marker="*", color=colors[label], label=label)
        axis.annotate(label, (row["D_E"], row["R_G"]), xytext=(5, 5), textcoords="offset points")
    axis.set_xlabel(r"Parameter-exposure deficit $D_E$ (lost update paths / parameter / batch)")
    axis.set_ylabel(r"Functional coverage radius $R_G$")
    axis.set_title("Geometry–parameter exposure Pareto preview")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.colorbar(scatter, ax=axis, label=r"Resource radius $R_R$")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_activation_bands(bands: pd.DataFrame, output: Path) -> None:
    elastic = bands.loc[
        bands["parameter_group"].isin(ELASTIC_GROUPS)
        & bands["newly_active_parameters"].gt(0)
    ].copy()
    pivot = elastic.pivot(
        index="parameter_group", columns="activation_width", values="newly_active_parameters"
    ).fillna(0).reindex(ELASTIC_GROUPS)
    values = np.log10(pivot.to_numpy(float) + 1.0)
    fig, axis = plt.subplots(figsize=(10, 4.5))
    image = axis.imshow(values, aspect="auto", cmap="magma")
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    axis.set_xticks(range(len(pivot.columns)), [f"{value:.2f}" for value in pivot.columns], rotation=45)
    axis.set_xlabel("First registered width that activates parameter band")
    axis.set_title("Exact nested-prefix parameter activation bands (log10 count + 1)")
    fig.colorbar(image, ax=axis, label="log10(parameters + 1)")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    clean = frame.copy()
    for column in clean:
        clean[column] = clean[column].map(
            lambda value: f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)
        )
    header = "| " + " | ".join(map(str, clean.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(clean.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in clean.to_numpy()]
    return "\n".join([header, separator, *rows])


def run_parameter_exposure_preview(development_root: str | Path, output_dir: str | Path) -> dict:
    development_root, output_dir = Path(development_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = development_root / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing RQ2 resolved config: {config_path}")
    config = yaml.safe_load(config_path.read_text())
    if tuple(map(float, config["compression"]["eval_widths"])) != GRID:
        raise ValueError("Parameter exposure preview requires the registered 16-width grid")

    geometry_coordinate, resource_coordinate, flops, pure_geo, hashes = _coordinates(development_root)
    active = exact_active_parameter_counts(config)
    bands = activation_bands(active)
    active.to_csv(output_dir / "parameter_active_counts_by_width_layer.csv", index=False)
    bands.to_csv(output_dir / "parameter_activation_bands.csv", index=False)

    uniform_R_G, _ = _radius(UNIFORM_ANCHORS, geometry_coordinate)
    uniform_R_R, _ = _radius(UNIFORM_ANCHORS, resource_coordinate)
    rows, layer_rows = [], []
    for a1, a2 in itertools.combinations(GRID[1:-1], 2):
        anchors = (0.25, a1, a2, 1.0)
        anchor_text = ",".join(f"{value:.2f}" for value in anchors)
        R_G, mean_G = _radius(anchors, geometry_coordinate)
        R_R, mean_R = _radius(anchors, resource_coordinate)
        R_W, mean_W = _radius(anchors, {width: width for width in GRID})
        exposure, detail = candidate_exposure_metrics(anchors, bands)
        rows.append({
            "a1": a1, "a2": a2, "anchors": anchor_text,
            "R_G": R_G, "R_R": R_R, "R_width": R_W,
            "mean_geometry_distance": mean_G,
            "mean_resource_distance": mean_R,
            "mean_width_distance": mean_W,
            "normalized_R_G": R_G / uniform_R_G,
            "normalized_R_R": R_R / uniform_R_R,
            **exposure,
            "subnet_flops_per_batch": sum(flops[width] for width in anchors),
        })
        detail.insert(0, "anchors", anchor_text)
        detail.insert(0, "a2", a2)
        detail.insert(0, "a1", a1)
        layer_rows.append(detail)
    candidates = pd.DataFrame(rows)
    if len(candidates) != 91:
        raise RuntimeError(f"Expected 91 endpoint-locked anchor sets, got {len(candidates)}")
    candidates["pareto_RG_DE"] = _pareto_mask(
        candidates["R_G"].to_numpy(float), candidates["D_E"].to_numpy(float)
    )
    candidates["pareto_RG_RR_DE"] = np.ones(len(candidates), dtype=bool)
    values = candidates[["R_G", "R_R", "D_E"]].to_numpy(float)
    for index, point in enumerate(values):
        weak = np.all(values <= point + 1e-12, axis=1)
        strict = np.any(values < point - 1e-12, axis=1)
        if np.any(weak & strict):
            candidates.loc[index, "pareto_RG_RR_DE"] = False

    # Reconstruct the prior Hybrid-v2 normalized-minimax choice for reference only.
    candidates["hybrid_v2_objective"] = np.maximum(
        candidates["normalized_R_G"], candidates["normalized_R_R"]
    )
    candidates["hybrid_v2_sum"] = candidates["normalized_R_G"] + candidates["normalized_R_R"]
    prior = candidates.assign(
        objective_key=candidates["hybrid_v2_objective"].round(12),
        sum_key=candidates["hybrid_v2_sum"].round(12),
    ).sort_values(["objective_key", "sum_key", "a1", "a2"], kind="mergesort").iloc[0]
    hybrid_v2 = (0.25, float(prior.a1), float(prior.a2), 1.0)

    candidates["is_uniform"] = candidates["anchors"].eq(
        ",".join(f"{value:.2f}" for value in UNIFORM_ANCHORS)
    )
    candidates["is_puregeo_v1"] = candidates["anchors"].eq(
        ",".join(f"{value:.2f}" for value in pure_geo)
    )
    candidates["is_hybrid_v2"] = candidates["anchors"].eq(
        ",".join(f"{value:.2f}" for value in hybrid_v2)
    )
    candidates = candidates.sort_values(["D_E", "R_G", "a1", "a2"], kind="mergesort")
    candidates.to_csv(output_dir / "parameter_exposure_all_anchor_sets.csv", index=False)
    frontier = candidates.loc[candidates["pareto_RG_DE"]].sort_values(["D_E", "R_G"])
    frontier.to_csv(output_dir / "parameter_exposure_pareto_RG_DE.csv", index=False)
    candidates.loc[candidates["pareto_RG_RR_DE"]].sort_values(
        ["D_E", "R_G", "R_R"]
    ).to_csv(output_dir / "parameter_exposure_pareto_RG_RR_DE.csv", index=False)
    layer_detail = pd.concat(layer_rows, ignore_index=True)
    layer_detail.to_csv(output_dir / "parameter_exposure_by_anchor_set_layer.csv", index=False)

    references = candidates.loc[
        candidates[["is_uniform", "is_puregeo_v1", "is_hybrid_v2"]].any(axis=1)
    ].copy()
    references.to_csv(output_dir / "parameter_exposure_reference_sets.csv", index=False)
    reference_names = {
        ",".join(f"{value:.2f}" for value in UNIFORM_ANCHORS): "Uniform",
        ",".join(f"{value:.2f}" for value in pure_geo): "PureGeo-v1",
        ",".join(f"{value:.2f}" for value in hybrid_v2): "Hybrid-v2",
    }
    reference_layers = layer_detail.loc[layer_detail["anchors"].isin(reference_names)].copy()
    reference_layers["method"] = reference_layers["anchors"].map(reference_names)
    reference_layers.to_csv(output_dir / "parameter_exposure_reference_layers.csv", index=False)

    correlations = []
    for proxy in ("R_width", "R_R", "subnet_flops_per_batch"):
        correlations.append({
            "proxy": proxy,
            "pearson_with_D_E": candidates[[proxy, "D_E"]].corr(method="pearson").iloc[0, 1],
            "spearman_with_D_E": candidates[[proxy, "D_E"]].corr(method="spearman").iloc[0, 1],
        })
    pd.DataFrame(correlations).to_csv(output_dir / "parameter_exposure_proxy_correlations.csv", index=False)

    _plot_frontier(candidates, pure_geo, hybrid_v2, output_dir / "parameter_exposure_pareto.png")
    _plot_activation_bands(bands, output_dir / "parameter_activation_bands.png")

    shown = references[[
        "anchors", "R_G", "R_R", "D_E", "relative_lost_uniform_exposure",
        "fraction_parameters_underexposed", "worst_group", "worst_group_D_E",
    ]].copy()
    shown.insert(0, "method", shown["anchors"].map(reference_names))
    report = f"""# Parameter-exposure Pareto preview

This is a CPU-only, pre-training structural diagnostic. No checkpoint weights,
images, accuracy, predictions, specialization gaps, or test data were read.

## Locked metric

For scalar parameter `p`, `E_A(p)` is the number of the four anchor forwards in
which that scalar is active. The primary deficit is exactly:

`D_E(A) = mean_p max(0, E_Uniform(p) - E_A(p))`.

Activation is computed from the real prefix slices in every convolution, shared
BN affine tensor, and projection tensor. The always-active classifier is included
in the exact denominator requested; `elastic_D_E` is also exported to show the
undiluted width-dependent value.

`E_A` counts direct CE/KD loss-gradient paths. It intentionally does not call
weight decay or momentum an exposure: an inactive tensor slice can still change
through optimizer regularization/state even when it receives no data-loss gradient.

## Reference sets

{_markdown_table(shown)}

## New two-objective Pareto frontier

{_markdown_table(frontier[["anchors", "R_G", "D_E", "R_R", "relative_lost_uniform_exposure", "worst_group"]])}

The frontier is a preview, not a frozen Hybrid-v3 selection. A constraint `tau`
must be preregistered without looking at confirmatory accuracy before choosing a
new anchor set from this table.
"""
    (output_dir / "parameter_exposure_report.md").write_text(report)

    artifact = {
        "status": "preview_only_not_frozen",
        "metric": "D_E(A)=mean_p max(0,E_Uniform(p)-E_A(p))",
        "parameter_scope": "all trainable parameters; elastic-only sensitivity also exported",
        "exposure_semantics": "direct CE/KD loss-gradient paths; excludes weight-decay and momentum-only updates",
        "activation_source": "exact SlimmableConv2d/SlimmableBatchNorm2d/SharedProjection prefix rules",
        "candidate_count": len(candidates),
        "pareto_RG_DE_count": int(candidates["pareto_RG_DE"].sum()),
        "pareto_RG_RR_DE_count": int(candidates["pareto_RG_RR_DE"].sum()),
        "uniform_anchors": list(UNIFORM_ANCHORS),
        "puregeo_v1_anchors": list(pure_geo),
        "hybrid_v2_anchors": list(hybrid_v2),
        "new_anchor_set_selected": False,
        "accuracy_or_test_data_used": False,
        "source_artifact_sha256": hashes,
    }
    (output_dir / "parameter_exposure_preview.json").write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact
