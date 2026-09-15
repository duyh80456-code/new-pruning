"""CPU-only audit of functional/resource coverage for RQ2 anchor placement."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rq2_anchor_placement import GRID, UNIFORM_ANCHORS
from rq2_finalgeo_selector import EXPECTED_PUREGEO
from rq2_parameter_exposure import _coordinates, _radius


FINALGEO_ANCHORS = (0.25, 0.40, 0.70, 1.00)
REFERENCE_SETS = {
    "Uniform": UNIFORM_ANCHORS,
    "PureGeo": EXPECTED_PUREGEO,
    "FinalGeo": FINALGEO_ANCHORS,
}


def _resource_metrics(anchors: tuple[float, ...]) -> tuple[float, float, float]:
    distances = np.asarray([min(abs(c - a) for a in anchors) for c in GRID], float)
    adjacent_gaps = np.diff(np.asarray(anchors, float))
    return float(distances.max()), float(distances.mean()), float(adjacent_gaps.max())


def _pareto_mask(frame: pd.DataFrame, columns=("R_G", "R_res")) -> np.ndarray:
    values = frame.loc[:, list(columns)].to_numpy(float)
    mask = np.ones(len(frame), dtype=bool)
    for index, point in enumerate(values):
        dominates = np.all(values <= point + 1e-15, axis=1) & np.any(
            values < point - 1e-15, axis=1
        )
        dominates[index] = False
        mask[index] = not dominates.any()
    return mask


def enumerate_anchor_sets(
    k: int,
    geometry_coordinate: dict[float, float],
    flops: dict[float, float],
) -> pd.DataFrame:
    """Enumerate endpoint-locked K-anchor sets on the registered dense grid."""
    if k not in (4, 5):
        raise ValueError("This audit is registered only for K=4 or K=5")
    uniform_compute = float(sum(flops[width] for width in UNIFORM_ANCHORS))
    uniform_R_G, _ = _radius(UNIFORM_ANCHORS, geometry_coordinate)
    uniform_R_res, _, _ = _resource_metrics(UNIFORM_ANCHORS)
    rows = []
    for interior in itertools.combinations(GRID[1:-1], k - 2):
        anchors = (GRID[0], *interior, GRID[-1])
        R_G, mean_G = _radius(anchors, geometry_coordinate)
        R_res, mean_res, max_gap = _resource_metrics(anchors)
        labels = [name for name, values in REFERENCE_SETS.items() if tuple(values) == anchors]
        rows.append({
            "K": k,
            "anchors": ",".join(f"{width:.2f}" for width in anchors),
            **{f"a{index}": width for index, width in enumerate(anchors)},
            "R_G": R_G,
            "mean_functional_distance": mean_G,
            "R_res": R_res,
            "mean_resource_distance": mean_res,
            "max_gap": max_gap,
            "subnet_flops_per_batch": float(sum(flops[width] for width in anchors)),
            "compute_ratio_vs_uniform_K4": float(
                sum(flops[width] for width in anchors) / uniform_compute
            ),
            "R_G_better_than_uniform": bool(R_G < uniform_R_G - 1e-15),
            "R_res_no_worse_than_uniform": bool(R_res <= uniform_R_res + 1e-15),
            "has_040": 0.40 in anchors,
            "reference_set": labels[0] if labels else "",
        })
    frame = pd.DataFrame(rows)
    expected = 91 if k == 4 else 364
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} K={k} candidates, got {len(frame)}")
    frame["joint_improvement_feasible"] = (
        frame["R_G_better_than_uniform"] & frame["R_res_no_worse_than_uniform"]
    )
    frame["functional_resource_pareto"] = _pareto_mask(frame)
    return frame.sort_values(["R_G", "R_res", "anchors"], kind="mergesort").reset_index(drop=True)


def widthwise_audit(
    validation_metrics: pd.DataFrame,
    geometry_coordinate: dict[float, float],
) -> pd.DataFrame:
    required = {"seed", "method", "budget", "accuracy"}
    if not required.issubset(validation_metrics.columns):
        raise ValueError(f"Interim metrics missing columns: {sorted(required - set(validation_metrics))}")
    selected = validation_metrics.loc[
        validation_metrics["seed"].astype(int).isin([3, 4])
        & validation_metrics["method"].isin(["uniform", "finalgeo"])
    ].copy()
    counts = selected.groupby(["seed", "method"])["budget"].nunique()
    if set(counts.index) != {(3, "uniform"), (3, "finalgeo"), (4, "uniform"), (4, "finalgeo")}:
        raise RuntimeError("Width audit requires complete Uniform/FinalGeo validation for seeds 3 and 4")
    if not (counts == len(GRID)).all():
        raise RuntimeError("Every seed/method must cover the complete 16-width grid")
    if "split" in selected and set(selected["split"]) != {"validation_5k"}:
        raise RuntimeError("Width audit accepts validation-only interim metrics, not test results")

    accuracy = selected.pivot(index="budget", columns=["method", "seed"], values="accuracy")
    rows = []
    for width in GRID:
        d_g_u = min(abs(geometry_coordinate[width] - geometry_coordinate[a]) for a in UNIFORM_ANCHORS)
        d_g_f = min(abs(geometry_coordinate[width] - geometry_coordinate[a]) for a in FINALGEO_ANCHORS)
        d_r_u = min(abs(width - a) for a in UNIFORM_ANCHORS)
        d_r_f = min(abs(width - a) for a in FINALGEO_ANCHORS)
        row = {
            "width": width,
            "d_G_uniform": d_g_u,
            "d_G_finalgeo": d_g_f,
            "delta_d_G_finalgeo_minus_uniform": d_g_f - d_g_u,
            "d_res_uniform": d_r_u,
            "d_res_finalgeo": d_r_f,
            "delta_d_res_finalgeo_minus_uniform": d_r_f - d_r_u,
            "resource_coverage_worse_in_finalgeo": d_r_f > d_r_u + 1e-15,
            "functional_coverage_better_in_finalgeo": d_g_f < d_g_u - 1e-15,
        }
        for seed in (3, 4):
            uniform = float(accuracy.loc[width, ("uniform", seed)])
            finalgeo = float(accuracy.loc[width, ("finalgeo", seed)])
            row[f"uniform_validation_accuracy_seed_{seed}"] = uniform
            row[f"finalgeo_validation_accuracy_seed_{seed}"] = finalgeo
            row[f"delta_accuracy_seed_{seed}"] = finalgeo - uniform
        row["mean_delta_accuracy_seed_3_4"] = np.mean(
            [row["delta_accuracy_seed_3"], row["delta_accuracy_seed_4"]]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_figures(k4: pd.DataFrame, widthwise: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(k4["R_res"], k4["R_G"], color="lightgrey", s=28, label="K=4 candidates")
    frontier = k4.loc[k4["functional_resource_pareto"]]
    ax.scatter(frontier["R_res"], frontier["R_G"], color="steelblue", s=42, label="Pareto frontier")
    for name, color in (("Uniform", "black"), ("PureGeo", "firebrick"), ("FinalGeo", "darkgreen")):
        row = k4.loc[k4["reference_set"].eq(name)].iloc[0]
        ax.scatter(row["R_res"], row["R_G"], color=color, s=110, zorder=5)
        ax.annotate(name, (row["R_res"], row["R_G"]), xytext=(5, 5), textcoords="offset points")
    ax.set(xlabel="Resource coverage radius R_res", ylabel="Functional coverage radius R_G",
           title="K=4 functional-resource coverage trade-off")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / "rq2_k4_functional_resource_pareto.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(widthwise["width"], widthwise["delta_d_res_finalgeo_minus_uniform"], marker="o")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(ylabel="FinalGeo - Uniform resource distance",
                title="Width-wise resource-coverage change and validation accuracy")
    for seed, color in ((3, "tab:blue"), (4, "tab:orange")):
        axes[1].plot(widthwise["width"], widthwise[f"delta_accuracy_seed_{seed}"],
                     marker="o", color=color, label=f"seed {seed}")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(xlabel="Width", ylabel="FinalGeo - Uniform validation accuracy")
    axes[1].legend()
    for ax in axes: ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "rq2_widthwise_resource_coverage_accuracy.png", dpi=200)
    plt.close(fig)


def run_resource_coverage_audit(
    development_root: str | Path,
    interim_validation_csv: str | Path,
    output_dir: str | Path,
) -> dict:
    development_root, output_dir = Path(development_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_coordinate, _, flops, pure_geo, source_hashes = _coordinates(development_root)
    if tuple(pure_geo) != EXPECTED_PUREGEO:
        raise RuntimeError(f"Unexpected PureGeo anchors: {pure_geo}")
    validation = pd.read_csv(interim_validation_csv)
    k4 = enumerate_anchor_sets(4, geometry_coordinate, flops)
    k5 = enumerate_anchor_sets(5, geometry_coordinate, flops)
    widthwise = widthwise_audit(validation, geometry_coordinate)

    uniform = k4.loc[k4["reference_set"].eq("Uniform")].iloc[0]
    forced_040 = k4.loc[k4["has_040"]]
    forced_feasible = forced_040.loc[forced_040["R_res_no_worse_than_uniform"]]
    k5_joint = k5.loc[k5["joint_improvement_feasible"]].sort_values(
        ["R_G", "compute_ratio_vs_uniform_K4", "anchors"], kind="mergesort"
    )
    audit = {
        "status": "CPU_RESOURCE_COVERAGE_AUDIT_COMPLETE",
        "uses_test_results": False,
        "development_geometry_seeds": [0, 1, 2],
        "validation_accuracy_seeds": [3, 4],
        "K4_candidate_count": len(k4),
        "K4_with_040_count": len(forced_040),
        "K4_with_040_and_R_res_no_worse_than_uniform_count": len(forced_feasible),
        "K4_with_040_resource_feasible_exists": bool(len(forced_feasible)),
        "K5_candidate_count": len(k5),
        "K5_joint_improvement_count": len(k5_joint),
        "K5_joint_improvement_exists": bool(len(k5_joint)),
        "best_K5_joint_candidate": None if k5_joint.empty else {
            key: (bool(k5_joint.iloc[0][key]) if isinstance(k5_joint.iloc[0][key], (bool, np.bool_))
                  else float(k5_joint.iloc[0][key]) if isinstance(k5_joint.iloc[0][key], (float, np.floating))
                  else int(k5_joint.iloc[0][key]) if isinstance(k5_joint.iloc[0][key], (int, np.integer))
                  else k5_joint.iloc[0][key])
            for key in ("anchors", "R_G", "R_res", "compute_ratio_vs_uniform_K4", "has_040")
        },
        "uniform_R_G": float(uniform["R_G"]),
        "uniform_R_res": float(uniform["R_res"]),
        "geometry_source_hashes": source_hashes,
    }
    k4.to_csv(output_dir / "rq2_k4_resource_geometry_candidates.csv", index=False)
    k5.to_csv(output_dir / "rq2_k5_resource_geometry_candidates.csv", index=False)
    widthwise.to_csv(output_dir / "rq2_widthwise_resource_geometry_accuracy.csv", index=False)
    (output_dir / "rq2_resource_coverage_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    _write_figures(k4, widthwise, output_dir)

    references = k4.loc[k4["reference_set"].ne(""), [
        "reference_set", "anchors", "R_G", "R_res", "mean_resource_distance",
        "max_gap", "compute_ratio_vs_uniform_K4", "functional_resource_pareto",
    ]]
    report = [
        "# RQ2 resource-coverage audit", "",
        "This audit uses frozen development geometry and seed-3/4 validation accuracy only. ",
        "It does not use the CIFAR-100 test split and does not select a new method.", "",
        "## Reference anchor sets", "", "```", references.to_string(index=False), "```", "",
        "## Structural feasibility", "",
        f"- K=4 candidates containing 0.40 with R_res <= Uniform: {len(forced_feasible)}.",
        f"- K=5 candidates with R_G < Uniform and R_res <= Uniform: {len(k5_joint)}.",
    ]
    if not k5_joint.empty:
        report += ["", "Best functional-radius K=5 joint-feasible candidate:", "",
                   "```", k5_joint.head(1).to_string(index=False), "```"]
    (output_dir / "rq2_resource_coverage_audit.md").write_text("\n".join(report) + "\n")
    return audit
