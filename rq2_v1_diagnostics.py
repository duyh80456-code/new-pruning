from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GRID = tuple(round(0.25 + 0.05 * index, 2) for index in range(16))
PRIMARY_REPRESENTATION = "learned_projection"
ANCHOR_INSPECTION_WIDTHS = (0.25, 0.40, 0.50, 0.60, 0.75, 1.00)


def _markdown_table(frame: pd.DataFrame) -> str:
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


def _load_dense(root: Path) -> pd.DataFrame:
    path = root / "rq2_dense_metrics_all.csv"
    required = {"seed", "method", "budget", "accuracy", "flops", "params"}
    dense = pd.read_csv(path)
    if not required.issubset(dense.columns):
        raise ValueError(f"{path} missing {sorted(required-set(dense.columns))}")
    dense = dense.rename(columns={"budget": "width"})
    dense["method"] = dense["method"].str.lower()
    expected = len(GRID) * 2 * 2
    if len(dense) != expected or set(dense["method"]) != {"uniform", "geo"}:
        raise RuntimeError(f"Expected {expected} dense Uniform/PureGeo rows, got {len(dense)}")
    return dense.sort_values(["seed", "method", "width"]).reset_index(drop=True)


def _load_geometry(root: Path) -> pd.DataFrame:
    path = root / "rq2_geometry_all.csv"
    required = {"seed", "method", "representation", "budget_start", "budget_end", "G"}
    geometry = pd.read_csv(path)
    if not required.issubset(geometry.columns):
        raise ValueError(f"{path} missing {sorted(required-set(geometry.columns))}")
    geometry = geometry.rename(columns={
        "budget_start": "width",
        "G": "local_wasserstein_sensitivity",
    })
    geometry["method"] = geometry["method"].str.lower()
    return geometry.sort_values(["representation", "seed", "method", "width"]).reset_index(drop=True)


def _paired_delta(frame, value, keys, candidate="geo", baseline="uniform"):
    wide = frame.pivot(index=keys, columns="method", values=value).reset_index()
    if candidate not in wide or baseline not in wide:
        raise RuntimeError(f"Cannot pair {candidate} and {baseline} for {value}")
    wide[f"delta_{value}"] = wide[candidate] - wide[baseline]
    return wide


def _load_training(root: Path) -> pd.DataFrame:
    frames = []
    for seed in (1, 2):
        path = root / "geo" / f"seed_{seed}" / "training_metrics_1_100.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path).rename(columns={
            "loss": "train_loss",
            "accuracy": "train_accuracy",
            "learning_rate": "lr",
        })
        frame["seed"] = seed
        # The shared RQ2 trainer did not run validation/checkpoint selection per
        # epoch. Keep the requested field explicit rather than inventing it.
        frame["val_accuracy"] = np.nan
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[
            "seed", "epoch", "width", "train_loss", "train_accuracy", "lr", "val_accuracy"
        ])
    result = pd.concat(frames, ignore_index=True)
    return result[[
        "seed", "epoch", "width", "train_loss", "train_accuracy", "lr", "val_accuracy"
    ]].sort_values(["seed", "width", "epoch"])


def _training_tail(training: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, width), group in training.groupby(["seed", "width"]):
        group = group.sort_values("epoch")
        tail = group.tail(10)
        slope = float(np.polyfit(tail["epoch"], tail["train_loss"], 1)[0]) if len(tail) > 1 else np.nan
        rows.append({
            "seed": int(seed), "width": float(width),
            "final_epoch": int(group["epoch"].iloc[-1]),
            "final_train_loss": float(group["train_loss"].iloc[-1]),
            "final_train_accuracy": float(group["train_accuracy"].iloc[-1]),
            "last10_mean_train_loss": float(tail["train_loss"].mean()),
            "last10_loss_slope_per_epoch": slope,
            "validation_curve_available": False,
        })
    return pd.DataFrame(rows)


def _prediction_diagnostics(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    transitions, class_rows = [], []
    for seed in (1, 2):
        paths = {
            method: root / "predictions" / method / f"seed_{seed}" / "predictions_all_widths.csv"
            for method in ("uniform", "geo")
        }
        if not all(path.is_file() for path in paths.values()):
            continue
        uniform = pd.read_csv(paths["uniform"])
        geo = pd.read_csv(paths["geo"])
        keys = ["seed", "budget", "sample_id", "label"]
        paired = uniform[keys + ["correct", "prediction"]].merge(
            geo[keys + ["correct", "prediction"]], on=keys, validate="one_to_one",
            suffixes=("_uniform", "_geo"),
        )
        paired["geo_gain"] = paired["correct_geo"] - paired["correct_uniform"]
        for width, group in paired.groupby("budget"):
            transitions.append({
                "seed": seed, "width": float(width), "n": len(group),
                "uniform_accuracy": float(group["correct_uniform"].mean()),
                "puregeo_accuracy": float(group["correct_geo"].mean()),
                "delta_accuracy": float(group["geo_gain"].mean()),
                "uniform_wrong_geo_right": int((group["geo_gain"] == 1).sum()),
                "uniform_right_geo_wrong": int((group["geo_gain"] == -1).sum()),
                "both_right": int(((group["correct_uniform"] == 1) & (group["correct_geo"] == 1)).sum()),
                "both_wrong": int(((group["correct_uniform"] == 0) & (group["correct_geo"] == 0)).sum()),
            })
        class_summary = paired.groupby(["budget", "label"], as_index=False).agg(
            uniform_accuracy=("correct_uniform", "mean"),
            puregeo_accuracy=("correct_geo", "mean"),
            n=("sample_id", "size"),
        )
        class_summary["delta_accuracy"] = (
            class_summary["puregeo_accuracy"] - class_summary["uniform_accuracy"]
        )
        class_summary.insert(0, "seed", seed)
        class_rows.append(class_summary.rename(columns={"budget": "width", "label": "class_id"}))
    transition_frame = pd.DataFrame(transitions)
    class_frame = pd.concat(class_rows, ignore_index=True) if class_rows else pd.DataFrame()
    return transition_frame, class_frame


def _region_summary(accuracy_delta: pd.DataFrame, geometry_delta: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "low_cliff": (0.30, 0.45),
        "around_removed_075": (0.65, 0.85),
        "high_width": (0.75, 1.00),
    }
    rows = []
    for seed in (1, 2):
        for name, (lower, upper) in definitions.items():
            accuracy = accuracy_delta.loc[
                accuracy_delta["seed"].eq(seed)
                & accuracy_delta["width"].between(lower, upper)
            ]
            geometry = geometry_delta.loc[
                geometry_delta["seed"].eq(seed)
                & geometry_delta["width"].between(lower, min(upper, 0.95))
            ]
            rows.append({
                "seed": seed, "region": name, "width_min": lower, "width_max": upper,
                "mean_delta_accuracy": float(accuracy["delta_accuracy"].mean()),
                "worst_delta_accuracy": float(accuracy["delta_accuracy"].min()),
                "mean_delta_G": float(geometry["delta_local_wasserstein_sensitivity"].mean()),
                "fraction_widths_accuracy_improved": float((accuracy["delta_accuracy"] > 0).mean()),
                "fraction_edges_G_reduced": float(
                    (geometry["delta_local_wasserstein_sensitivity"] < 0).mean()
                ),
            })
    result = pd.DataFrame(rows)
    pooled = result.groupby("region", as_index=False).agg({
        "width_min": "first", "width_max": "first", "mean_delta_accuracy": "mean",
        "worst_delta_accuracy": "min", "mean_delta_G": "mean",
        "fraction_widths_accuracy_improved": "mean", "fraction_edges_G_reduced": "mean",
    })
    pooled.insert(0, "seed", "pooled_mean")
    return pd.concat([result, pooled], ignore_index=True)


def _plots(dense, accuracy_delta, geometry_delta, training, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for (method, seed), group in dense.groupby(["method", "seed"]):
        axes[0].plot(group["width"], group["accuracy"], marker="o", label=f"{method} seed {seed}")
    axes[0].set_ylabel("Test accuracy"); axes[0].legend(ncol=2); axes[0].grid(alpha=0.25)
    for seed, group in accuracy_delta.groupby("seed"):
        axes[1].plot(group["width"], 100 * group["delta_accuracy"], marker="o", label=f"seed {seed}")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axvline(0.75, color="crimson", linestyle="--", label="removed Uniform anchor 0.75")
    axes[1].set(xlabel="Width", ylabel="PureGeo − Uniform (percentage points)")
    axes[1].legend(); axes[1].grid(alpha=0.25); fig.tight_layout()
    fig.savefig(output_dir / "rq2_v1_dense_accuracy_tradeoff.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for seed, group in geometry_delta.groupby("seed"):
        axes[0].plot(group["width"], group["uniform"], marker="o", alpha=0.7, label=f"Uniform seed {seed}")
        axes[0].plot(group["width"], group["geo"], marker="o", alpha=0.7, linestyle="--",
                     label=f"PureGeo seed {seed}")
        axes[1].plot(group["width"], group["delta_local_wasserstein_sensitivity"],
                     marker="o", label=f"seed {seed}")
    axes[0].set_ylabel("Local Wasserstein sensitivity G(c)")
    axes[0].legend(ncol=2); axes[0].grid(alpha=0.25)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axvline(0.75, color="crimson", linestyle="--")
    axes[1].set(xlabel="Interval start width", ylabel="ΔG = PureGeo − Uniform")
    axes[1].legend(); axes[1].grid(alpha=0.25); fig.tight_layout()
    fig.savefig(output_dir / "rq2_v1_dense_geometry_tradeoff.png", dpi=200); plt.close(fig)

    if len(training):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        for (seed, width), group in training.groupby(["seed", "width"]):
            label = f"s{seed} w={width:.2f}"
            axes[0].plot(group["epoch"], group["train_loss"], label=label)
            axes[1].plot(group["epoch"], group["train_accuracy"], label=label)
        axes[0].set(xlabel="Epoch", ylabel="Train loss")
        axes[1].set(xlabel="Epoch", ylabel="Train accuracy")
        for ax in axes:
            ax.axvline(50, color="black", linestyle="--", alpha=0.5)
            ax.grid(alpha=0.25)
        axes[1].legend(ncol=2, fontsize=8); fig.tight_layout()
        fig.savefig(output_dir / "rq2_v1_geo_training_convergence.png", dpi=200); plt.close(fig)


def run_rq2_v1_diagnostics(root: str | Path, output_dir: str | Path) -> dict:
    root, output_dir = Path(root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dense = _load_dense(root)
    geometry = _load_geometry(root)
    selection = json.loads((root / "protocol" / "selected_anchors.json").read_text())
    uniform_anchors = tuple(map(float, selection["uniform_anchors"]))
    geo_anchors = tuple(map(float, selection["selected_anchors"]))
    for method, anchors in (("uniform", uniform_anchors), ("geo", geo_anchors)):
        mask = dense["method"].eq(method)
        dense.loc[mask, "nearest_training_anchor_distance"] = dense.loc[mask, "width"].map(
            lambda width: min(abs(width - anchor) for anchor in anchors)
        )
        dense.loc[mask, "is_train_anchor"] = dense.loc[mask, "width"].round(2).isin(anchors)
    dense[["seed", "method", "width", "accuracy", "flops", "params",
           "is_train_anchor", "nearest_training_anchor_distance"]].to_csv(
        output_dir / "rq2_v1_dense_accuracy.csv", index=False
    )
    geometry.to_csv(output_dir / "rq2_v1_dense_geometry.csv", index=False)
    accuracy_delta = _paired_delta(dense, "accuracy", ["seed", "width"])
    support_delta = _paired_delta(
        dense, "nearest_training_anchor_distance", ["seed", "width"]
    )[["seed", "width", "delta_nearest_training_anchor_distance"]]
    accuracy_delta = accuracy_delta.merge(support_delta, on=["seed", "width"], validate="one_to_one")
    accuracy_delta.to_csv(output_dir / "rq2_v1_accuracy_deltas.csv", index=False)
    learned = geometry.loc[geometry["representation"].eq(PRIMARY_REPRESENTATION)]
    geometry_delta = _paired_delta(
        learned, "local_wasserstein_sensitivity", ["seed", "width", "budget_end"]
    )
    geometry_delta.to_csv(output_dir / "rq2_v1_geometry_deltas.csv", index=False)
    anchor_accuracy = dense.loc[dense["width"].round(2).isin(ANCHOR_INSPECTION_WIDTHS), [
        "seed", "method", "width", "accuracy", "is_train_anchor",
        "nearest_training_anchor_distance",
    ]]
    anchor_accuracy.to_csv(output_dir / "rq2_v1_anchor_accuracy.csv", index=False)
    compute = pd.read_csv(root / "protocol" / "anchor_training_compute.csv")
    compute["phase_1_epochs"] = 50
    compute["phase_2_epochs"] = 50
    compute["total_epochs"] = 100
    compute.to_csv(output_dir / "rq2_v1_training_compute.csv", index=False)
    training = _load_training(root)
    training.to_csv(output_dir / "rq2_v1_geo_training_curves.csv", index=False)
    tail = _training_tail(training)
    tail.to_csv(output_dir / "rq2_v1_geo_training_tail_summary.csv", index=False)
    transitions, classes = _prediction_diagnostics(root)
    transitions.to_csv(output_dir / "rq2_v1_sample_transitions.csv", index=False)
    classes.to_csv(output_dir / "rq2_v1_class_accuracy_deltas.csv", index=False)
    regions = _region_summary(accuracy_delta, geometry_delta)
    regions.to_csv(output_dir / "rq2_v1_region_summary.csv", index=False)
    _plots(dense, accuracy_delta, geometry_delta, training, output_dir)
    availability = {
        "dense_accuracy_available": True,
        "dense_geometry_available": True,
        "anchor_compute_available": True,
        "geo_train_metrics_by_epoch_available": bool(len(training)),
        "geo_validation_accuracy_by_epoch_available": False,
        "reason_validation_unavailable": (
            "RQ2-v1 shared training used the final epoch and did not evaluate/save validation "
            "accuracy each epoch; it cannot be reconstructed from final checkpoints."
        ),
        "paired_predictions_available": bool(len(transitions)),
        "raw_logits_available": False,
    }
    (output_dir / "rq2_v1_artifact_availability.json").write_text(
        json.dumps(availability, indent=2) + "\n"
    )
    report = [
        "# RQ2-v1 PureGeo failure diagnostics", "",
        f"Uniform anchors: `{list(uniform_anchors)}`", "",
        f"PureGeo anchors: `{list(geo_anchors)}`", "",
        "## Regional trade-off", "", _markdown_table(regions), "",
        "## Accuracy at requested support points", "", _markdown_table(anchor_accuracy), "",
        "## Training compute", "", _markdown_table(compute), "",
        "## Convergence evidence available", "", _markdown_table(tail), "",
        "Per-epoch validation accuracy is **not available** and is not imputed. Train loss/accuracy "
        "can identify obvious non-convergence, but cannot by itself exclude a generalization or "
        "checkpoint-selection explanation.", "",
        "Interpretation should jointly inspect accuracy deltas, ΔG, changes in nearest-anchor "
        "distance around 0.75, and the breadth of per-sample/class transitions. These diagnostics "
        "distinguish evidence patterns but do not by themselves prove causality.",
    ]
    (output_dir / "rq2_v1_diagnostic_report.md").write_text("\n".join(report) + "\n")
    return {"availability": availability, "uniform_anchors": uniform_anchors, "geo_anchors": geo_anchors}
