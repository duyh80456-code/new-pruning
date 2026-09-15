"""Two-fold cross-batch validation of RQ2 importance-corrected variance."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rq2_anchor_placement import GRID, _sha256
from rq2_gradient_variance_v3 import (
    NUM_INTERIORS,
    INTERIOR_SLOTS,
    _read_square_matrix,
    capped_neyman_allocation,
    importance_corrected_variance,
)
from rq2_probabilistic_support import INTERIOR_WIDTHS, maximum_entropy_pairs


def _worker_shards(interaction_root: Path) -> list[tuple[Path, dict]]:
    rows = []
    for metadata_path in (interaction_root / "worker_shards").glob("*/metadata.json"):
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("training_performed") is not False
            or metadata.get("test_used") is not False
            or metadata.get("weights_unchanged") is not True
            or metadata.get("bn_buffers_unchanged_during_probe") is not True
        ):
            raise RuntimeError(f"Worker shard violates read-only protocol: {metadata_path}")
        rows.append((metadata_path.parent, metadata))
    rows.sort(key=lambda item: min(item[1]["global_batch_indices"]))
    if len(rows) != 2:
        raise FileNotFoundError(f"Expected two saved 8-batch worker shards, found {len(rows)}")
    indices = [set(map(int, metadata["global_batch_indices"])) for _, metadata in rows]
    if indices[0] & indices[1] or sorted(indices[0] | indices[1]) != list(range(16)):
        raise RuntimeError(f"Worker shards must be disjoint and cover batches 0..15: {indices}")
    if any(len(item) != 8 for item in indices):
        raise RuntimeError("Two-fold validation requires exactly eight batches per worker shard")
    return rows


def _shard_dot_and_moments(root: Path) -> tuple[np.ndarray, np.ndarray]:
    full = _read_square_matrix(root / "gradient_dot_matrix.csv")
    indices = [GRID.index(width) for width in INTERIOR_WIDTHS]
    dot = full[np.ix_(indices, indices)]
    norms = pd.read_csv(root / "gradient_norms_by_batch.csv")
    norms["width"] = norms["width"].round(2)
    moments = (
        norms.loc[norms.width.isin(INTERIOR_WIDTHS)]
        .assign(squared_norm=lambda frame: np.square(frame.gradient_norm))
        .groupby("width")["squared_norm"].mean()
        .reindex(INTERIOR_WIDTHS).to_numpy(float)
    )
    raw_diagonal = np.diag(dot).copy()
    relative_error = np.max(
        np.abs(raw_diagonal - moments)
        / np.maximum.reduce([np.abs(raw_diagonal), np.abs(moments), np.full_like(moments, 1e-30)])
    )
    if not np.isfinite(moments).all() or np.any(moments <= 0) or relative_error > 1e-2:
        raise RuntimeError(
            f"Invalid shard second moments or material dot/norm mismatch: {relative_error:.6g}"
        )
    dot[np.diag_indices_from(dot)] = moments
    return dot, moments


def _frozen_policies(interaction_root: Path) -> dict[str, tuple[np.ndarray, pd.DataFrame]]:
    theory = interaction_root / "frozen_policy" / "theory-allocation-probe"
    preview = interaction_root / "frozen_policy" / "probabilistic-support-preview"
    theory_marginals = pd.read_csv(theory / "policy_family_marginals.csv")
    geometry = theory_marginals.loc[np.isclose(theory_marginals.p, 1.0)].sort_values("width")
    resource = pd.read_csv(preview / "support_allocation_marginals.csv").sort_values("width")
    uniform_pi = np.full(NUM_INTERIORS, INTERIOR_SLOTS / NUM_INTERIORS)
    return {
        "uniform_dynamic": (uniform_pi, maximum_entropy_pairs(uniform_pi)[0]),
        "resource": (
            resource.pi_resource_matched_compute.to_numpy(float),
            pd.read_csv(preview / "resource_pair_distribution.csv"),
        ),
        "geometry": (
            geometry.pi.to_numpy(float),
            pd.read_csv(theory / "pair_distribution_p100.csv"),
        ),
    }


def run_cross_batch_variance_validation(
    interaction_root: str | Path,
    output_dir: str | Path,
) -> dict:
    """Fit gradient oracle on one 8-batch shard and evaluate on the other, then swap."""
    interaction_root, output_dir = Path(interaction_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = _worker_shards(interaction_root)
    shard_values = [_shard_dot_and_moments(root) for root, _ in shards]
    frozen = _frozen_policies(interaction_root)
    target_weights = np.full(NUM_INTERIORS, 1.0 / NUM_INTERIORS)
    variance_rows, oracle_rows = [], []
    for fold, (train_index, heldout_index) in enumerate(((0, 1), (1, 0))):
        _, train_moments = shard_values[train_index]
        heldout_dot, heldout_moments = shard_values[heldout_index]
        oracle_pi = capped_neyman_allocation(train_moments, target_weights)
        oracle_pairs = maximum_entropy_pairs(oracle_pi)[0]
        policies = {**frozen, "gradient_oracle_crossfit": (oracle_pi, oracle_pairs)}
        train_batches = shards[train_index][1]["global_batch_indices"]
        heldout_batches = shards[heldout_index][1]["global_batch_indices"]
        for width, train_moment, heldout_moment, pi in zip(
            INTERIOR_WIDTHS, train_moments, heldout_moments, oracle_pi
        ):
            oracle_rows.append({
                "fold": fold,
                "fit_batches": ",".join(map(str, train_batches)),
                "heldout_batches": ",".join(map(str, heldout_batches)),
                "width": width,
                "fit_gradient_second_moment": train_moment,
                "heldout_gradient_second_moment": heldout_moment,
                "crossfit_oracle_pi": pi,
            })
        fold_values = {}
        for policy, (pi, pairs) in policies.items():
            values = importance_corrected_variance(
                heldout_dot, target_weights, pi, pairs
            )
            fold_values[policy] = values["importance_corrected_variance"]
            variance_rows.append({
                "fold": fold,
                "fit_batches": ",".join(map(str, train_batches)),
                "heldout_batches": ",".join(map(str, heldout_batches)),
                "policy": policy,
                **values,
                "minimum_pi": float(pi.min()),
                "maximum_pi": float(pi.max()),
            })
        uniform = fold_values["uniform_dynamic"]
        for row in variance_rows[-len(policies):]:
            row["variance_ratio_vs_uniform_dynamic"] = (
                row["importance_corrected_variance"] / uniform
            )

    variance = pd.DataFrame(variance_rows)
    oracle = pd.DataFrame(oracle_rows)
    variance.to_csv(output_dir / "cross_batch_variance_by_fold.csv", index=False)
    oracle.to_csv(output_dir / "cross_batch_oracle_marginals.csv", index=False)
    summary_table = variance.groupby("policy", as_index=False).agg(
        mean_heldout_variance=("importance_corrected_variance", "mean"),
        std_heldout_variance=("importance_corrected_variance", "std"),
        mean_ratio_vs_uniform_dynamic=("variance_ratio_vs_uniform_dynamic", "mean"),
        worst_ratio_vs_uniform_dynamic=("variance_ratio_vs_uniform_dynamic", "max"),
    ).sort_values("mean_heldout_variance")
    summary_table.to_csv(output_dir / "cross_batch_variance_summary.csv", index=False)

    piv = variance.pivot(index="fold", columns="policy", values="importance_corrected_variance")
    ordering = (
        (piv.geometry < piv.resource) & (piv.resource < piv.uniform_dynamic)
    )
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(2)
    width = 0.2
    for index, policy in enumerate(
        ("uniform_dynamic", "resource", "geometry", "gradient_oracle_crossfit")
    ):
        values = piv[policy] / piv.uniform_dynamic
        axis.bar(x + (index - 1.5) * width, values, width=width, label=policy)
    axis.axhline(1, color="black", linewidth=1)
    axis.set(xticks=x, xticklabels=["fit 0–7 / eval 8–15", "fit 8–15 / eval 0–7"],
             ylabel="Held-out variance / Uniform-dynamic")
    axis.grid(axis="y", alpha=0.25); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output_dir / "cross_batch_variance.png", dpi=200); plt.close(fig)

    summary = {
        "status": "RQ2_V3_CROSS_BATCH_VALIDATION_COMPLETE",
        "training_performed": False,
        "model_or_checkpoint_loaded": False,
        "gpu_used": False,
        "test_used": False,
        "accuracy_used": False,
        "folds": [
            {"fit_batches": list(range(0, 8)), "heldout_batches": list(range(8, 16))},
            {"fit_batches": list(range(8, 16)), "heldout_batches": list(range(0, 8))},
        ],
        "gradient_oracle_cross_fitted": True,
        "frozen_geometry_and_resource_refit": False,
        "full_q_aware_variance_used": True,
        "geometry_below_resource_below_uniform_each_fold": bool(ordering.all()),
        "geometry_below_resource_below_uniform_fold_count": int(ordering.sum()),
        "geometry_below_resource_below_uniform_mean": bool(
            piv.geometry.mean() < piv.resource.mean() < piv.uniform_dynamic.mean()
        ),
        "interaction_metadata_sha256": _sha256(interaction_root / "metadata.json"),
        "interpretation": "held-out-batch variance diagnostic; not a training result or causal proof",
    }
    (output_dir / "cross_batch_variance_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary

