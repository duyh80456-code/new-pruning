"""CPU-only policy decomposition on the completed fresh seed-6 Gate B1 artifact.

This module never imports torch, reconstructs a model, loads a checkpoint, or
fits a predictor.  It compares six fixed-marginal pair policies using the SW
and gradient-Gram tensors that Gate B1 already exported.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from rq2_pairwise_external_gates import _prepare_fresh_table, frozen_predict
from rq2_pairwise_surrogate_regret import (
    INTERIOR_WIDTHS,
    NUM_PAIRS,
    PAIR_INDICES,
    UNIFORM_PI,
    _variance,
    gram_pair_scores,
    solve_pair_lp,
)


EPOCHS = (10, 50, 100)
POLICY_ORDER = (
    "uniform_pair",
    "raw_resource",
    "pure_sw",
    "frozen_resource",
    "frozen_resource_plus_sw",
    "gradient_oracle",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair_csv(root: Path) -> Path | None:
    for name in ("fresh_pair_structure.csv", "pair_structure.csv"):
        path = root / name
        if path.is_file():
            return path
    return None


def _gram_path(root: Path, epoch: int) -> Path | None:
    candidates = (
        root / "gradient_grams" / f"epoch_{epoch:03d}.npy",
        root / "grams" / f"seed_6_epoch_{epoch:03d}.npy",
    )
    return next((path for path in candidates if path.is_file()), None)


def _is_fresh_diagnostic_root(root: Path) -> bool:
    pair_path = _pair_csv(root)
    if pair_path is None or not all(_gram_path(root, epoch) for epoch in EPOCHS):
        return False
    try:
        frame = pd.read_csv(pair_path, usecols=[
            "seed", "epoch", "width_i", "width_j", "representation_sw",
            "gradient_euclidean_distance", "flops_i", "flops_j",
        ])
    except (OSError, ValueError):
        return False
    return (
        set(frame.seed.astype(int)) == {6}
        and set(frame.epoch.astype(int)) == set(EPOCHS)
        and frame.groupby("epoch").size().eq(NUM_PAIRS).all()
    )


def _diagnostic_identity(root: Path) -> tuple[str, ...]:
    pair_path = _pair_csv(root)
    assert pair_path is not None
    return (_sha256(pair_path),) + tuple(
        _sha256(_gram_path(root, epoch)) for epoch in EPOCHS
    )


def _deduplicate_roots(roots: list[Path]) -> list[Path]:
    by_identity: dict[tuple[str, ...], list[Path]] = {}
    for root in roots:
        by_identity.setdefault(_diagnostic_identity(root), []).append(root)
    if len(by_identity) > 1:
        raise RuntimeError(f"Conflicting fresh seed-6 diagnostic artifacts: {roots}")
    return [sorted(next(iter(by_identity.values())), key=lambda path: (len(str(path)), str(path)))[0]] if roots else []


def _extract_diagnostic_members(archive: Path, destination: Path) -> None:
    """Extract only small analysis inputs, never checkpoint weights."""
    wanted_names = {
        "fresh_pair_structure.csv", "pair_structure.csv", "fresh_state_metadata.json",
        "state_metadata.json", "frozen_pairwise_predictors.json", "gate_a_summary.json",
    }
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            normalized = member.filename.replace("\\", "/")
            basename = Path(normalized).name
            wanted_tensor = (
                ("/gradient_grams/" in f"/{normalized}" or "/grams/" in f"/{normalized}")
                and basename in {f"epoch_{epoch:03d}.npy" for epoch in EPOCHS}
                | {f"seed_6_epoch_{epoch:03d}.npy" for epoch in EPOCHS}
            )
            if basename not in wanted_names and not wanted_tensor:
                continue
            target = (destination / normalized).resolve()
            resolved = destination.resolve()
            if target != resolved and resolved not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def find_fresh_seed6_inputs(
    input_root: str | Path,
    materialized_root: str | Path,
) -> tuple[Path, Path]:
    """Find one seed-6 diagnostic root and its already-frozen predictor.

    Identical duplicated Kaggle inputs are deduplicated by content hash.
    Conflicting artifacts fail loudly instead of selecting one post hoc.
    """
    input_root, materialized_root = Path(input_root), Path(materialized_root)
    roots = sorted({
        path.parent for name in ("fresh_pair_structure.csv", "pair_structure.csv")
        for path in input_root.rglob(name) if _is_fresh_diagnostic_root(path.parent)
    })
    roots = _deduplicate_roots(roots)
    predictors = sorted(input_root.rglob("frozen_pairwise_predictors.json"))

    if not roots or not predictors:
        materialized_root.mkdir(parents=True, exist_ok=True)
        for index, archive in enumerate(sorted(input_root.rglob("*.zip"))):
            try:
                with zipfile.ZipFile(archive) as bundle:
                    names = bundle.namelist()
                    relevant = (
                        any(name.endswith(("fresh_pair_structure.csv", "pair_structure.csv")) for name in names)
                        or any(name.endswith("frozen_pairwise_predictors.json") for name in names)
                    )
            except (OSError, zipfile.BadZipFile):
                continue
            if relevant:
                _extract_diagnostic_members(archive, materialized_root / f"archive_{index:03d}")
        if not roots:
            roots = sorted({
                path.parent for name in ("fresh_pair_structure.csv", "pair_structure.csv")
                for path in materialized_root.rglob(name) if _is_fresh_diagnostic_root(path.parent)
            })
            roots = _deduplicate_roots(roots)
        if not predictors:
            predictors = sorted(materialized_root.rglob("frozen_pairwise_predictors.json"))

    if len(roots) != 1:
        raise FileNotFoundError(
            "Expected one complete seed-6 diagnostic root containing a 273-row pair table "
            "and gradient Gram matrices for epochs 10/50/100; no checkpoint is required. "
            f"found={roots}"
        )
    valid_predictors = []
    for path in predictors:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "FROZEN_BEFORE_FRESH_STATES":
            valid_predictors.append(path)
    by_hash: dict[str, list[Path]] = {}
    for path in valid_predictors:
        by_hash.setdefault(_sha256(path), []).append(path)
    if len(by_hash) != 1:
        raise FileNotFoundError(
            "Expected one unique already-frozen development predictor. "
            f"valid_candidates={valid_predictors}"
        )
    predictor = sorted(next(iter(by_hash.values())), key=lambda path: (len(str(path)), str(path)))[0]
    return roots[0], predictor


def _load_gram(root: Path, epoch: int) -> np.ndarray:
    path = _gram_path(root, epoch)
    if path is None:
        raise FileNotFoundError(f"Missing gradient Gram for epoch {epoch}")
    values = np.load(path).astype(np.float64)
    gram = values.mean(axis=0) if values.ndim == 3 else values
    if gram.shape != (len(INTERIOR_WIDTHS), len(INTERIOR_WIDTHS)):
        raise RuntimeError(f"Invalid Gram shape at epoch {epoch}: {values.shape}")
    if not np.isfinite(gram).all() or not np.allclose(gram, gram.T, atol=1e-7, rtol=1e-7):
        raise RuntimeError(f"Invalid/non-symmetric Gram matrix at epoch {epoch}")
    return gram


def _oracle_gap_captured(v_uniform: float, variance: float, v_oracle: float) -> float:
    denominator = v_uniform - v_oracle
    return (v_uniform - variance) / denominator if denominator > 1e-12 else np.nan


def _plot_results(output_dir: Path, long: pd.DataFrame, distances: pd.DataFrame) -> None:
    labels = [f"F{epoch}" for epoch in EPOCHS]
    pivot = long.pivot(index="epoch", columns="policy", values="exact_variance").reindex(EPOCHS)
    fig, axis = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(EPOCHS)); width = 0.13
    for index, policy in enumerate(POLICY_ORDER):
        axis.bar(x + (index - 2.5) * width, pivot[policy], width, label=policy)
    axis.set(xticks=x, xticklabels=labels, ylabel="Exact estimator variance")
    axis.grid(axis="y", alpha=0.25); axis.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(output_dir / "fresh_seed6_policy_variance.png", dpi=200)
    plt.close(fig)

    selected = distances.loc[distances.policy.isin(
        ["raw_resource", "pure_sw", "frozen_resource_plus_sw"]
    )]
    pivot = selected.pivot(index="epoch", columns="policy", values="l1_to_gradient_oracle").reindex(EPOCHS)
    fig, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(len(EPOCHS)); width = 0.24
    for index, policy in enumerate(pivot.columns):
        axis.bar(x + (index - 1) * width, pivot[policy], width, label=policy)
    axis.set(xticks=x, xticklabels=labels, ylabel="L1 distance to gradient-oracle policy")
    axis.grid(axis="y", alpha=0.25); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output_dir / "fresh_seed6_policy_l1_to_oracle.png", dpi=200)
    plt.close(fig)


def _diagnose(wide: pd.DataFrame, correlations: pd.DataFrame) -> tuple[str, str]:
    sw_uniform_wins = int((wide.V_pure_sw < wide.V_uniform_pair).sum())
    sw_raw_wins = int((wide.V_pure_sw < wide.V_raw_resource).sum())
    hybrid_frozen_r_wins = int(
        (wide.V_frozen_resource_plus_sw < wide.V_frozen_resource).sum()
    )
    positive_sw_correlations = int((correlations.spearman_sw2_gradient_distance2 > 0).sum())
    if sw_uniform_wins <= 1 or positive_sw_correlations <= 1:
        return (
            "CASE_C_PURE_SW_FRESH_TRANSFER_WEAK",
            "Pure SW fails to beat Uniform in a majority of fresh states or loses positive pair-structure correlation.",
        )
    if sw_raw_wins <= 1:
        return (
            "CASE_A_SW_SIGNAL_BUT_RESOURCE_NEARER_ORACLE",
            "Pure SW beats Uniform in a majority of states, but raw resource pairing is at least as strong in a majority.",
        )
    if hybrid_frozen_r_wins <= 1:
        return (
            "CASE_B_SW_TRANSFERS_BUT_FROZEN_HYBRID_CALIBRATION_FAILS",
            "Pure SW beats raw resource in a majority, while the frozen learned hybrid fails to beat Frozen-R in a majority.",
        )
    return (
        "MIXED_OR_BETTER_THAN_DECLARED_CASES",
        "Pure SW and the learned hybrid both retain some fresh-state value; inspect per-state results before any new gate.",
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    columns = list(frame.columns)
    def render(value):
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.8g}"
        return str(value)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def run_fresh_seed6_policy_decomposition(
    fresh_root: str | Path,
    frozen_predictor_path: str | Path,
    output_dir: str | Path,
) -> dict:
    fresh_root, frozen_predictor_path, output_dir = map(
        Path, (fresh_root, frozen_predictor_path, output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = json.loads(frozen_predictor_path.read_text())
    if predictor.get("status") != "FROZEN_BEFORE_FRESH_STATES":
        raise RuntimeError("Predictor was not frozen before fresh-state evaluation")
    if 6 in set(map(int, predictor.get("development_seeds", []))):
        raise RuntimeError("Seed 6 appears in the development predictor")

    pair_path = _pair_csv(fresh_root)
    if pair_path is None:
        raise FileNotFoundError("Fresh seed-6 pair table is missing")
    table = _prepare_fresh_table(pd.read_csv(pair_path))
    if set(table.seed.astype(int)) != {6} or set(table.epoch.astype(int)) != set(EPOCHS):
        raise RuntimeError("Diagnostic input must be exactly seed 6 at epochs 10/50/100")

    uniform_q = np.full(NUM_PAIRS, 1.0 / NUM_PAIRS)
    variance_rows, policy_rows, distance_rows, correlation_rows = [], [], [], []
    wide_rows = []
    max_gram_pair_error = 0.0
    raw_width_resource_l1 = []

    for epoch in EPOCHS:
        state = table.loc[table.epoch.astype(int).eq(epoch)].sort_values(
            ["width_i", "width_j"]
        ).reset_index(drop=True)
        if len(state) != NUM_PAIRS:
            raise RuntimeError(f"Epoch {epoch} does not contain exactly 91 pairs")
        expected_pairs = [(INTERIOR_WIDTHS[i], INTERIOR_WIDTHS[j]) for i, j in PAIR_INDICES]
        actual_pairs = list(zip(state.width_i.round(2), state.width_j.round(2)))
        if actual_pairs != expected_pairs:
            raise RuntimeError(f"Pair ordering/content mismatch at epoch {epoch}")

        gram = _load_gram(fresh_root, epoch)
        oracle_scores, _ = gram_pair_scores(gram)
        csv_gradient = state.gradient_distance_sq.to_numpy(float)
        relative_error = np.max(
            np.abs(csv_gradient - oracle_scores) / np.maximum(np.abs(oracle_scores), 1e-30)
        )
        max_gram_pair_error = max(max_gram_pair_error, float(relative_error))
        if relative_error > 1e-5:
            raise RuntimeError(
                f"Pair table and Gram-derived gradient distances disagree at epoch {epoch}: {relative_error}"
            )

        raw_resource_scores = state.log_flops_distance_sq.to_numpy(float)
        raw_width_scores = state.width_distance_sq.to_numpy(float)
        sw_scores = state.sw_sq.to_numpy(float)
        frozen_resource_scores = frozen_predict(predictor, "Resource", state)
        frozen_hybrid_scores = frozen_predict(predictor, "Resource+SW", state)
        policies = {
            "uniform_pair": uniform_q,
            "raw_resource": solve_pair_lp(raw_resource_scores, maximize=True),
            "pure_sw": solve_pair_lp(sw_scores, maximize=True),
            "frozen_resource": solve_pair_lp(frozen_resource_scores, maximize=True),
            "frozen_resource_plus_sw": solve_pair_lp(frozen_hybrid_scores, maximize=True),
            "gradient_oracle": solve_pair_lp(oracle_scores, maximize=True),
        }
        q_width = solve_pair_lp(raw_width_scores, maximize=True)
        raw_width_resource_l1.append(float(np.abs(q_width - policies["raw_resource"]).sum()))
        variances = {name: _variance(gram, q) for name, q in policies.items()}
        tolerance = 1e-8 * max(1.0, max(abs(value) for value in variances.values()))
        if any(variances["gradient_oracle"] > value + tolerance for value in variances.values()):
            raise RuntimeError(f"Gradient-oracle LP is not variance-optimal at epoch {epoch}")

        oracle_q = policies["gradient_oracle"]
        for policy, q in policies.items():
            variance_rows.append({
                "state": f"F{epoch}", "seed": 6, "epoch": epoch, "policy": policy,
                "exact_variance": variances[policy],
                "delta_vs_uniform": variances[policy] - variances["uniform_pair"],
                "oracle_gap_captured": _oracle_gap_captured(
                    variances["uniform_pair"], variances[policy], variances["gradient_oracle"]
                ),
            })
            distance_rows.append({
                "state": f"F{epoch}", "seed": 6, "epoch": epoch, "policy": policy,
                "l1_to_gradient_oracle": float(np.abs(q - oracle_q).sum()),
            })
            for pair_index, ((i, j), probability) in enumerate(zip(PAIR_INDICES, q)):
                policy_rows.append({
                    "state": f"F{epoch}", "seed": 6, "epoch": epoch,
                    "policy": policy, "pair_index": pair_index,
                    "width_i": INTERIOR_WIDTHS[i], "width_j": INTERIOR_WIDTHS[j],
                    "probability": float(probability),
                })
        rho_sw = spearmanr(sw_scores, oracle_scores)
        rho_resource = spearmanr(raw_resource_scores, oracle_scores)
        correlation_rows.append({
            "state": f"F{epoch}", "seed": 6, "epoch": epoch,
            "spearman_sw2_gradient_distance2": float(rho_sw.statistic),
            "spearman_sw2_p": float(rho_sw.pvalue),
            "spearman_raw_resource_gradient_distance2": float(rho_resource.statistic),
            "spearman_raw_resource_p": float(rho_resource.pvalue),
        })
        wide_rows.append({
            "state": f"F{epoch}", "seed": 6, "epoch": epoch,
            **{f"V_{name}": value for name, value in variances.items()},
            "L1_raw_resource_to_oracle": float(np.abs(policies["raw_resource"] - oracle_q).sum()),
            "L1_pure_sw_to_oracle": float(np.abs(policies["pure_sw"] - oracle_q).sum()),
            "L1_frozen_resource_to_oracle": float(np.abs(policies["frozen_resource"] - oracle_q).sum()),
            "L1_hybrid_to_oracle": float(np.abs(policies["frozen_resource_plus_sw"] - oracle_q).sum()),
            "pure_sw_beats_uniform": bool(variances["pure_sw"] < variances["uniform_pair"]),
            "pure_sw_beats_raw_resource": bool(variances["pure_sw"] < variances["raw_resource"]),
            "hybrid_beats_frozen_resource": bool(
                variances["frozen_resource_plus_sw"] < variances["frozen_resource"]
            ),
        })

    long = pd.DataFrame(variance_rows)
    assignments = pd.DataFrame(policy_rows)
    distances = pd.DataFrame(distance_rows)
    correlations = pd.DataFrame(correlation_rows)
    wide = pd.DataFrame(wide_rows)
    decision, interpretation = _diagnose(wide, correlations)

    long.to_csv(output_dir / "fresh_seed6_policy_variance_long.csv", index=False)
    wide.to_csv(output_dir / "fresh_seed6_policy_decomposition.csv", index=False)
    assignments.to_csv(output_dir / "fresh_seed6_pair_policy_assignments.csv", index=False)
    distances.to_csv(output_dir / "fresh_seed6_policy_l1_to_oracle.csv", index=False)
    correlations.to_csv(output_dir / "fresh_seed6_sw_gradient_correlations.csv", index=False)
    _plot_results(output_dir, long, distances)

    summary = {
        "status": "FRESH_SEED6_POLICY_DECOMPOSITION_COMPLETE",
        "decision": decision,
        "interpretation": interpretation,
        "fresh_seed": 6,
        "states": [f"F{epoch}" for epoch in EPOCHS],
        "policies": list(POLICY_ORDER),
        "raw_resource_definition": "maximize squared log-FLOPs distance under fixed uniform marginals",
        "raw_width_vs_log_flops_policy_max_l1": float(max(raw_width_resource_l1)),
        "pure_sw_beats_uniform_states": int(wide.pure_sw_beats_uniform.sum()),
        "pure_sw_beats_raw_resource_states": int(wide.pure_sw_beats_raw_resource.sum()),
        "frozen_hybrid_beats_frozen_resource_states": int(wide.hybrid_beats_frozen_resource.sum()),
        "positive_sw_gradient_spearman_states": int(
            (correlations.spearman_sw2_gradient_distance2 > 0).sum()
        ),
        "max_relative_pair_distance_error_csv_vs_gram": float(max_gram_pair_error),
        "predictors_refit": False,
        "checkpoint_loaded": False,
        "model_forward_or_backward": False,
        "gpu_required": False,
        "accuracy_used": False,
        "gate_c_authorized": False,
        "gate_state": {
            "Gate A": "PASS",
            "Development LOSO/B0": "positive",
            "Fresh Gate B1 seed 6": "NO-GO",
            "Frozen Resource+SW transfer": "FAIL",
            "Pure SW fresh transfer": decision,
            "Gate C end-to-end": "NOT AUTHORIZED",
        },
        "sources": {
            "pair_structure": str(pair_path),
            "pair_structure_sha256": _sha256(pair_path),
            "frozen_predictor": str(frozen_predictor_path),
            "frozen_predictor_sha256": _sha256(frozen_predictor_path),
            "gradient_grams": {
                str(epoch): {"path": str(_gram_path(fresh_root, epoch)), "sha256": _sha256(_gram_path(fresh_root, epoch))}
                for epoch in EPOCHS
            },
        },
    }
    (output_dir / "fresh_seed6_policy_decomposition_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    report_lines = [
        "# Fresh seed-6 policy decomposition", "",
        f"**Decision:** `{decision}`", "", interpretation, "",
        "No checkpoint was loaded, no predictor was refit, and no GPU/model execution occurred.", "",
        "## State results", "", _markdown_table(wide), "",
        "## Gate state", "",
    ] + [f"- {key}: **{value}**" for key, value in summary["gate_state"].items()]
    (output_dir / "fresh_seed6_policy_decomposition_summary.md").write_text(
        "\n".join(report_lines) + "\n"
    )
    return summary
