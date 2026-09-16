"""Fresh seed-7 Pure-SW replication using the locked Gate B1 protocol."""

from __future__ import annotations

import json
import shutil
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rq2_fresh_seed_gate_b1 as base
from rq2_pairwise_surrogate_regret import (
    INTERIOR_WIDTHS,
    NUM_PAIRS,
    PAIR_INDICES,
    _variance,
    gram_pair_scores,
    solve_pair_lp,
)


FRESH_SEED = 7
METHOD = "fresh_uniform_fixed_puresw_replication"
CHECKPOINT_EPOCHS = (10, 50, 100)
POLICIES = ("uniform_pair", "raw_resource", "pure_sw", "gradient_oracle")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_cifar100_root(input_root: str | Path) -> Path:
    """Return the torchvision root containing one valid cifar-100-python folder."""
    input_root = Path(input_root)
    candidates = sorted({
        path.parent for path in input_root.rglob("cifar-100-python")
        if path.is_dir() and all((path / name).is_file() for name in ("train", "test", "meta"))
    })
    dedicated = [path for path in candidates if str(path).startswith("/kaggle/input/datasets/")]
    preferred = dedicated or candidates
    if len(preferred) != 1:
        raise FileNotFoundError(
            "Attach exactly one CIFAR-100 dataset containing "
            f"cifar-100-python/{{train,test,meta}}; found={preferred}"
        )
    return preferred[0]


def find_gate_a_summary(input_root: str | Path) -> Path:
    """Resolve one semantically unique frozen Gate-A FLOPs audit."""
    candidates = []
    identities = {}
    for path in Path(input_root).rglob("gate_a_summary.json"):
        try:
            payload = json.loads(path.read_text())
            flops = payload["flops"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
        if len(flops) == len(INTERIOR_WIDTHS) and all(float(value) > 0 for value in flops.values()):
            candidates.append(path)
            identity = tuple(
                float(flops[f"{width:.2f}"]) for width in INTERIOR_WIDTHS
            )
            identities.setdefault(identity, []).append(path)
    if len(identities) != 1:
        raise FileNotFoundError(
            "Attach one semantically unique frozen gate_a_summary.json with FLOPs for 14 widths; "
            f"found={candidates}"
        )
    return sorted(next(iter(identities.values())), key=lambda path: (len(str(path)), str(path)))[0]


def _configure_base() -> None:
    """Configure the already-audited seed-6 machinery for an isolated process."""
    base.FRESH_SEED = FRESH_SEED
    base.METHOD = METHOD


def load_config(config_path: str | Path, dataset_root: str | Path) -> dict:
    _configure_base()
    return base.load_fresh_config(config_path, dataset_root)


def materialize_progress(input_root, destination, extraction_root) -> Path:
    _configure_base()
    return base.materialize_fresh_progress(input_root, destination, extraction_root)


def train_trajectory(config: dict, root: str | Path) -> Path:
    _configure_base()
    return base.train_fresh_trajectory(config, root)


def extract_state(root, epoch, dataset_root, gate_a_summary, output_dir, device="cuda:0"):
    _configure_base()
    return base.extract_fresh_state(
        root, epoch, dataset_root, gate_a_summary, output_dir, device
    )


def _gap_captured(v_uniform: float, value: float, v_oracle: float) -> float:
    denominator = v_uniform - v_oracle
    return (v_uniform - value) / denominator if denominator > 1e-12 else np.nan


def _bootstrap_difference(
    grams: np.ndarray,
    left_q: np.ndarray,
    right_q: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    per_batch = np.asarray([
        _variance(gram, left_q) - _variance(gram, right_q) for gram in grams
    ])
    rng = np.random.default_rng(seed)
    means = np.asarray([
        per_batch[rng.integers(0, len(per_batch), size=len(per_batch))].mean()
        for _ in range(int(draws))
    ])
    return tuple(map(float, np.quantile(means, [0.025, 0.975])))


def merge_and_evaluate(
    root: str | Path,
    worker_dirs: list[str | Path],
    bootstrap_draws: int = 10000,
    bootstrap_seed: int = 20260919,
) -> dict:
    """Merge three read-only states and evaluate four parameter-free policies."""
    root = Path(root)
    workers = [Path(path) for path in worker_dirs]
    root.mkdir(parents=True, exist_ok=True)
    metadata = [json.loads((path / "metadata.json").read_text()) for path in workers]
    if {int(item["seed"]) for item in metadata} != {FRESH_SEED}:
        raise RuntimeError("Workers are not from fresh seed 7")
    if {int(item["epoch"]) for item in metadata} != set(CHECKPOINT_EPOCHS):
        raise RuntimeError("Workers must cover epochs 10/50/100 exactly")
    if len({tuple(item["gradient_probe_ids"]) for item in metadata}) != 1:
        raise RuntimeError("Gradient samples/order differ across checkpoints")
    if len({tuple(item["geometry_probe_ids"]) for item in metadata}) != 1:
        raise RuntimeError("Geometry samples/order differ across checkpoints")

    for name in ("sw_matrices", "gradient_grams", "grams"):
        (root / name).mkdir(exist_ok=True)
    pair_frames = []
    for worker, item in zip(workers, metadata):
        epoch = int(item["epoch"])
        pair_frames.append(pd.read_csv(worker / f"pair_structure_epoch_{epoch:03d}.csv"))
        shutil.copy2(
            worker / f"sw_matrix_epoch_{epoch:03d}.npy",
            root / "sw_matrices" / f"epoch_{epoch:03d}.npy",
        )
        source_gram = worker / f"gradient_gram_epoch_{epoch:03d}.npy"
        shutil.copy2(source_gram, root / "gradient_grams" / f"epoch_{epoch:03d}.npy")
        shutil.copy2(
            source_gram,
            root / "grams" / f"seed_{FRESH_SEED}_epoch_{epoch:03d}.npy",
        )
    pairs = pd.concat(pair_frames, ignore_index=True).sort_values(
        ["epoch", "width_i", "width_j"]
    )
    if len(pairs) != len(CHECKPOINT_EPOCHS) * NUM_PAIRS:
        raise RuntimeError("Merged pair table must contain 273 rows")
    pairs.to_csv(root / "pair_structure.csv", index=False)

    uniform_q = np.full(NUM_PAIRS, 1.0 / NUM_PAIRS)
    result_rows, policy_rows, bootstrap_rows = [], [], []
    for epoch in CHECKPOINT_EPOCHS:
        state = pairs.loc[pairs.epoch.astype(int).eq(epoch)].sort_values(
            ["width_i", "width_j"]
        ).reset_index(drop=True)
        expected = [(INTERIOR_WIDTHS[i], INTERIOR_WIDTHS[j]) for i, j in PAIR_INDICES]
        if list(zip(state.width_i.round(2), state.width_j.round(2))) != expected:
            raise RuntimeError(f"Pair alignment mismatch at epoch {epoch}")
        grams = np.load(root / "gradient_grams" / f"epoch_{epoch:03d}.npy").astype(np.float64)
        if grams.ndim != 3 or grams.shape[1:] != (14, 14):
            raise RuntimeError(f"Expected an 8x14x14 Gram stack, got {grams.shape}")
        mean_gram = grams.mean(axis=0)
        oracle_scores, _ = gram_pair_scores(mean_gram)
        sw_scores = np.square(state.representation_sw.to_numpy(float))
        resource_scores = np.square(
            np.log(state.flops_i.to_numpy(float)) - np.log(state.flops_j.to_numpy(float))
        )
        q = {
            "uniform_pair": uniform_q,
            "raw_resource": solve_pair_lp(resource_scores, maximize=True),
            "pure_sw": solve_pair_lp(sw_scores, maximize=True),
            "gradient_oracle": solve_pair_lp(oracle_scores, maximize=True),
        }
        variance = {name: _variance(mean_gram, values) for name, values in q.items()}
        tolerance = 1e-8 * max(1.0, max(map(abs, variance.values())))
        if any(variance["gradient_oracle"] > value + tolerance for value in variance.values()):
            raise RuntimeError("Gradient oracle is not variance-optimal")
        ci_sw_u = _bootstrap_difference(
            grams, q["pure_sw"], q["uniform_pair"], bootstrap_draws, bootstrap_seed + epoch
        )
        ci_sw_r = _bootstrap_difference(
            grams, q["pure_sw"], q["raw_resource"], bootstrap_draws,
            bootstrap_seed + 1000 + epoch,
        )
        bootstrap_rows.extend([
            {"state": f"F{epoch}", "epoch": epoch, "contrast": "V_SW_minus_V_uniform",
             "estimate": variance["pure_sw"] - variance["uniform_pair"],
             "ci_2_5": ci_sw_u[0], "ci_97_5": ci_sw_u[1]},
            {"state": f"F{epoch}", "epoch": epoch, "contrast": "V_SW_minus_V_resource",
             "estimate": variance["pure_sw"] - variance["raw_resource"],
             "ci_2_5": ci_sw_r[0], "ci_97_5": ci_sw_r[1]},
        ])
        result_rows.append({
            "state": f"F{epoch}", "seed": FRESH_SEED, "epoch": epoch,
            "V_uniform": variance["uniform_pair"],
            "V_resource": variance["raw_resource"],
            "V_SW": variance["pure_sw"],
            "V_oracle": variance["gradient_oracle"],
            "relative_delta_SW_minus_uniform": (
                variance["pure_sw"] - variance["uniform_pair"]
            ) / max(abs(variance["uniform_pair"]), 1e-30),
            "relative_delta_SW_minus_resource": (
                variance["pure_sw"] - variance["raw_resource"]
            ) / max(abs(variance["raw_resource"]), 1e-30),
            "SW_oracle_gap_captured": _gap_captured(
                variance["uniform_pair"], variance["pure_sw"], variance["gradient_oracle"]
            ),
            "resource_oracle_gap_captured": _gap_captured(
                variance["uniform_pair"], variance["raw_resource"], variance["gradient_oracle"]
            ),
            "L1_SW_to_oracle": float(np.abs(q["pure_sw"] - q["gradient_oracle"]).sum()),
            "L1_resource_to_oracle": float(np.abs(q["raw_resource"] - q["gradient_oracle"]).sum()),
            "SW_beats_uniform": bool(variance["pure_sw"] < variance["uniform_pair"]),
            "SW_beats_resource": bool(variance["pure_sw"] < variance["raw_resource"]),
        })
        for policy, values in q.items():
            for pair_index, ((i, j), probability) in enumerate(zip(PAIR_INDICES, values)):
                policy_rows.append({
                    "state": f"F{epoch}", "seed": FRESH_SEED, "epoch": epoch,
                    "policy": policy, "pair_index": pair_index,
                    "width_i": INTERIOR_WIDTHS[i], "width_j": INTERIOR_WIDTHS[j],
                    "probability": float(probability),
                })

    evaluation = root / "evaluation"
    evaluation.mkdir(exist_ok=True)
    results = pd.DataFrame(result_rows)
    results.to_csv(evaluation / "fresh_seed7_puresw_replication.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(
        evaluation / "fresh_seed7_pair_policies.csv", index=False
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        evaluation / "fresh_seed7_bootstrap_contrasts.csv", index=False
    )
    x = np.arange(len(results)); width = 0.2
    fig, axis = plt.subplots(figsize=(9, 5))
    for index, (column, label) in enumerate((
        ("V_uniform", "Uniform"), ("V_resource", "Raw Resource"),
        ("V_SW", "Pure SW"), ("V_oracle", "Gradient oracle"),
    )):
        axis.bar(x + (index - 1.5) * width, results[column], width, label=label)
    axis.set(
        xticks=x, xticklabels=results.state, ylabel="Exact fixed-marginal estimator variance"
    )
    axis.grid(axis="y", alpha=0.25); axis.legend()
    fig.tight_layout(); fig.savefig(evaluation / "fresh_seed7_puresw_variance.png", dpi=200)
    plt.close(fig)
    wins = int(results.SW_beats_uniform.sum())
    decision = "PASS" if wins >= 2 else "NO_GO"
    summary = {
        "status": "FRESH_SEED7_PURE_SW_REPLICATION_COMPLETE",
        "decision": decision,
        "primary_gate": "V_SW < V_uniform in at least 2/3 checkpoints",
        "SW_beats_uniform_states": wins,
        "SW_beats_resource_states": int(results.SW_beats_resource.sum()),
        "accuracy_used_for_gate": False,
        "fixed_uniform_marginals": True,
        "marginal_probability": 1.0 / 7.0,
        "predictor_fitted_or_used": False,
        "three_way_development_pilot_authorized": bool(decision == "PASS"),
        "gate_c_end_to_end_authorized": False,
        "fresh_seed": FRESH_SEED,
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
    }
    (evaluation / "fresh_seed7_puresw_replication_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    report = [
        "# Fresh seed-7 Pure-SW replication", "",
        f"**Decision:** `{decision}`", "",
        "Primary gate: Pure-SW exact variance is below Uniform at least 2/3 checkpoints.", "",
        "Accuracy and learned predictors were not used. All pair policies retain marginal pi_i = 1/7.", "",
        "| state | V_uniform | V_resource | V_SW | V_oracle | SW beats Uniform | SW beats Resource |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in results.itertuples():
        report.append(
            f"| {row.state} | {row.V_uniform:.8g} | {row.V_resource:.8g} | "
            f"{row.V_SW:.8g} | {row.V_oracle:.8g} | {row.SW_beats_uniform} | "
            f"{row.SW_beats_resource} |"
        )
    (evaluation / "fresh_seed7_puresw_replication_summary.md").write_text(
        "\n".join(report) + "\n"
    )
    state_metadata = {
        "status": "FRESH_PAIRWISE_STATES_COMPLETE", "seeds": [FRESH_SEED],
        "epochs": list(CHECKPOINT_EPOCHS), "trajectory": METHOD,
        "predictor_used_during_extraction": False, "test_used": False,
    }
    (root / "state_metadata.json").write_text(json.dumps(state_metadata, indent=2) + "\n")
    return summary


__all__ = [
    "FRESH_SEED", "CHECKPOINT_EPOCHS", "METHOD", "load_config",
    "materialize_progress", "train_trajectory", "extract_state", "merge_and_evaluate",
    "find_cifar100_root", "find_gate_a_summary",
]
