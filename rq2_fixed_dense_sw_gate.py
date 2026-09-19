"""Frozen E10 soft-SW pair gate and exact shape-matched permutation control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from rq2_pair_policies import UniformPairPolicy, validate_pair_probabilities
from rq2_pairwise_surrogate_regret import (
    INTERIOR_WIDTHS, NUM_PAIRS, PAIR_INDICES, UNIFORM_PI, incidence_matrix,
)

KAPPAS = (0.1, 1.0, 10.0)
METHODS = ("fixed_dense_sw", "fixed_dense_shuffled")
PAIR_LOOKUP = {pair: index for index, pair in enumerate(PAIR_INDICES)}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_e10_sw(root: str | Path) -> np.ndarray:
    root = Path(root)
    if not (root / "common_warmup/epoch_010.pt").is_file():
        raise FileNotFoundError("The common E10 checkpoint is required")
    path = root / "pure_sw/sw_policies/epoch_010.npz"
    with np.load(path) as payload:
        sw = np.asarray(payload["sw_matrix"], dtype=float)
        widths = np.asarray(payload["widths"], dtype=float)
    if sw.shape != (14, 14) or not np.allclose(widths, INTERIOR_WIDTHS):
        raise RuntimeError("E10 SW matrix uses a different interior-width grid")
    if not np.isfinite(sw).all() or np.any(sw < 0) or not np.allclose(sw, sw.T):
        raise RuntimeError("Invalid E10 SW matrix")
    return sw


def score_vector(sw: np.ndarray) -> np.ndarray:
    return np.asarray([float(sw[i, j] ** 2) for i, j in PAIR_INDICES])


def soft_sw_policy(sw: np.ndarray, kappa: float) -> np.ndarray:
    """Maximize <q, SW²> + κ Std(SW²) H(q), with fixed 1/7 marginals."""
    scores = score_vector(sw)
    deviation = float(np.std(scores))
    if deviation <= 0 or not np.isfinite(deviation):
        raise RuntimeError("E10 SW scores have no finite variation")
    kappa = float(kappa)
    if kappa <= 0 or not np.isfinite(kappa):
        raise ValueError("kappa must be positive and finite")
    normalized = scores / deviation
    incidence = incidence_matrix()
    initial = UniformPairPolicy().probabilities

    def objective(q):
        return float(-normalized @ q + kappa * np.sum(q * np.log(q)))

    def jacobian(q):
        return -normalized + kappa * (np.log(q) + 1.0)

    result = minimize(
        objective, initial, jac=jacobian, method="SLSQP",
        bounds=[(1e-12, 1.0)] * NUM_PAIRS,
        constraints={"type": "eq", "fun": lambda q: incidence @ q - UNIFORM_PI,
                     "jac": lambda q: incidence},
        options={"ftol": 1e-12, "maxiter": 4000, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"Soft-SW solve failed for kappa={kappa}: {result.message}")
    return validate_pair_probabilities(result.x)


def policy_summary(q: np.ndarray, sw: np.ndarray) -> dict:
    q = validate_pair_probabilities(q)
    uniform = UniformPairPolicy().probabilities
    positive = q[q > 0]
    return {
        "entropy": float(-(positive * np.log(positive)).sum()),
        "effective_support": float(1.0 / np.square(q).sum()),
        "support_size": int(np.count_nonzero(q > 1e-8)),
        "l1_to_uniform": float(np.abs(q - uniform).sum()),
        "max_probability": float(q.max()),
        "expected_sw2_score": float(score_vector(sw) @ q),
        "marginal_error": float(np.max(np.abs(incidence_matrix() @ q - UNIFORM_PI))),
    }


def permute_width_labels(q: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    q = validate_pair_probabilities(q)
    permutation = np.asarray(permutation, dtype=int)
    if sorted(permutation.tolist()) != list(range(14)):
        raise ValueError("Expected a permutation of the 14 width labels")
    shuffled = np.zeros_like(q)
    for index, (i, j) in enumerate(PAIR_INDICES):
        pair = tuple(sorted((int(permutation[i]), int(permutation[j]))))
        shuffled[PAIR_LOOKUP[pair]] = q[index]
    return validate_pair_probabilities(shuffled)


def decorrelated_control(q: np.ndarray, sw: np.ndarray, seed: int = 20260919):
    """Freeze one width-label permutation with near-zero SW-score association."""
    scores = score_vector(sw)
    rng = np.random.default_rng(seed)
    candidates = []
    for index in range(256):
        permutation = rng.permutation(14)
        shuffled = permute_width_labels(q, permutation)
        correlation = float(spearmanr(scores, shuffled).statistic)
        candidates.append((abs(correlation), index, permutation, shuffled, correlation))
    _, index, permutation, shuffled, correlation = min(candidates, key=lambda row: (row[0], row[1]))
    summary_sw, summary_shuffle = policy_summary(q, sw), policy_summary(shuffled, sw)
    for key in ("entropy", "effective_support", "support_size", "max_probability", "marginal_error"):
        if not np.isclose(summary_sw[key], summary_shuffle[key], atol=1e-10):
            raise RuntimeError(f"Matched control changed {key}")
    return shuffled, permutation, {"permutation_candidate": int(index),
                                    "spearman_sw2_vs_control_q": correlation,
                                    "l1_to_sw_q": float(np.abs(shuffled - q).sum())}


def run_tau_probe(root: str | Path, output_dir: str | Path) -> pd.DataFrame:
    root, output_dir = Path(root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sw = load_e10_sw(root)
    deviation = float(np.std(score_vector(sw)))
    rows = []
    for kappa in KAPPAS:
        q = soft_sw_policy(sw, kappa)
        rows.append({"kappa": kappa, "tau": kappa * deviation, **policy_summary(q, sw)})
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "fixed_dense_sw_tau_probe.csv", index=False)
    source = {
        "status": "E10_TAU_PROBE_ONLY_NO_TRAINING",
        "common_e10_sha256": sha256(root / "common_warmup/epoch_010.pt"),
        "sw_e10_sha256": sha256(root / "pure_sw/sw_policies/epoch_010.npz"),
        "candidate_kappas": list(KAPPAS),
        "score": "SW_E10_squared",
        "temperature_rule": "tau=kappa*Std(91 SW_E10_squared scores)",
        "accuracy_used": False, "test_used": False,
    }
    (output_dir / "fixed_dense_sw_tau_source.json").write_text(json.dumps(source, indent=2) + "\n")
    return table


def freeze_policy(root: str | Path, probe_dir: str | Path, selected_kappa: float):
    root, probe_dir = Path(root), Path(probe_dir)
    source_path = probe_dir / "fixed_dense_sw_tau_source.json"
    source = json.loads(source_path.read_text())
    if source["common_e10_sha256"] != sha256(root / "common_warmup/epoch_010.pt"):
        raise RuntimeError("E10 checkpoint differs from the offline probe")
    if source["sw_e10_sha256"] != sha256(root / "pure_sw/sw_policies/epoch_010.npz"):
        raise RuntimeError("E10 SW differs from the offline probe")
    uniform_provenance_path = root / "uniform/training_provenance.json"
    if not (root / "uniform/checkpoints/epoch_100.pt").is_file() or not uniform_provenance_path.is_file():
        raise FileNotFoundError("Completed Uniform E100 anchor with provenance is required")
    uniform_provenance = json.loads(uniform_provenance_path.read_text())
    if uniform_provenance.get("common_epoch10_sha256") != source["common_e10_sha256"]:
        raise RuntimeError("Uniform anchor was not trained from the same common E10 checkpoint")
    pure_sw_provenance_path = root / "pure_sw/training_provenance.json"
    if not pure_sw_provenance_path.is_file():
        raise FileNotFoundError("Pure-SW provenance is required to authenticate E10 SW")
    pure_sw_provenance = json.loads(pure_sw_provenance_path.read_text())
    if pure_sw_provenance.get("common_epoch10_sha256") != source["common_e10_sha256"]:
        raise RuntimeError("E10 SW source was not trained from the same common E10 checkpoint")
    kappa = float(selected_kappa)
    if kappa not in KAPPAS:
        raise ValueError(f"Select one pre-probed kappa from {KAPPAS}")
    sw = load_e10_sw(root)
    q_sw = soft_sw_policy(sw, kappa)
    q_shuffle, permutation, control_info = decorrelated_control(q_sw, sw)
    protocol = {
        "status": "FROZEN_BEFORE_FIXED_DENSE_SW_E2E", "seed": 3,
        "selected_kappa": kappa, "tau": float(kappa * np.std(score_vector(sw))),
        "selection_source": "offline_E10_tau_probe_user_choice",
        "accuracy_used_to_choose_kappa": False, "test_used": False,
        "common_e10_sha256": source["common_e10_sha256"],
        "uniform_e100_sha256": sha256(root / "uniform/checkpoints/epoch_100.pt"),
        "sw_e10_sha256": source["sw_e10_sha256"],
        "tau_probe_sha256": sha256(probe_dir / "fixed_dense_sw_tau_probe.csv"),
        "score": "SW_E10_squared", "fixed_marginal": 1.0 / 7.0,
        "shuffled_control": "deterministic_width_label_permutation_of_q_sw",
        "control_permutation_seed": 20260919,
        "control_permutation": permutation.tolist(),
        "control_diagnostics": control_info,
        "policy_probabilities": {"fixed_dense_sw": q_sw.tolist(),
                                 "fixed_dense_shuffled": q_shuffle.tolist()},
        "policy_summaries": {"fixed_dense_sw": policy_summary(q_sw, sw),
                             "fixed_dense_shuffled": policy_summary(q_shuffle, sw)},
        "policy_frozen_at_epoch": 10, "training_epochs": [11, 100],
    }
    path = root / "fixed_dense_sw_frozen_protocol.json"
    encoded = json.dumps(protocol, indent=2) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError("Existing fixed-dense-SW freeze differs; refusing overwrite")
    if not path.exists():
        path.write_text(encoded)
    pd.concat([
        pd.DataFrame({"method": method, "pair_index": np.arange(NUM_PAIRS),
                      "width_i": [INTERIOR_WIDTHS[i] for i, _ in PAIR_INDICES],
                      "width_j": [INTERIOR_WIDTHS[j] for _, j in PAIR_INDICES],
                      "probability": q})
        for method, q in (("fixed_dense_sw", q_sw), ("fixed_dense_shuffled", q_shuffle))
    ]).to_csv(root / "fixed_dense_sw_policy_summary.csv", index=False)
    return protocol


def load_frozen_policy(root: str | Path) -> dict:
    root = Path(root)
    protocol = json.loads((root / "fixed_dense_sw_frozen_protocol.json").read_text())
    if protocol.get("status") != "FROZEN_BEFORE_FIXED_DENSE_SW_E2E":
        raise RuntimeError("Fixed-dense-SW policy is not frozen")
    if protocol["common_e10_sha256"] != sha256(root / "common_warmup/epoch_010.pt"):
        raise RuntimeError("Wrong common E10 checkpoint")
    if protocol["sw_e10_sha256"] != sha256(root / "pure_sw/sw_policies/epoch_010.npz"):
        raise RuntimeError("Wrong E10 SW source")
    for method in METHODS:
        validate_pair_probabilities(protocol["policy_probabilities"][method])
    return protocol


def summarize_gate(root: str | Path) -> pd.DataFrame:
    """Compact validation-only comparison; never turns one seed into a claim."""
    root = Path(root)
    methods = ("uniform",) + METHODS
    rows = []
    for method in methods:
        path = root / method / "dense_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        dense = pd.read_csv(path)
        final = dense.loc[dense.epoch.eq(100)]
        if len(final) != 16 or set(np.round(final.width, 2)) != set(np.round(np.arange(.25, 1.001, .05), 2)):
            raise RuntimeError(f"Incomplete dense E100 validation for {method}")
        interior = final.loc[final.width.isin(INTERIOR_WIDTHS)]
        rows.append({
            "seed": 3, "method": method, "split": "validation_5k",
            "dense_mean_accuracy": float(final.accuracy.mean()),
            "interior_mean_accuracy": float(interior.accuracy.mean()),
            "worst_interior_accuracy": float(interior.accuracy.min()),
            "low_accuracy": float(final.loc[final.width.between(.30, .45), "accuracy"].mean()),
            "mid_accuracy": float(final.loc[final.width.between(.55, .70), "accuracy"].mean()),
            "high_accuracy": float(final.loc[final.width.between(.80, .95), "accuracy"].mean()),
            "full_width_accuracy": float(final.loc[final.width.eq(1.0), "accuracy"].iloc[0]),
        })
    table = pd.DataFrame(rows)
    table.to_csv(root / "fixed_dense_sw_gate_comparison.csv", index=False)
    indexed = table.set_index("method")
    conclusion = {
        "status": "SEED3_DEVELOPMENT_GATE_COMPLETE",
        "primary_comparison": "fixed_dense_sw_vs_fixed_dense_shuffled",
        "interior_mean_sw_minus_shuffled": float(
            indexed.loc["fixed_dense_sw", "interior_mean_accuracy"]
            - indexed.loc["fixed_dense_shuffled", "interior_mean_accuracy"]
        ),
        "interior_mean_sw_minus_uniform": float(
            indexed.loc["fixed_dense_sw", "interior_mean_accuracy"]
            - indexed.loc["uniform", "interior_mean_accuracy"]
        ),
        "test_used": False,
        "single_seed_not_confirmatory": True,
    }
    (root / "fixed_dense_sw_gate_summary.json").write_text(json.dumps(conclusion, indent=2) + "\n")
    return table
