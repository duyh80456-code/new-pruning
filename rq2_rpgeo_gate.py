"""Offline exact-face gate for Resource-Preserving Geometry refinement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rq2_e2e_pairwise_pilot import _flops
from rq2_gradient_variance_v3 import importance_corrected_variance
from rq2_pair_policies import (
    ResourcePairPolicy,
    SWPairPolicy,
    pair_table,
    solve_resource_preserving_geo,
)
from rq2_pairwise_surrogate_regret import (
    TARGET_WEIGHTS,
    UNIFORM_PI,
    gram_pair_scores,
    solve_pair_lp,
)


GATE_STATES = (
    ("common_warmup", 10),
    ("resource", 50),
    ("resource", 100),
    ("pure_sw", 50),
    ("pure_sw", 100),
)
RETENTION_PROBE_GRID = (0.9999, 0.999, 0.995, 0.99, 0.98, 0.95)


def _variance(gram: np.ndarray, q: np.ndarray, name: str) -> float:
    return float(importance_corrected_variance(
        gram, TARGET_WEIGHTS, UNIFORM_PI, pair_table(q, name)
    )["importance_corrected_variance"])


def run_rpgeo_offline_gate(
    pilot_root: str | Path,
    gate_a_summary: str | Path,
    output_dir: str | Path,
) -> dict:
    """Evaluate exact-face RP-Geo on frozen states without updating a model."""
    pilot_root, output_dir = Path(pilot_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flops = _flops(gate_a_summary)
    q_resource = ResourcePairPolicy(flops).probabilities
    anti_monotone = np.count_nonzero(q_resource > 1e-10) == 7
    rows, policy_rows = [], []
    for method, epoch in GATE_STATES:
        state = f"{method}_E{epoch}"
        source = pilot_root / "diagnostics" / state
        gram_path, sw_path = source / "gradient_grams.npy", source / "sw_matrix.npy"
        if not gram_path.is_file() or not sw_path.is_file():
            raise FileNotFoundError(
                f"Missing frozen diagnostic state {state}; expected {gram_path} and {sw_path}"
            )
        grams = np.load(gram_path)
        gram = np.asarray(grams, dtype=float).mean(axis=0) if grams.ndim == 3 else grams
        sw = np.asarray(np.load(sw_path), dtype=float)
        q_rg, diagnostics = solve_resource_preserving_geo(
            sw, flops, resource_retention=1.0
        )
        q_sw = SWPairPolicy(sw).probabilities
        q_oracle = solve_pair_lp(gram_pair_scores(gram)[0], maximize=True)
        variances = {
            "resource": _variance(gram, q_resource, "resource"),
            "resource_geo": _variance(gram, q_rg, "resource_geo"),
            "pure_sw": _variance(gram, q_sw, "pure_sw"),
            "gradient_oracle": _variance(gram, q_oracle, "gradient_oracle"),
        }
        variance_tolerance = 1e-8 * max(1.0, abs(variances["resource"]))
        row = {
            "state": state,
            "method": method,
            "epoch": epoch,
            **diagnostics,
            "V_resource": variances["resource"],
            "V_resource_geo": variances["resource_geo"],
            "V_pure_sw": variances["pure_sw"],
            "V_oracle": variances["gradient_oracle"],
            "resource_geo_variance_delta": (
                variances["resource_geo"] - variances["resource"]
            ),
            "exact_face_has_freedom": bool(diagnostics["l1_vs_resource"] > 1e-7),
            "resource_geo_beats_resource": bool(
                variances["resource_geo"] < variances["resource"] - variance_tolerance
            ),
        }
        rows.append(row)
        for name, q in (
            ("resource", q_resource), ("resource_geo", q_rg),
            ("pure_sw", q_sw), ("gradient_oracle", q_oracle),
        ):
            frame = pair_table(q, name)
            frame.insert(0, "state", state)
            policy_rows.extend(frame.to_dict("records"))
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "rpgeo_offline_gate.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(output_dir / "rpgeo_gate_pair_policies.csv", index=False)
    freedom_count = int(table.exact_face_has_freedom.sum())
    win_count = int(table.resource_geo_beats_resource.sum())
    if freedom_count == len(table) and win_count == len(table):
        decision = "GO"
    elif freedom_count == 0 or win_count == 0:
        decision = "NO_GO"
    else:
        decision = "REVIEW_REQUIRED"
    summary = {
        "status": "RPGEO_OFFLINE_EXACT_FACE_GATE_COMPLETE",
        "decision": decision,
        "resource_retention": 1.0,
        "states": list(table.state),
        "state_count": int(len(table)),
        "exact_face_freedom_states": freedom_count,
        "variance_win_states": win_count,
        "all_resource_constraints_verified": bool(
            (table.resource_retention_achieved >= 1.0 - 1e-7).all()
        ),
        "training_authorized": decision == "GO",
        "decision_rule": (
            "GO only when exact-face freedom and strict exact-variance improvement hold "
            "on every frozen state; mixed evidence requires review and does not auto-train"
        ),
        "accuracy_used": False,
        "model_updates": 0,
        "resource_antimonotone_seven_pair_solution_verified": bool(anti_monotone),
    }
    (output_dir / "rpgeo_offline_gate.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def run_rpgeo_retention_probe(
    pilot_root: str | Path,
    gate_a_summary: str | Path,
    output_dir: str | Path,
    retentions: tuple[float, ...] = RETENTION_PROBE_GRID,
) -> dict:
    """Build a mechanistic Resource-retention/variance Pareto table; never train."""
    pilot_root, output_dir = Path(pilot_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    retention_values = tuple(float(value) for value in retentions)
    if (
        not retention_values or len(set(retention_values)) != len(retention_values)
        or any(not 0.0 < value < 1.0 for value in retention_values)
    ):
        raise ValueError("Retention probe requires unique values strictly between zero and one")
    flops = _flops(gate_a_summary)
    q_resource = ResourcePairPolicy(flops).probabilities
    rows = []
    for method, epoch in GATE_STATES:
        state = f"{method}_E{epoch}"
        source = pilot_root / "diagnostics" / state
        grams = np.load(source / "gradient_grams.npy")
        gram = np.asarray(grams, float).mean(axis=0) if grams.ndim == 3 else grams
        sw = np.asarray(np.load(source / "sw_matrix.npy"), float)
        v_resource = _variance(gram, q_resource, "resource")
        for retention in retention_values:
            q_rg, diagnostics = solve_resource_preserving_geo(
                sw, flops, resource_retention=retention
            )
            v_rg = _variance(gram, q_rg, f"resource_geo_{retention:g}")
            tolerance = 1e-8 * max(1.0, abs(v_resource))
            rows.append({
                "state": state, "method": method, "epoch": epoch,
                "resource_retention_target": retention,
                **diagnostics,
                "V_resource": v_resource,
                "V_resource_geo": v_rg,
                "variance_delta": v_rg - v_resource,
                "variance_win": bool(v_rg < v_resource - tolerance),
                "has_meaningful_movement": bool(diagnostics["l1_vs_resource"] > 1e-7),
                "geometry_gain": diagnostics["geo_rg"] - diagnostics["geo_resource"],
            })
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "rpgeo_retention_probe_by_state.csv", index=False)
    aggregate = table.groupby("resource_retention_target", as_index=False).agg(
        states=("state", "nunique"),
        variance_win_states=("variance_win", "sum"),
        movement_states=("has_meaningful_movement", "sum"),
        mean_variance_delta=("variance_delta", "mean"),
        worst_variance_delta=("variance_delta", "max"),
        mean_l1_vs_resource=("l1_vs_resource", "mean"),
        mean_geometry_gain=("geometry_gain", "mean"),
        minimum_resource_retention_achieved=("resource_retention_achieved", "min"),
    ).sort_values("resource_retention_target", ascending=False)
    aggregate["all_state_mechanistic_pass"] = (
        aggregate.variance_win_states.eq(len(GATE_STATES))
        & aggregate.movement_states.eq(len(GATE_STATES))
    )
    aggregate.to_csv(output_dir / "rpgeo_retention_pareto.csv", index=False)
    eligible = aggregate.loc[aggregate.all_state_mechanistic_pass]
    candidate = (
        None if eligible.empty else float(eligible.resource_retention_target.max())
    )
    summary = {
        "status": "RPGEO_NEAR_OPTIMAL_RETENTION_PROBE_COMPLETE",
        "retention_grid": list(retention_values),
        "selection_uses_accuracy": False,
        "candidate_rule": (
            "largest pre-enumerated retention with meaningful movement and strict exact-variance "
            "improvement on every frozen state"
        ),
        "mechanistic_candidate_retention": candidate,
        "candidate_requires_separate_freeze_before_e2e": True,
        "training_authorized": False,
        "model_updates": 0,
    }
    (output_dir / "rpgeo_retention_probe.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


__all__ = [
    "GATE_STATES", "RETENTION_PROBE_GRID", "run_rpgeo_offline_gate",
    "run_rpgeo_retention_probe",
]
