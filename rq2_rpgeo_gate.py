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
    }
    (output_dir / "rpgeo_offline_gate.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


__all__ = ["GATE_STATES", "run_rpgeo_offline_gate"]
