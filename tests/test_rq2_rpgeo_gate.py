import json

import numpy as np
import pandas as pd

from rq2_pairwise_surrogate_regret import INTERIOR_WIDTHS
from rq2_rpgeo_gate import (
    GATE_STATES,
    find_rpgeo_retention_probe,
    freeze_rpgeo_retention,
    run_rpgeo_offline_gate,
    run_rpgeo_retention_probe,
)
from rq2_e2e_pairwise_pilot import load_frozen_rpgeo_retention


def test_rpgeo_offline_gate_writes_audited_exact_face_results(tmp_path):
    root = tmp_path / "pilot"
    for state_index, (method, epoch) in enumerate(GATE_STATES):
        folder = root / "diagnostics" / f"{method}_E{epoch}"
        folder.mkdir(parents=True)
        diagonal = np.linspace(1.0, 2.0, len(INTERIOR_WIDTHS))
        gram = np.diag(diagonal)
        np.save(folder / "gradient_grams.npy", np.stack([gram, gram * 1.01]))
        coordinates = np.asarray(INTERIOR_WIDTHS)
        sw = np.abs(
            np.sin((state_index + 1) * coordinates[:, None])
            - np.sin((state_index + 1) * coordinates[None, :])
        )
        np.save(folder / "sw_matrix.npy", sw)
    gate_a = tmp_path / "gate_a_summary.json"
    gate_a.write_text(json.dumps({
        "flops": {
            f"{width:.2f}": 1e8 * (0.1 + width ** 2) for width in INTERIOR_WIDTHS
        }
    }))
    output = tmp_path / "gate"
    summary = run_rpgeo_offline_gate(root, gate_a, output)
    assert summary["status"] == "RPGEO_OFFLINE_EXACT_FACE_GATE_COMPLETE"
    assert summary["decision"] in {"GO", "NO_GO", "REVIEW_REQUIRED"}
    assert summary["model_updates"] == 0
    assert summary["decision"] == "NO_GO"
    assert summary["resource_antimonotone_seven_pair_solution_verified"] is True
    table = pd.read_csv(output / "rpgeo_offline_gate.csv")
    assert len(table) == len(GATE_STATES)
    assert (table.resource_retention_achieved >= 1.0 - 1e-7).all()
    assert table.l1_vs_resource.max() < 1e-7
    assert (output / "rpgeo_gate_pair_policies.csv").is_file()
    probe = run_rpgeo_retention_probe(
        root, gate_a, tmp_path / "probe", retentions=(0.999, 0.99)
    )
    assert probe["training_authorized"] is False
    assert probe["selection_uses_accuracy"] is False
    probe_dir = tmp_path / "probe"
    pareto_path = probe_dir / "rpgeo_retention_pareto.csv"
    assert pareto_path.is_file()
    pareto = pd.read_csv(pareto_path)
    assert "mean_oracle_gap_captured" in pareto
    assert "minimum_oracle_gap_captured" in pareto
    assert "oracle_gap_defined_states" in pareto
    assert "mean_resource_sacrifice" in pareto
    pareto["all_state_mechanistic_pass"] = False
    pareto.to_csv(pareto_path, index=False)
    with np.testing.assert_raises_regex(RuntimeError, "must pass movement"):
        freeze_rpgeo_retention(
            probe_dir, float(pareto.iloc[0].resource_retention_target),
            tmp_path / "invalid_freeze.json",
        )

    # Build an eligible synthetic Pareto row to exercise the immutable freeze path.
    pareto.loc[pareto.index[0], "all_state_mechanistic_pass"] = True
    pareto.to_csv(pareto_path, index=False)

    # Stage C accepts only a separately frozen, hash-bound artifact.
    root_probe = root / "rpgeo_retention_probe"
    root_probe.mkdir()
    for name in (
        "rpgeo_retention_probe.json",
        "rpgeo_retention_pareto.csv",
        "rpgeo_retention_probe_by_state.csv",
    ):
        (root_probe / name).write_bytes((probe_dir / name).read_bytes())
    frozen_path = root / "rpgeo_frozen_retention.json"
    selected_retention = float(pareto.iloc[0].resource_retention_target)
    frozen = freeze_rpgeo_retention(root_probe, selected_retention, frozen_path)
    assert frozen["selection_frozen"] is True
    assert frozen["frozen_before_rpgeo_e2e"] is True
    assert frozen["selection_source"] == "mechanistic_pareto_development"
    assert frozen["accuracy_used"] is False
    assert frozen["resource_retention"] == selected_retention
    assert load_frozen_rpgeo_retention(root)["resource_retention"] == selected_retention

    # Source mutation after freeze must invalidate the Stage-C gate.
    with (root_probe / "rpgeo_retention_probe_by_state.csv").open("a") as handle:
        handle.write("tampered\n")
    with np.testing.assert_raises_regex(RuntimeError, "source mismatch"):
        load_frozen_rpgeo_retention(root)

    attached = tmp_path / "attached"
    duplicate = attached / "notebook-output" / "rpgeo_retention_probe"
    duplicate.mkdir(parents=True)
    for name in (
        "rpgeo_retention_probe.json",
        "rpgeo_retention_pareto.csv",
        "rpgeo_retention_probe_by_state.csv",
    ):
        (duplicate / name).write_bytes((probe_dir / name).read_bytes())
    assert find_rpgeo_retention_probe(attached, tmp_path / "unused") == duplicate
