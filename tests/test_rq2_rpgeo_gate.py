import json

import numpy as np
import pandas as pd

from rq2_pairwise_surrogate_regret import INTERIOR_WIDTHS
from rq2_rpgeo_gate import (
    GATE_STATES,
    run_rpgeo_offline_gate,
    run_rpgeo_retention_probe,
)


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
    assert (tmp_path / "probe" / "rpgeo_retention_pareto.csv").is_file()
