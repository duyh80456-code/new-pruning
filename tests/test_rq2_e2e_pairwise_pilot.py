import json
from pathlib import Path

import pandas as pd
import scripts.run_e2e_pairwise_pilot as runner

from rq2_e2e_pairwise_pilot import (
    DIAGNOSTIC_STATES,
    METHODS,
    RPGEO_DIAGNOSTIC_STATES,
    finalize,
    finalize_rpgeo_extension,
    finalize_screening,
)
from rq2_anchor_placement import GRID


def test_finalize_e2e_pairwise_pilot_is_development_only(tmp_path):
    rows = []
    for method_index, method in enumerate(METHODS):
        folder = tmp_path / method
        folder.mkdir()
        for epoch in range(10, 101, 10):
            for width in GRID:
                rows.append({
                    "method": method, "seed": 3, "epoch": epoch,
                    "split": "validation_5k", "width": width,
                    "accuracy": 0.5 + 0.01 * method_index + 0.001 * width,
                    "loss": 1.0,
                })
        pd.DataFrame([row for row in rows if row["method"] == method]).to_csv(
            folder / "dense_metrics.csv", index=False
        )
        pd.DataFrame([
            {
                "method": method, "seed": 3, "epoch": epoch,
                "train_loss": 1.0, "learning_rate": 0.1,
                "pair_uniform_draw_sha256": f"matched-{epoch}",
                "train_sample_order_sha256": f"data-{epoch}",
            }
            for epoch in range(11, 101)
        ]).to_csv(folder / "metrics.csv", index=False)
    for method, epoch in DIAGNOSTIC_STATES:
        folder = tmp_path / "diagnostics" / f"{method}_E{epoch}"
        folder.mkdir(parents=True)
        pd.DataFrame([{
            "state": f"{method}_E{epoch}", "method": method, "epoch": epoch,
            "V_uniform": 3.0, "V_resource": 2.5, "V_SW": 2.0,
            "V_oracle": 1.0, "SW_beats_uniform": True, "SW_beats_resource": True,
        }]).to_csv(folder / "variance.csv", index=False)
    result = finalize(tmp_path)
    assert result["status"] == "E2E_PAIRWISE_PILOT_COMPLETE"
    assert result["primary_dense_accuracy_pass"] is True
    assert result["single_development_seed_only"] is True
    assert result["confirmatory_claim_authorized"] is False
    assert result["matched_pair_uniform_draw_stream_verified"] is True
    assert result["matched_training_sample_order_verified"] is True
    assert (tmp_path / "method_summary.csv").is_file()
    assert (tmp_path / "trajectory_variance_diagnostics.csv").is_file()
    method_summary = pd.read_csv(tmp_path / "method_summary.csv")
    assert "interior_mean_accuracy" in method_summary.columns
    screening = finalize_screening(tmp_path)
    assert screening["status"] == "E2E_PAIRWISE_RESOURCE_PURE_SW_SCREENING_COMPLETE"
    assert screening["full_pilot_complete"] is False
    assert screening["uniform_pending"] is True
    assert screening["matched_training_sample_order_verified"] is True
    assert (tmp_path / "screening_method_summary.csv").is_file()

    method = "resource_geo"
    folder = tmp_path / method
    folder.mkdir()
    pd.DataFrame([
        {
            "method": method, "seed": 3, "epoch": epoch,
            "split": "validation_5k", "width": width,
            "accuracy": 0.54 + 0.001 * width, "loss": 1.0,
        }
        for epoch in range(10, 101, 10) for width in GRID
    ]).to_csv(folder / "dense_metrics.csv", index=False)
    pd.DataFrame([
        {
            "method": method, "seed": 3, "epoch": epoch,
            "train_loss": 1.0, "learning_rate": 0.1,
            "pair_uniform_draw_sha256": f"matched-{epoch}",
            "train_sample_order_sha256": f"data-{epoch}",
        }
        for epoch in range(11, 101)
    ]).to_csv(folder / "metrics.csv", index=False)
    for diagnostic_method, epoch in RPGEO_DIAGNOSTIC_STATES:
        diagnostic = tmp_path / "diagnostics" / f"{diagnostic_method}_E{epoch}"
        if not diagnostic.exists():
            diagnostic.mkdir(parents=True)
            pd.DataFrame([{
                "state": f"{diagnostic_method}_E{epoch}",
                "method": diagnostic_method, "epoch": epoch,
                "V_uniform": 3.0, "V_resource": 2.5, "V_SW": 2.0,
                "V_resource_geo": 1.8, "V_oracle": 1.0,
            }]).to_csv(diagnostic / "variance.csv", index=False)
    extension = finalize_rpgeo_extension(tmp_path)
    assert extension["status"] == "RPGEO_FOUR_WAY_DEVELOPMENT_COMPLETE"
    assert extension["rpgeo_beats_resource_dense_mean"] is True
    assert extension["confirmatory_claim_authorized"] is False


def test_completion_overlaps_uniform_with_existing_diagnostics(monkeypatch, tmp_path):
    calls = []

    def fake_branches(root, dataset_root, gate_a_summary, gpu_ids, methods):
        calls.append(("branches", tuple(gpu_ids), tuple(methods)))
        return pd.DataFrame([{"job": "uniform"}])

    def fake_diagnostics(root, dataset_root, gate_a_summary, gpu_ids, jobs):
        calls.append(("diagnostics", tuple(gpu_ids), tuple(jobs)))
        return pd.DataFrame([{"job": str(job)} for job in jobs])

    monkeypatch.setattr(runner, "run_branches", fake_branches)
    monkeypatch.setattr(runner, "run_diagnostics", fake_diagnostics)
    branch_runtime, diagnostic_runtime = runner.run_completion(
        tmp_path, tmp_path, tmp_path / "gate.json", gpu_ids=(0, 1)
    )
    assert not branch_runtime.empty and not diagnostic_runtime.empty
    assert ("branches", (0,), ("uniform",)) in calls
    existing = next(call for call in calls if call[0] == "diagnostics" and call[1] == (1,))
    assert {method for method, _ in existing[2]} == {"common_warmup", "resource", "pure_sw"}
    uniform = next(call for call in calls if call[0] == "diagnostics" and call[1] == (0, 1))
    assert {method for method, _ in uniform[2]} == {"uniform"}


def test_rpgeo_extension_stops_training_when_frozen_gate_is_not_go(monkeypatch, tmp_path):
    for path in (
        tmp_path / "common_warmup" / "epoch_010.pt",
        tmp_path / "resource" / "checkpoints" / "epoch_100.pt",
        tmp_path / "pure_sw" / "checkpoints" / "epoch_100.pt",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
    branch_calls = []

    def fake_branches(root, dataset_root, gate_a_summary, gpu_ids, methods):
        branch_calls.append(tuple(methods))
        return pd.DataFrame([{"job": methods[0]}])

    monkeypatch.setattr(runner, "run_branches", fake_branches)
    monkeypatch.setattr(
        runner, "run_diagnostics",
        lambda *args, **kwargs: pd.DataFrame([{"job": "diagnostic"}]),
    )
    monkeypatch.setattr(
        runner, "run_rpgeo_offline_gate",
        lambda *args, **kwargs: {"decision": "NO_GO", "training_authorized": False},
    )
    monkeypatch.setattr(
        runner, "run_rpgeo_retention_probe",
        lambda *args, **kwargs: {
            "status": "RPGEO_NEAR_OPTIMAL_RETENTION_PROBE_COMPLETE",
            "training_authorized": False,
        },
    )
    result, _, diagnostics = runner.run_rpgeo_extension(
        tmp_path, tmp_path, tmp_path / "gate.json", gpu_ids=(0, 1)
    )
    assert result["status"] == "RPGEO_EXTENSION_STOPPED_AT_GATE"
    assert result["retention_probe"]["training_authorized"] is False
    assert branch_calls == [("uniform",)]
    assert diagnostics.empty
