"""Modal orchestration for the one-seed CIFAR-100 geometry smoke test.

All research logic is delegated to scripts.run_experiment. This module only
defines the remote environment, validates scientific invariants, persists output,
and returns compact notebook-friendly results.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal


APP_NAME = "new-pruning-wasserstein-smoke"
VOLUME_NAME = "new-pruning-smoke-data"
PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = Path("/workspace")
PERSISTENT_ROOT = Path("/persistent")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "numpy>=1.26,<2",
        "pandas>=2.0",
        "scipy>=1.11",
        "scikit-learn>=1.4",
        "matplotlib>=3.8",
        "PyYAML>=6.0",
        "thop>=0.1.1",
    )
    .env({"MPLBACKEND": "Agg", "PYTHONPATH": str(REMOTE_ROOT)})
    .add_local_dir(
        PROJECT_ROOT,
        remote_path=str(REMOTE_ROOT),
        ignore=[
            ".git/**",
            ".pytest_cache/**",
            "**/__pycache__/**",
            "data/**",
            "outputs/**",
        ],
    )
)


def _json_number(value):
    value = float(value)
    return value if value == value and abs(value) != float("inf") else None


def _assert_exact_configuration(config: dict) -> None:
    anchors = [float(x) for x in config["compression"]["train_widths"]]
    expected_anchors = [0.25, 0.50, 0.75, 1.00]
    expected_grid = [round(0.25 + 0.05 * index, 2) for index in range(16)]
    eval_widths = [float(x) for x in config["compression"]["eval_widths"]]
    assert config["dataset"]["name"].lower() == "cifar100"
    assert config["dataset"].get("fake_data") is False
    assert anchors == expected_anchors
    assert eval_widths == expected_grid
    assert int(config["experiment"]["seeds"][0]) == 0
    assert len(config["experiment"]["seeds"]) == 1
    assert int(config["training"]["epochs"]) == 20
    assert config["geometry"]["method"] == "sliced_wasserstein"
    assert config["geometry"].get("control_methods") == ["euclidean_mean"]
    assert int(config["geometry"]["num_projections"]) == 128
    assert config["oracle"]["enabled"] is False
    assert set(anchors).intersection(set(eval_widths) - set(anchors)) == set()


def _save_required_plots(metrics, local, matrix, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    anchors = metrics[metrics["is_train_anchor"]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(metrics["budget"], metrics["accuracy"], marker="o", label="all budgets")
    ax.scatter(
        anchors["budget"], anchors["accuracy"], s=80, marker="*", color="crimson", label="training anchors"
    )
    ax.set(xlabel="Width budget", ylabel="Accuracy", title="Accuracy vs width budget")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_budget.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(local["budget_start"], local["G_width"], marker="o")
    ax.set(
        xlabel="Interval start c",
        ylabel="G(c) = SW(c, c+0.05) / 0.05",
        title="Local Sliced Wasserstein sensitivity",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "local_wasserstein_sensitivity.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(local["budget_start"], local["P"], marker="o", color="darkorange")
    ax.set(
        xlabel="Interval start c",
        ylabel="P(c) = |Acc(c+0.05)-Acc(c)| / 0.05",
        title="Local accuracy sensitivity",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "local_accuracy_sensitivity.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    points = ax.scatter(local["G_width"], local["P"], c=local["budget_start"], cmap="viridis", s=55)
    ax.set(xlabel="Wasserstein sensitivity G(c)", ylabel="Accuracy sensitivity P(c)", title="Geometry vs performance")
    ax.grid(alpha=0.25)
    fig.colorbar(points, ax=ax, label="Interval start c")
    fig.tight_layout()
    fig.savefig(output_dir / "geometry_vs_performance.png", dpi=180)
    plt.close(fig)

    assert np.isfinite(matrix).all()


def _interpret(local, cliff_count: int, spearman_value: float | None) -> tuple[str, list[str]]:
    import numpy as np

    geometry = local["G_width"].to_numpy(float)
    performance = local["P"].to_numpy(float)
    relative_range = float(np.ptp(geometry) / max(abs(np.mean(geometry)), 1e-12))
    top_count = max(3, len(geometry) // 4)
    top_indices = np.argsort(geometry)[-top_count:]
    high_geometry_has_more_degradation = float(np.median(performance[top_indices])) > float(
        np.median(performance)
    )
    signals = {
        "geometry_not_nearly_flat": relative_range > 0.15,
        "high_geometry_tends_to_match_high_degradation": high_geometry_has_more_degradation,
        "notable_rank_association": spearman_value is not None and abs(spearman_value) >= 0.30,
        "compression_cliff_candidate_present": cliff_count > 0,
    }
    count = sum(signals.values())
    if count >= 3:
        label = "PROMISING SIGNAL"
    elif count >= 1:
        label = "WEAK / INCONCLUSIVE SIGNAL"
    else:
        label = "NO OBVIOUS SIGNAL"
    evidence = [f"{name}: {value}" for name, value in signals.items()]
    evidence.append(f"relative range of G(c): {relative_range:.4f}")
    return label, evidence


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=6 * 60 * 60,
    volumes={str(PERSISTENT_ROOT): volume},
)
def run_smoke(run_name: str) -> dict:
    import os
    import sys

    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from scipy.stats import pearsonr, spearmanr

    sys.path.insert(0, str(REMOTE_ROOT))
    os.chdir(REMOTE_ROOT)

    from data import build_loaders
    from research_utils import load_config
    from scripts.run_experiment import run

    assert torch.cuda.is_available(), "CUDA is unavailable; refusing to run the CIFAR-100 smoke test"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", run_name):
        raise ValueError("run_name must contain only letters, digits, dot, underscore, or hyphen")

    config = load_config(REMOTE_ROOT / "configs" / "modal_smoke.yaml")
    _assert_exact_configuration(config)
    run_dir = PERSISTENT_ROOT / "outputs" / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Output already exists: {run_dir}; choose a new run_name")
    config["experiment"]["output_dir"] = str(run_dir)
    resolved_path = Path("/tmp/modal_smoke_resolved.yaml")
    resolved_path.write_text(yaml.safe_dump(config, sort_keys=False))

    print("GPU name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    print("PyTorch version:", torch.__version__)
    print("training widths:", config["compression"]["train_widths"])
    print("evaluation widths:", config["compression"]["eval_widths"])
    print("seed:", config["experiment"]["seeds"][0])
    print("epochs:", config["training"]["epochs"])
    print("dataset mode:", config["dataset"]["name"], "fake_data=", config["dataset"]["fake_data"])
    print("AMP: disabled (the existing baseline training loop has no AMP option)")

    # Fail before allocating training time if CIFAR-100 cannot be downloaded/read.
    train_loader, val_loader, feature_loader = build_loaders(config, seed=0)
    assert len(train_loader.dataset) == 50_000
    assert len(val_loader.dataset) == 10_000
    assert len(feature_loader.dataset) == 2_000
    volume.commit()

    run(resolved_path)

    seed_dir = run_dir / "seed_0"
    result_dir = seed_dir / "results"
    checkpoint = seed_dir / "checkpoint.pt"
    assert checkpoint.is_file() and checkpoint.stat().st_size > 0, "Checkpoint was not saved"

    metrics = pd.read_csv(result_dir / "budget_metrics.csv").sort_values("budget")
    training = pd.read_csv(seed_dir / "training_metrics.csv")
    local = pd.read_csv(result_dir / "local_sensitivity.csv")
    control = pd.read_csv(result_dir / "control_metric_correlations.csv")
    cliffs = pd.read_csv(result_dir / "compression_cliffs.csv")
    matrix = np.load(result_dir / "wasserstein_matrix.npy")

    expected_anchors = {0.25, 0.50, 0.75, 1.00}
    assert set(training["width"].astype(float).unique()) == expected_anchors
    assert len(metrics) == 16 and int(metrics["is_train_anchor"].sum()) == 4
    assert np.isfinite(metrics[["accuracy", "loss", "flops", "params"]].to_numpy()).all()
    assert np.all(np.diff(metrics["flops"].to_numpy(float)) > 0), "FLOPs must rise with width"
    assert matrix.shape == (16, 16) and np.isfinite(matrix).all()
    assert np.allclose(matrix, matrix.T) and np.allclose(np.diag(matrix), 0)

    feature_paths = sorted((seed_dir / "features").glob("features_budget_*.pt"))
    assert len(feature_paths) == 16
    reference_ids = None
    for path in feature_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        ids = payload["sample_ids"]
        assert torch.isfinite(payload["features"]).all()
        assert float(payload["features"].std()) > 1e-8, f"Degenerate features at {path.name}"
        if reference_ids is None:
            reference_ids = ids
        else:
            assert torch.equal(reference_ids, ids), "Feature sample IDs/order differ across budgets"

    pearson_result = pearsonr(local["G_width"], local["P"])
    spearman_result = spearmanr(local["G_width"], local["P"])
    pearson_value = _json_number(pearson_result.statistic)
    pearson_p = _json_number(pearson_result.pvalue)
    spearman_value = _json_number(spearman_result.statistic)
    spearman_p = _json_number(spearman_result.pvalue)
    control_row = control.loc[control["method"] == "euclidean_mean"].iloc[0]
    control_spearman = _json_number(control_row["spearman_with_accuracy_sensitivity"])
    control_p = _json_number(control_row["spearman_p_value"])
    interpretation, evidence = _interpret(local, len(cliffs), spearman_value)

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    _save_required_plots(metrics, local, matrix, plot_dir)

    full_accuracy = float(metrics.loc[np.isclose(metrics["budget"], 1.0), "accuracy"].iloc[0])
    quarter_accuracy = float(metrics.loc[np.isclose(metrics["budget"], 0.25), "accuracy"].iloc[0])
    seen_mean = float(metrics.loc[metrics["is_train_anchor"], "accuracy"].mean())
    unseen_mean = float(metrics.loc[~metrics["is_train_anchor"], "accuracy"].mean())
    report_lines = [
        "# Modal Wasserstein geometry smoke-test summary",
        "",
        "- Dataset: CIFAR-100 (real data; FakeData disabled)",
        "- Backbone: slimmable CIFAR ResNet-18",
        f"- GPU: {torch.cuda.get_device_name(0)}",
        "- Seed: 0",
        "- Epochs: 20",
        "- Training anchors: 0.25, 0.50, 0.75, 1.00",
        "- Evaluation budgets: 0.25, 0.30, ..., 1.00",
        "",
        f"- Full-width accuracy: {full_accuracy:.6f}",
        f"- 25% width accuracy: {quarter_accuracy:.6f}",
        f"- Mean seen accuracy: {seen_mean:.6f}",
        f"- Mean unseen accuracy: {unseen_mean:.6f}",
        f"- Pearson G vs P: r={pearson_value}, p={pearson_p}",
        f"- Spearman G vs P: rho={spearman_value}, p={spearman_p}",
        f"- Euclidean-mean control: Spearman rho={control_spearman}, p={control_p}",
        f"- Detected compression cliffs: {len(cliffs)}",
        f"- Interpretation: **{interpretation}**",
        "",
        "Evidence used by the descriptive multi-signal rule:",
        *[f"- {item}" for item in evidence],
        "",
        "> This is a one-seed, short-training smoke test intended only to assess whether the research hypothesis warrants a full experiment.",
        "",
        "No oracle, CFM, geometry regularization, adaptive width sampling, or hyperparameter search was run.",
    ]
    modal_report = run_dir / "reports" / "modal_smoke_summary.md"
    modal_report.parent.mkdir(parents=True, exist_ok=True)
    modal_report.write_text("\n".join(report_lines) + "\n")
    volume.commit()

    compact_table = metrics[["budget", "is_train_anchor", "accuracy", "flops", "params"]].copy()
    compact_table["seen_unseen"] = np.where(compact_table["is_train_anchor"], "seen", "unseen")
    return {
        "run_name": run_name,
        "volume_name": VOLUME_NAME,
        "volume_output_path": f"outputs/{run_name}",
        "modal_report_path": f"outputs/{run_name}/reports/modal_smoke_summary.md",
        "gpu": torch.cuda.get_device_name(0),
        "accuracy_table": compact_table.to_dict(orient="records"),
        "local_sensitivity": local.to_dict(orient="records"),
        "wasserstein_matrix": matrix.tolist(),
        "budgets": metrics["budget"].astype(float).tolist(),
        "correlations": {
            "pearson_r": pearson_value,
            "pearson_p_value": pearson_p,
            "spearman_rho": spearman_value,
            "spearman_p_value": spearman_p,
            "euclidean_mean_spearman_rho": control_spearman,
            "euclidean_mean_spearman_p_value": control_p,
        },
        "compression_cliffs": cliffs.to_dict(orient="records"),
        "interpretation": interpretation,
        "interpretation_evidence": evidence,
        "report_markdown": "\n".join(report_lines),
    }


@app.local_entrypoint()
def main(run_name: str = "modal-smoke-seed0") -> None:
    result = run_smoke.remote(run_name)
    print(json.dumps(result, indent=2))
