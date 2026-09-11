from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from baseline_artifacts import compute_anchor_geometry_prior, find_confirmatory_root
from data import build_confirmatory_loaders
from evaluation import evaluate_grid
from geometry.analysis import analyze_seed, correlation_summary
from geometry.reporting import write_smoke_report
from models import slimmable_resnet18
from research_utils import load_config, resolve_device, seed_everything
from training import train_shared_model


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and (value != value or abs(value) == float("inf")):
        return None
    return value


def finalize_geoweighting(root_output: Path, config: dict) -> Path:
    # The multi-GPU launcher runs one seed in each temporary worker directory.
    # Recreate the frozen prior at the final root so the deliverable is complete
    # regardless of whether execution was sequential or scheduled across GPUs.
    baseline_root = find_confirmatory_root(config["baseline_artifacts"]["input_root"])
    weighting = config["geometry_weighting"]
    prior, _ = compute_anchor_geometry_prior(
        baseline_root,
        [float(width) for width in config["compression"]["train_widths"]],
        alpha=float(weighting["alpha"]),
        beta=float(weighting["beta"]),
        epsilon=float(weighting["epsilon"]),
    )
    prior["baseline_root"] = str(baseline_root)
    prior.to_csv(root_output / "frozen_anchor_geometry_prior.csv", index=False)
    central_frames, summaries = [], []
    for seed_value in config["experiment"]["seeds"]:
        seed_dir = root_output / f"seed_{int(seed_value)}"
        central_frames.append(pd.read_csv(seed_dir / "results" / "central_analysis.csv"))
        with (seed_dir / "results" / "analysis_summary.json").open() as handle:
            summaries.append(json.load(handle))
    combined = pd.concat(central_frames, ignore_index=True)
    combined.to_csv(root_output / "central_analysis_all_seeds.csv", index=False)
    pooled = combined[["local_wasserstein_sensitivity", "local_accuracy_sensitivity"]].dropna()
    aggregate = correlation_summary(
        pooled["local_wasserstein_sensitivity"],
        pooled["local_accuracy_sensitivity"],
        int(config["geometry"]["bootstrap_samples"]),
        int(config["geometry"]["projection_seed"]),
    )
    with (root_output / "aggregate_correlations.json").open("w") as handle:
        json.dump(_json_safe(aggregate), handle, indent=2, allow_nan=False)
    return write_smoke_report(root_output, summaries, combined)


def run(config_path: str | Path) -> Path:
    config = load_config(config_path)
    root_output = Path(config["experiment"]["output_dir"])
    root_output.mkdir(parents=True, exist_ok=True)
    with (root_output / "resolved_config.yaml").open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    baseline_root = find_confirmatory_root(config["baseline_artifacts"]["input_root"])
    weighting = config["geometry_weighting"]
    prior, weights = compute_anchor_geometry_prior(
        baseline_root,
        [float(width) for width in config["compression"]["train_widths"]],
        alpha=float(weighting["alpha"]),
        beta=float(weighting["beta"]),
        epsilon=float(weighting["epsilon"]),
    )
    prior["baseline_root"] = str(baseline_root)
    prior.to_csv(root_output / "frozen_anchor_geometry_prior.csv", index=False)
    print(f"[GeoWeighting] baseline artifacts: {baseline_root}", flush=True)
    print(f"[GeoWeighting] frozen weights: {weights}", flush=True)
    device = resolve_device(config["experiment"].get("device", "auto"))
    for seed_value in config["experiment"]["seeds"]:
        seed = int(seed_value)
        seed_everything(seed)
        seed_dir = root_output / f"seed_{seed}"
        loaders = build_confirmatory_loaders(config, training_seed=seed)
        model = slimmable_resnet18(
            num_classes=int(config["dataset"]["num_classes"]),
            supported_widths=config["compression"]["eval_widths"],
            projection_dim=int(config["model"]["projection_dim"]),
        ).to(device)
        train_shared_model(
            model,
            loaders.train,
            config,
            device,
            seed_dir,
            anchor_loss_weights=weights,
        )
        evaluate_grid(
            model,
            loaders.train,
            loaders.test,
            loaders.geometry,
            config,
            device,
            seed,
            seed_dir,
            calibration_loader=loaders.calibration,
        )
        analyze_seed(seed_dir, config, seed)
    return finalize_geoweighting(root_output, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen-prior GeoWeighting A1")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = run(args.config)
    print(f"Completed. Report: {report}")


if __name__ == "__main__":
    main()
