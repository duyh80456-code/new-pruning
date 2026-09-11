from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from data import build_confirmatory_loaders
from evaluation import evaluate_grid
from geometry.analysis import analyze_seed, correlation_summary
from geometry.reporting import write_smoke_report
from models import slimmable_resnet18
from research_utils import load_config, resolve_device, seed_everything
from specialization import (
    evaluate_selected_specialized_models,
    select_all_specialized_checkpoints,
)
from training import train_shared_model


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and (value != value or abs(value) == float("inf")):
        return None
    return value


def _attach_specialization(central: pd.DataFrame, specialized: pd.DataFrame) -> pd.DataFrame:
    result = central.copy()
    result["specialized_accuracy_if_available"] = float("nan")
    result["specialization_gap_if_available"] = float("nan")
    result["best_specialization_epoch_if_available"] = float("nan")
    result["best_validation_accuracy_if_available"] = float("nan")
    for row in specialized.itertuples(index=False):
        mask = (result["budget"].astype(float) - float(row.width)).abs() < 1e-8
        result.loc[mask, "specialized_accuracy_if_available"] = row.specialized_test_accuracy
        result.loc[mask, "specialization_gap_if_available"] = row.specialization_gap
        result.loc[mask, "best_specialization_epoch_if_available"] = row.best_epoch
        result.loc[mask, "best_validation_accuracy_if_available"] = row.best_validation_accuracy
    return result


def finalize_confirmatory(root_output: Path, config: dict) -> Path:
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
    if config["model"].get("backbone") != "slimmable_resnet18":
        raise ValueError("Confirmatory protocol requires slimmable_resnet18")
    root_output = Path(config["experiment"]["output_dir"])
    root_output.mkdir(parents=True, exist_ok=True)
    with (root_output / "resolved_config.yaml").open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    device = resolve_device(config["experiment"].get("device", "auto"))

    for seed_value in config["experiment"]["seeds"]:
        seed = int(seed_value)
        print(f"[confirmatory] seed {seed} starting", flush=True)
        seed_everything(seed)
        seed_dir = root_output / f"seed_{seed}"
        loaders = build_confirmatory_loaders(config, training_seed=seed)
        model = slimmable_resnet18(
            num_classes=int(config["dataset"]["num_classes"]),
            supported_widths=config["compression"]["eval_widths"],
            projection_dim=int(config["model"]["projection_dim"]),
        ).to(device)
        train_shared_model(model, loaders.train, config, device, seed_dir)

        # All checkpoint choices happen on validation data before test_loader is used.
        selected = select_all_specialized_checkpoints(
            model, loaders, config, device, seed, seed_dir
        )
        baseline_metrics = evaluate_grid(
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
        specialized = evaluate_selected_specialized_models(
            model,
            selected,
            baseline_metrics,
            loaders.test,
            loaders.calibration,
            config,
            device,
            seed_dir,
        )
        central, _ = analyze_seed(seed_dir, config, seed)
        central = _attach_specialization(central, specialized)
        central.to_csv(seed_dir / "results" / "central_analysis.csv", index=False)
        print(f"[confirmatory] seed {seed} completed", flush=True)
    return finalize_confirmatory(root_output, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe confirmatory specialization")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = run(args.config)
    print(f"Completed. Report: {report}")


if __name__ == "__main__":
    main()
