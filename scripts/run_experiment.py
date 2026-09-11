from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from data import build_loaders
from evaluation import evaluate_grid
from geometry.analysis import analyze_seed, correlation_summary
from geometry.reporting import write_smoke_report
from models import slimmable_resnet18
from oracles import run_oracles
from research_utils import load_config, resolve_device, seed_everything
from training import train_shared_model


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and (value != value or abs(value) == float("inf")):
        return None
    return value


def run(config_path: str | Path) -> Path:
    config = load_config(config_path)
    root_output = Path(config["experiment"]["output_dir"])
    root_output.mkdir(parents=True, exist_ok=True)
    with (root_output / "resolved_config.yaml").open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    device = resolve_device(config["experiment"].get("device", "auto"))
    all_central, summaries = [], []
    for seed_value in config["experiment"]["seeds"]:
        seed = int(seed_value)
        seed_everything(seed)
        seed_dir = root_output / f"seed_{seed}"
        train_loader, val_loader, feature_loader = build_loaders(config, seed)
        model = slimmable_resnet18(
            num_classes=int(config["dataset"]["num_classes"]),
            supported_widths=config["compression"]["eval_widths"],
            projection_dim=int(config["model"]["projection_dim"]),
        ).to(device)
        train_shared_model(model, train_loader, config, device, seed_dir)
        metrics = evaluate_grid(
            model,
            train_loader,
            val_loader,
            feature_loader,
            config,
            device,
            seed,
            seed_dir,
        )
        if config.get("oracle", {}).get("enabled", False):
            run_oracles(
                model,
                train_loader,
                val_loader,
                feature_loader,
                metrics,
                config,
                device,
                seed_dir,
            )
        central, summary = analyze_seed(seed_dir, config, seed)
        all_central.append(central)
        summaries.append(summary)
    combined = pd.concat(all_central, ignore_index=True)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the width/geometry smoke-test pipeline")
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    report = run(args.config)
    print(f"Completed. Report: {report}")


if __name__ == "__main__":
    main()
