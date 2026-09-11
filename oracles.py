from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import torch

from evaluation import calibrate_batch_norm, evaluate_width, extract_features
from geometry.distances import compute_distribution_distance
from profiling import profile_subnet
from research_utils import budget_tag
from training import fine_tune_oracle


def run_oracles(
    shared_model,
    train_loader,
    val_loader,
    feature_loader,
    baseline_metrics: pd.DataFrame,
    config: dict,
    device: torch.device,
    seed_dir: Path,
) -> pd.DataFrame:
    """Fine-tune selected unseen widths independently from the shared checkpoint."""
    records = []
    geometry_cfg = config["geometry"]
    for width in [float(x) for x in config["oracle"]["widths"]]:
        if width in set(float(x) for x in config["compression"]["train_widths"]):
            raise ValueError(f"Oracle width {width} is a training anchor")
        model = copy.deepcopy(shared_model).to(device)
        fine_tune_oracle(model, width, train_loader, config, device)
        calibrate_batch_norm(
            model,
            train_loader,
            width,
            device,
            int(config["evaluation"]["bn_calibration_batches"]),
        )
        metrics = evaluate_width(model, val_loader, width, device)
        oracle_path = seed_dir / "oracle_features" / f"features_budget_{budget_tag(width)}.pt"
        oracle_payload = extract_features(
            model,
            feature_loader,
            width,
            device,
            config["features"]["normalize"],
            oracle_path,
        )
        baseline_payload = torch.load(
            seed_dir / "features" / f"features_budget_{budget_tag(width)}.pt",
            map_location="cpu",
            weights_only=False,
        )
        if not torch.equal(oracle_payload["sample_ids"], baseline_payload["sample_ids"]):
            raise RuntimeError("Oracle and shared-model feature sample IDs differ")
        distance = compute_distribution_distance(
            baseline_payload["features"],
            oracle_payload["features"],
            method=geometry_cfg["method"],
            num_projections=int(geometry_cfg["num_projections"]),
            seed=int(geometry_cfg["projection_seed"]),
        )
        baseline_accuracy = float(
            baseline_metrics.loc[
                np_isclose(baseline_metrics["budget"], width), "accuracy"
            ].iloc[0]
        )
        flops, params = profile_subnet(model, width)
        records.append(
            {
                "budget": width,
                "baseline_accuracy": baseline_accuracy,
                "oracle_accuracy": metrics["accuracy"],
                "oracle_gap": metrics["accuracy"] - baseline_accuracy,
                "baseline_to_oracle_wasserstein": distance,
                "flops": flops,
                "params": params,
                "oracle_protocol": "independent selected-width fine-tune from shared checkpoint",
            }
        )
        checkpoint_dir = seed_dir / "oracles"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / f"oracle_budget_{budget_tag(width)}.pt")
    frame = pd.DataFrame(records)
    frame.to_csv(seed_dir / "results" / "oracle_metrics.csv", index=False)
    return frame


def np_isclose(series, value: float):
    # Kept local to avoid importing NumPy in callers and to tolerate CSV float roundoff.
    return (series.astype(float) - value).abs() < 1e-8
