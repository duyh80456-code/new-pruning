"""Why does Uniform/Sandwich beat structured pair samplers?

The module is deliberately diagnostic-only.  It consumes the completed RQ2
pilot artifacts, never changes a checkpoint, and separates:

* temporal predictive value of representation SW;
* pair-support diversity and realized pair exposure;
* optimizer-momentum alignment and local two-update stability; and
* stage-wise training behavior.

Only checkpoints that were pre-registered by the original run are used.  In
particular, pair-gradient structure exists at epochs 10/50/100, whereas SW and
dense metrics exist every ten epochs.  Outputs label this asymmetry rather
than pretending intermediate gradient checkpoints exist.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr, spearmanr

from rq2_cross_subnet_interaction import _state_digest, interaction_parameters
from rq2_e2e_pairwise_pilot import (
    ALL_METHODS,
    GRID,
    PILOT_SEED,
    _flops,
    _optimizer_scheduler,
    _sw_matrix,
)
from rq2_gradient_variance_v3 import importance_corrected_variance
from rq2_pair_policies import (
    ResourceGeoPairPolicy,
    ResourcePairPolicy,
    SWPairPolicy,
    UniformPairPolicy,
)
from rq2_pairwise_surrogate_regret import (
    INTERIOR_WIDTHS,
    PAIR_INDICES,
    TARGET_WEIGHTS,
    UNIFORM_PI,
)
from rq2_quick_trajectory_diagnostic import (
    NUM_BATCHES,
    _batch_gradient_matrix,
    _fixed_probe_batches,
)
from s1_width import make_model


METHODS = tuple(ALL_METHODS)
PAIR_I = np.asarray([i for i, _ in PAIR_INDICES], dtype=np.int64)
PAIR_J = np.asarray([j for _, j in PAIR_INDICES], dtype=np.int64)
STAGES = ((10, 30), (30, 50), (50, 70), (70, 100))


def _safe_corr(x, y, kind: str) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) <= 1e-15 or np.std(y) <= 1e-15:
        return float("nan")
    result = pearsonr(x, y) if kind == "pearson" else spearmanr(x, y)
    return float(result.statistic)


def _pair_vector(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (14, 14) or not np.isfinite(matrix).all():
        raise RuntimeError(f"Expected a finite 14x14 matrix, got {matrix.shape}")
    return matrix[PAIR_I, PAIR_J]


def _diagnostic_dir(root: Path, method: str, epoch: int) -> Path:
    candidates = []
    for namespace in ("diagnostics_rpgeo_frozen", "diagnostics"):
        path = root / namespace / f"{method}_E{epoch}"
        if (path / "gradient_grams.npy").is_file() and (path / "sw_matrix.npy").is_file():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No complete diagnostic for {method} E{epoch}")
    return candidates[0]


def _state_sw(root: Path, method: str, epoch: int) -> np.ndarray:
    if method == "common_warmup":
        return np.load(_diagnostic_dir(root, method, epoch) / "sw_matrix.npy")
    policy = root / method / "sw_policies" / f"epoch_{epoch:03d}.npz"
    if policy.is_file():
        with np.load(policy) as payload:
            return np.asarray(payload["sw_matrix"], dtype=np.float64)
    return np.load(_diagnostic_dir(root, method, epoch) / "sw_matrix.npy")


def _gradient_pair_targets(root: Path, method: str, epoch: int) -> pd.DataFrame:
    grams = np.load(_diagnostic_dir(root, method, epoch) / "gradient_grams.npy")
    gram = np.asarray(grams, dtype=np.float64).mean(axis=0)
    if gram.shape != (14, 14):
        raise RuntimeError(f"Unexpected gradient Gram shape for {method} E{epoch}: {gram.shape}")
    diagonal = np.diag(gram).clip(min=1e-30)
    distance2 = diagonal[PAIR_I] + diagonal[PAIR_J] - 2.0 * gram[PAIR_I, PAIR_J]
    cosine = gram[PAIR_I, PAIR_J] / np.sqrt(diagonal[PAIR_I] * diagonal[PAIR_J])
    return pd.DataFrame({
        "pair_index": np.arange(len(PAIR_INDICES)),
        "width_i": np.asarray(INTERIOR_WIDTHS)[PAIR_I],
        "width_j": np.asarray(INTERIOR_WIDTHS)[PAIR_J],
        "gradient_distance2": distance2.clip(min=0.0),
        "gradient_cosine": cosine.clip(-1.0, 1.0),
    })


def temporal_predictive_audit(root: str | Path, output_dir: str | Path) -> dict:
    root, output_dir = Path(root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporal_rows, future_gradient_rows, width_rows, loss_rows = [], [], [], []

    dense_by_method = {
        method: pd.read_csv(root / method / "dense_metrics.csv") for method in METHODS
    }
    for method in METHODS:
        sw = {epoch: _state_sw(root, method, epoch) for epoch in range(10, 101, 10)}
        training_metrics = pd.read_csv(root / method / "metrics.csv").set_index("epoch")
        for epoch in range(10, 100, 10):
            current = _pair_vector(sw[epoch])
            for lag in (10, 20):
                future_epoch = epoch + lag
                if future_epoch > 100:
                    continue
                future = _pair_vector(sw[future_epoch])
                temporal_rows.append({
                    "method": method, "epoch": epoch, "future_epoch": future_epoch,
                    "lag": lag, "num_pairs": len(current),
                    "pearson": _safe_corr(current, future, "pearson"),
                    "spearman": _safe_corr(current, future, "spearman"),
                    "normalized_rmse": float(np.sqrt(np.mean((current - future) ** 2)) /
                                             max(np.std(future), 1e-12)),
                    "top_quartile_overlap": float(len(
                        set(np.argsort(current)[-23:]) & set(np.argsort(future)[-23:])
                    ) / 23.0),
                })

            dense = dense_by_method[method]
            current_acc = dense.loc[dense.epoch.eq(epoch)].set_index("width")["accuracy"]
            centrality = sw[epoch].mean(axis=1)
            for lag in (10, 20):
                future_epoch = epoch + lag
                if future_epoch > 100:
                    continue
                future_acc = dense.loc[dense.epoch.eq(future_epoch)].set_index("width")["accuracy"]
                widths = list(INTERIOR_WIDTHS)
                delta = np.asarray([future_acc.loc[w] - current_acc.loc[w] for w in widths])
                width_rows.append({
                    "method": method, "epoch": epoch, "future_epoch": future_epoch,
                    "lag": lag, "num_widths": len(widths),
                    "pearson_SW_centrality_vs_future_accuracy_gain": _safe_corr(
                        centrality, delta, "pearson"
                    ),
                    "spearman_SW_centrality_vs_future_accuracy_gain": _safe_corr(
                        centrality, delta, "spearman"
                    ),
                })
                if epoch in training_metrics.index and future_epoch in training_metrics.index:
                    loss_rows.append({
                        "method": method, "epoch": epoch,
                        "future_epoch": future_epoch, "lag": lag,
                        "mean_pair_SW": float(current.mean()),
                        "max_pair_SW": float(current.max()),
                        "train_loss": float(training_metrics.loc[epoch, "train_loss"]),
                        "future_train_loss": float(
                            training_metrics.loc[future_epoch, "train_loss"]
                        ),
                        "future_loss_reduction": float(
                            training_metrics.loc[epoch, "train_loss"]
                            - training_metrics.loc[future_epoch, "train_loss"]
                        ),
                    })

        for future_epoch in (50, 100):
            gradient = _gradient_pair_targets(root, method, future_epoch)
            for lag in (0, 10, 20):
                epoch = future_epoch - lag
                if epoch < 10:
                    continue
                sw2 = _pair_vector(sw[epoch]) ** 2
                future_gradient_rows.append({
                    "method": method, "epoch": epoch, "future_epoch": future_epoch,
                    "lag": lag, "num_pairs": len(sw2),
                    "pearson_SW2_vs_future_gradient_distance2": _safe_corr(
                        sw2, gradient.gradient_distance2, "pearson"
                    ),
                    "spearman_SW2_vs_future_gradient_distance2": _safe_corr(
                        sw2, gradient.gradient_distance2, "spearman"
                    ),
                    "pearson_SW2_vs_future_negative_cosine": _safe_corr(
                        sw2, -gradient.gradient_cosine, "pearson"
                    ),
                    "spearman_SW2_vs_future_negative_cosine": _safe_corr(
                        sw2, -gradient.gradient_cosine, "spearman"
                    ),
                })

    temporal = pd.DataFrame(temporal_rows)
    future_gradient = pd.DataFrame(future_gradient_rows)
    width_forecast = pd.DataFrame(width_rows)
    loss_forecast = pd.DataFrame(loss_rows)
    temporal.to_csv(output_dir / "temporal_sw_forecast.csv", index=False)
    future_gradient.to_csv(output_dir / "sw_to_future_gradient.csv", index=False)
    width_forecast.to_csv(output_dir / "sw_to_future_width_accuracy.csv", index=False)
    loss_forecast.to_csv(output_dir / "sw_to_future_training_loss.csv", index=False)
    return {
        "temporal_rows": len(temporal),
        "future_gradient_rows": len(future_gradient),
        "width_forecast_rows": len(width_forecast),
        "loss_forecast_rows": len(loss_forecast),
    }


def diversity_and_stage_audit(root: str | Path, output_dir: str | Path) -> dict:
    root, output_dir = Path(root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diversity_rows, exposure_rows, stage_rows = [], [], []

    for method in METHODS:
        pair_stats = pd.read_csv(root / method / "pair_stats.csv")
        metrics = pd.read_csv(root / method / "metrics.csv")
        dense = pd.read_csv(root / method / "dense_metrics.csv")
        for epoch, frame in pair_stats.groupby("epoch", sort=True):
            expected = frame.policy_probability.to_numpy(float)
            counts = frame["count"].to_numpy(float)
            realized = counts / max(counts.sum(), 1.0)
            diversity_rows.append({
                "method": method, "epoch": int(epoch),
                "expected_entropy": float(-(expected[expected > 0] * np.log(expected[expected > 0])).sum()),
                "expected_support": int((expected > 1e-12).sum()),
                "expected_effective_support": float(1.0 / np.square(expected).sum()),
                "realized_entropy": float(-(realized[realized > 0] * np.log(realized[realized > 0])).sum()),
                "realized_support": int((counts > 0).sum()),
                "realized_effective_support": float(1.0 / max(np.square(realized).sum(), 1e-30)),
                "max_pair_probability": float(expected.max()),
            })

        cumulative = pair_stats.copy()
        cumulative["cumulative_count"] = cumulative.groupby("pair_index")["count"].cumsum()
        for future_epoch in (50, 100):
            gradient = _gradient_pair_targets(root, method, future_epoch).set_index("pair_index")
            for lag in (10, 20):
                epoch = future_epoch - lag
                exposure = cumulative.loc[cumulative.epoch.eq(epoch)].set_index("pair_index")
                if exposure.empty:
                    continue
                count = exposure.loc[gradient.index, "cumulative_count"].to_numpy(float)
                exposure_rows.append({
                    "method": method, "epoch": epoch, "future_epoch": future_epoch,
                    "lag": lag,
                    "pearson_exposure_vs_future_gradient_distance2": _safe_corr(
                        count, gradient.gradient_distance2, "pearson"
                    ),
                    "spearman_exposure_vs_future_gradient_distance2": _safe_corr(
                        count, gradient.gradient_distance2, "spearman"
                    ),
                    "pearson_exposure_vs_future_gradient_cosine": _safe_corr(
                        count, gradient.gradient_cosine, "pearson"
                    ),
                    "spearman_exposure_vs_future_gradient_cosine": _safe_corr(
                        count, gradient.gradient_cosine, "spearman"
                    ),
                })

        diversity = pd.DataFrame(diversity_rows).loc[lambda x: x.method.eq(method)]
        for start, end in STAGES:
            epoch_metrics = metrics.loc[metrics.epoch.gt(start) & metrics.epoch.le(end)]
            epoch_diversity = diversity.loc[
                diversity.epoch.gt(start) & diversity.epoch.le(end)
            ]
            sw_epochs = [epoch for epoch in range(start, end + 1, 10) if epoch <= 100]
            sw_values = np.concatenate([
                _pair_vector(_state_sw(root, method, epoch)) for epoch in sw_epochs
            ])
            start_dense = dense.loc[dense.epoch.eq(start)]
            end_dense = dense.loc[dense.epoch.eq(end)]
            stage_rows.append({
                "method": method, "stage": f"{start}-{end}",
                "start_epoch": start, "end_epoch": end,
                "mean_train_loss": float(epoch_metrics.train_loss.mean()),
                "train_loss_change": float(epoch_metrics.train_loss.iloc[-1] - epoch_metrics.train_loss.iloc[0]),
                "mean_expected_entropy": float(epoch_diversity.expected_entropy.mean()),
                "mean_realized_effective_support": float(epoch_diversity.realized_effective_support.mean()),
                "mean_pair_SW": float(sw_values.mean()), "max_pair_SW": float(sw_values.max()),
                "dense_mean_start": float(start_dense.accuracy.mean()),
                "dense_mean_end": float(end_dense.accuracy.mean()),
                "dense_mean_gain": float(end_dense.accuracy.mean() - start_dense.accuracy.mean()),
                "interior_mean_gain": float(
                    end_dense.loc[end_dense.width.isin(INTERIOR_WIDTHS), "accuracy"].mean()
                    - start_dense.loc[start_dense.width.isin(INTERIOR_WIDTHS), "accuracy"].mean()
                ),
            })

    diversity = pd.DataFrame(diversity_rows)
    exposure = pd.DataFrame(exposure_rows)
    stages = pd.DataFrame(stage_rows)
    diversity.to_csv(output_dir / "pair_diversity_by_epoch.csv", index=False)
    exposure.to_csv(output_dir / "pair_exposure_vs_future_gradient.csv", index=False)
    stages.to_csv(output_dir / "stage_summary.csv", index=False)
    return {
        "diversity_rows": len(diversity), "exposure_rows": len(exposure),
        "stage_rows": len(stages),
    }


def _checkpoint(root: Path, method: str, epoch: int) -> Path:
    if method == "common_warmup" and epoch == 10:
        return root / "common_warmup" / "epoch_010.pt"
    return root / method / "checkpoints" / f"epoch_{epoch:03d}.pt"


def _momentum_vector(optimizer, named_parameters, zero: bool = False) -> torch.Tensor:
    parts = []
    for _, parameter in named_parameters:
        if zero:
            parts.append(torch.zeros_like(parameter).reshape(-1))
            continue
        state = optimizer.state.get(parameter, {})
        buffer = state.get("momentum_buffer")
        parts.append(
            torch.zeros_like(parameter).reshape(-1)
            if buffer is None else buffer.detach().reshape(-1)
        )
    return torch.cat(parts)


def _cosine(numerator, norm_a, norm_b):
    denominator = np.sqrt(np.maximum(norm_a, 0.0) * np.maximum(norm_b, 0.0))
    return np.divide(
        numerator, denominator, out=np.full_like(np.asarray(numerator, float), np.nan),
        where=denominator > 1e-30,
    )


def _momentum_metrics_from_gram(
    gradient_gram: np.ndarray,
    gradient_dot_momentum: np.ndarray,
    gradient_dot_parameters: np.ndarray,
    momentum_norm2: float,
    momentum_dot_parameters: float,
    parameter_norm2: float,
    momentum: float,
    weight_decay: float,
    policy_probabilities: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coefficient = np.zeros((len(PAIR_INDICES), len(GRID)), dtype=np.float64)
    coefficient[:, 0] = 0.25
    coefficient[:, -1] = 0.25
    coefficient[np.arange(len(PAIR_INDICES)), PAIR_I + 1] = 0.25
    coefficient[np.arange(len(PAIR_INDICES)), PAIR_J + 1] = 0.25

    a_theta = coefficient @ gradient_dot_parameters
    u_dot_m = coefficient @ gradient_dot_momentum + weight_decay * momentum_dot_parameters
    u_gram = coefficient @ gradient_gram @ coefficient.T
    u_gram += weight_decay * (a_theta[:, None] + a_theta[None, :])
    u_gram += (weight_decay ** 2) * parameter_norm2
    u_norm2 = np.diag(u_gram).clip(min=0.0)

    width_norm2 = np.diag(gradient_gram).clip(min=0.0)
    width_cos = _cosine(gradient_dot_momentum, width_norm2, momentum_norm2)
    pair_cos = _cosine(u_dot_m, u_norm2, momentum_norm2)
    m1_norm2 = (
        momentum ** 2 * momentum_norm2 + 2.0 * momentum * u_dot_m + u_norm2
    ).clip(min=0.0)
    m1_dot_m = momentum * momentum_norm2 + u_dot_m
    first_stability = _cosine(m1_dot_m, m1_norm2, momentum_norm2)

    # Rows are first pair a, columns are next pair b.
    sequence_dot = (
        momentum ** 3 * momentum_norm2
        + 2.0 * momentum ** 2 * u_dot_m[:, None]
        + momentum * u_norm2[:, None]
        + momentum * u_dot_m[None, :]
        + u_gram
    )
    m2_norm2 = (
        momentum ** 4 * momentum_norm2
        + momentum ** 2 * u_norm2[:, None]
        + u_norm2[None, :]
        + 2.0 * momentum ** 3 * u_dot_m[:, None]
        + 2.0 * momentum ** 2 * u_dot_m[None, :]
        + 2.0 * momentum * u_gram
    ).clip(min=0.0)
    sequence_cos = _cosine(sequence_dot, m2_norm2, m1_norm2[:, None])

    width = pd.DataFrame({
        "width": GRID,
        "gradient_norm": np.sqrt(width_norm2),
        "cos_momentum_gradient": width_cos,
    })
    pair = pd.DataFrame({
        "pair_index": np.arange(len(PAIR_INDICES)),
        "width_i": np.asarray(INTERIOR_WIDTHS)[PAIR_I],
        "width_j": np.asarray(INTERIOR_WIDTHS)[PAIR_J],
        "pair_update_norm": np.sqrt(u_norm2),
        "cos_momentum_pair_update": pair_cos,
        "cos_momentum_after_one_update": first_stability,
    })
    policy_rows = []
    for policy, probabilities in policy_probabilities.items():
        q = np.asarray(probabilities, dtype=np.float64)
        joint = q[:, None] * q[None, :]

        def weighted_finite_mean(values, weights):
            values, weights = np.asarray(values, float), np.asarray(weights, float)
            finite = np.isfinite(values)
            if not finite.any() or float(weights[finite].sum()) <= 0:
                return float("nan")
            return float(np.sum(weights[finite] * values[finite]) / weights[finite].sum())

        policy_rows.append({
            "evaluated_policy": policy,
            "expected_cos_momentum_pair_update": weighted_finite_mean(pair_cos, q),
            "expected_cos_momentum_after_one_update": weighted_finite_mean(first_stability, q),
            "expected_two_update_momentum_stability": weighted_finite_mean(sequence_cos, joint),
            "probability_two_update_negative_cosine": float(np.nansum(joint * (sequence_cos < 0))),
            "expected_pair_update_norm": float(np.sum(q * np.sqrt(u_norm2))),
        })
    return width, pair, pd.DataFrame(policy_rows)


def momentum_probe_state(
    root: str | Path,
    dataset_root: str | Path,
    gate_a_summary: str | Path,
    method: str,
    epoch: int,
    output_dir: str | Path,
    device: str = "cuda:0",
) -> dict:
    root, output_dir = Path(root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _checkpoint(root, method, int(epoch))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = yaml.safe_load((root / "resolved_config.yaml").read_text())
    config["dataset"]["root"] = str(dataset_root)
    config["dataset"]["download"] = False
    config["dataset"]["num_workers"] = 0
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = make_model(config, torch_device)
    model.load_state_dict(payload["model"])
    optimizer, _ = _optimizer_scheduler(
        model, config, 0.01 if int(epoch) >= 51 else 0.1
    )
    optimizer.load_state_dict(payload["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value): state[key] = value.to(torch_device)
    parameters = interaction_parameters(model)
    parameter_before = _state_digest(model.named_parameters())
    buffer_before = _state_digest(model.named_buffers())
    theta = torch.cat([parameter.detach().reshape(-1) for _, parameter in parameters])
    saved_momentum = _momentum_vector(optimizer, parameters, zero=False)
    batches, probe_ids = _fixed_probe_batches(config, Path(dataset_root), torch_device)
    sw = _state_sw(root, method, int(epoch))
    flops = _flops(gate_a_summary)
    resource = ResourcePairPolicy(flops)
    freeze = json.loads((root / "rpgeo_frozen_retention.json").read_text())
    policies = {
        "uniform": UniformPairPolicy().probabilities,
        "resource": resource.probabilities,
        "pure_sw": SWPairPolicy(sw).probabilities,
        "resource_geo": ResourceGeoPairPolicy(
            flops, sw, float(freeze["resource_retention"])
        ).probabilities,
    }
    momentum_modes = ("saved", "protocol_reset_zero") if int(epoch) == 50 else ("saved",)
    width_frames, pair_frames, policy_frames = [], [], []
    gram_frames = []
    training = config["training"]
    for batch_index, (images, labels) in enumerate(batches):
        model.eval()
        matrix = _batch_gradient_matrix(
            model, images.to(torch_device), labels.to(torch_device), parameters, config
        )
        gram = (matrix @ matrix.T).detach().double().cpu().numpy()
        g_theta = (matrix @ theta).detach().double().cpu().numpy()
        theta_norm2 = float(torch.dot(theta, theta).detach().double().cpu())
        gram_frames.append(gram)
        for mode in momentum_modes:
            momentum_vector = (
                torch.zeros_like(saved_momentum) if mode == "protocol_reset_zero"
                else saved_momentum
            )
            g_m = (matrix @ momentum_vector).detach().double().cpu().numpy()
            m_norm2 = float(torch.dot(momentum_vector, momentum_vector).detach().double().cpu())
            m_theta = float(torch.dot(momentum_vector, theta).detach().double().cpu())
            width, pair, policy = _momentum_metrics_from_gram(
                gram, g_m, g_theta, m_norm2, m_theta, theta_norm2,
                float(training["momentum"]), float(training["weight_decay"]), policies,
            )
            for frame in (width, pair, policy):
                frame.insert(0, "batch", batch_index)
                frame.insert(0, "momentum_mode", mode)
                frame.insert(0, "epoch", int(epoch))
                frame.insert(0, "checkpoint_method", method)
            width_frames.append(width); pair_frames.append(pair); policy_frames.append(policy)
        print(
            f"[momentum probe] {method} E{epoch}: batch {batch_index+1}/{NUM_BATCHES}",
            flush=True,
        )

    width_table = pd.concat(width_frames, ignore_index=True)
    pair_table = pd.concat(pair_frames, ignore_index=True)
    policy_table = pd.concat(policy_frames, ignore_index=True)
    width_table.to_csv(output_dir / "width_momentum_alignment.csv", index=False)
    pair_table.to_csv(output_dir / "pair_momentum_alignment.csv", index=False)
    policy_table.to_csv(output_dir / "policy_momentum_stability.csv", index=False)
    np.save(output_dir / "gradient_grams.npy", np.stack(gram_frames))
    if parameter_before != _state_digest(model.named_parameters()):
        raise RuntimeError("Momentum diagnostic changed model parameters")
    if buffer_before != _state_digest(model.named_buffers()):
        raise RuntimeError("Momentum diagnostic changed model buffers")
    metadata = {
        "status": "UNIFORM_WIN_MOMENTUM_DIAGNOSTIC_COMPLETE",
        "checkpoint_method": method, "epoch": int(epoch), "seed": PILOT_SEED,
        "optimizer_steps": 0, "model_updates": 0,
        "probe_ids_sha256": hashlib.sha256(
            np.asarray(probe_ids, dtype=np.int64).tobytes()
        ).hexdigest(),
        "num_batches": len(batches),
        "momentum_modes": list(momentum_modes),
        "epoch_50_reset_handling": "report_saved_pre_reset_and_protocol_zero_post_reset",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def finalize_uniform_win_diagnostic(root: str | Path, output_dir: str | Path) -> dict:
    root, output_dir = Path(root), Path(output_dir)
    state_dirs = sorted((output_dir / "momentum_states").glob("*_E*"))
    policy = pd.concat([
        pd.read_csv(path / "policy_momentum_stability.csv") for path in state_dirs
    ], ignore_index=True)
    width = pd.concat([
        pd.read_csv(path / "width_momentum_alignment.csv") for path in state_dirs
    ], ignore_index=True)
    pair = pd.concat([
        pd.read_csv(path / "pair_momentum_alignment.csv") for path in state_dirs
    ], ignore_index=True)
    policy_summary = policy.groupby(
        ["checkpoint_method", "epoch", "momentum_mode", "evaluated_policy"], as_index=False
    ).mean(numeric_only=True)
    width_summary = width.groupby(
        ["checkpoint_method", "epoch", "momentum_mode", "width"], as_index=False
    ).mean(numeric_only=True)
    pair_summary = pair.groupby(
        ["checkpoint_method", "epoch", "momentum_mode", "pair_index", "width_i", "width_j"],
        as_index=False,
    ).mean(numeric_only=True)
    policy_summary.to_csv(output_dir / "momentum_policy_summary.csv", index=False)
    width_summary.to_csv(output_dir / "momentum_width_summary.csv", index=False)
    pair_summary.to_csv(output_dir / "momentum_pair_summary.csv", index=False)

    temporal = pd.read_csv(output_dir / "temporal_sw_forecast.csv")
    future_gradient = pd.read_csv(output_dir / "sw_to_future_gradient.csv")
    future_loss = pd.read_csv(output_dir / "sw_to_future_training_loss.csv")
    diversity = pd.read_csv(output_dir / "pair_diversity_by_epoch.csv")
    stage = pd.read_csv(output_dir / "stage_summary.csv")
    summary = {
        "status": "UNIFORM_WIN_MECHANISM_DIAGNOSTICS_COMPLETE",
        "training_performed": False,
        "optimizer_steps": 0,
        "methods": list(METHODS),
        "checkpoint_states": ["common_warmup_E10"] + [
            f"{method}_E{epoch}" for method in METHODS for epoch in (50, 100)
        ],
        "temporal_SW_mean_spearman_lag10": float(
            temporal.loc[temporal.lag.eq(10), "spearman"].mean()
        ),
        "SW_to_future_gradient_mean_spearman_lag10_20": float(
            future_gradient.loc[future_gradient.lag.isin([10, 20]),
                                "spearman_SW2_vs_future_gradient_distance2"].mean()
        ),
        "mean_SW_vs_future_loss_reduction_spearman": _safe_corr(
            future_loss.mean_pair_SW, future_loss.future_loss_reduction, "spearman"
        ),
        "uniform_expected_effective_support": float(
            diversity.loc[diversity.method.eq("uniform"), "expected_effective_support"].mean()
        ),
        "stage_rows": len(stage),
        "interpretation_guard": (
            "Exploratory one-seed mechanism audit; 91 pairs within a state are dependent. "
            "No new algorithm or accuracy-tuned policy is authorized by this diagnostic alone."
        ),
    }
    (output_dir / "uniform_win_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


__all__ = [
    "METHODS", "temporal_predictive_audit", "diversity_and_stage_audit",
    "momentum_probe_state", "finalize_uniform_win_diagnostic",
]
