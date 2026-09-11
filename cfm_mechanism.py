from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from geometry.distances import compute_distribution_distance
from research_utils import budget_tag, seed_everything


class ConditionalVectorField(nn.Module):
    def __init__(self, dimension: int, hidden: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dimension),
        )

    def forward(self, z, time, source_budget, target_budget):
        condition = torch.stack([time, source_budget, target_budget], dim=1)
        return self.network(torch.cat([z, condition], dim=1))


class ConditionalTransportMLP(nn.Module):
    def __init__(self, dimension: int, hidden: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dimension),
        )

    def forward(self, z, source_budget, target_budget):
        condition = torch.stack([source_budget, target_budget], dim=1)
        return self.network(torch.cat([z, condition], dim=1))


def load_feature_bank(baseline_root: str | Path, seed: int, budgets: list[float]):
    feature_dir = Path(baseline_root) / f"seed_{seed}" / "features"
    bank, reference_ids = {}, None
    for budget in budgets:
        payload = torch.load(
            feature_dir / f"features_budget_{budget_tag(budget)}.pt",
            map_location="cpu",
            weights_only=False,
        )
        ids = payload["sample_ids"].long()
        if reference_ids is None:
            reference_ids = ids
        elif not torch.equal(reference_ids, ids):
            raise RuntimeError("Cached feature sample IDs/order differ across budgets")
        features = payload["features"].float()
        if not bool(torch.isfinite(features).all()):
            raise FloatingPointError(f"Non-finite cached features at budget {budget}")
        bank[float(budget)] = F.normalize(features, dim=1)
    return bank, reference_ids


def fixed_sample_split(sample_count: int, seed: int, train_fraction: float, val_fraction: float):
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("Invalid train/validation fractions")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(sample_count, generator=generator)
    train_end = int(sample_count * train_fraction)
    val_end = train_end + int(sample_count * val_fraction)
    return permutation[:train_end], permutation[train_end:val_end], permutation[val_end:]


def _known_pairs(known_budgets: list[float]):
    return [(source, target) for source in known_budgets for target in known_budgets if source != target]


def _best_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


@torch.no_grad()
def _validation_losses(vector_field, transport_mlp, bank, indices, pairs, device):
    vector_field.eval(); transport_mlp.eval()
    cfm_losses, mlp_losses = [], []
    for source, target in pairs:
        z0 = bank[source][indices].to(device)
        z1 = bank[target][indices].to(device)
        source_c = torch.full((len(indices),), source, device=device)
        target_c = torch.full((len(indices),), target, device=device)
        for time_value in (0.25, 0.5, 0.75):
            time = torch.full((len(indices),), time_value, device=device)
            zt = (1.0 - time[:, None]) * z0 + time[:, None] * z1
            velocity = vector_field(zt, time, source_c, target_c)
            cfm_losses.append(F.mse_loss(velocity, z1 - z0).item())
        prediction = F.normalize(transport_mlp(z0, source_c, target_c), dim=1)
        mlp_losses.append(F.mse_loss(prediction, z1).item())
    return float(np.mean(cfm_losses)), float(np.mean(mlp_losses))


def train_mechanism_models(bank, known_budgets, train_indices, validation_indices, config, device):
    dimension = next(iter(bank.values())).shape[1]
    hidden = int(config["cfm"]["hidden_dim"])
    vector_field = ConditionalVectorField(dimension, hidden).to(device)
    transport_mlp = ConditionalTransportMLP(dimension, hidden).to(device)
    cfm_optimizer = torch.optim.AdamW(
        vector_field.parameters(), lr=float(config["cfm"]["learning_rate"])
    )
    mlp_optimizer = torch.optim.AdamW(
        transport_mlp.parameters(), lr=float(config["cfm"]["learning_rate"])
    )
    pairs = _known_pairs(known_budgets)
    batch_size = int(config["cfm"]["batch_size"])
    epochs = int(config["cfm"]["epochs"])
    best_cfm = {"epoch": 0, "loss": float("inf"), "state": None}
    best_mlp = {"epoch": 0, "loss": float("inf"), "state": None}
    history = []
    rng = torch.Generator().manual_seed(int(config["cfm"]["training_seed"]))
    for epoch in range(1, epochs + 1):
        vector_field.train(); transport_mlp.train()
        order = train_indices[torch.randperm(len(train_indices), generator=rng)]
        cfm_sum = mlp_sum = 0.0
        updates = 0
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            pair_indices = torch.randint(len(pairs), (len(batch_indices),), generator=rng)
            z0 = torch.stack(
                [bank[pairs[int(pair_id)][0]][int(idx)] for idx, pair_id in zip(batch_indices, pair_indices)]
            ).to(device)
            z1 = torch.stack(
                [bank[pairs[int(pair_id)][1]][int(idx)] for idx, pair_id in zip(batch_indices, pair_indices)]
            ).to(device)
            source_c = torch.tensor(
                [pairs[int(pair_id)][0] for pair_id in pair_indices], device=device
            )
            target_c = torch.tensor(
                [pairs[int(pair_id)][1] for pair_id in pair_indices], device=device
            )
            time = torch.rand(len(batch_indices), generator=rng).to(device)
            zt = (1.0 - time[:, None]) * z0 + time[:, None] * z1
            cfm_optimizer.zero_grad(set_to_none=True)
            cfm_loss = F.mse_loss(vector_field(zt, time, source_c, target_c), z1 - z0)
            if not bool(torch.isfinite(cfm_loss)):
                raise FloatingPointError(f"Non-finite CFM loss at epoch {epoch}")
            cfm_loss.backward(); cfm_optimizer.step()
            mlp_optimizer.zero_grad(set_to_none=True)
            mlp_prediction = F.normalize(transport_mlp(z0, source_c, target_c), dim=1)
            mlp_loss = F.mse_loss(mlp_prediction, z1)
            if not bool(torch.isfinite(mlp_loss)):
                raise FloatingPointError(f"Non-finite conditional-MLP loss at epoch {epoch}")
            mlp_loss.backward(); mlp_optimizer.step()
            cfm_sum += float(cfm_loss.detach()); mlp_sum += float(mlp_loss.detach()); updates += 1
        validation_cfm, validation_mlp = _validation_losses(
            vector_field, transport_mlp, bank, validation_indices, pairs, device
        )
        if validation_cfm < best_cfm["loss"]:
            best_cfm = {"epoch": epoch, "loss": validation_cfm, "state": _best_state(vector_field)}
        if validation_mlp < best_mlp["loss"]:
            best_mlp = {"epoch": epoch, "loss": validation_mlp, "state": _best_state(transport_mlp)}
        history.append(
            {"epoch": epoch, "train_cfm_loss": cfm_sum / updates, "validation_cfm_loss": validation_cfm,
             "train_mlp_loss": mlp_sum / updates, "validation_mlp_loss": validation_mlp}
        )
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(
                f"mechanism epoch {epoch}/{epochs} | val_cfm={validation_cfm:.6f}, "
                f"val_mlp={validation_mlp:.6f}", flush=True
            )
    vector_field.load_state_dict(best_cfm["state"])
    transport_mlp.load_state_dict(best_mlp["state"])
    return vector_field, transport_mlp, pd.DataFrame(history), best_cfm, best_mlp


@torch.no_grad()
def integrate_flow(model, initial, source_budget, target_budget, steps, device):
    model.eval()
    z = initial.to(device)
    source = torch.full((len(z),), source_budget, device=device)
    target = torch.full((len(z),), target_budget, device=device)
    dt = 1.0 / steps
    for step in range(steps):
        time = torch.full((len(z),), (step + 0.5) * dt, device=device)
        z = z + dt * model(z, time, source, target)
    return F.normalize(z, dim=1).cpu()


@torch.no_grad()
def predict_conditional_mlp(model, initial, source_budget, target_budget, device):
    model.eval()
    initial = initial.to(device)
    source = torch.full((len(initial),), source_budget, device=device)
    target = torch.full((len(initial),), target_budget, device=device)
    return F.normalize(model(initial, source, target), dim=1).cpu()


def _blend_predictions(lower_prediction, upper_prediction, fraction):
    return F.normalize((1.0 - fraction) * lower_prediction + fraction * upper_prediction, dim=1)


def evaluate_holdout(
    vector_field, transport_mlp, bank, known_budgets, holdout, test_indices, config, device
):
    lower = max(budget for budget in known_budgets if budget < holdout)
    upper = min(budget for budget in known_budgets if budget > holdout)
    fraction = (holdout - lower) / (upper - lower)
    z_lower = bank[lower][test_indices]
    z_upper = bank[upper][test_indices]
    truth = bank[holdout][test_indices]
    nearest = z_lower if holdout - lower <= upper - holdout else z_upper
    linear = _blend_predictions(z_lower, z_upper, fraction)
    cfm = _blend_predictions(
        integrate_flow(vector_field, z_lower, lower, holdout, int(config["cfm"]["integration_steps"]), device),
        integrate_flow(vector_field, z_upper, upper, holdout, int(config["cfm"]["integration_steps"]), device),
        fraction,
    )
    mlp = _blend_predictions(
        predict_conditional_mlp(transport_mlp, z_lower, lower, holdout, device),
        predict_conditional_mlp(transport_mlp, z_upper, upper, holdout, device),
        fraction,
    )
    records = []
    for method, prediction in {
        "shared_nearest": nearest,
        "linear_interpolation": linear,
        "conditional_mlp": mlp,
        "cfm": cfm,
    }.items():
        records.append(
            {
                "method": method,
                "holdout_width": holdout,
                "lower_width": lower,
                "upper_width": upper,
                "sliced_wasserstein": compute_distribution_distance(
                    prediction.numpy(), truth.numpy(), method="sliced_wasserstein",
                    num_projections=int(config["geometry"]["num_projections"]),
                    seed=int(config["geometry"]["projection_seed"]),
                ),
                "paired_mse": float(F.mse_loss(prediction, truth)),
                "paired_cosine": float(F.cosine_similarity(prediction, truth).mean()),
            }
        )
    return pd.DataFrame(records)


def run_cfm_mechanism(baseline_root: str | Path, config: dict, output_dir: str | Path):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    budgets = sorted({float(x) for task in config["cfm"]["tasks"] for x in task["known"] + [task["holdout"]]})
    all_results = []
    for seed in config["experiment"]["seeds"]:
        seed = int(seed); seed_everything(seed)
        bank, sample_ids = load_feature_bank(baseline_root, seed, budgets)
        train_indices, validation_indices, test_indices = fixed_sample_split(
            len(sample_ids), int(config["cfm"]["sample_split_seed"]),
            float(config["cfm"]["train_fraction"]), float(config["cfm"]["validation_fraction"]),
        )
        for task in config["cfm"]["tasks"]:
            holdout = float(task["holdout"]); known = [float(x) for x in task["known"]]
            vector_field, mlp, history, best_cfm, best_mlp = train_mechanism_models(
                bank, known, train_indices, validation_indices, config, torch.device(config["experiment"]["device"])
            )
            tag = budget_tag(holdout); task_dir = output_dir / f"seed_{seed}" / f"holdout_{tag}"
            task_dir.mkdir(parents=True, exist_ok=True)
            history.to_csv(task_dir / "training_history.csv", index=False)
            torch.save({"model": vector_field.state_dict(), "best_epoch": best_cfm["epoch"]}, task_dir / "cfm_best.pt")
            torch.save({"model": mlp.state_dict(), "best_epoch": best_mlp["epoch"]}, task_dir / "conditional_mlp_best.pt")
            result = evaluate_holdout(vector_field, mlp, bank, known, holdout, test_indices, config, torch.device(config["experiment"]["device"]))
            result.insert(0, "seed", seed)
            result["cfm_best_epoch"] = best_cfm["epoch"]
            result["mlp_best_epoch"] = best_mlp["epoch"]
            result.to_csv(task_dir / "metrics.csv", index=False)
            all_results.append(result)
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(output_dir / "cfm_mechanism_results.csv", index=False)
    return combined
