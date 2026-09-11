import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from georeg_curvature import (
    ANCHORS,
    curvature_components,
    fixed_projection_directions,
    paired_accuracy_bootstrap,
    paired_curvature_bootstrap,
    load_training_checkpoint,
    train_segment,
)
from scripts.run_georeg_rq3 import validate_rq3_config


def test_rq3_config_locks_unseen_protocol():
    with open("configs/kaggle_georeg_rq3.yaml") as handle:
        config = yaml.safe_load(handle)
    validate_rq3_config(config)
    assert config["dataset"]["num_workers"] == 0
    assert config["georeg"]["train_projection_seed"] != config["georeg"]["heldout_projection_seed"]


def test_fixed_projection_directions_are_reproducible_and_unit_norm():
    first = fixed_projection_directions(8, 5, 17)
    second = fixed_projection_directions(8, 5, 17)
    different = fixed_projection_directions(8, 5, 18)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert torch.allclose(torch.linalg.vector_norm(first, dim=0), torch.ones(5))


def test_curvature_loss_is_zero_for_stationary_path_and_positive_when_bent():
    generator = torch.Generator().manual_seed(3)
    base = torch.randn(16, 8, generator=generator)
    directions = fixed_projection_directions(8, 6, 11)
    stationary = {width: base.clone() for width in ANCHORS}
    dgeo, curvature, motion = curvature_components(stationary, directions)
    assert torch.isclose(curvature, torch.zeros(()), atol=1e-10)
    assert torch.isclose(motion, torch.zeros(()), atol=1e-10)
    assert torch.isclose(dgeo, torch.zeros(()), atol=1e-10)
    bent = dict(stationary)
    bent[0.5] = torch.randn(16, 8, generator=generator)
    bent_dgeo, bent_curvature, bent_motion = curvature_components(bent, directions)
    assert bent_dgeo > 0 and bent_curvature > 0 and bent_motion > 0


def test_paired_curvature_bootstrap_recomputes_statistic_from_rows():
    generator = torch.Generator().manual_seed(9)
    baseline = {
        width: torch.randn(30, 7, generator=generator) + width for width in ANCHORS
    }
    candidate = {width: values.clone() for width, values in baseline.items()}
    directions = fixed_projection_directions(7, 4, 5)
    result = paired_curvature_bootstrap(
        baseline, candidate, directions, replicates=20, seed=1
    )
    assert result["delta_D_geo"] == 0.0
    assert result["ci_low"] == 0.0 and result["ci_high"] == 0.0


def test_paired_accuracy_bootstrap_preserves_sample_and_width_pairing():
    rows = []
    for sample_id in range(20):
        for budget in (0.3, 0.4):
            rows.append({"sample_id": sample_id, "budget": budget, "correct": 0})
    baseline = pd.DataFrame(rows)
    candidate = baseline.copy()
    candidate["correct"] = 1
    result = paired_accuracy_bootstrap(
        baseline, candidate, [0.3, 0.4], replicates=20, seed=2
    )
    assert np.isclose(result["delta_mean_unseen_accuracy"], 1.0)
    assert np.isclose(result["ci_low"], 1.0) and np.isclose(result["ci_high"], 1.0)


class _TinyGeoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = nn.ModuleList([nn.Linear(4, 4)])
        self.projection = nn.Linear(4, 3)
        self.classifier = nn.Linear(3, 2)
        self.width = 1.0

    def set_width(self, width):
        self.width = float(width)

    def forward_features(self, inputs):
        hidden = torch.tanh(self.layer4[-1](inputs))
        offset = hidden.new_tensor([0.0, 0.2, -0.1, 0.3]) * self.width**2
        return self.projection(hidden * (0.5 + self.width) + offset)


def _tiny_training_objects(seed):
    torch.manual_seed(seed)
    model = _TinyGeoModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    return model, optimizer, scheduler


def test_training_checkpoint_replays_branch_rng_and_optimizer_state(tmp_path):
    inputs = torch.randn(12, 4, generator=torch.Generator().manual_seed(4))
    labels = torch.randint(0, 2, (12,), generator=torch.Generator().manual_seed(5))
    ids = torch.arange(12)
    config = {
        "training": {"epochs": 2, "kd_lambda": 1.0, "kd_temperature": 2.0},
        "georeg": {"epsilon": 1e-8},
    }
    directions = fixed_projection_directions(3, 2, 6)

    def loader():
        return DataLoader(
            TensorDataset(inputs, labels, ids), batch_size=4, shuffle=True,
            generator=torch.Generator().manual_seed(8), num_workers=0,
        )

    first_loader = loader()
    model, optimizer, scheduler = _tiny_training_objects(10)
    _, _, checkpoint = train_segment(
        model, optimizer, scheduler, first_loader, config, torch.device("cpu"), directions,
        1, 1, 0.0, tmp_path / "warmup",
    )
    states = []
    for branch in ("a", "b"):
        branch_loader = loader()
        branch_model, branch_optimizer, branch_scheduler = _tiny_training_objects(999)
        loaded_epoch = load_training_checkpoint(
            checkpoint, branch_model, branch_optimizer, branch_scheduler,
            branch_loader, torch.device("cpu"),
        )
        train_segment(
            branch_model, branch_optimizer, branch_scheduler, branch_loader, config,
            torch.device("cpu"), directions, loaded_epoch + 1, 2, 0.1,
            tmp_path / branch,
        )
        states.append({name: value.detach().clone() for name, value in branch_model.state_dict().items()})
    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(states[0][name], states[1][name]) for name in states[0])
