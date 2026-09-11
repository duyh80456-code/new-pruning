from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data import _stratified_cifar_split
from scripts.run_confirmatory import _attach_specialization
from specialization import select_best_specialized_checkpoint


def test_stratified_confirmatory_split_is_fixed_and_balanced():
    targets = [class_id for class_id in range(100) for _ in range(500)]
    train_a, validation_a = _stratified_cifar_split(targets, 5000, seed=17)
    train_b, validation_b = _stratified_cifar_split(targets, 5000, seed=17)
    assert train_a == train_b and validation_a == validation_b
    assert len(train_a) == 45_000 and len(validation_a) == 5_000
    assert set(train_a).isdisjoint(validation_a)
    validation_targets = np.asarray(targets)[validation_a]
    assert all((validation_targets == class_id).sum() == 50 for class_id in range(100))


class _TinyWidthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 3)

    def set_width(self, width):
        self.width = float(width)

    def forward(self, inputs):
        return self.layer(inputs)


def test_specialization_selects_and_persists_best_validation_checkpoint(tmp_path):
    inputs = torch.randn(12, 4)
    targets = torch.randint(0, 3, (12,))
    sample_ids = torch.arange(12)
    loader = DataLoader(
        TensorDataset(inputs, targets, sample_ids),
        batch_size=4,
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
    )
    deterministic_loader = DataLoader(
        TensorDataset(inputs, targets, sample_ids), batch_size=4, shuffle=False
    )
    config = {
        "training": {"momentum": 0.0, "weight_decay": 0.0},
        "specialization": {"epochs": 2, "learning_rate": 0.01},
        "evaluation": {"bn_calibration_batches": 2},
    }
    selected = select_best_specialized_checkpoint(
        _TinyWidthModel(),
        0.4,
        loader,
        deterministic_loader,
        deterministic_loader,
        config,
        torch.device("cpu"),
        seed=0,
        output_dir=tmp_path,
    )
    assert Path(selected["checkpoint"]).is_file()
    assert 0 <= selected["epoch"] <= 2
    history = __import__("pandas").read_csv(
        tmp_path / "results" / "specialization_history_budget_040.csv"
    )
    assert len(history) == 3 and history["epoch"].tolist() == [0, 1, 2]


def test_attach_specialization_uses_only_selected_width_rows():
    pd = __import__("pandas")
    central = pd.DataFrame({"budget": [0.3, 0.4, 0.6], "accuracy": [0.5, 0.6, 0.7]})
    specialized = pd.DataFrame(
        {
            "width": [0.4],
            "specialized_test_accuracy": [0.65],
            "specialization_gap": [0.05],
            "best_epoch": [7],
            "best_validation_accuracy": [0.66],
        }
    )
    result = _attach_specialization(central, specialized)
    selected = result.loc[result["budget"] == 0.4].iloc[0]
    assert np.isclose(selected["specialization_gap_if_available"], 0.05)
    assert result["specialization_gap_if_available"].notna().sum() == 1
