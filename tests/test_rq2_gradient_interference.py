import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rq2_gradient_interference import (
    DIAGNOSTIC_WIDTHS,
    compare_methods,
    diagnose_model,
    diagnostic_parameters,
    resolve_checkpoints,
    summarize_pairwise,
)


class TinyDiagnosticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.projection = nn.Linear(4, 3)
        self.classifier = nn.Linear(3, 2)
        self.width = 1.0

    def set_width(self, width):
        self.width = float(width)

    def forward(self, images):
        hidden = torch.tanh(self.backbone(images.flatten(1))) * self.width
        return self.classifier(torch.tanh(self.projection(hidden)))


def _config():
    return {"training": {"kd_lambda": 1.0, "kd_temperature": 2.0}}


def test_parameter_scopes_exclude_projection_and_primary_classifier():
    model = TinyDiagnosticModel()
    selected, backbone_numel = diagnostic_parameters(model)
    names = [name for name, _ in selected]
    assert not any(name.startswith("projection.") for name in names)
    assert names[:2] == ["backbone.weight", "backbone.bias"]
    assert names[-2:] == ["classifier.weight", "classifier.bias"]
    assert backbone_numel == model.backbone.weight.numel() + model.backbone.bias.numel()


def test_diagnose_model_produces_aligned_batchwise_pairs_and_norms():
    torch.manual_seed(3)
    model = TinyDiagnosticModel().eval()
    images = torch.randn(8, 1, 2, 2)
    labels = torch.randint(0, 2, (8,))
    ids = torch.arange(100, 108)
    loader = DataLoader(TensorDataset(images, labels, ids), batch_size=4, shuffle=False)
    pairs, norms = diagnose_model(
        model, loader, _config(), torch.device("cpu"), "Uniform", 1
    )
    pairs_per_batch = len(DIAGNOSTIC_WIDTHS) * (len(DIAGNOSTIC_WIDTHS) - 1) // 2
    assert len(pairs) == 2 * 2 * pairs_per_batch
    assert len(norms) == 2 * 2 * len(DIAGNOSTIC_WIDTHS)
    assert set(pairs["scope"]) == {"backbone", "backbone_classifier"}
    assert np.isfinite(pairs["cosine"]).all()
    assert np.isfinite(norms["gradient_norm"]).all()
    assert (norms["gradient_norm"] > 0).all()
    assert model.training is False
    assert all(parameter.grad is None for parameter in model.parameters())


def test_summaries_and_puregeo_minus_uniform_have_expected_sign():
    rows = []
    for method, shift in (("Uniform", 0.0), ("PureGeo", -0.60)):
        for batch, cosine in enumerate((0.5 + shift, -0.1 + shift)):
            rows.append({
                "method": method, "seed": 1, "scope": "backbone", "batch": batch,
                "width_i": 0.4, "width_j": 0.75, "cosine": cosine,
                "is_conflict": cosine < 0, "negative_cosine": max(-cosine, 0.0),
            })
    summary = summarize_pairwise(pd.DataFrame(rows))
    comparison = compare_methods(summary)
    assert len(comparison) == 1
    assert np.isclose(
        comparison.iloc[0]["delta_mean_cosine_puregeo_minus_uniform"], -0.60
    )
    assert comparison.iloc[0]["delta_conflict_rate_puregeo_minus_uniform"] > 0


def test_checkpoint_resolution_uses_frozen_rq2_paths(tmp_path):
    uniform, rq2 = tmp_path / "uniform", tmp_path / "rq2"
    expected = []
    for path in (
        uniform / "shared/seed_1/checkpoint.pt",
        uniform / "shared/seed_2/checkpoint.pt",
        rq2 / "geo/seed_1/checkpoint.pt",
        rq2 / "geo/seed_2/checkpoint.pt",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
        expected.append(path)
    assert list(resolve_checkpoints(uniform, rq2).values()) == expected
