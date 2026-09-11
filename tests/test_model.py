import torch

from models import slimmable_resnet18
from models.slimmable_ops import SlimmableConv2d, active_channels


WIDTHS = [0.25, 0.30, 0.50, 0.75, 1.0]


def test_exact_floor_rule_and_nested_prefixes():
    assert active_channels(10, 0.25) == 2
    layer = SlimmableConv2d(8, 10, 3)
    x = torch.randn(1, 8, 8, 8)
    previous = 0
    for width in WIDTHS:
        layer.set_width(width)
        y = layer(x[:, : active_channels(8, width)])
        assert y.shape[1] == active_channels(10, width)
        assert y.shape[1] >= previous
        previous = y.shape[1]


def test_resnet_fixed_feature_dimension_for_all_widths():
    model = slimmable_resnet18(num_classes=10, supported_widths=WIDTHS, projection_dim=16)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        for width in WIDTHS:
            model.set_width(width)
            assert model.forward_features(x).shape == (2, 16)
            assert model(x).shape == (2, 10)


def test_unknown_width_is_rejected():
    model = slimmable_resnet18(supported_widths=WIDTHS)
    try:
        model.set_width(0.4)
    except ValueError as error:
        assert "outside configured" in str(error)
    else:
        raise AssertionError("Unconfigured width should be rejected")
