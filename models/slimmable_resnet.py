"""CIFAR ResNet-18 adapted from torchvision's ResNet implementation.

Upstream source: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
The block/stage topology remains recognizable; dynamic prefix-sliced operators,
a CIFAR stem, and a fixed-dimensional projection are the only material changes.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from .slimmable_ops import (
    SharedProjection,
    SlimmableBatchNorm2d,
    SlimmableConv2d,
    WidthModule,
)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int, widths: Iterable[float]) -> None:
        super().__init__()
        self.conv1 = SlimmableConv2d(inplanes, planes, 3, stride=stride, padding=1)
        self.bn1 = SlimmableBatchNorm2d(planes, widths)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SlimmableConv2d(planes, planes, 3, padding=1)
        self.bn2 = SlimmableBatchNorm2d(planes, widths)
        self.downsample: nn.Module | None = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                SlimmableConv2d(inplanes, planes, 1, stride=stride),
                SlimmableBatchNorm2d(planes, widths),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class SlimmableResNet(nn.Module):
    def __init__(
        self,
        layers: list[int],
        num_classes: int,
        supported_widths: Iterable[float],
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        widths = tuple(sorted(set(float(w) for w in supported_widths)))
        if 1.0 not in widths:
            raise ValueError("supported_widths must contain 1.0")
        self.supported_widths = widths
        self.width = 1.0
        self.inplanes = 64
        self.conv1 = SlimmableConv2d(3, 64, 3, stride=1, padding=1, scale_in=False)
        self.bn1 = SlimmableBatchNorm2d(64, widths)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, layers[0], 1, widths)
        self.layer2 = self._make_layer(128, layers[1], 2, widths)
        self.layer3 = self._make_layer(256, layers[2], 2, widths)
        self.layer4 = self._make_layer(512, layers[3], 2, widths)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = SharedProjection(512, projection_dim)
        self.classifier = nn.Linear(projection_dim, num_classes)

    def _make_layer(
        self, planes: int, blocks: int, stride: int, widths: Iterable[float]
    ) -> nn.Sequential:
        result = [BasicBlock(self.inplanes, planes, stride, widths)]
        self.inplanes = planes
        result.extend(BasicBlock(self.inplanes, planes, 1, widths) for _ in range(1, blocks))
        return nn.Sequential(*result)

    def set_width(self, width: float) -> None:
        width = float(width)
        if not any(abs(width - w) < 1e-8 for w in self.supported_widths):
            raise ValueError(f"Width {width:g} is outside configured evaluation grid")
        self.width = width
        for module in self.modules():
            if module is not self and isinstance(module, WidthModule):
                module.set_width(width)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.projection(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.forward_features(x))

    def active_parameter_count(self, width: float | None = None) -> int:
        if width is not None:
            self.set_width(width)
        dynamic = sum(
            m.active_parameter_count()
            for m in self.modules()
            if m is not self and hasattr(m, "active_parameter_count")
        )
        return dynamic + sum(p.numel() for p in self.classifier.parameters())


def slimmable_resnet18(
    num_classes: int = 100,
    supported_widths: Iterable[float] = (0.25, 0.5, 0.75, 1.0),
    projection_dim: int = 128,
) -> SlimmableResNet:
    return SlimmableResNet([2, 2, 2, 2], num_classes, supported_widths, projection_dim)
