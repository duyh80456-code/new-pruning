"""Modern PyTorch port of the core operators from JiahuiYu/slimmable_networks.

Source: https://github.com/JiahuiYu/slimmable_networks/blob/master/models/slimmable_ops.py
Adaptations: model-local width state, exact floor channel rule, and explicit BN
banks for the configured evaluation grid. See THIRD_PARTY.md.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def width_key(width: float) -> str:
    return f"w{int(round(float(width) * 1000)):04d}"


def active_channels(max_channels: int, width: float) -> int:
    """The experiment's required nested prefix rule: floor(width * C)."""
    return max(1, math.floor(max_channels * float(width) + 1e-9))


class WidthModule(nn.Module):
    width: float

    def set_width(self, width: float) -> None:
        self.width = float(width)


class SlimmableConv2d(WidthModule):
    def __init__(
        self,
        max_in_channels: int,
        max_out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        scale_in: bool = True,
        scale_out: bool = True,
    ) -> None:
        super().__init__()
        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (1, 1)
        self.scale_in = scale_in
        self.scale_out = scale_out
        self.width = 1.0
        self.weight = nn.Parameter(
            torch.empty(max_out_channels, max_in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(max_out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            fan_in = self.max_in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def active_in_channels(self) -> int:
        return active_channels(self.max_in_channels, self.width) if self.scale_in else self.max_in_channels

    @property
    def active_out_channels(self) -> int:
        return active_channels(self.max_out_channels, self.width) if self.scale_out else self.max_out_channels

    def forward(self, x: Tensor) -> Tensor:
        in_channels = x.shape[1]
        out_channels = self.active_out_channels
        weight = self.weight[:out_channels, :in_channels]
        bias = self.bias[:out_channels] if self.bias is not None else None
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, 1)

    def active_parameter_count(self) -> int:
        n = self.active_out_channels * self.active_in_channels * math.prod(self.kernel_size)
        return n + (self.active_out_channels if self.bias is not None else 0)


class SlimmableBatchNorm2d(WidthModule):
    """Shared affine parameters plus width-specific running-statistic banks.

    Only configured widths are valid. This prevents accidental use/training of an
    unrecorded budget and makes unseen-width calibration explicit.
    """

    def __init__(
        self,
        max_features: int,
        supported_widths: Iterable[float],
        eps: float = 1e-5,
        momentum: float | None = 0.1,
    ) -> None:
        super().__init__()
        self.max_features = max_features
        self.supported_widths = tuple(float(w) for w in supported_widths)
        self.eps = eps
        self.momentum = momentum
        self.width = 1.0
        self.weight = nn.Parameter(torch.ones(max_features))
        self.bias = nn.Parameter(torch.zeros(max_features))
        self.banks = nn.ModuleDict(
            {
                width_key(w): nn.BatchNorm2d(
                    active_channels(max_features, w), affine=False, momentum=momentum
                )
                for w in self.supported_widths
            }
        )

    def _bank(self) -> nn.BatchNorm2d:
        key = width_key(self.width)
        if key not in self.banks:
            raise ValueError(
                f"Width {self.width:g} is not configured. Supported widths: {self.supported_widths}"
            )
        return self.banks[key]

    def reset_current_stats(self) -> None:
        self._bank().reset_running_stats()

    def forward(self, x: Tensor) -> Tensor:
        bank = self._bank()
        c = x.shape[1]
        exponential_average_factor = 0.0 if bank.momentum is None else bank.momentum
        if self.training:
            bank.num_batches_tracked.add_(1)
            if bank.momentum is None:
                exponential_average_factor = 1.0 / float(bank.num_batches_tracked)
        return F.batch_norm(
            x,
            bank.running_mean,
            bank.running_var,
            self.weight[:c],
            self.bias[:c],
            self.training,
            exponential_average_factor,
            bank.eps,
        )

    def active_parameter_count(self) -> int:
        return 2 * active_channels(self.max_features, self.width)


class SharedProjection(WidthModule):
    """A fixed-output projection whose input is the active feature prefix."""

    def __init__(self, max_in_features: int, out_features: int) -> None:
        super().__init__()
        self.max_in_features = max_in_features
        self.out_features = out_features
        self.width = 1.0
        self.weight = nn.Parameter(torch.empty(out_features, max_in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight[:, : x.shape[1]], self.bias)

    def active_parameter_count(self) -> int:
        return self.out_features * active_channels(self.max_in_features, self.width) + self.out_features
