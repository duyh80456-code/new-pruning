from __future__ import annotations

import copy

import torch

from models.slimmable_ops import SharedProjection, SlimmableConv2d


def _count_dynamic_conv(module: SlimmableConv2d, inputs, output) -> None:
    batch = output.shape[0]
    spatial = output.shape[2] * output.shape[3]
    kernel_ops = module.kernel_size[0] * module.kernel_size[1] * inputs[0].shape[1]
    module.total_ops += torch.DoubleTensor([batch * output.shape[1] * spatial * kernel_ops])


def _count_projection(module: SharedProjection, inputs, output) -> None:
    module.total_ops += torch.DoubleTensor([inputs[0].shape[0] * inputs[0].shape[1] * output.shape[1]])


def profile_subnet(model, width: float, input_size: tuple[int, int, int] = (3, 32, 32)) -> tuple[int, int]:
    """Profile active operations with THOP; return (FLOPs, active parameters).

    THOP counts a multiply-accumulate as one MAC. We report two FLOPs per MAC and
    use that convention consistently for all budgets.
    """
    try:
        from thop import profile
    except ImportError as exc:
        raise RuntimeError("THOP is required for FLOPs/MAC profiling: pip install thop") from exc
    was_training = model.training
    model.set_width(width)
    model.eval()
    dummy = torch.zeros(1, *input_size, device=next(model.parameters()).device)
    macs, _ = profile(
        model,
        inputs=(dummy,),
        custom_ops={SlimmableConv2d: _count_dynamic_conv, SharedProjection: _count_projection},
        verbose=False,
    )
    if was_training:
        model.train()
    return int(2 * macs), int(model.active_parameter_count(width))
