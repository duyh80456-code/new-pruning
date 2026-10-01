"""DYNAS on US-Net: a learning rate and a momentum per kind of sub-network.

Jeon et al., "Subnet-Aware Dynamic Supernet Training for Neural
Architecture Search", CVPR 2025, arXiv 2503.10740. Code:
github.com/cvlab-yonsei/DYNAS, exps/NAS-Bench-201-algos/train_spos.py and
utils/LR_scheduler.py. Two parts, both in the optimizer:

  CaLR  each sub-network's learning rate decays as (1 - t/T)^p, with the
        exponent linear in the log of its parameter count: p = gamma_max
        (4) at the smallest, 1/gamma_max at the largest. Small networks
        converge fast and are held back; large ones keep a high rate.
  MS    sub-networks are split into groups, and each group keeps its own
        SGD momentum buffer over the shared weights, so the momentum one
        kind of sub-network built up does not push the next kind.

Their supernet trains one sub-network per step and steps the optimizer of
its group. US-Net's sandwich runs four widths per step and sums their
gradients into one update. Here each width takes its own update through
its own group's optimizer instead, the DYNAS way, and three things keep
the step comparable to US-Net's:

  the base learning rate is US-Net's: four sequential steps at lr move the
        weights by the same first-order amount as one step on the sum;
  weight decay is divided by the number of widths a step runs, because
        SGD applies it on every optimizer step and there are now four;
  the gradient is clipped per width at grad_clip, as DYNAS clips per
        sub-network step.

Groups are equal bins of the width range; their groups came from the
operation on one edge of NAS-Bench-201, which has no counterpart here. The
5-epoch linear warm-up of this project's protocol is kept and applies to
every group alike, before CaLR takes over. The teacher's soft target is
computed before any width updates and is detached, as in US-Net.
"""
import math

import torch

from models.slimmable_ops import USConv2d, USLinear, make_divisible
from utils.config import FLAGS


def params_at(model, width):
    """weights a sub-network of this width reads, convs and classifier"""
    total = 0
    for m in model.modules():
        if isinstance(m, USConv2d):
            cin = (make_divisible(m.in_channels_max * width / m.ratio[0])
                   * m.ratio[0] if m.us[0] else m.in_channels_max)
            cout = (make_divisible(m.out_channels_max * width / m.ratio[1])
                    * m.ratio[1] if m.us[1] else m.out_channels_max)
            groups = cin if m.depthwise else 1
            total += cout * (cin // groups) * m.kernel_size[0] \
                * m.kernel_size[1]
        elif isinstance(m, USLinear):
            fin = (make_divisible(m.in_features_max * width) if m.us[0]
                   else m.in_features_max)
            fout = (make_divisible(m.out_features_max * width) if m.us[1]
                    else m.out_features_max)
            total += fin * fout
    return total


class Dynas(object):
    def __init__(self, model, make_optimizer, steps_per_iteration):
        self.groups = getattr(FLAGS, 'dynas_groups', 4)
        self.low, self.high = FLAGS.width_mult_range
        self.optimizers = [make_optimizer(model) for _ in range(self.groups)]
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                group['weight_decay'] /= steps_per_iteration
        gamma = getattr(FLAGS, 'dynas_max_coeff', 4.0)
        bare = model.module if hasattr(model, 'module') else model
        c_min = params_at(bare, self.low)
        c_max = params_at(bare, self.high)
        # train_spos.py: w = -(r_max - r_min) / (log C_max - log C_min),
        # tau = r_min - w log C_max, p = w log C + tau
        r_max, r_min = gamma, 1.0 / gamma
        self.slope = -(r_max - r_min) / (math.log(c_max) - math.log(c_min))
        self.tau = r_min - self.slope * math.log(c_max)
        self.model = bare
        self.exponents = {}

    def group_of(self, width):
        spot = (width - self.low) / (self.high - self.low) * self.groups
        return min(self.groups - 1, max(0, int(spot)))

    def exponent(self, width):
        key = round(width, 6)
        if key not in self.exponents:
            self.exponents[key] = (
                self.slope * math.log(params_at(self.model, width))
                + self.tau)
        return self.exponents[key]

    def lr(self, width, epoch, batch_idx):
        """warm-up as everywhere else, then (1 - t/T)^p"""
        warmup = getattr(FLAGS, 'lr_warmup_epochs', 0)
        per_epoch = FLAGS.data_size_train / FLAGS.batch_size
        current = epoch * per_epoch + batch_idx + 1
        if getattr(FLAGS, 'lr_warmup', False) and epoch < warmup:
            return FLAGS.lr * current / (warmup * per_epoch)
        span = (FLAGS.num_epochs - warmup) * per_epoch
        left = max(0.0, 1.0 - (current - warmup * per_epoch) / span)
        return FLAGS.lr * left ** self.exponent(width)

    def step(self, model, width, epoch, batch_idx):
        """one update for this width from the gradient now in .grad"""
        optimizer = self.optimizers[self.group_of(width)]
        rate = self.lr(width, epoch, batch_idx)
        for group in optimizer.param_groups:
            group['lr'] = rate
        clip = getattr(FLAGS, 'grad_clip', 0)
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        model.zero_grad()
        return rate

    def state_dict(self):
        return [o.state_dict() for o in self.optimizers]

    def load_state_dict(self, states):
        for optimizer, state in zip(self.optimizers, states):
            optimizer.load_state_dict(state)
