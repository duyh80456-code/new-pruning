"""Mixup and cutmix for the ViT runs, as in DeiT and timm

Each training batch is mixed with a permutation of itself: by mixup
(alpha mixup) or cutmix (alpha cutmix), one or the other chosen per batch
with probability mix_switch_prob for cutmix. The labels stay integer and
the loss is the matching mixture of two losses, lam * L(y) + (1 - lam) *
L(y_perm), which is the soft-target cross entropy for a criterion linear
in its target. Inplace distillation is untouched: the widest width sees
the mixed image like every student, and its soft output is their target.
Off unless mixup or cutmix is set.
"""
import numpy as np
import torch

from utils.config import FLAGS


def mixing_on():
    return (getattr(FLAGS, 'mixup', 0.0) > 0
            or getattr(FLAGS, 'cutmix', 0.0) > 0)


def mix_batch(input, target):
    """the mixed batch, the partner labels and the weight on the own ones"""
    mixup = getattr(FLAGS, 'mixup', 0.0)
    cutmix = getattr(FLAGS, 'cutmix', 0.0)
    use_cutmix = cutmix > 0 and (
        mixup <= 0
        or np.random.rand() < getattr(FLAGS, 'mix_switch_prob', 0.5))
    perm = torch.randperm(input.size(0), device=input.device)
    if use_cutmix:
        lam = np.random.beta(cutmix, cutmix)
        height, width = input.shape[2:]
        cut = np.sqrt(1.0 - lam)
        ch, cw = int(height * cut), int(width * cut)
        cy, cx = np.random.randint(height), np.random.randint(width)
        y0, y1 = np.clip(cy - ch // 2, 0, height), np.clip(cy + ch // 2, 0,
                                                           height)
        x0, x1 = np.clip(cx - cw // 2, 0, width), np.clip(cx + cw // 2, 0,
                                                          width)
        input = input.clone()
        input[:, :, y0:y1, x0:x1] = input[perm, :, y0:y1, x0:x1]
        # the area actually pasted, after clipping at the border
        lam = 1.0 - (y1 - y0) * (x1 - x0) / float(height * width)
    else:
        lam = np.random.beta(mixup, mixup)
        input = lam * input + (1.0 - lam) * input[perm]
    return input, target[perm], float(lam)


class MixedCriterion(object):
    """criterion(output, target) on a mixed batch"""

    def __init__(self, criterion, partner, lam):
        self.criterion = criterion
        self.partner = partner
        self.lam = lam

    def __call__(self, output, target):
        return (self.lam * self.criterion(output, target)
                + (1.0 - self.lam) * self.criterion(output, self.partner))
