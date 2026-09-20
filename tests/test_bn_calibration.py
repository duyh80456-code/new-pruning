"""BN post-statistics, the step US-Net's accuracy at any width rests on

A 100-epoch run reached 75.4% at width 1.0 during training and then reported
1.0% at every width after calibration. The statistics were reset and never
refilled, so the network normalized by zero mean and unit variance and read
back at chance.

Runs on the CPU in a few seconds.
"""
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not any(arg.startswith('app:') for arg in sys.argv):
    sys.argv.append('app:apps/cifar100_a_kl.yml')

import torch

from models.slimmable_ops import USBatchNorm2d, bn_calibration_init
from utils.config import FLAGS


FAILURES = []


def check(name, condition, detail=''):
    status = 'pass' if condition else 'FAIL'
    print('{:5} {}{}'.format(status, name, '  ' + detail if detail else ''))
    if not condition:
        FAILURES.append(name)


def build_model():
    import importlib
    model_lib = importlib.import_module('models.us_resnet')
    return model_lib.Model(FLAGS.num_classes, input_size=FLAGS.image_size)


def batches(n, generator):
    """inputs with a mean and a spread that are nothing like 0 and 1"""
    for _ in range(n):
        yield 3.0 + 2.0 * torch.randn(16, 3, 32, 32, generator=generator)


def calibrate(model, widths, n_batches, generator):
    """exactly the sequence train.py runs for phase 'cal'"""
    model.eval()
    model.apply(bn_calibration_init)
    with torch.no_grad():
        for data in batches(n_batches, generator):
            for width in sorted(widths, reverse=True):
                model.apply(lambda m: setattr(m, 'width_mult', width))
                model(data)


def test_statistics_are_refilled():
    generator = torch.Generator().manual_seed(0)
    model = build_model()
    widths = FLAGS.width_mult_list

    calibrate(model, widths, 8, generator)

    reset_like = 0
    total = 0
    for module in model.modules():
        if not isinstance(module, USBatchNorm2d):
            continue
        for bn in module.bn:
            total += 1
            untouched = (bn.running_mean.abs().max().item() < 1e-6
                         and (bn.running_var - 1.0).abs().max().item() < 1e-6)
            reset_like += int(untouched)
    check(
        'calibration writes statistics back',
        reset_like == 0,
        '{} of {} still at the reset values'.format(reset_like, total))


def test_eval_agrees_with_batch_statistics():
    """the property calibration exists to provide

    With the statistics calibrated on this distribution, evaluating with them
    must give what the batch statistics gave. If they were never refilled the
    two are unrelated and the logits diverge.
    """
    generator = torch.Generator().manual_seed(1)
    model = build_model()
    widths = FLAGS.width_mult_list
    calibrate(model, widths, 8, generator)

    probe = next(iter(batches(1, generator)))
    worst = 0.0
    for width in sorted(widths, reverse=True):
        model.apply(lambda m: setattr(m, 'width_mult', width))
        model.eval()
        with torch.no_grad():
            calibrated = model(probe)
        model.train()
        with torch.no_grad():
            from_batch = model(probe)
        relative = ((calibrated - from_batch).norm()
                    / from_batch.norm().clamp_min(1e-12)).item()
        worst = max(worst, relative)
        print('        width {:.2f}: relative difference {:.3f}'.format(
            width, relative))
    check(
        'calibrated statistics reproduce the batch statistics',
        worst < 0.25,
        'worst relative difference = {:.3f}'.format(worst))


def main():
    print('torch', torch.__version__)
    print('widths', FLAGS.width_mult_list)
    print()
    test_statistics_are_refilled()
    test_eval_agrees_with_batch_statistics()
    print()
    if FAILURES:
        print('{} failed: {}'.format(len(FAILURES), ', '.join(FAILURES)))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
