"""How hard each feature loss pulls, on the real model at the real widths.

    python scripts/calibrate_feature_weight.py

Four losses on one axis are only a comparison if they apply comparable
pressure. At feature_weight 1.0 they do not: measured here, the entropic
transport term contributes a gradient of 3.65 and MMD 0.39, so a branch
that lost would leave "this loss is worse" and "this term was ten times
weaker" indistinguishable.

The loss value is the obvious thing to equalize and it is the wrong one.
What reaches the weights is the gradient, and the two criteria disagree by
a factor of three for mse and nine for mmd.

What this cannot do is stay true. It is one measurement on an untrained
model, and the ratios drift as the widths converge. It is a starting point
chosen by measurement rather than by taste, not a schedule.
"""
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
if not any(arg.startswith('app:') for arg in sys.argv):
    sys.argv.append('app:apps/cifar100_k_feature_pair.yml')

import torch  # noqa: E402

from utils.config import FLAGS  # noqa: E402
from utils.loss_ops import FeatureMMDLoss  # noqa: E402
from utils.loss_ops import FeatureMSELoss  # noqa: E402
from utils.loss_ops import FeatureSlicedWassersteinLoss  # noqa: E402
from utils.loss_ops import FeatureWassersteinLoss  # noqa: E402

REFERENCE = 'wasserstein'
BATCHES = 4
BATCH = 64
# max, min, and two middles: the shape the sandwich rule draws every step
WIDTHS = [1.0, 0.25, 0.55, 0.8]


def losses():
    return [
        ('mse', FeatureMSELoss(align='prefix')),
        ('mmd', FeatureMMDLoss(
            align='prefix',
            bandwidth=getattr(FLAGS, 'mmd_bandwidth', 1.0))),
        ('sliced', FeatureSlicedWassersteinLoss(
            align='prefix',
            n_projections=getattr(FLAGS, 'sliced_projections', 128))),
        ('wasserstein', FeatureWassersteinLoss(
            align='prefix',
            eps=getattr(FLAGS, 'sinkhorn_eps', 0.2),
            n_iters=getattr(FLAGS, 'sinkhorn_iters', 100))),
    ]


def cloud_at(model, batch, width):
    model.apply(lambda m: setattr(m, 'width_mult', width))
    return model(batch)[1][0]


def main():
    FLAGS.return_features = True
    FLAGS.feature_layers = ['final']
    torch.manual_seed(0)
    model = importlib.import_module(FLAGS.model).Model(
        FLAGS.num_classes, input_size=FLAGS.image_size)
    model.train()

    totals = {name: [0.0, 0.0] for name, _ in losses()}
    for step in range(BATCHES):
        batch = torch.randn(BATCH, 3, FLAGS.image_size, FLAGS.image_size)
        for name, loss_fn in losses():
            model.zero_grad(set_to_none=True)
            clouds = {w: cloud_at(model, batch, w) for w in WIDTHS}
            # the same seed on both sides so the sliced projections do not
            # vary between the value and the gradient being attributed to it
            torch.manual_seed(100 + step)
            term = sum(loss_fn(clouds[w], clouds[1.0]) for w in WIDTHS[1:])
            torch.manual_seed(100 + step)
            term = term + loss_fn(clouds[WIDTHS[2]], clouds[WIDTHS[3]])
            value = term.item()
            term.backward()
            grad = torch.sqrt(sum(
                p.grad.pow(2).sum() for p in model.parameters()
                if p.grad is not None)).item()
            totals[name][0] += value / BATCHES
            totals[name][1] += grad / BATCHES
    model.zero_grad(set_to_none=True)

    reference_value, reference_grad = totals[REFERENCE]
    print('{:<13}{:>10}{:>10}{:>14}{:>13}'.format(
        'feature_loss', 'value', 'grad', 'weight/value', 'weight/grad'))
    for name, _ in losses():
        value, grad = totals[name]
        print('{:<13}{:>10.4f}{:>10.4f}{:>14.1f}{:>13.1f}'.format(
            name, value, grad,
            reference_value / max(value, 1e-12),
            reference_grad / max(grad, 1e-12)))
    print()
    print('weight/grad is what apps/cifar100_{q,r,s}_*.yml use, rounded.')
    print('Measured on an untrained model over {} batches of {}.'.format(
        BATCHES, BATCH))


if __name__ == '__main__':
    main()
