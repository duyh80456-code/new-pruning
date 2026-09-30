"""feature_pre_relu: the transport reads the last block before its ReLU.

Built to fail if the flag is silently ignored. CR (K on ResNet-50 at seed
2026) died with the final BN shift pushed negative on the channels the
transport reads: after the ReLU those channels were zero for every input,
all-zero clouds transport to each other at no cost, and nothing flowed
back. The test rebuilds that state by hand and checks:

  off, the dead state is a fixed point: the feature is all zero, the
      transport between two widths is zero, and no gradient reaches the
      last block;
  on,  the same weights give a feature that varies with the input and a
      gradient that reaches the last block;
  on or off, the logits are the same, so the classifier path is untouched.

Runs on the CPU in under a minute: python tests/test_pre_relu.py
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = [sys.argv[0], 'app:apps/smoke_by_k_r50.yml']

import torch  # noqa: E402

from utils.config import FLAGS  # noqa: E402
from utils.loss_ops import build_feature_criterion  # noqa: E402


def check(cond, what):
    if not cond:
        raise AssertionError(what)
    print('ok   ', what)


def run(model, x, width, pre_relu):
    FLAGS.feature_pre_relu = pre_relu
    model.apply(lambda m: setattr(m, 'width_mult', width))
    return model(x)


def main():
    FLAGS.return_features = True
    FLAGS.reset_parameters = True
    torch.manual_seed(0)
    model = importlib.import_module('models.us_resnet').Model(100, 32)
    model.train()
    last = model.features[len(model.features) - 2]

    # the CR state: the closing BNs of the last block shifted far below
    # zero, so its ReLU output is zero for every input at every width
    with torch.no_grad():
        last.body[-1].bias.fill_(-50.0)
        if len(last.shortcut):
            last.shortcut[-1].bias.fill_(-50.0)

    x = torch.randn(16, 3, 32, 32)
    criterion = build_feature_criterion()

    # off: the dead state is a fixed point
    model.zero_grad()
    logits_off, (s_off,) = run(model, x, 0.35, False)
    _, (t_off,) = run(model, x, 1.0, False)
    check(bool((s_off == 0).all()), 'off: the dead feature is zero')
    loss = criterion((s_off,), (t_off.detach(),))
    check(abs(float(loss)) < 1e-6,
          'off: dead clouds transport at no cost ({:.2e})'.format(
              float(loss)))
    if loss.requires_grad:
        loss.backward()
    grad = last.body[-2].weight.grad
    check(grad is None or float(grad.abs().sum()) == 0.0,
          'off: no gradient reaches the last block')

    # on: the same weights, a live signal
    model.zero_grad()
    logits_on, (s_on,) = run(model, x, 0.35, True)
    _, (t_on,) = run(model, x, 1.0, True)
    check(float(s_on.std(0).mean()) > 1e-3,
          'on: the feature varies with the input')
    loss = criterion((s_on,), (t_on.detach(),))
    loss.backward()
    grad = last.body[-2].weight.grad
    check(grad is not None and float(grad.abs().sum()) > 0.0,
          'on: the gradient reaches the last block')

    check(torch.allclose(logits_off, logits_on),
          'the logits do not depend on the flag')

    # and off must be the model as it was: 'final' is the pooled output
    with torch.no_grad():
        FLAGS.return_features = False
        plain = run(model, x, 0.5, False)
        FLAGS.return_features = True
        logits, (final,) = run(model, x, 0.5, False)
    check(torch.allclose(plain, logits), 'off: logits as before')
    with torch.no_grad():
        FLAGS.feature_pre_relu = False
        pooled = model.features(x).flatten(1)
    check(torch.allclose(final, pooled),
          'off: final is the pooled post-ReLU feature')
    print('all checks passed')


if __name__ == '__main__':
    main()
