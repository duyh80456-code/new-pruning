"""The frozen narrow block has to survive a real optimizer step unchanged.

BT zeroed the gradient over the width-0.25 block and trusted that to hold
it still. It did not: SGD adds weight_decay * w inside step() and momentum
carries it forward, so the block shrank toward zero and width 0.25 ended
at chance. The check that shipped with it looked at the gradient. This one
looks at the weights, after the step, with the optimizer train.py builds,
and first shows that the step without restore_frozen does move them, so a
pass here is not a check that cannot fail.

Runs on the CPU in a few seconds: python tests/test_freeze.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = [sys.argv[0], 'app:apps/smoke_bt_narrow_first_frozen.yml']

import torch  # noqa: E402

import train as T  # noqa: E402
from utils.config import FLAGS  # noqa: E402


def check(cond, what):
    if not cond:
        raise AssertionError(what)
    print('ok   ', what)


def run(restore):
    import importlib
    FLAGS.reset_parameters = True
    FLAGS.width_scalars = False
    FLAGS.return_features = False
    torch.manual_seed(0)
    model = importlib.import_module('models.us_resnet').Model(100, 32)
    model.train()
    optimizer = T.get_optimizer(model)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 100, (8,))
    epoch = FLAGS.narrow_first_epochs + 1
    first = None
    for _ in range(3):
        optimizer.zero_grad()
        for width in (1.0, 0.25, 0.6):
            model.apply(lambda m: setattr(m, 'width_mult', width))
            torch.nn.functional.cross_entropy(model(x), y).backward()
        held = T.freeze_narrow_prefix(model, epoch)
        if first is None:
            first = [(t, i, v.clone()) for t, i, v in held]
            start = {n: p.detach().clone() for n, p in model.named_parameters()}
        optimizer.step()
        if restore:
            T.restore_frozen(held)
    frozen = max((t.detach()[i] - v).abs().max().item() for t, i, v in first)
    total = sum((p.detach() - start[n]).pow(2).sum()
                for n, p in model.named_parameters()).sqrt().item()
    return frozen, total, len(first)


if __name__ == '__main__':
    check(FLAGS.freeze_prefix_at and FLAGS.narrow_first_epochs,
          'the config under test freezes a block')
    moved, _, _ = run(restore=False)
    check(moved > 1e-6,
          'without restore_frozen the zero-gradient block still moves '
          '({:.2e}), so this test can fail'.format(moved))
    moved, total, count = run(restore=True)
    check(moved == 0.0,
          'with restore_frozen all {} frozen slices are bit-identical after '
          '3 steps'.format(count))
    check(total > 1e-3,
          'the rest of the network still trains ({:.2e})'.format(total))
    print('all checks passed')
