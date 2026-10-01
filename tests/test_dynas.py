"""DYNAS on US-Net: utils/dynas.py and its wiring in train.py.

Checks the two parts against the authors' train_spos.py:
  CaLR  the exponent is gamma_max at the narrowest width and 1/gamma_max
        at the widest, linear in log parameters between; the rate is the
        warm-up first, then (1 - t/T)^p;
  MS    each group has its own momentum buffer: a step in one group leaves
        every other group's buffer untouched.
And the US-Net adaptation: weight decay divided by the widths per step.

Then the path that runs: a smoke config with dynas on goes through
train.py and must leave momentum in more than one group's optimizer. With
the wiring removed, only the shared optimizer would hold momentum and
this fails.

Runs on the CPU, the last part on the GPU if there is one:
python tests/test_dynas.py
"""
import importlib
import io
import math
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.argv = [sys.argv[0], 'app:apps/smoke_bx_a_r50.yml']

import torch  # noqa: E402

from utils.config import FLAGS  # noqa: E402
from utils.dynas import Dynas, params_at  # noqa: E402


def check(cond, what):
    if not cond:
        raise AssertionError(what)
    print('ok   ', what)


def sgd(model):
    return torch.optim.SGD([{'params': p, 'weight_decay': 5e-4, 'lr': 0.1,
                             'momentum': 0.9} for p in model.parameters()])


def unit():
    FLAGS.width_mult_list = FLAGS.width_mult_range
    FLAGS.data_size_train = 50000  # set by the data loader in train.py
    torch.manual_seed(0)
    model = importlib.import_module('models.us_resnet').Model(100, 32)
    dynas = Dynas(model, sgd, 4)

    check(abs(dynas.exponent(0.25) - 4.0) < 1e-9,
          'exponent 4 at the narrowest ({:.4f})'.format(dynas.exponent(0.25)))
    check(abs(dynas.exponent(1.0) - 0.25) < 1e-9,
          'exponent 1/4 at the widest ({:.4f})'.format(dynas.exponent(1.0)))
    mid = dynas.exponent(0.5)
    expected = 4.0 + (0.25 - 4.0) * (
        math.log(params_at(model, 0.5)) - math.log(params_at(model, 0.25))) / (
        math.log(params_at(model, 1.0)) - math.log(params_at(model, 0.25)))
    check(abs(mid - expected) < 1e-9 and 0.25 < mid < 4.0,
          'exponent linear in log parameters ({:.4f} at 0.5)'.format(mid))
    check([dynas.group_of(w) for w in (0.25, 0.4, 0.6, 0.8, 1.0)]
          == [0, 0, 1, 2, 3], 'width bins to groups')

    check(all(g['weight_decay'] == 5e-4 / 4
              for o in dynas.optimizers for g in o.param_groups),
          'weight decay divided by the widths per step')

    warm = FLAGS.lr_warmup_epochs
    check(abs(dynas.lr(0.25, 0, 0) - dynas.lr(1.0, 0, 0)) < 1e-12,
          'warm-up is the same for every width')
    late = FLAGS.num_epochs - 1
    check(dynas.lr(0.25, late, 0) < dynas.lr(1.0, late, 0),
          'late in training the narrow width has the lower rate')
    check(abs(dynas.lr(1.0, warm, 0) / FLAGS.lr - 1) < 0.01,
          'the rate starts from the base rate after warm-up')

    # momentum separation: step group 0, then check group 3 is empty
    x = torch.randn(4, 3, 32, 32)
    model.train()
    model.apply(lambda m: setattr(m, 'width_mult', 0.25))
    model(x).sum().backward()
    dynas.step(model, 0.25, warm, 0)
    filled = [sum(1 for s in o.state.values() if 'momentum_buffer' in s)
              for o in dynas.optimizers]
    check(filled[0] > 0 and filled[1:] == [0, 0, 0],
          'a step at 0.25 fills group 0 only ({})'.format(filled))
    check(all(p.grad is None or float(p.grad.abs().sum()) == 0
              for p in model.parameters()),
          'gradients cleared after the step')


def through_train():
    """the run must go through the dynas path in train.py"""
    if not torch.cuda.is_available():
        print('skip  no GPU: the train.py path is not exercised here')
        return
    base = io.open(os.path.join(ROOT, 'apps', 'smoke_db_dynas_r50.yml'),
                   encoding='utf-8').read()
    log = tempfile.mkdtemp().replace('\\', '/')
    body = base.replace('log_dir: logs/smoke_db_dynas_r50',
                        'log_dir: ' + log)
    path = os.path.join(log, 'smoke.yml')
    io.open(path, 'w', encoding='utf-8').write(body)
    out = subprocess.run([sys.executable, 'train.py', 'app:' + path],
                         cwd=ROOT, capture_output=True, text=True)
    check(out.returncode == 0, 'train.py ran with dynas on' + (
        '' if out.returncode == 0 else '\n' + out.stdout[-3000:]
        + out.stderr[-3000:]))
    ckpt = torch.load(os.path.join(log, 'latest_checkpoint.pt'),
                      map_location='cpu', weights_only=False)
    states = ckpt.get('dynas') or []
    filled = [len(s['state']) for s in states]
    check(sum(1 for n in filled if n) >= 2,
          'momentum in more than one group after training ({})'.format(
              filled))
    check(len(ckpt['optimizer']['state']) == 0,
          'the shared optimizer never stepped')


if __name__ == '__main__':
    unit()
    through_train()
    print('all checks passed')
