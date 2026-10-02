"""WKD-L (Lv et al., NeurIPS 2024) as the inplace distillation loss.

  1. utils/loss_ops.WKDLogitLoss against the authors' own
     wkd_logit_loss_with_speration, on the same logits, labels, cost and
     gamma. Their repository carries no license, so it is not vendored
     here: point WKD_REFERENCE at a clone of github.com/JiamingLv/WKD and
     this part runs; without one it is skipped and says so.
  2. the schedule: gamma constant, then cosine to zero from 62.5% of
     training, as their 150-of-240.
  3. the path that runs: DG's smoke config through train.py. The loss
     reads the teacher logits and the cost that train.py sets each step
     and fails on None, so a clean run is the wiring check; it must also
     not reproduce A's smoke numbers.

python tests/test_wkd.py   (part 3 needs a GPU)
"""
import io
import math
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.argv = [sys.argv[0], 'app:apps/smoke_dg_wkd_r50.yml']

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from utils.config import FLAGS  # noqa: E402
from utils.loss_ops import WKDLogitLoss, wkd_gamma  # noqa: E402

REFERENCE = os.environ.get('WKD_REFERENCE', '')


def check(cond, what):
    if not cond:
        raise AssertionError(what)
    print('ok   ', what)


def reference_functions():
    path = os.path.join(REFERENCE, 'mdistiller', 'distillers', 'WKD.py')
    if not REFERENCE or not os.path.exists(path):
        return None
    text = io.open(path, encoding='utf-8').read()
    # the three loss functions only: everything from the feature helpers
    # on needs the rest of their package
    text = text[:text.index('def adaptive_avg_std_pool2d')]
    text = '\n'.join(line for line in text.splitlines()
                     if not line.startswith('from .'))
    space = {}
    exec(compile(text, path, 'exec'), space)
    return space


def against_reference():
    ref = reference_functions()
    if ref is None:
        print('skip  WKD_REFERENCE not set: no comparison with the authors')
        return
    torch.manual_seed(0)
    n, c = 32, 100
    student = torch.randn(n, c) * 3
    teacher = torch.randn(n, c) * 3
    label = torch.randint(0, c, (n,))
    weight = torch.randn(c, 64)

    ours = WKDLogitLoss(temperature=8.0, reg=0.05, n_iters=10)
    ours.set_cost(weight)
    ours.set_teacher(teacher, label)
    ours.set_gamma(600.0)
    mine = ours(student).mean()

    normed = F.normalize(weight, p=2, dim=-1)
    dist = 1 - torch.exp(-(1 - normed.matmul(normed.t())))
    # their wkd_logit_loss applies relu and the 1e-8 floor itself
    theirs = ref['wkd_logit_loss_with_speration'](
        student, teacher, label, 8.0, 600.0, dist, 0.05, 10)
    check(torch.allclose(mine, theirs, rtol=1e-5, atol=1e-5),
          'equals the authors\' loss ({:.6f} vs {:.6f})'.format(
              float(mine), float(theirs)))

    # and the gradient reaches the student
    student.requires_grad_(True)
    ours(student).mean().backward()
    check(float(student.grad.abs().sum()) > 0, 'gradient reaches the student')


def schedule():
    check(wkd_gamma(1) == FLAGS.wkd_weight, 'gamma at the start is the weight')
    start = int(0.625 * FLAGS.num_epochs)
    check(wkd_gamma(start) == FLAGS.wkd_weight, 'constant up to 62.5%')
    check(abs(wkd_gamma(FLAGS.num_epochs)) < 1e-9, 'zero at the end')


def through_train():
    if not torch.cuda.is_available():
        print('skip  no GPU: train.py path not exercised')
        return
    outs = {}
    for name in ('dg_wkd_r50', 'bx_a_r50'):
        base = io.open(os.path.join(ROOT, 'apps', 'smoke_{}.yml'.format(name)),
                       encoding='utf-8').read()
        log = tempfile.mkdtemp().replace('\\', '/')
        body = re.sub(r'^log_dir: .*$', 'log_dir: ' + log, base, flags=re.M)
        path = os.path.join(log, 'smoke.yml')
        io.open(path, 'w', encoding='utf-8').write(body)
        out = subprocess.run([sys.executable, 'train.py', 'app:' + path],
                             cwd=ROOT, capture_output=True, text=True)
        text = out.stdout + out.stderr
        check(out.returncode == 0, '{} ran through train.py'.format(name) + (
            '' if out.returncode == 0 else '\n' + text[-3000:]))
        found = re.search(r'\ttrain\t0\.25\t1/\d+: loss: ([0-9.]+)', text)
        outs[name] = float(found.group(1))
        losses = [float(v) for v in re.findall(r'loss: ([-0-9.eE+naif]+)',
                                               text)]
        check(all(math.isfinite(v) for v in losses),
              '{}: every logged loss finite'.format(name))
    check(outs['dg_wkd_r50'] != outs['bx_a_r50'],
          'the narrow width trains on a different loss than A '
          '({:.4f} vs {:.4f})'.format(outs['dg_wkd_r50'], outs['bx_a_r50']))


if __name__ == '__main__':
    against_reference()
    schedule()
    through_train()
    print('all checks passed')
