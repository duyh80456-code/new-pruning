"""Checks for the four published methods wired in behind flags.

Each check is built to fail if its flag is silently ignored: the stable
sampler has to hit both edges of both halves and nothing outside them,
the isolated narrowest width has to be blind to the leading channels it
used to read, the EMA update has to follow its formula exactly, and the
NASViT merge has to reproduce a hand-computed case in which the conflict
step actually fires.

Runs on the CPU in a few seconds: python tests/test_published_methods.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = [sys.argv[0], 'app:apps/smoke_a_kl.yml']

import torch  # noqa: E402

from utils.config import FLAGS  # noqa: E402

FLAGS.width_mult_list_test = [0.25, 0.3, 1.0]


def check(cond, what):
    if not cond:
        raise AssertionError(what)
    print('ok   ', what)


def stable_sampling():
    from utils.loss_ops import training_widths
    FLAGS.width_sampling = 'stable'
    FLAGS.stable_granularity = 0.05
    FLAGS.num_sample_training = 4
    random.seed(0)
    upper, lower = set(), set()
    for _ in range(4000):
        widths = training_widths(50, 0.25, 1.0)
        check_len = len(widths) == 4 and widths[:2] == [1.0, 0.25]
        if not check_len:
            raise AssertionError('stable sampling returned {}'.format(widths))
        upper.add(widths[2])
        lower.add(widths[3])
    want_upper = {round(0.05 * i, 10) for i in range(12, 20)}
    want_lower = {round(0.05 * i, 10) for i in range(6, 12)}
    check(upper == want_upper,
          'stable sampling: upper half is exactly 0.60 to 0.95 on the grid')
    check(lower == want_lower,
          'stable sampling: lower half is exactly 0.30 to 0.55 on the grid')
    FLAGS.width_sampling = 'uniform'


def isolated_activation():
    import importlib
    FLAGS.reset_parameters = True
    FLAGS.width_scalars = False
    FLAGS.return_features = False
    mod = importlib.import_module('models.us_resnet')
    from models.slimmable_ops import USBatchNorm2d, USConv2d, USLinear
    torch.manual_seed(0)
    x = torch.randn(4, 3, 32, 32)

    def run(model, width):
        model.train()
        model.apply(lambda m: setattr(m, 'width_mult', width))
        with torch.no_grad():
            return model(x)

    def scramble_leading(model):
        """overwrite the first half of every sliceable dimension"""
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, USConv2d):
                    half_out = m.weight.size(0) // 2
                    m.weight[:half_out].normal_()
                    if m.us[0]:
                        m.weight[:, :m.weight.size(1) // 2].normal_()
                elif isinstance(m, USLinear):
                    m.weight[:, :m.weight.size(1) // 2].normal_()
                elif isinstance(m, USBatchNorm2d):
                    m.weight[:m.weight.size(0) // 2].normal_()
                    m.bias[:m.bias.size(0) // 2].normal_()

    for flag in (True, False):
        FLAGS.isolate_smallest = flag
        torch.manual_seed(1)
        model = mod.Model(100, 32)
        before = {w: run(model, w) for w in (0.25, 0.3)}
        scramble_leading(model)
        after = {w: run(model, w) for w in (0.25, 0.3)}
        moved = {w: not torch.allclose(before[w], after[w]) for w in before}
        if flag:
            check(not moved[0.25],
                  'isolated: width 0.25 ignores the leading half entirely')
            check(moved[0.3],
                  'isolated: width 0.30 still reads the leading channels')
        else:
            check(moved[0.25],
                  'not isolated: width 0.25 reads the leading channels')
    FLAGS.isolate_smallest = False


def ema_update():
    import train as T
    FLAGS.ema_teacher_decay = 0.9
    torch.manual_seed(0)
    online = torch.nn.Sequential(torch.nn.Linear(3, 2),
                                 torch.nn.BatchNorm1d(2))
    T._EMA.clear()
    ema = T.ema_teacher(online)
    start = {k: v.clone() for k, v in ema.state_dict().items()}
    with torch.no_grad():
        for p in online.parameters():
            p.add_(1.0)
        online[1].running_mean.add_(2.0)
        online[1].num_batches_tracked.add_(5)
    T.update_ema_teacher(online)
    now = ema.state_dict()
    live = online.state_dict()
    exact = all(
        torch.allclose(now[k], 0.9 * start[k] + 0.1 * live[k])
        for k in now if now[k].dtype.is_floating_point)
    check(exact, 'EMA: every float entry, buffers included, is 0.9 old + 0.1 new')
    check(int(now['1.num_batches_tracked']) == 5,
          'EMA: integer buffers are copied, not averaged')
    check(all(not p.requires_grad for p in ema.parameters()),
          'EMA: the teacher takes no gradient')
    T._EMA.clear()
    FLAGS.ema_teacher_decay = 0.0


def nasvit_merge():
    import train as T
    FLAGS.conflict_from_epoch = 25
    p = torch.nn.Parameter(torch.zeros(2))
    q = torch.nn.Parameter(torch.zeros(2))

    class Two(torch.nn.Module):
        def __init__(self):
            super(Two, self).__init__()
            self.p, self.q = p, q
    model = Two()

    def merged(epoch, largest_p, small_p, largest_q, small_q):
        merge = T.GradientConflict()
        p.grad, q.grad = largest_p.clone(), largest_q.clone()
        merge.hold(model)
        p.grad, q.grad = small_p.clone(), small_q.clone()
        merge.merge(model, epoch)
        return p.grad.clone(), q.grad.clone()

    gl_p, gs_p = torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0])
    gl_q, gs_q = torch.tensor([1.0, 1.0]), torch.tensor([1.0, 0.0])
    # before the switch: the plain sum, whatever the angle
    out_p, out_q = merged(10, gl_p, gs_p, gl_q, gs_q)
    check(torch.allclose(out_p, gs_p + gl_p) and torch.allclose(out_q, gs_q + gl_q),
          'NASViT: before conflict_from_epoch the gradient is the plain sum')
    # after: p conflicts, <gs, gl> = -1, |gs|^2 = 2, step = 0.5
    out_p, out_q = merged(30, gl_p, gs_p, gl_q, gs_q)
    check(torch.allclose(out_p, gs_p * 1.5 + gl_p),
          'NASViT: a conflicting subnet gradient is scaled by 1 + 0.5 here')
    # q agrees, <gs, gl> = 1 > 0, clamp gives 0
    check(torch.allclose(out_q, gs_q + gl_q),
          'NASViT: an agreeing subnet gradient is left alone')
    # the clamp at 2: <gs, gl> = -10, |gs|^2 = 1, raw step 10
    out_p, _ = merged(30, torch.tensor([10.0, 0.0]), torch.tensor([-1.0, 0.0]),
                      gl_q, gs_q)
    check(torch.allclose(out_p, torch.tensor([-1.0, 0.0]) * 3.0
                         + torch.tensor([10.0, 0.0])),
          'NASViT: the step is clamped at 2')
    FLAGS.conflict_from_epoch = None


if __name__ == '__main__':
    stable_sampling()
    isolated_activation()
    ema_update()
    nasvit_merge()
    print('all checks passed')
