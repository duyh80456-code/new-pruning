"""sam_rho: Sharpness-Aware Minimization around the whole sandwich step.

On the CPU: sam_ascend moves the weights by exactly rho along the unit
gradient and sam_descend puts them back; on a quadratic the second-pass
gradient is the gradient at the perturbed point, which is what SAM steps
with.

With --gpu, through train.py itself on DD's smoke config: sam_rho 0 has to
give the same training losses as before the step was wrapped (the stored
reference is the unwrapped run), and sam_rho > 0 has to draw the widths
once per step, not once per pass, which is checked by counting calls to
training_widths.

    python tests/test_sam.py          CPU checks
    python tests/test_sam.py --gpu    plus the train.py checks
"""
import io
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def check(cond, what):
    if not cond:
        raise AssertionError(what)
    print('ok   ', what)


def cpu_checks():
    sys.argv = [sys.argv[0], 'app:apps/smoke_a_kl.yml']
    import torch
    import train as T
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(10))
    v = torch.nn.Parameter(torch.randn(3, 4))
    model = torch.nn.Module()
    model.w, model.v = w, v
    a = torch.randn(10)
    loss = lambda: ((w - a) ** 2).sum() + (v ** 4).sum()  # noqa: E731
    loss().backward()
    before = [w.detach().clone(), v.detach().clone()]
    grads = [w.grad.clone(), v.grad.clone()]
    moved = T.sam_ascend(model, 0.05)
    shift = torch.sqrt(sum(((p.detach() - b) ** 2).sum()
                           for p, b in zip((w, v), before)))
    check(abs(shift.item() - 0.05) < 1e-6,
          'ascent moves the weights by rho ({:.6f})'.format(shift.item()))
    norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
    check(all(torch.allclose(p.detach() - b, 0.05 * g / norm, atol=1e-7)
              for p, b, g in zip((w, v), before, grads)),
          'along the unit gradient')
    perturbed = [w.detach().clone(), v.detach().clone()]
    w.grad = None
    v.grad = None
    loss().backward()
    expect_w = 2 * (perturbed[0] - a)
    expect_v = 4 * perturbed[1] ** 3
    T.sam_descend(moved)
    check(torch.allclose(w.detach(), before[0], atol=1e-7)
          and torch.allclose(v.detach(), before[1], atol=1e-7),
          'descent puts the weights back')
    check(torch.allclose(w.grad, expect_w) and torch.allclose(v.grad,
                                                              expect_v),
          'the gradient left for the step is the one at the perturbed point')


def run_train(config, rho):
    """train.py on config with sam_rho, counting training_widths calls"""
    import subprocess
    log = tempfile.mkdtemp().replace('\\', '/')
    body = io.open(config, encoding='utf-8').read()
    body = re.sub(r'log_dir: \S+', 'log_dir: ' + log, body)
    body += '\nsam_rho: {}\n'.format(rho)
    path = os.path.join(log, 'smoke.yml')
    io.open(path, 'w', encoding='utf-8').write(body)
    driver = '''
import sys
sys.path.insert(0, {root!r})
sys.argv = ['train.py', 'app:{path}']
import train, utils.loss_ops as L
calls = [0]
real = L.training_widths
def counted(*a, **k):
    calls[0] += 1
    return real(*a, **k)
train.training_widths = counted
train.main()
print('TRAINING_WIDTHS_CALLS', calls[0])
'''.format(root=ROOT, path=path)
    out = subprocess.run([sys.executable, '-c', driver], cwd=ROOT,
                         capture_output=True, text=True)
    text = out.stdout + out.stderr
    if out.returncode != 0:
        raise AssertionError(text[-3000:])
    losses = re.findall(r'\ttrain\t1\.0\t\d+/\d+: loss: ([0-9.]+)', text)
    calls = int(re.search(r'TRAINING_WIDTHS_CALLS (\d+)', text).group(1))
    return losses, calls


def gpu_checks():
    config = os.path.join(ROOT, 'apps', 'smoke_dd_k_late_r50.yml')
    plain, calls_plain = run_train(config, 0.0)
    sam, calls_sam = run_train(config, 0.05)
    check(calls_sam == calls_plain and calls_plain > 0,
          'sam draws the widths once per step ({} calls, {} without)'.format(
              calls_sam, calls_plain))
    check(plain != sam, 'sam_rho changes training ({} vs {})'.format(
        plain, sam))
    # the unwrapped train.py, same config, same machine
    reference = os.environ.get('SAM_REFERENCE')
    if reference:
        check(plain == reference.split(','),
              'sam_rho 0 trains as the unwrapped step did ({})'.format(
                  plain))


if __name__ == '__main__':
    on_gpu = '--gpu' in sys.argv  # cpu_checks rewrites sys.argv
    cpu_checks()
    if on_gpu:
        gpu_checks()
    print('all checks passed')
