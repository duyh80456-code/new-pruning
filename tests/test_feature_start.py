"""feature_start_epoch: the feature transport terms are off until then.

Runs DC's smoke config through train.py: three epochs, the terms start at
epoch 2. The horizontal term is the one train.py logs (pair_loss on the
widest width's train line), so epoch 1's line must not carry it and
epoch 2's must. If the flag were ignored, epoch 1 would print pair_loss
and this fails; if the terms never came back, epoch 2 would not.

Needs a GPU, as train.py does: python tests/test_feature_start.py
"""
import io
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(cond, what):
    if not cond:
        raise AssertionError(what)
    print('ok   ', what)


def main():
    base = io.open(os.path.join(ROOT, 'apps', 'smoke_dc_k_late_r50_seed2.yml'),
                   encoding='utf-8').read()
    check('feature_start_epoch: 2' in base and 'num_epochs: 3' in base,
          'smoke config starts the terms at epoch 2 of 3')
    log = tempfile.mkdtemp().replace('\\', '/')
    body = base.replace('log_dir: logs/smoke_dc_k_late_r50_seed2',
                        'log_dir: ' + log)
    path = os.path.join(log, 'smoke.yml')
    io.open(path, 'w', encoding='utf-8').write(body)
    out = subprocess.run([sys.executable, 'train.py', 'app:' + path],
                         cwd=ROOT, capture_output=True, text=True)
    text = out.stdout + out.stderr
    check(out.returncode == 0, 'train.py ran' + (
        '' if out.returncode == 0 else '\n' + text[-3000:]))
    widest = {}
    for line in text.splitlines():
        found = re.search(r'\ttrain\t1\.0\t(\d+)/3:', line)
        if found:
            widest[int(found.group(1))] = line
    check(set(widest) == {1, 2}, 'two training epochs logged')
    check('pair_loss' not in widest[1],
          'epoch 1: no transport term ({})'.format(widest[1].strip()[-60:]))
    check('pair_loss' in widest[2],
          'epoch 2: the transport term is back')
    print('all checks passed')


if __name__ == '__main__':
    main()
