"""amp: the network runs in fp16 while training, and nowhere else.

  on,  training: the convs compute in fp16, the logits and features the
       losses see come back in fp32;
  on,  eval (validation, BN calibration): everything stays fp32;
  off: fp32 throughout.
Then the path that runs: K's smoke config with amp on goes through
train.py (split pair backward, Sinkhorn, BN calibration), finishes with
finite losses, and its checkpoint carries the GradScaler state, which is
only written when the flag is read as on.

Needs a GPU: python tests/test_amp.py
"""
import importlib
import io
import math
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.argv = [sys.argv[0], 'app:apps/smoke_by_k_r50.yml']

import torch  # noqa: E402

from utils.config import FLAGS  # noqa: E402
import train  # noqa: E402


def check(cond, what):
    if not cond:
        raise AssertionError(what)
    print('ok   ', what)


def unit():
    FLAGS.width_mult_list = FLAGS.width_mult_range
    FLAGS.return_features = True
    model = importlib.import_module('models.us_resnet').Model(100, 32).cuda()
    model.apply(lambda m: setattr(m, 'width_mult', 0.5))
    seen = []
    first = next(m for m in model.modules()
                 if isinstance(m, torch.nn.Conv2d))
    first.register_forward_hook(lambda m, i, o: seen.append(o.dtype))
    x = torch.randn(8, 3, 32, 32, device='cuda')

    FLAGS.amp = True
    model.train()
    logits, feats = train.run_model(model, x)
    check(seen[-1] == torch.float16, 'on, training: conv runs in fp16')
    check(logits.dtype == torch.float32
          and all(f.dtype == torch.float32 for f in feats),
          'on, training: logits and features handed back in fp32')
    model.eval()
    # eval reads the BN statistics kept for the widths in the list
    model.apply(lambda m: setattr(m, 'width_mult', 1.0))
    with torch.no_grad():
        train.run_model(model, x)
    check(seen[-1] == torch.float32, 'on, eval: conv stays fp32')

    FLAGS.amp = False
    model.train()
    train.run_model(model, x)
    check(seen[-1] == torch.float32, 'off: conv runs in fp32')


def through_train():
    base = io.open(os.path.join(ROOT, 'apps', 'smoke_df_k_amp_r50.yml'),
                   encoding='utf-8').read()
    check('amp: True' in base, 'smoke config has amp on')
    log = tempfile.mkdtemp().replace('\\', '/')
    body = base.replace('log_dir: logs/smoke_df_k_amp_r50', 'log_dir: ' + log)
    path = os.path.join(log, 'smoke.yml')
    io.open(path, 'w', encoding='utf-8').write(body)
    out = subprocess.run([sys.executable, 'train.py', 'app:' + path],
                         cwd=ROOT, capture_output=True, text=True)
    text = out.stdout + out.stderr
    check(out.returncode == 0, 'K with amp ran through train.py' + (
        '' if out.returncode == 0 else '\n' + text[-3000:]))
    losses = [float(v) for v in re.findall(r'loss: ([-0-9.eE+naif]+)', text)]
    check(losses and all(math.isfinite(v) for v in losses),
          'every logged loss finite ({} lines)'.format(len(losses)))
    ckpt = torch.load(os.path.join(log, 'latest_checkpoint.pt'),
                      map_location='cpu', weights_only=False)
    check(ckpt.get('scaler') is not None and ckpt['scaler'].get('scale'),
          'checkpoint carries the loss scale ({})'.format(
              (ckpt.get('scaler') or {}).get('scale')))


if __name__ == '__main__':
    unit()
    through_train()
    print('all checks passed')
