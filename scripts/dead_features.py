"""dead-feature statistics per width on a trained checkpoint

    python scripts/dead_features.py <checkpoint> <config> <depth> <out.json>

e.g. python scripts/dead_features.py latest_checkpoint.zip \
         apps/cifar100_cr_k_r50_seed2.yml 50 dead_cr.json

Measured on CR (K at seed 2026, collapsed, epoch 79): every width up to
0.60 has no positive pre-activation at the last block and sits at chance,
and 1091 of width 1.0's 1092 dead channels lie in its first 1232, the
prefix the narrow widths share. That is the all-zero fixed point the
feature transport cannot leave (see models/us_resnet.py and
feature_start_epoch in train.py).

Builds the model from the config, loads the weights (a checkpoint dict
with 'model', or a bare state dict), then for every test width: BN
recalibration on bn_cal_batch_num train batches (the protocol), and on
the CIFAR-100 test set:
  acc        top-1
  dead       fraction of channels whose pooled post-ReLU feature is 0 on
             every test image
  zero_img   fraction of images whose whole pooled feature vector is 0
  norm       mean L2 norm of the pooled feature
  spread     mean pairwise distance inside a batch (the s of the cost)
  pos_pre    fraction of positive entries of the last block's pre-ReLU map
  beta_dead / beta_live  mean BN shift (last BN of the last block) on the
             dead and on the live channels
  vs_full_dead  fraction of the widest net's first d channels that are
             dead too (the prefix the vertical term compares against)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ckpt_path, config, depth, out_path = (
    os.path.abspath(sys.argv[1]), sys.argv[2], int(sys.argv[3]),
    os.path.abspath(sys.argv[4]))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.argv = ['train.py', 'app:' + config]

import torch  # noqa: E402
import train as T  # noqa: E402
from utils.config import FLAGS  # noqa: E402
from models.slimmable_ops import bn_calibration_init, make_divisible  # noqa: E402

FLAGS.depth = depth
FLAGS.num_gpus_per_job = 1
FLAGS.data_loader_workers = 0
tests = FLAGS.width_mult_list_test
FLAGS.width_mult_list = FLAGS.width_mult_range.copy() + [
    w for w in tests if w not in FLAGS.width_mult_range]
T.set_random_seed(0)

model, wrapper = T.get_model()
state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state = state['model'] if 'model' in state else state
missing, unexpected = wrapper.load_state_dict(state, strict=False)
assert not unexpected, unexpected
# only the BN slots of widths the checkpoint never stored may be missing;
# calibration fills them
assert all('.bn.' in k for k in missing), missing[:5]

train_t, val_t, test_t = T.data_transforms()
train_set, val_set, test_set = T.dataset(train_t, val_t, test_t)
train_loader, val_loader, _ = T.data_loader(train_set, val_set, test_set)

net = wrapper.module
blocks = net.features
last = len(blocks) - 2  # the last residual block; -1 is the pool


def set_width(w):
    wrapper.apply(lambda m: setattr(m, 'width_mult', w))


@torch.no_grad()
def calibrate(w):
    wrapper.eval()
    wrapper.apply(bn_calibration_init)
    set_width(w)
    for i, (x, _) in enumerate(train_loader):
        if i == FLAGS.bn_cal_batch_num:
            break
        wrapper(x.cuda())
    wrapper.eval()


@torch.no_grad()
def features(w):
    """pooled post-ReLU features, pre-ReLU positive fraction, correct"""
    set_width(w)
    feats, correct, pos, total = [], 0, 0, 0
    for x, y in val_loader:
        x, y = x.cuda(), y.cuda()
        h = x
        for i in range(last):
            h = blocks[i](h)
        pre = blocks[last].preact(h)
        pos += (pre > 0).sum().item()
        total += pre.numel()
        f = torch.relu(pre).mean(dim=(2, 3))
        logits = net.head()(f)
        correct += (logits.argmax(1) == y).sum().item()
        feats.append(f)
    return torch.cat(feats), correct, pos / total


def last_bn_shift(w):
    bn = blocks[last].body[-1]
    d = make_divisible(bn.num_features_max * w / bn.ratio) * bn.ratio
    return bn.bias.detach()[:d]


rows = {}
full = None
for w in sorted(tests, reverse=True):
    calibrate(w)
    f, correct, pos_pre = features(w)
    n, d = f.shape
    dead = (f.abs().sum(0) == 0)
    beta = last_bn_shift(w).to(f.device)
    sample = f[:512]
    row = {
        'channels': d,
        'acc': 100.0 * correct / n,
        'dead': dead.float().mean().item(),
        'zero_img': (f.abs().sum(1) == 0).float().mean().item(),
        'norm': f.norm(dim=1).mean().item(),
        'spread': torch.cdist(sample, sample).mean().item(),
        'pos_pre': pos_pre,
        'beta_dead': beta[dead].mean().item() if dead.any() else None,
        'beta_live': beta[~dead].mean().item() if (~dead).any() else None,
    }
    if full is None:
        full = dead
    else:
        row['vs_full_dead'] = full[:d].float().mean().item()
    rows[str(w)] = row
    print(w, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in row.items()}), flush=True)

with open(out_path, 'w') as handle:
    json.dump(rows, handle, indent=1)
