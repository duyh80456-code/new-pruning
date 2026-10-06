"""dead channels at every ReLU of a trained checkpoint, per width

    python scripts/dead_layers.py <checkpoint> <config> <depth> <out.json>

e.g. python scripts/dead_layers.py latest_checkpoint.zip \
         apps/cifar100_cr_k_r50_seed2.yml 50 results/dead_layers_cr.json

scripts/dead_features.py reads the last block only, where the feature
transport acts. This reads every ReLU in the network, so a collapse can
be placed: confined to the last block, where the transport terms pull,
or spread through the trunk, as a general dying-ReLU would be.

A channel is dead at a ReLU when its input is <= 0 at every position of
every test image: it passes nothing forward and, through the ReLU, takes
no gradient back. For every test width, after BN recalibration on
bn_cal_batch_num train batches (the protocol):

  acc       top-1 on the CIFAR-100 test set
  layers    one entry per ReLU, in forward order: its name, the channels
            it has at this width, and the fraction of them dead

At width 1.0 each layer's dead channels are also split by where they sit:
inside the prefix width 0.25 uses, inside the prefix width 0.60 uses, and
in the rest, which only the wider widths have. A healthy ReLU network has
some dead channels anyway, so these numbers mean something only beside
the same measurement on a run that did not collapse.
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
from models.slimmable_ops import bn_calibration_init  # noqa: E402

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
relus = [(name, m) for name, m in net.named_modules()
         if isinstance(m, torch.nn.ReLU)]

# per ReLU, the largest input each channel reaches over the test set; a
# pre-hook sees the input before an inplace ReLU overwrites it
peak = {}


def watch(name):
    def hook(module, inputs):
        x = inputs[0].detach()
        top = x.transpose(0, 1).reshape(x.size(1), -1).amax(dim=1)
        peak[name] = top if name not in peak else torch.maximum(
            peak[name], top)
    return hook


order = []


def first_call(name):
    def hook(module, inputs):
        if name not in order:
            order.append(name)
    return hook


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
def measure(w):
    set_width(w)
    peak.clear()
    handles = [m.register_forward_pre_hook(watch(name))
               for name, m in relus]
    correct, n = 0, 0
    for x, y in val_loader:
        out = wrapper(x.cuda())
        correct += (out.argmax(1) == y.cuda()).sum().item()
        n += y.numel()
    for h in handles:
        h.remove()
    return 100.0 * correct / n, {k: (v <= 0) for k, v in peak.items()}


# forward order of the ReLUs, read off one call at full width
handles = [m.register_forward_pre_hook(first_call(name)) for name, m in relus]
set_width(1.0)
with torch.no_grad():
    wrapper.eval()
    wrapper(torch.zeros(2, 3, 32, 32).cuda())
for h in handles:
    h.remove()

rows = {}
dead_at = {}
for w in sorted(tests, reverse=True):
    calibrate(w)
    acc, dead = measure(w)
    dead_at[w] = dead
    rows[str(w)] = {
        'acc': acc,
        'layers': [{'name': name, 'channels': int(dead[name].numel()),
                    'dead': dead[name].float().mean().item()}
                   for name in order],
    }
    worst = max(rows[str(w)]['layers'], key=lambda r: r['dead'])
    print('{:.2f}  acc {:6.2f}  mean dead {:.3f}  worst {} {:.3f}'.format(
        w, acc, sum(r['dead'] for r in rows[str(w)]['layers']) / len(order),
        worst['name'], worst['dead']), flush=True)

# where width 1.0's dead channels sit, against the prefixes narrow widths use
split = []
for name in order:
    full = dead_at[1.0][name]
    entry = {'name': name, 'channels': int(full.numel())}
    for w in (0.25, 0.6):
        c = int(dead_at[w][name].numel())
        entry['prefix_{}'.format(w)] = {
            'channels': c, 'dead': full[:c].float().mean().item()}
    c60 = int(dead_at[0.6][name].numel())
    rest = full[c60:]
    entry['rest'] = {'channels': int(rest.numel()),
                     'dead': rest.float().mean().item() if rest.numel()
                     else None}
    split.append(entry)

with open(out_path, 'w') as handle:
    json.dump({'checkpoint': os.path.basename(ckpt_path), 'config': config,
               'relu_order': order, 'widths': rows, 'full_width_split': split},
              handle, indent=1)
print('wrote', out_path)
