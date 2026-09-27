"""Does split_pair_backward give the gradient the deferred path gives?

Runs train.py's own run_one_epoch on one fixed batch with the same widths
drawn every time, and records the gradient at optimizer.step() instead of
taking the step. Everything that made end-to-end runs disagree with
themselves - data order, augmentation, the trajectory - is outside this.

Needs a CUDA card, which is why it is not among the CPU suites the
notebooks run. Measured when split_pair_backward went in: relative gap
4.6e-8 on ResNet-18 and 3.8e-8 on ResNet-50 at batch 256, against 3e-1
and 9e-2 for dropping the pair term entirely, and 2.2e-1 for a split that
charged only one side of each pair - so a real mistake shows six orders
of magnitude above the float noise.

usage: python tests/gpu_split_pair.py <config> [batch]
"""
import os
import random
import runpy
import sys

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path.insert(0, os.getcwd())
config = sys.argv[1]
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 32
sys.argv = ['train.py', 'app:' + config]

import torch  # noqa: E402
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

T = runpy.run_path('train.py', run_name='harness')
FLAGS = T['FLAGS']
# set by the data loader, which this harness replaces with one batch
FLAGS.data_size_train = 50000

T['set_random_seed']()
model, wrapper = T['get_model']()
criterion = torch.nn.CrossEntropyLoss(reduction='none')
soft = T['build_soft_criterion']()
pair = T['build_pair_criterion']()
fvert = T['build_feature_criterion']()
fpair = T['build_feature_pair_criterion']()
FLAGS.return_features = fvert is not None or fpair is not None
start = {k: v.clone() for k, v in wrapper.state_dict().items()}

g = torch.Generator().manual_seed(7)
loader = [(torch.randn(batch, 3, 32, 32, generator=g),
           torch.randint(0, 100, (batch,), generator=g))]


class Recorder(object):
    """an optimizer that remembers the gradient instead of stepping"""
    param_groups = [{'lr': 0.1}]

    def __init__(self):
        self.grads = None

    def zero_grad(self, set_to_none=True):
        wrapper.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.grads = {n: p.grad.detach().clone()
                      for n, p in wrapper.named_parameters()
                      if p.grad is not None}


def gradient(split, weight=1.0):
    wrapper.load_state_dict(start)
    FLAGS.split_pair_backward = split
    FLAGS.horizontal_weight = weight
    random.seed(11)
    torch.manual_seed(11)
    rec = Recorder()
    T['run_one_epoch'](
        10, loader, wrapper, criterion, rec, T['get_meters']('train'),
        phase='train', soft_criterion=soft, pair_criterion=pair,
        feature_criterion=fvert, feature_pair_criterion=fpair)
    torch.cuda.synchronize()
    return rec.grads


def gap(a, b):
    num = sum((a[k] - b[k]).pow(2).sum() for k in a).sqrt().item()
    den = sum(a[k].pow(2).sum() for k in a).sqrt().item()
    return num / den


torch.cuda.reset_peak_memory_stats()
old = gradient(False)
peak_old = torch.cuda.max_memory_allocated() / 2 ** 30
again = gradient(False)
torch.cuda.reset_peak_memory_stats()
new = gradient(True)
peak_new = torch.cuda.max_memory_allocated() / 2 ** 30
none = gradient(False, weight=0.0)

print('config', os.path.basename(config), '| depth',
      getattr(FLAGS, 'depth', 18), '| batch', batch,
      '| params with grad', len(old))
print('relative L2 gap, deferred vs deferred again : %.2e' % gap(old, again))
print('relative L2 gap, deferred vs split          : %.2e' % gap(old, new))
print('relative L2 gap, deferred vs no pair term   : %.2e' % gap(old, none))
print('peak memory  deferred %.2f GB   split %.2f GB' % (peak_old, peak_new))
