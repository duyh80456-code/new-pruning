"""Kaggle notebooks for published methods run from their authors' code.

Each notebook clones the authors' repository at a pinned commit, applies a
patch from third_party/patches/ (CIFAR-100, a CIFAR ResNet-50, our training
recipe, resume, and evaluation at the sixteen widths printed in our line
format), and runs two branches, one per T4. The patch and its README list
every change; nothing else of the authors' code is touched.

    python scripts/build_external_notebooks.py
"""
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAGGLE = os.path.join(ROOT, 'kaggle')

REPO_URL = 'https://github.com/duyh80456-code/new-pruning.git'
REPO_BRANCH = 'nhan'

# BX, A on ResNet-50 at seed 1995, the reference every table is read against
A_REF = {0.25: 75.26, 0.30: 75.35, 0.35: 75.54, 0.40: 76.01, 0.45: 76.30,
         0.50: 76.77, 0.55: 76.63, 0.60: 77.03, 0.65: 77.13, 0.70: 77.20,
         0.75: 77.39, 0.80: 77.56, 0.85: 77.60, 0.90: 77.99, 0.95: 78.01,
         1.00: 78.04}

JOSLIM = {
    'number': 39,
    'name': 'kaggle_kd_39_joslim',
    'title': 'Joslim from the authors\' code: Joslim and its US-Net mode',
    'upstream': 'https://github.com/enyac-group/Joslim.git',
    'commit': '9b743d9b50ade9fc2cd33d255dfbb936e4894733',
    'dir': 'Joslim',
    'patch': 'joslim.patch',
    'pip': 'botorch==0.16.1 gpytorch==1.15.2 linear_operator==0.6.1',
    'intro': '''Joslim (Chin et al., ECML-PKDD 2021) trains a slimmable network and,
during training, searches per-layer width configurations with Bayesian
optimisation instead of shrinking every layer by the same factor. Run here
from the authors' repository (enyac-group/Joslim at 9b743d9) with
`third_party/patches/joslim.patch` applied; `third_party/patches/joslim_README.md`
lists every change.

| | GPU | what |
|---|---|---|
| joslim_r50 | 0 | Joslim, 200 visited architectures (tau 195) |
| slim_r50 | 1 | the same code in its US-Net mode (`--slim --slim_uniform`): a cross-check of our BX in an independent codebase |

Both: CIFAR-100, CIFAR ResNet-50 (3x3 stem, no max-pool), 100 epochs, batch
256, SGD nesterov lr 0.2 cosine with 5 warm-up epochs, wd 5e-4, widths
0.25-1.00. Read at the sixteen uniform widths after BN recalibration on 20
batches, as every other run. Joslim's own design point is per-layer widths,
so its Pareto set (accuracy against FLOPs) is printed as well.

One change of substance: upstream scored every sampled architecture on test
batches, and that score drives the search. The patch scores them on a fixed
5,000-image subset of the training set; the test set is read only at the end.

## More than one session

On a T4 the US-Net mode takes about 10-13 hours and Joslim longer, since its
Bayesian optimisation runs on the CPU. Each run checkpoints every epoch.
When the session ends, Save Version, then in a new run attach that version's
output (pick the version by number, not Latest) and run again: the notebook
finds the checkpoints and both runs continue. Attach the CIFAR-100 dataset
as well.
''',
    'branches': [
        ('joslim_r50', '0', '--tau 195 --prior_points 20'),
        ('slim_r50', '1', '--slim --slim_uniform --tau 1'),
    ],
}

LCS = {
    'number': 40,
    'name': 'kaggle_kd_40_lcs',
    'title': 'LCS from the authors\' code: published norm and BatchNorm',
    'upstream': 'https://github.com/apple/learning-compressible-subspaces.git',
    'commit': 'e6d3924368faccbdfd3d89c4a4735dba947275c9',
    'dir': 'learning-compressible-subspaces',
    'patch': 'lcs.patch',
    'pip': 'pyyaml',
    'intro': '''LCS (Nunez et al., WACV 2023) learns a line in weight space whose points
are networks of different widths: width 0.25 at one end, 1.00 at the other,
and every width between them by interpolation. Run here from the authors'
repository (apple/learning-compressible-subspaces at e6d3924) with
`third_party/patches/lcs.patch` applied; `third_party/patches/lcs_README.md`
lists every change. In structured mode only the norm layers are lines, the
convolutions are shared, so the stored model is 23.76M parameters, 0.2% more
than one ResNet-50.

| | GPU | what |
|---|---|---|
| lcs_l_bn | 1 | LCS with BatchNorm, recalibrated at each width on 20 batches: our protocol |
| lcs_l_in | 0 | LCS with its published instance norm, which needs no recalibration |

Both: CIFAR-100, CIFAR ResNet-50 (3x3 stem, no max-pool), method `lcs_l`,
100 epochs, batch 256, SGD nesterov lr 0.2 with the authors' per-epoch
warm-up and cosine, wd 5e-4, widths 0.25-1.00, read at the sixteen widths.

## More than one session

On a T4 the BatchNorm run takes about 9-13 hours and the instance-norm run
about twice that. Each run checkpoints every epoch. When the session ends,
Save Version, then in a new run attach that version's output (pick the
version by number, not Latest) and run again: the notebook finds the
checkpoints and both runs continue. Attach the CIFAR-100 dataset as well.
''',
    'branches': [
        ('lcs_l_in', '0', ''),
        ('lcs_l_bn', '1', '--norm BN --recal_batches 20'),
    ],
}


def source(text):
    lines = text.strip('\n').split('\n')
    return [line + '\n' for line in lines[:-1]] + [lines[-1]]


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': source(text)}


def markdown(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': source(text)}


SETUP = '''import os, queue, re, shutil, subprocess, sys, threading, time
import torch

n_gpu = torch.cuda.device_count()
print('torch', torch.__version__, '| gpus', n_gpu)
if n_gpu == 0:
    raise SystemExit('No GPU. Set Accelerator to GPU T4 x2 and run again.')

WORK = '/kaggle/working'
OURS = os.path.join(WORK, 'new-pruning')
if not os.path.isdir(OURS):
    subprocess.run(['git', 'clone', '-b', REPO_BRANCH, REPO_URL, OURS],
                   check=True)
print('our code at', subprocess.run(
    ['git', '-C', OURS, 'log', '-1', '--format=%h %s'],
    capture_output=True, text=True).stdout.strip())

CODE = os.path.join(WORK, UPSTREAM_DIR)
if not os.path.isdir(CODE):
    subprocess.run(['git', 'clone', UPSTREAM, CODE], check=True)
    subprocess.run(['git', '-C', CODE, 'checkout', '-q', COMMIT], check=True)
    subprocess.run(['git', '-C', CODE, 'apply',
                    os.path.join(OURS, 'third_party', 'patches', PATCH)],
                   check=True)
print('upstream at', subprocess.run(
    ['git', '-C', CODE, 'log', '-1', '--format=%h %ad', '--date=short'],
    capture_output=True, text=True).stdout.strip(), '+', PATCH)

# install without letting pip replace Kaggle's torch
base = torch.__version__.split('+')[0]
with open(os.path.join(WORK, 'constraints.txt'), 'w') as handle:
    handle.write('torch=={}\\n'.format(base))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-c',
                os.path.join(WORK, 'constraints.txt')] + PIP.split(),
               check=True)
'''

DATA = '''# the folder that contains cifar-100-python/, or a place to download it
DATA = None
for root, dirs, _ in os.walk('/kaggle/input', followlinks=True):
    if 'cifar-100-python' in dirs:
        DATA = root
        break
if DATA is None:
    DATA = os.path.join(WORK, 'data')
    os.makedirs(DATA, exist_ok=True)
    print('no CIFAR-100 attached: it will be downloaded into', DATA)
else:
    print('CIFAR-100 from', DATA)
'''

RUN = '''VAL_LINE = re.compile(
    r'val\\s+([0-9.]+)\\s+-1/\\d+:\\s+loss:\\s+([0-9.eE+-]+),\\s+'
    r'top1_error:\\s+([0-9.]+)')
results = {}


def run_pinned(jobs):
    """one command per card, output tagged and interleaved"""
    lines = queue.Queue()
    procs = {}

    def pump(label, proc):
        for line in proc.stdout:
            lines.put((label, line.rstrip('\\n')))
        proc.wait()
        lines.put((label, None))

    for label, gpu, command in jobs:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, **WORKER_ENV)
        proc = subprocess.Popen(command, cwd=CODE, env=env, text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        procs[label] = proc
        threading.Thread(target=pump, args=(label, proc),
                         daemon=True).start()
        print('[{}] gpu {}: {}'.format(label, gpu, ' '.join(command)),
              flush=True)
    started, remaining = time.time(), len(jobs)
    while remaining:
        label, line = lines.get()
        if line is None:
            remaining -= 1
            print('[{}] exit code {} after {:.0f} min'.format(
                label, procs[label].returncode,
                (time.time() - started) / 60), flush=True)
            continue
        found = VAL_LINE.search(line)
        if found:
            width, loss, top1 = found.groups()
            results.setdefault(label, {})[round(float(width), 2)] = (
                float(loss), float(top1))
        print('[{}] {}'.format(label, line), flush=True)
    return {label: proc.returncode for label, proc in procs.items()}
'''

TABLE = '''A_REF = {a_ref}
labels = [b[0] for b in BRANCHES]
print('{{:>7}}{{:>9}}'.format('width', 'A') + ''.join(
    '{{:>12}}{{:>8}}'.format(b[:11], 'vs A') for b in labels))
for width in sorted(A_REF):
    row = '{{:>7.2f}}{{:>9.2f}}'.format(width, A_REF[width])
    for label in labels:
        entry = results.get(label, {{}}).get(width)
        if entry is None:
            row += '{{:>12}}{{:>8}}'.format('-', '-')
            continue
        accuracy = 100.0 * (1.0 - entry[1])
        row += '{{:>12.2f}}{{:>+8.2f}}'.format(accuracy, accuracy - A_REF[width])
    print(row)
reference = sum(A_REF.values()) / len(A_REF)
print('\\n{{:24}} mean {{:.2f}}'.format('A (BX)', reference))
for label in labels:
    table = results.get(label, {{}})
    if len(table) == len(A_REF):
        mean = sum(100.0 * (1.0 - v[1]) for v in table.values()) / len(table)
        print('{{:24}} mean {{:.2f}}   vs A {{:+.2f}}'.format(
            label, mean, mean - reference))
    else:
        print('{{:24}} {{}} of 16 widths read'.format(label, len(table)))
'''

JOSLIM_RESUME = '''CK = os.path.join(WORK, 'ckpt')
os.makedirs(CK, exist_ok=True)
for label, _, _ in BRANCHES:
    target = os.path.join(CK, label + '.pt')
    if os.path.exists(target):
        continue
    for root, _, files in os.walk('/kaggle/input', followlinks=True):
        if label + '.pt' in files:
            shutil.copy(os.path.join(root, label + '.pt'), target)
            print('restored', label, 'from', root)
            break
    print(label, 'resumes from its checkpoint' if os.path.exists(target)
          else 'starts at epoch 1')
'''

JOSLIM_TRAIN = '''COMMON = ['--dataset', 'CIFAR100', '--datapath', DATA,
          '--network', 'slim_resnet50_cifar', '--epochs', '100',
          '--warmup', '5', '--baselr', '0.2', '--scheduler', 'cosine_decay',
          '--batch_size', '256', '--wd', '5e-4', '--mmt', '0.9',
          '--nesterov', '--label_smoothing', '0', '--lower_channel', '0.25',
          '--num_sampled_arch', '2', '--baseline', '-3',
          '--print_freq', '100', '--ckpt_dir', CK]
codes = run_pinned([
    (label, gpu, [sys.executable, '-u', 'joslim.py', '--name', label]
     + COMMON + extra.split())
    for label, gpu, extra in BRANCHES])
print(codes)
'''

JOSLIM_EVAL = '''# the sixteen uniform widths for both runs, then Joslim's Pareto set
EVAL = ['--dataset', 'CIFAR100', '--datapath', DATA,
        '--network', 'slim_resnet50_cifar', '--batch_size', '256',
        '--lower_channel', '0.25', '--ckpt_dir', CK]
run_pinned([(label, gpu, [sys.executable, '-u', 'eval_checkpoints.py',
                          '--name', label] + EVAL + ['--uniform', '--tag', label])
            for label, gpu, _ in BRANCHES])
run_pinned([('joslim_r50_pareto', '0',
             [sys.executable, '-u', 'eval_checkpoints.py',
              '--name', 'joslim_r50'] + EVAL)])
'''

LCS_RESUME = '''CK = os.path.join(WORK, 'ckpt')
for label, _, _ in BRANCHES:
    target = os.path.join(CK, label, 'last.pt')
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.exists(target):
        continue
    for root, _, files in os.walk('/kaggle/input', followlinks=True):
        if 'last.pt' in files and os.path.basename(root) == label:
            shutil.copy(os.path.join(root, 'last.pt'), target)
            print('restored', label, 'from', root)
            break
    print(label, 'resumes from its checkpoint' if os.path.exists(target)
          else 'starts at epoch 1')
'''

LCS_TRAIN = '''WIDTHS = ','.join('{:.2f}'.format(0.25 + 0.05 * i) for i in range(16))
COMMON = ['--model', 'cresnet50', '--dataset', 'cifar100', '--method',
          'lcs_l', '--data_dir', DATA, '--epochs', '100',
          '--batch_size', '256', '--learning_rate', '0.2', '--momentum',
          '0.9', '--nesterov', '--weight_decay', '5e-4',
          '--width_factor_limits', '0.25,1.0', '--eval_width_factors',
          WIDTHS, '--skip_upstream_test']
# training ends with the sixteen protocol lines
codes = run_pinned([
    (label, gpu, [sys.executable, '-u', 'train_structured.py'] + COMMON
     + extra.split() + ['--save_dir', os.path.join(WORK, label),
                        '--ckpt_dir', os.path.join(CK, label),
                        '--log_prefix', label])
    for label, gpu, extra in BRANCHES])
print(codes)
'''

KEEP = '''# keep the checkpoints in the output, under a name the next session finds
print('checkpoints in', CK, sorted(os.listdir(CK)))
'''


def build(spec, steps):
    head = '# {}. {}\n\n{}'.format(spec['number'], spec['title'],
                                   spec['intro'])
    branches = repr(spec['branches'])
    config = '''# Fixed for this notebook.
REPO_URL = {repo!r}
REPO_BRANCH = {branch!r}
UPSTREAM = {upstream!r}
COMMIT = {commit!r}
UPSTREAM_DIR = {dir!r}
PATCH = {patch!r}
PIP = {pip!r}
# (label, gpu, extra flags)
BRANCHES = {branches}
WORKER_ENV = {{'JOSLIM_WORKERS': '2', 'LCS_WORKERS': '2'}}
'''.format(repo=REPO_URL, branch=REPO_BRANCH, upstream=spec['upstream'],
           commit=spec['commit'], dir=spec['dir'], patch=spec['patch'],
           pip=spec['pip'], branches=branches)
    cells = [markdown(head), code(config), code(SETUP), code(DATA),
             code(RUN)]
    cells += [code(step) for step in steps]
    cells += [markdown('## Results, against A (BX, 76.86)\n\nSend back this '
                       'table and the `val ... -1/100` lines above it.'),
              code(TABLE.format(a_ref=repr(A_REF))), code(KEEP)]
    nb = {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                       'name': 'python3'},
        'language_info': {'name': 'python'}},
        'nbformat': 4, 'nbformat_minor': 5}
    path = os.path.join(KAGGLE, spec['name'] + '.ipynb')
    with io.open(path, 'w', encoding='utf-8', newline='\n') as handle:
        json.dump(nb, handle, indent=1, ensure_ascii=False)
        handle.write('\n')
    print('wrote', path)


if __name__ == '__main__':
    build(JOSLIM, [JOSLIM_RESUME, JOSLIM_TRAIN, JOSLIM_EVAL])
    build(LCS, [LCS_RESUME, LCS_TRAIN])
