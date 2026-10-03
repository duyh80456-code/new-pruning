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

# DD, K as kept (feature transport from epoch 6), seed 1995
K_REF = {0.25: 75.63, 0.30: 76.29, 0.35: 76.54, 0.40: 76.97, 0.45: 77.67,
         0.50: 77.43, 0.55: 77.58, 0.60: 77.62, 0.65: 77.83, 0.70: 78.02,
         0.75: 78.00, 0.80: 78.01, 0.85: 78.09, 0.90: 78.23, 0.95: 78.08,
         1.00: 77.95}

UPSTREAMS = {
    'joslim': ('https://github.com/enyac-group/Joslim.git',
               '9b743d9b50ade9fc2cd33d255dfbb936e4894733', 'Joslim',
               'joslim.patch',
               'botorch==0.16.1 gpytorch==1.15.2 linear_operator==0.6.1'),
    'lcs': ('https://github.com/apple/learning-compressible-subspaces.git',
            'e6d3924368faccbdfd3d89c4a4735dba947275c9',
            'learning-compressible-subspaces', 'lcs.patch', 'pyyaml'),
}

SPEC = {
    'number': 39,
    'name': 'kaggle_kd_39_joslim_lcs',
    'title': 'Joslim and LCS from the authors\' code, on our protocol',
    'intro': '''Two published slimmable methods run from their authors' repositories
rather than re-implemented, both on exactly the protocol of every other row:
CIFAR-100 at 32x32, CIFAR ResNet-50 (3x3 stem, no max-pool), 100 epochs,
batch 256, SGD nesterov lr 0.2 cosine with warm-up, wd 5e-4, widths
0.25-1.00, read at the sixteen widths after BN recalibration on 20 batches.

| | GPU | method | code |
|---|---|---|---|
| joslim_r50 | 0 | Joslim (Chin et al., ECML-PKDD 2021): per-layer widths searched by Bayesian optimisation during training | enyac-group/Joslim at 9b743d9 + `third_party/patches/joslim.patch` |
| lcs_l_bn | 1 | LCS (Nunez et al., WACV 2023): a line in weight space from width 0.25 to 1.00, here with BatchNorm | apple/learning-compressible-subspaces at e6d3924 + `third_party/patches/lcs.patch` |

Each patch adds only what the protocol needs (CIFAR-100, the CIFAR
ResNet-50, the recipe, resume, the sixteen-width evaluation in our line
format); `third_party/patches/*_README.md` list every change. One change of
substance in Joslim: upstream scored every sampled architecture on test
batches, and that score drives its search; the patch scores them on a fixed
5,000-image subset of the training set. Joslim's own design point is
per-layer widths, so its Pareto set (accuracy against FLOPs) is printed too.

## More than one session

On a T4 LCS takes about 9-13 hours and Joslim longer, since its Bayesian
optimisation runs on the CPU. Both checkpoint every epoch. When the session
ends, Save Version, then in a new run attach that version's output (pick the
version by number, not Latest) and run again: the notebook finds the
checkpoints and both continue. Attach the CIFAR-100 dataset as well.
''',
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
if n_gpu < 2:
    raise SystemExit('Needs GPU T4 x2: one card per method.')

WORK = '/kaggle/working'
OURS = os.path.join(WORK, 'new-pruning')
if not os.path.isdir(OURS):
    subprocess.run(['git', 'clone', '-b', REPO_BRANCH, REPO_URL, OURS],
                   check=True)
print('our code at', subprocess.run(
    ['git', '-C', OURS, 'log', '-1', '--format=%h %s'],
    capture_output=True, text=True).stdout.strip())

CODE = {}
packages = []
for key, (url, commit, folder, patch, pip) in UPSTREAMS.items():
    CODE[key] = os.path.join(WORK, folder)
    if not os.path.isdir(CODE[key]):
        subprocess.run(['git', 'clone', url, CODE[key]], check=True)
        subprocess.run(['git', '-C', CODE[key], 'checkout', '-q', commit],
                       check=True)
        subprocess.run(['git', '-C', CODE[key], 'apply', os.path.join(
            OURS, 'third_party', 'patches', patch)], check=True)
    print(key, 'at', subprocess.run(
        ['git', '-C', CODE[key], 'log', '-1', '--format=%h %ad',
         '--date=short'], capture_output=True, text=True).stdout.strip(),
        '+', patch)
    packages += pip.split()

# install without letting pip replace Kaggle's torch
base = torch.__version__.split('+')[0]
with open(os.path.join(WORK, 'constraints.txt'), 'w') as handle:
    handle.write('torch=={}\\n'.format(base))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-c',
                os.path.join(WORK, 'constraints.txt')] + packages, check=True)
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

    for label, gpu, cwd, command in jobs:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, **WORKER_ENV)
        proc = subprocess.Popen(command, cwd=cwd, env=env, text=True,
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
K_REF = {k_ref}
labels = ['joslim_r50', 'lcs_l_bn']
print('{{:>7}}{{:>8}}{{:>8}}'.format('width', 'A', 'K') + ''.join(
    '{{:>12}}{{:>7}}{{:>7}}'.format(b[:11], 'vs A', 'vs K') for b in labels))
for width in sorted(A_REF):
    row = '{{:>7.2f}}{{:>8.2f}}{{:>8.2f}}'.format(
        width, A_REF[width], K_REF[width])
    for label in labels:
        entry = results.get(label, {{}}).get(width)
        if entry is None:
            row += '{{:>12}}{{:>7}}{{:>7}}'.format('-', '-', '-')
            continue
        accuracy = 100.0 * (1.0 - entry[1])
        row += '{{:>12.2f}}{{:>+7.2f}}{{:>+7.2f}}'.format(
            accuracy, accuracy - A_REF[width], accuracy - K_REF[width])
    print(row)
reference = sum(A_REF.values()) / len(A_REF)
k_reference = sum(K_REF.values()) / len(K_REF)
print('\\n{{:24}} mean {{:.2f}}'.format('A (BX)', reference))
print('{{:24}} mean {{:.2f}}'.format('K (DD)', k_reference))
for label in labels:
    table = results.get(label, {{}})
    if len(table) == len(A_REF):
        mean = sum(100.0 * (1.0 - v[1]) for v in table.values()) / len(table)
        print('{{:24}} mean {{:.2f}}   vs A {{:+.2f}}   vs K {{:+.2f}}'.format(
            label, mean, mean - reference, mean - k_reference))
    else:
        print('{{:24}} {{}} of 16 widths read'.format(label, len(table)))
'''

RESUME = '''CK = os.path.join(WORK, 'ckpt')
# Joslim keeps ckpt/joslim_r50.pt, LCS keeps ckpt/lcs_l_bn/last.pt
wanted = {'joslim_r50': ('joslim_r50.pt', None),
          'lcs_l_bn': ('last.pt', 'lcs_l_bn')}
for label, (name, folder) in wanted.items():
    target = os.path.join(CK, folder or '', name)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if not os.path.exists(target):
        for root, _, files in os.walk('/kaggle/input', followlinks=True):
            if name in files and (folder is None
                                  or os.path.basename(root) == folder):
                shutil.copy(os.path.join(root, name), target)
                print('restored', label, 'from', root)
                break
    print(label, 'resumes from its checkpoint' if os.path.exists(target)
          else 'starts at epoch 1')
'''

TRAIN = '''JOSLIM = ['--dataset', 'CIFAR100', '--datapath', DATA,
          '--network', 'slim_resnet50_cifar', '--epochs', '100',
          '--warmup', '5', '--baselr', '0.2', '--scheduler', 'cosine_decay',
          '--batch_size', '256', '--wd', '5e-4', '--mmt', '0.9',
          '--nesterov', '--label_smoothing', '0', '--lower_channel', '0.25',
          '--num_sampled_arch', '2', '--baseline', '-3',
          '--print_freq', '100', '--ckpt_dir', CK,
          '--tau', '195', '--prior_points', '20']
WIDTHS = ','.join('{:.2f}'.format(0.25 + 0.05 * i) for i in range(16))
LCS = ['--model', 'cresnet50', '--dataset', 'cifar100', '--method', 'lcs_l',
       '--data_dir', DATA, '--epochs', '100', '--batch_size', '256',
       '--learning_rate', '0.2', '--momentum', '0.9', '--nesterov',
       '--weight_decay', '5e-4', '--width_factor_limits', '0.25,1.0',
       '--eval_width_factors', WIDTHS, '--skip_upstream_test',
       '--norm', 'BN', '--recal_batches', '20',
       '--save_dir', os.path.join(WORK, 'lcs_l_bn'),
       '--ckpt_dir', os.path.join(CK, 'lcs_l_bn'), '--log_prefix', 'lcs_l_bn']
# LCS ends its run with the sixteen protocol lines; Joslim is read below
codes = run_pinned([
    ('joslim_r50', '0', CODE['joslim'],
     [sys.executable, '-u', 'joslim.py', '--name', 'joslim_r50'] + JOSLIM),
    ('lcs_l_bn', '1', CODE['lcs'],
     [sys.executable, '-u', 'train_structured.py'] + LCS)])
print(codes)
'''

EVAL = '''# Joslim at the sixteen uniform widths, then its own Pareto set
EVAL = ['--name', 'joslim_r50', '--dataset', 'CIFAR100', '--datapath', DATA,
        '--network', 'slim_resnet50_cifar', '--batch_size', '256',
        '--lower_channel', '0.25', '--ckpt_dir', CK]
run_pinned([('joslim_r50', '0', CODE['joslim'],
             [sys.executable, '-u', 'eval_checkpoints.py'] + EVAL
             + ['--uniform', '--tag', 'joslim_r50'])])
run_pinned([('joslim_r50_pareto', '0', CODE['joslim'],
             [sys.executable, '-u', 'eval_checkpoints.py'] + EVAL)])
'''

KEEP = '''# keep the checkpoints in the output, under a name the next session finds
print('checkpoints in', CK, sorted(os.listdir(CK)))
'''


def build(spec, steps):
    head = '# {}. {}\n\n{}'.format(spec['number'], spec['title'],
                                   spec['intro'])
    config = '''# Fixed for this notebook.
REPO_URL = {repo!r}
REPO_BRANCH = {branch!r}
# key: (repository, pinned commit, folder, patch, extra pip packages)
UPSTREAMS = {upstreams}
WORKER_ENV = {{'JOSLIM_WORKERS': '2', 'LCS_WORKERS': '2'}}
'''.format(repo=REPO_URL, branch=REPO_BRANCH,
           upstreams=repr(UPSTREAMS))
    cells = [markdown(head), code(config), code(SETUP), code(DATA),
             code(RUN)]
    cells += [code(step) for step in steps]
    cells += [markdown('## Results, against A (BX, 76.86) and K (DD, 77.50)\n\nSend back this '
                       'table and the `val ... -1/100` lines above it.'),
              code(TABLE.format(a_ref=repr(A_REF), k_ref=repr(K_REF))), code(KEEP)]
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
    build(SPEC, [RESUME, TRAIN, EVAL])
