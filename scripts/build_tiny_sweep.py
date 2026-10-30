"""configs and notebooks 117-120: SlimOT on Tiny ImageNet, hyperparameters
chosen on images held out of the train split

    python scripts/build_tiny_sweep.py

Every branch is EQ (SlimOT, ResNet-18, Tiny ImageNet, seed 1995) with
holdout_per_class 50: 50 train images of each class are held out and are
what the run reports, so the official val split, which is the test set of
the paper, plays no part in the choice. JE changes nothing else and is the
reference; each other branch changes one hyperparameter. One seed each.
"""
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = 'eq_k_heads4_r18_tiny'

# (letter, branch, key, old, new, what)
SWEEP = [
    ('JE', 'je_k_heads4_r18_tiny_holdout', None, None, None,
     'no other change: the reference every other branch is read against'),
    ('JF', 'jf_k_heads4_r18_tiny_holdout_eps010', 'sinkhorn_eps', '0.2', '0.1',
     'the Sinkhorn entropic blur at 0.1 instead of 0.2'),
    ('JG', 'jg_k_heads4_r18_tiny_holdout_eps050', 'sinkhorn_eps', '0.2', '0.5',
     'the Sinkhorn entropic blur at 0.5 instead of 0.2'),
    ('JH', 'jh_k_heads4_r18_tiny_holdout_peer_half', 'horizontal_weight', '1.0', '0.5',
     'Peer Transport at half weight, 0.5 instead of 1.0'),
    ('JI', 'ji_k_heads4_r18_tiny_holdout_peer_double', 'horizontal_weight', '1.0', '2.0',
     'Peer Transport at double weight, 2.0 instead of 1.0'),
    ('JJ', 'jj_k_heads4_r18_tiny_holdout_teacher_half', 'feature_weight', '1.0', '0.5',
     'Teacher Transport at half weight, 0.5 instead of 1.0'),
    ('JK', 'jk_k_heads2_r18_tiny_holdout', 'head_groups', '4', '2',
     'two band heads instead of four'),
    ('JL', 'jl_k_heads8_r18_tiny_holdout', 'head_groups', '4', '8',
     'eight band heads instead of four'),
]


def swap(text, key, old, new):
    line_old = '{}: {}\n'.format(key, old)
    line_new = '{}: {}\n'.format(key, new)
    if text.count(line_old) != 1:
        raise SystemExit('{!r} found {} times'.format(line_old, text.count(line_old)))
    return text.replace(line_old, line_new)


def body_of(text):
    """the config after its leading comment block"""
    lines = text.split('\n')
    i = 0
    while i < len(lines) and (lines[i].startswith('#') or not lines[i].strip()):
        i += 1
    return '\n'.join(lines[i:])


HOLDOUT = ('dataset_dir: data\n', 'dataset_dir: data\n'
           '# 50 train images of each class held out and reported instead of\n'
           '# the official val split (data/tinyimagenet.py, holdout)\n'
           'holdout_per_class: 50\n')

for letter, branch, key, old, new, what in SWEEP:
    for kind in ('cifar100', 'smoke'):
        src = os.path.join(ROOT, 'apps', '{}_{}.yml'.format(kind, BASE))
        text = io.open(src, encoding='utf-8').read()
        if key:
            text = swap(text, key, old, new)
        if text.count(HOLDOUT[0]) != 1:
            raise SystemExit('dataset_dir not found once in ' + src)
        text = text.replace(HOLDOUT[0], HOLDOUT[1])
        old_log = 'log_dir: logs/{}_{}\n'.format(kind, BASE)
        if text.count(old_log) != 1:
            raise SystemExit('log_dir not found in ' + src)
        text = text.replace(old_log, 'log_dir: logs/{}_{}\n'.format(kind, branch))
        if kind == 'cifar100':
            header = (
                '# =========================== Branch {} ===========================\n'
                '# SlimOT on ResNet-18, Tiny ImageNet (EQ, seed 1995, 59.27 on the\n'
                '# official val split), trained on 450 train images a class and read\n'
                '# on the 50 held out, with {}.\n'
                '# Hyperparameter choice for Tiny ImageNet without touching its test\n'
                '# set. One seed. Read against JE, never against EQ: the two are\n'
                '# scored on different images.\n\n'
            ).format(letter, what)
        else:
            header = '# smoke for Branch {}\n\n'.format(letter)
        out = os.path.join(ROOT, 'apps', '{}_{}.yml'.format(kind, branch))
        with io.open(out, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(header + body_of(text))
        print('wrote', out)


def lines(text):
    parts = text.split('\n')
    return [p + '\n' for p in parts[:-1]] + [parts[-1]]


# notebooks from notebook 72's layout, the last Tiny ImageNet pair
template = json.load(io.open(os.path.join(ROOT, 'kaggle', 'kaggle_kd_72_fq_fr.ipynb'),
                             encoding='utf-8'))
RESULTS = '''# Every branch here is scored on the same 10,000 held-out train images,
# not on the official val split, so the numbers are not comparable with
# EQ or with any table in the paper. Read each branch against JE.
widths = sorted({w for table in results.values() for w in table})
header = '{:>7}'.format('width')
for branch in BRANCHES:
    header += '{:>12}'.format(branch[:2].upper())
print(header)
for width in widths:
    row = '{:>7.2f}'.format(width)
    for branch in BRANCHES:
        entry = results.get(branch, {}).get(width)
        row += '{:>12}'.format('-' if entry is None
                               else '{:.2f}'.format(100.0 * (1.0 - entry[1])))
    print(row)
print()
for branch in BRANCHES:
    table = results.get(branch, {})
    if table:
        accuracies = [100.0 * (1.0 - v[1]) for v in table.values()]
        print('{:42} mean {:.2f} over {} widths'.format(
            branch, sum(accuracies) / len(accuracies), len(accuracies)))

out = os.path.join(WORK, 'logs')
for branch in BRANCHES:
    log_dir = 'logs/cifar100_{}'.format(branch)
    if os.path.isdir(log_dir):
        shutil.copytree(log_dir, os.path.join(out, 'cifar100_' + branch),
                        dirs_exist_ok=True)
print('\\ncheckpoints copied to', out)'''

for number, (left, right) in zip((117, 118, 119, 120), zip(SWEEP[0::2], SWEEP[1::2])):
    nb = json.loads(json.dumps(template))
    md = ''.join(nb['cells'][0]['source'])
    keep = md[md.index('## Before you start'):]
    head = (
        '# {n}. SlimOT on Tiny ImageNet, hyperparameters on held-out train images\n\n'
        'Two branches, one per card, over more than one session - see below for\n'
        'how to carry it forward. Nothing in this notebook needs editing: run it as it is.\n\n'
        'SlimOT on ResNet-18, Tiny ImageNet (EQ, seed 1995), trained on 450 train\n'
        'images of each class and scored on the 50 held out, so the official val\n'
        'split, the test set of the paper, plays no part in choosing. One seed\n'
        'each: across seeds SlimOT moves by about 0.5 here, so read a gap under\n'
        'about 0.8 against JE as no difference.\n\n'
        '| | |\n|---|---|\n'
        '| {l1} | {w1} |\n'
        '| {l2} | {w2} |\n\n'
        '**How to read it.** Each branch against JE (notebook 117), never against\n'
        'EQ or a table of the paper: those are scored on other images.\n\n'
        '## Data\n\n'
        'Attach a Tiny ImageNet dataset (the one notebook 57 used), and CIFAR-100\n'
        'as well; the notebook\'s data cell expects it.\n\n'
        '## Timing\n\n'
        'About 525 s an epoch for the method on a T4 with the full train split;\n'
        'a tenth fewer images here. Plan on two sessions. When the first ends,\n'
        'Save Version, attach that version\'s output (pick the version number,\n'
        'not Latest) and run again: the notebook finds the checkpoints.\n\n'
        '## `{b1}`\n\nSee apps/cifar100_{b1}.yml for every setting and why.\n\n'
        '## `{b2}`\n\nSee apps/cifar100_{b2}.yml for every setting and why.\n\n\n'
    ).format(n=number, l1=left[0], w1=left[5], l2=right[0], w2=right[5],
             b1=left[1], b2=right[1])
    nb['cells'][0]['source'] = lines(head + keep)
    cfg = ''.join(nb['cells'][1]['source'])
    cfg = cfg.replace('Notebook 72 of 83.', 'Notebook {}.'.format(number))
    cfg = cfg.replace("['fq_a_r18_tiny_seed3', 'fr_k_heads4_r18_tiny_seed3']",
                      "['{}', '{}']".format(left[1], right[1]))
    assert left[1] in cfg and 'Notebook {}.'.format(number) in cfg, cfg
    nb['cells'][1]['source'] = lines(cfg)
    nb['cells'][11]['source'] = lines(
        '## Results, against JE\n\nJE is SlimOT unchanged on the same held-out '
        'images; a branch helps only if it is clearly above JE.')
    assert 'shutil.copytree' in ''.join(nb['cells'][12]['source'])
    nb['cells'][12]['source'] = lines(RESULTS)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'], cell['execution_count'] = [], None
    name = 'kaggle_kd_{}_{}_{}.ipynb'.format(number, left[0].lower(), right[0].lower())
    path = os.path.join(ROOT, 'kaggle', name)
    with io.open(path, 'w', encoding='utf-8', newline='\n') as handle:
        json.dump(nb, handle, indent=1, ensure_ascii=False)
        handle.write('\n')
    print('wrote', path)
