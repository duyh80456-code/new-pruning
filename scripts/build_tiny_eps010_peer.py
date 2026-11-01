"""configs and notebooks 125-126: SlimOT on Tiny ImageNet with the Sinkhorn
blur at 0.1 and Peer Transport at 0.5, three seeds on the full train split

    python scripts/build_tiny_eps010_peer.py

JM (blur 0.1, Peer Transport 0.5) was the best branch of the held-out
sweep (58.49 against JE 57.79, +0.70, one seed), and JF (blur 0.1 alone)
the best single change (58.41). Notebooks 123-124 run JF's change at three
seeds (JQ, JR, JS); JT, JU and JV run JM's, so the two can be compared
seed by seed. Each is EQ, FP or FR (SlimOT, ResNet-18, Tiny ImageNet,
seeds 1995, 2026 and 2006) with sinkhorn_eps 0.1 and horizontal_weight
0.5, on the full train split and scored on the official val split.
"""
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (letter, branch, source branch, seed, source letter)
RUNS = [
    ('JT', 'jt_k_heads4_r18_tiny_eps010_peer_half', 'eq_k_heads4_r18_tiny', 1995, 'EQ'),
    ('JU', 'ju_k_heads4_r18_tiny_eps010_peer_half_seed2', 'fp_k_heads4_r18_tiny_seed2', 2026, 'FP'),
    ('JV', 'jv_k_heads4_r18_tiny_eps010_peer_half_seed3', 'fr_k_heads4_r18_tiny_seed3', 2006, 'FR'),
]


def swap(text, old, new):
    if text.count(old) != 1:
        raise SystemExit('{!r} found {} times'.format(old, text.count(old)))
    return text.replace(old, new)


def body_of(text):
    """the config after its leading comment block"""
    lines = text.split('\n')
    i = 0
    while i < len(lines) and (lines[i].startswith('#') or not lines[i].strip()):
        i += 1
    return '\n'.join(lines[i:])


for letter, branch, source, seed, source_letter in RUNS:
    for kind in ('cifar100', 'smoke'):
        src = os.path.join(ROOT, 'apps', '{}_{}.yml'.format(kind, source))
        text = io.open(src, encoding='utf-8').read()
        text = swap(text, 'sinkhorn_eps: 0.2\n', 'sinkhorn_eps: 0.1\n')
        text = swap(text, 'horizontal_weight: 1.0\n', 'horizontal_weight: 0.5\n')
        text = swap(text, 'random_seed: {}\n'.format(seed),
                    'random_seed: {}\n'.format(seed))
        text = swap(text, 'log_dir: logs/{}_{}\n'.format(kind, source),
                    'log_dir: logs/{}_{}\n'.format(kind, branch))
        if kind == 'cifar100':
            header = (
                '# =========================== Branch {} ===========================\n'
                '# {} (SlimOT + WBH, ResNet-18, Tiny ImageNet, seed {}) with the\n'
                '# Sinkhorn entropic blur at 0.1 instead of 0.2 and Peer Transport at\n'
                '# 0.5 instead of 1.0, otherwise unchanged. The pair was the best\n'
                '# branch on the held-out train images (JM 58.49 against JE 57.79,\n'
                '# one seed). Full train split, official val split: read against {}\n'
                '# at the same seed, and against the blur alone (JQ, JR, JS).\n\n'
            ).format(letter, source_letter, seed, source_letter)
        else:
            header = '# smoke for Branch {}\n\n'.format(letter)
        out = os.path.join(ROOT, 'apps', '{}_{}.yml'.format(kind, branch))
        with io.open(out, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(header + body_of(text))
        print('wrote', out)


def lines(text):
    parts = text.split('\n')
    return [p + '\n' for p in parts[:-1]] + [parts[-1]]


# top-1 per width on the official val split, from results/runs.json
REFERENCES = '''# SlimOT as in the paper (blur 0.2, Peer Transport 1.0), same seed:
# EQ (1995), FP (2026), FR (2006)
SLIMOT = {
    'jt_k_heads4_r18_tiny_eps010_peer_half': [54.93, 56.03, 56.56, 57.92, 58.43, 59.18, 59.72, 59.8, 59.47, 60.05, 60.41, 60.9, 61.18, 60.9, 61.23, 61.62],
    'ju_k_heads4_r18_tiny_eps010_peer_half_seed2': [54.77, 55.85, 56.91, 57.66, 58.09, 58.6, 58.73, 59.11, 59.41, 59.62, 59.8, 60.18, 60.19, 60.65, 60.65, 60.79],
    'jv_k_heads4_r18_tiny_eps010_peer_half_seed3': [54.61, 55.47, 56.61, 57.09, 57.98, 58.81, 59.41, 59.49, 59.69, 59.95, 60.12, 60.4, 60.38, 60.39, 60.78, 61.2],
}
# US-Net (ES, FO, FQ) and AlphaNet (GE, GF, GG), same three seeds
USNET = {
    'jt_k_heads4_r18_tiny_eps010_peer_half': [54.98, 55.5, 55.96, 56.43, 57.48, 57.26, 57.69, 58.27, 58.79, 59.18, 59.41, 59.39, 59.72, 59.54, 59.98, 60.46],
    'ju_k_heads4_r18_tiny_eps010_peer_half_seed2': [54.79, 55.89, 56.34, 57.18, 57.41, 57.95, 58.4, 58.92, 59.38, 59.61, 59.41, 59.81, 60.45, 60.38, 60.59, 61.08],
    'jv_k_heads4_r18_tiny_eps010_peer_half_seed3': [55.25, 55.75, 56.64, 57.05, 57.82, 58.79, 59.0, 59.27, 59.86, 59.93, 59.93, 60.67, 60.9, 60.89, 61.02, 61.23],
}
ALPHANET = {
    'jt_k_heads4_r18_tiny_eps010_peer_half': [55.62, 56.03, 56.96, 57.42, 58.16, 58.99, 59.83, 59.54, 60.18, 60.5, 60.67, 60.44, 60.8, 60.92, 60.75, 60.86],
    'ju_k_heads4_r18_tiny_eps010_peer_half_seed2': [54.75, 55.7, 56.55, 57.31, 58.4, 59.09, 59.38, 59.75, 59.9, 59.74, 60.04, 60.29, 60.53, 60.61, 60.87, 60.98],
    'jv_k_heads4_r18_tiny_eps010_peer_half_seed3': [55.32, 56.16, 56.91, 57.21, 58.29, 58.89, 59.58, 59.33, 59.46, 59.79, 59.63, 60.0, 60.58, 60.61, 60.53, 60.72],
}
WIDTHS = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]


def mean(values):
    return sum(values) / len(values)


for branch in BRANCHES:
    table = results.get(branch, {})
    if not table:
        print(branch, 'no results')
        continue
    ours = {round(w, 2): 100.0 * (1.0 - v[1]) for w, v in table.items()}
    print(branch)
    print('{:>7}{:>9}{:>9}{:>9}{:>9}{:>9}'.format(
        'width', 'JM set', 'SlimOT', 'vs', 'US-Net', 'AlphaNet'))
    for i, width in enumerate(WIDTHS):
        if width not in ours:
            continue
        print('{:>7.2f}{:>9.2f}{:>9.2f}{:>+9.2f}{:>9.2f}{:>9.2f}'.format(
            width, ours[width], SLIMOT[branch][i], ours[width] - SLIMOT[branch][i],
            USNET[branch][i], ALPHANET[branch][i]))
    accuracies = [ours[w] for w in WIDTHS if w in ours]
    above = sum(ours[w] > SLIMOT[branch][i] for i, w in enumerate(WIDTHS) if w in ours)
    print('mean {:.2f}   SlimOT {:.2f} ({:+.2f}, above at {}/{} widths)'
          '   US-Net {:.2f}   AlphaNet {:.2f}\\n'.format(
              mean(accuracies), mean(SLIMOT[branch]),
              mean(accuracies) - mean(SLIMOT[branch]), above, len(accuracies),
              mean(USNET[branch]), mean(ALPHANET[branch])))

out = os.path.join(WORK, 'logs')
for branch in BRANCHES:
    log_dir = 'logs/cifar100_{}'.format(branch)
    if os.path.isdir(log_dir):
        shutil.copytree(log_dir, os.path.join(out, 'cifar100_' + branch),
                        dirs_exist_ok=True)
print('checkpoints copied to', out)'''

# notebooks from notebook 72's layout, the last Tiny ImageNet pair
template = json.load(io.open(os.path.join(ROOT, 'kaggle', 'kaggle_kd_72_fq_fr.ipynb'),
                             encoding='utf-8'))
for number, group in ((125, RUNS[:2]), (126, RUNS[2:])):
    nb = json.loads(json.dumps(template))
    md = ''.join(nb['cells'][0]['source'])
    keep = md[md.index('## Before you start'):]
    rows = ''.join('| {} | {} at seed {}, blur 0.1 and Peer Transport 0.5 |\n'.format(
        letter, source_letter, seed) for letter, _, _, seed, source_letter in group)
    sections = ''.join('## `{b}`\n\nSee apps/cifar100_{b}.yml for every setting and why.\n\n'
                       .format(b=branch) for _, branch, _, _, _ in group)
    count = 'Two branches, one per card' if len(group) == 2 else 'One branch'
    head = (
        '# {n}. SlimOT on Tiny ImageNet, blur 0.1 and Peer Transport 0.5, three seeds\n\n'
        '{count}, over more than one session - see below for\n'
        'how to carry it forward. Nothing in this notebook needs editing: run it as it is.\n\n'
        'The blur at 0.1 with Peer Transport at 0.5 was the best branch on\n'
        'held-out train images (JM 58.49 against JE 57.79, one seed). Here it\n'
        'runs on the full train split and is scored on the official val split,\n'
        'at the three seeds of EQ, FP and FR; notebooks 125 and 126 together\n'
        'give all three. Notebooks 123 and 124 run the blur alone at the same\n'
        'seeds, so the two can be compared seed by seed.\n\n'
        '| | |\n|---|---|\n{rows}\n'
        '**How to read it.** Each branch against SlimOT at the same seed (the\n'
        '"SlimOT" column), then the mean over the three seeds against SlimOT\n'
        '58.99, AlphaNet 59.05 and US-Net 58.58. Across seeds SlimOT moves by\n'
        'about 0.5 here.\n\n'
        '## Data\n\n'
        'Attach a Tiny ImageNet dataset (the one notebook 57 used), and CIFAR-100\n'
        'as well; the notebook\'s data cell expects it.\n\n'
        '## Timing\n\n'
        'About 525 s an epoch for the method on a T4, so about 14.5 hours.\n'
        'Plan on two sessions. When the first ends, Save Version, attach that\n'
        'version\'s output (pick the version number, not Latest) and run again:\n'
        'the notebook finds the checkpoints.\n\n'
        '{sections}\n'
    ).format(n=number, count=count, rows=rows, sections=sections)
    nb['cells'][0]['source'] = lines(head + keep)
    cfg = ''.join(nb['cells'][1]['source'])
    cfg = cfg.replace('Notebook 72 of 83.', 'Notebook {}.'.format(number))
    cfg = cfg.replace("['fq_a_r18_tiny_seed3', 'fr_k_heads4_r18_tiny_seed3']",
                      '[{}]'.format(', '.join("'{}'".format(b) for _, b, _, _, _ in group)))
    assert group[0][1] in cfg and 'Notebook {}.'.format(number) in cfg, cfg
    nb['cells'][1]['source'] = lines(cfg)
    nb['cells'][11]['source'] = lines(
        '## Results, against SlimOT at the same seed\n\nThe "SlimOT" column is '
        'SlimOT as in the paper at the same seed; US-Net and AlphaNet are at the '
        'same seed too.')
    assert 'shutil.copytree' in ''.join(nb['cells'][12]['source'])
    nb['cells'][12]['source'] = lines(REFERENCES)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'], cell['execution_count'] = [], None
    name = 'kaggle_kd_{}_{}.ipynb'.format(
        number, '_'.join(letter.lower() for letter, _, _, _, _ in group))
    path = os.path.join(ROOT, 'kaggle', name)
    with io.open(path, 'w', encoding='utf-8', newline='\n') as handle:
        json.dump(nb, handle, indent=1, ensure_ascii=False)
        handle.write('\n')
    print('wrote', path)
