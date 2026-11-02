"""configs and notebooks 132-133: SlimOT with one-way Peer Transport, three
seeds, CIFAR-100, ResNet-50

    python scripts/build_peer_one_way.py

The ablation for the claim that Peer Transport helps because it is two-way.
KE, KF and KG are DO, EI and FC (SlimOT, seeds 1995, 2026 and 2006) with
peer_one_way: the gradient through the wider width of each Peer pair is
stopped, so only the narrower one is pulled toward it, as inplace
distillation pulls a narrow width toward the widest. The narrower width's
gradient is exactly what it gets in SlimOT; only the wider width's part is
gone. Read against SlimOT (77.84) and SlimOT w/o Peer Transport (EO, FG,
FK: 77.16) over the same three seeds.
"""
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (letter, branch, source branch, seed, SlimOT letter, w/o Peer letter)
RUNS = [
    ('KE', 'ke_k_heads4_peer_one_way_r50', 'do_k_late_heads4_r50', 1995, 'DO', 'EO'),
    ('KF', 'kf_k_heads4_peer_one_way_r50_seed2', 'ei_k_heads4_r50_seed2', 2026, 'EI', 'FG'),
    ('KG', 'kg_k_heads4_peer_one_way_r50_seed3', 'fc_k_heads4_r50_seed3', 2006, 'FC', 'FK'),
]
ONE_WAY = ('\n# =========================== One-way Peer ===========================\n'
           'peer_one_way: True\n')


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


for letter, branch, source, seed, slimot, _ in RUNS:
    for kind in ('cifar100', 'smoke'):
        src = os.path.join(ROOT, 'apps', '{}_{}.yml'.format(kind, source))
        text = io.open(src, encoding='utf-8').read()
        if 'peer_one_way' in text:
            raise SystemExit('{} already sets peer_one_way'.format(src))
        swap(text, 'split_pair_backward: True\n', 'split_pair_backward: True\n')
        text = swap(text, 'random_seed: {}\n'.format(seed),
                    'random_seed: {}\n'.format(seed))
        text = swap(text, 'log_dir: logs/{}_{}\n'.format(kind, source),
                    'log_dir: logs/{}_{}\n'.format(kind, branch))
        if kind == 'cifar100':
            header = (
                '# =========================== Branch {} ===========================\n'
                '# {} (SlimOT, ResNet-50, CIFAR-100, seed {}) with one-way Peer\n'
                '# Transport: the gradient through the wider width of each pair is\n'
                '# stopped, so only the narrower width is pulled toward it. The\n'
                '# narrower width gets the same gradient as in {}. Read against {}\n'
                '# and SlimOT w/o Peer Transport at the same seed.\n\n'
            ).format(letter, slimot, seed, slimot, slimot)
        else:
            header = '# smoke for Branch {}\n\n'.format(letter)
        out = os.path.join(ROOT, 'apps', '{}_{}.yml'.format(kind, branch))
        with io.open(out, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(header + body_of(text).rstrip('\n') + '\n' + ONE_WAY)
        print('wrote', out)

runs = {r['letter']: r for r in json.load(
    io.open(os.path.join(ROOT, 'results', 'runs.json'), encoding='utf-8'))['runs']}


def per_width(letter):
    return [round(v, 2) for v in runs[letter]['top1']]


def lines(text):
    parts = text.split('\n')
    return [p + '\n' for p in parts[:-1]] + [parts[-1]]


def references(group):
    def table(index):
        return '{\n' + ''.join("    '{}': {},\n".format(g[1], per_width(g[index]))
                               for g in group) + '}'
    return '''# Same seed: SlimOT (DO 1995, EI 2026, FC 2006) and SlimOT w/o Peer
# Transport (EO, FG, FK), top-1 per width from results/runs.json
SLIMOT = {s}
NO_PEER = {n}
WIDTHS = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]


def mean(values):
    return sum(values) / len(values)


for branch in BRANCHES:
    table = results.get(branch, {{}})
    if not table:
        print(branch, 'no results')
        continue
    ours = {{round(w, 2): 100.0 * (1.0 - v[1]) for w, v in table.items()}}
    print(branch)
    print('{{:>7}}{{:>9}}{{:>9}}{{:>9}}{{:>9}}'.format(
        'width', 'one-way', 'SlimOT', 'vs it', 'no Peer'))
    for i, width in enumerate(WIDTHS):
        if width not in ours:
            continue
        print('{{:>7.2f}}{{:>9.2f}}{{:>9.2f}}{{:>+9.2f}}{{:>9.2f}}'.format(
            width, ours[width], SLIMOT[branch][i], ours[width] - SLIMOT[branch][i],
            NO_PEER[branch][i]))
    accuracies = [ours[w] for w in WIDTHS if w in ours]
    below = sum(ours[w] < SLIMOT[branch][i] for i, w in enumerate(WIDTHS) if w in ours)
    print('mean {{:.2f}}   SlimOT {{:.2f}} ({{:+.2f}}, below it at {{}}/{{}} widths)'
          '   w/o Peer {{:.2f}}\\n'.format(
              mean(accuracies), mean(SLIMOT[branch]),
              mean(accuracies) - mean(SLIMOT[branch]), below, len(accuracies),
              mean(NO_PEER[branch])))

out = os.path.join(WORK, 'logs')
for branch in BRANCHES:
    log_dir = 'logs/cifar100_{{}}'.format(branch)
    if os.path.isdir(log_dir):
        shutil.copytree(log_dir, os.path.join(out, 'cifar100_' + branch),
                        dirs_exist_ok=True)
print('checkpoints copied to', out)'''.format(s=table(4), n=table(5))


template = json.load(io.open(os.path.join(ROOT, 'kaggle', 'kaggle_kd_88_hc_hd.ipynb'),
                             encoding='utf-8'))
for number, group in ((132, RUNS[:2]), (133, RUNS[2:])):
    nb = json.loads(json.dumps(template))
    md = ''.join(nb['cells'][0]['source'])
    keep = md[md.index('## Before you start'):]
    rows = ''.join('| {} | {} (SlimOT, seed {}) with one-way Peer Transport |\n'.format(
        g[0], g[4], g[3]) for g in group)
    sections = ''.join('## `{b}`\n\nSee apps/cifar100_{b}.yml for every setting and why.\n\n'
                       .format(b=g[1]) for g in group)
    count = 'Two branches, one per card' if len(group) == 2 else 'One branch'
    head = (
        '# {n}. SlimOT with one-way Peer Transport\n\n'
        '{count}, over more than one session - see below for\n'
        'how to carry it forward. Nothing in this notebook needs editing: run it as it is.\n\n'
        'Peer Transport pulls the two middle widths of each step toward each\n'
        'other, with gradient into both. Here the gradient through the wider\n'
        'width is stopped, so only the narrower width is pulled, as inplace\n'
        'distillation pulls a narrow width toward the widest. Notebooks 132 and\n'
        '133 together give seeds 1995, 2026 and 2006.\n\n'
        '| | |\n|---|---|\n{rows}\n'
        '**How to read it.** Each branch against SlimOT at the same seed, then\n'
        'the three-seed mean against SlimOT 77.84 and SlimOT w/o Peer Transport\n'
        '77.16. Near 77.84: the two-way flow adds nothing. Near 77.16: the gain\n'
        'of Peer Transport needs it.\n\n'
        '## Data\n\nAttach the CIFAR-100 dataset.\n\n'
        '## Timing\n\n'
        'SlimOT on ResNet-50 takes about 14 hours on a T4, so two sessions.\n'
        'When the first ends, Save Version, attach that version\'s output (pick\n'
        'the version number, not Latest) and run again: the notebook finds the\n'
        'checkpoints.\n\n'
        '{sections}'
    ).format(n=number, count=count, rows=rows, sections=sections)
    nb['cells'][0]['source'] = lines(head + keep)
    cfg = ''.join(nb['cells'][1]['source'])
    cfg = swap(cfg, 'Notebook 88 of 88.', 'Notebook {}.'.format(number))
    cfg = swap(cfg, "['hc_solar_grid_r50', 'hd_dynas_grid_r50']",
               '[{}]'.format(', '.join("'{}'".format(g[1]) for g in group)))
    nb['cells'][1]['source'] = lines(cfg)
    nb['cells'][11]['source'] = lines(
        '## Results, against SlimOT at the same seed\n\n'
        '"SlimOT" is the two-way method as in the paper; "no Peer" is SlimOT '
        'w/o Peer Transport, both at the same seed.')
    assert 'shutil.copytree' in ''.join(nb['cells'][12]['source'])
    nb['cells'][12]['source'] = lines(references(group))
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'], cell['execution_count'] = [], None
    name = 'kaggle_kd_{}_{}.ipynb'.format(number, '_'.join(g[0].lower() for g in group))
    path = os.path.join(ROOT, 'kaggle', name)
    with io.open(path, 'w', encoding='utf-8', newline='\n') as handle:
        json.dump(nb, handle, indent=1, ensure_ascii=False)
        handle.write('\n')
    print('wrote', path)
