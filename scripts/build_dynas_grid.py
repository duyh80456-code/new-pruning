"""configs and notebooks 127-131: DYNAS on the discrete width grid, three
seeds in each of the paper's three settings

    python scripts/build_dynas_grid.py

DYNAS was published on discrete supernets, and every DYNAS run in the
paper's tables draws continuous widths (no width_sampling in its configs,
so sample_width takes random.uniform). Each branch here is one of those
nine runs with the free widths of the sandwich drawn from the 16 test
widths, 0.25 to 1.00 in steps of 0.05, and nothing else changed. HD (built
for notebook 88, never run) is the ResNet-50 one at seed 1995. DS-Net needs
no such run: the authors' code already trains the 16 test widths as
discrete channel choices.
"""
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SETTINGS = {
    'r50': 'CIFAR-100, ResNet-50',
    'mbv2': 'CIFAR-100, MobileNetV2 with the last layer slimmed',
    'tiny': 'Tiny ImageNet, ResNet-18',
}
# (letter, branch, source branch, seed, setting); HD's configs exist
RUNS = [
    ('HD', 'hd_dynas_grid_r50', 'db_dynas_r50', 1995, 'r50'),
    ('JW', 'jw_dynas_grid_r50_seed2', 'ga_dynas_r50_seed2', 2026, 'r50'),
    ('JX', 'jx_dynas_grid_r50_seed3', 'gb_dynas_r50_seed3', 2006, 'r50'),
    ('JY', 'jy_dynas_grid_mbv2_slimlast', 'gq_dynas_mbv2_slimlast', 1995, 'mbv2'),
    ('JZ', 'jz_dynas_grid_mbv2_slimlast_seed2', 'gr_dynas_mbv2_slimlast_seed2', 2026, 'mbv2'),
    ('KA', 'ka_dynas_grid_mbv2_slimlast_seed3', 'gs_dynas_mbv2_slimlast_seed3', 2006, 'mbv2'),
    ('KB', 'kb_dynas_grid_r18_tiny', 'gh_dynas_r18_tiny', 1995, 'tiny'),
    ('KC', 'kc_dynas_grid_r18_tiny_seed2', 'gi_dynas_r18_tiny_seed2', 2026, 'tiny'),
    ('KD', 'kd_dynas_grid_r18_tiny_seed3', 'gj_dynas_r18_tiny_seed3', 2006, 'tiny'),
]
# same setting and seed: DYNAS continuous, US-Net, SlimOT as in the paper
SAME_SEED = {
    ('r50', 1995): ('DB', 'BX', 'DO'), ('r50', 2026): ('GA', 'CQ', 'EI'),
    ('r50', 2006): ('GB', 'FB', 'FC'),
    ('mbv2', 1995): ('GQ', 'EV', 'EW'), ('mbv2', 2026): ('GR', 'FS', 'FT'),
    ('mbv2', 2006): ('GS', 'FU', 'FV'),
    ('tiny', 1995): ('GH', 'ES', 'JQ'), ('tiny', 2026): ('GI', 'FO', 'JR'),
    ('tiny', 2006): ('GJ', 'FQ', 'JS'),
}
NOTEBOOKS = [(127, ['HD', 'JW']), (128, ['JX', 'JY']), (129, ['JZ', 'KA']),
             (130, ['KB', 'KC']), (131, ['KD'])]
TIMING = {
    'r50': 'DYNAS on ResNet-50 took about 10.5 hours on a T4',
    'mbv2': 'DYNAS on MobileNetV2 took about 6 hours on a T4',
    'tiny': 'DYNAS on Tiny ImageNet takes 8 to 12 hours on a T4',
}
GRID = ('\n# =========================== Discrete widths ===========================\n'
        'width_sampling: grid\nwidth_grid_step: 0.05\n')


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


runs = {r['letter']: r for r in json.load(
    io.open(os.path.join(ROOT, 'results', 'runs.json'), encoding='utf-8'))['runs']}


def per_width(letter):
    return [round(v, 2) for v in runs[letter]['top1']]


def three_seed_mean(setting, index):
    means = [sum(per_width(SAME_SEED[(setting, s)][index])) / 16.0
             for s in (1995, 2026, 2006)]
    return sum(means) / 3.0


for letter, branch, source, seed, setting in RUNS:
    if letter == 'HD':
        continue
    continuous = SAME_SEED[(setting, seed)][0]
    for kind in ('cifar100', 'smoke'):
        src = os.path.join(ROOT, 'apps', '{}_{}.yml'.format(kind, source))
        text = io.open(src, encoding='utf-8').read()
        if 'width_sampling' in text:
            raise SystemExit('{} already sets width_sampling'.format(src))
        text = swap(text, 'random_seed: {}\n'.format(seed),
                    'random_seed: {}\n'.format(seed))
        text = swap(text, 'log_dir: logs/{}_{}\n'.format(kind, source),
                    'log_dir: logs/{}_{}\n'.format(kind, branch))
        if kind == 'cifar100':
            header = (
                '# =========================== Branch {} ===========================\n'
                '# {} (DYNAS, {}, seed {}) on the 16 test widths, 0.25 to 1.00 in\n'
                '# steps of 0.05: the free widths of the sandwich are drawn from the\n'
                '# grid points between the ends, otherwise unchanged. DYNAS was\n'
                '# published on discrete supernets; {} draws continuous widths.\n'
                '# Read against {} at the same seed, and with the other two seeds\n'
                '# against its continuous three-seed mean ({:.2f}).\n\n'
            ).format(letter, continuous, SETTINGS[setting], seed, continuous,
                     continuous, three_seed_mean(setting, 0))
        else:
            header = '# smoke for Branch {}\n\n'.format(letter)
        out = os.path.join(ROOT, 'apps', '{}_{}.yml'.format(kind, branch))
        with io.open(out, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(header + body_of(text).rstrip('\n') + '\n' + GRID)
        print('wrote', out)


def lines(text):
    parts = text.split('\n')
    return [p + '\n' for p in parts[:-1]] + [parts[-1]]


def references(group):
    def table(index):
        return '{\n' + ''.join(
            "    '{}': {},\n".format(b, per_width(SAME_SEED[(st, s)][index]))
            for _, b, _, s, st in group) + '}'
    names = ', '.join('{} {}: {}'.format(st, s, '/'.join(SAME_SEED[(st, s)]))
                      for _, _, _, s, st in group)
    return '''# Same setting and seed, as DYNAS continuous / US-Net / SlimOT in the
# paper ({names}), top-1 per width from results/runs.json
CONTINUOUS = {c}
USNET = {u}
SLIMOT = {s}
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
    print('{{:>7}}{{:>9}}{{:>11}}{{:>9}}{{:>9}}{{:>9}}'.format(
        'width', 'grid', 'continuous', 'vs cont', 'US-Net', 'SlimOT'))
    for i, width in enumerate(WIDTHS):
        if width not in ours:
            continue
        print('{{:>7.2f}}{{:>9.2f}}{{:>11.2f}}{{:>+9.2f}}{{:>9.2f}}{{:>9.2f}}'.format(
            width, ours[width], CONTINUOUS[branch][i],
            ours[width] - CONTINUOUS[branch][i], USNET[branch][i], SLIMOT[branch][i]))
    accuracies = [ours[w] for w in WIDTHS if w in ours]
    below = sum(ours[w] < SLIMOT[branch][i] for i, w in enumerate(WIDTHS) if w in ours)
    print('mean {{:.2f}}   continuous {{:.2f}} ({{:+.2f}})   US-Net {{:.2f}}'
          '   SlimOT {{:.2f}}, above this at {{}}/{{}} widths\\n'.format(
              mean(accuracies), mean(CONTINUOUS[branch]),
              mean(accuracies) - mean(CONTINUOUS[branch]), mean(USNET[branch]),
              mean(SLIMOT[branch]), below, len(accuracies)))

out = os.path.join(WORK, 'logs')
for branch in BRANCHES:
    log_dir = 'logs/cifar100_{{}}'.format(branch)
    if os.path.isdir(log_dir):
        shutil.copytree(log_dir, os.path.join(out, 'cifar100_' + branch),
                        dirs_exist_ok=True)
print('checkpoints copied to', out)'''.format(names=names, c=table(0), u=table(1),
                                              s=table(2))


# notebooks from notebook 88's layout, the one HD was first built into; its
# code cells are the same as those of the Tiny ImageNet and MobileNetV2 ones
template = json.load(io.open(os.path.join(ROOT, 'kaggle', 'kaggle_kd_88_hc_hd.ipynb'),
                             encoding='utf-8'))
by_letter = {r[0]: r for r in RUNS}
for number, letters in NOTEBOOKS:
    group = [by_letter[l] for l in letters]
    nb = json.loads(json.dumps(template))
    md = ''.join(nb['cells'][0]['source'])
    keep = md[md.index('## Before you start'):]
    rows = ''.join('| {} | DYNAS on the 16 test widths, {}, seed {} (continuous: {}) |\n'.format(
        letter, SETTINGS[st], s, SAME_SEED[(st, s)][0]) for letter, _, _, s, st in group)
    means = ''.join('* {}: DYNAS continuous {:.2f}, US-Net {:.2f}, SlimOT {:.2f}\n'.format(
        SETTINGS[st], three_seed_mean(st, 0), three_seed_mean(st, 1), three_seed_mean(st, 2))
        for st in sorted({g[4] for g in group}, key=list(SETTINGS).index))
    sections = ''.join('## `{b}`\n\nSee apps/cifar100_{b}.yml for every setting and why.\n\n'
                       .format(b=b) for _, b, _, _, _ in group)
    count = 'Two branches, one per card' if len(group) == 2 else 'One branch'
    data = ''
    if any(g[4] == 'tiny' for g in group):
        data = ('## Data\n\n'
                'Attach a Tiny ImageNet dataset (the one notebook 57 used), and CIFAR-100\n'
                'as well; the notebook\'s data cell expects it.\n\n')
    else:
        data = '## Data\n\nAttach the CIFAR-100 dataset.\n\n'
    timing = '; '.join(sorted({TIMING[g[4]] for g in group}))
    head = (
        '# {n}. DYNAS on the discrete width grid\n\n'
        '{count}, over more than one session - see below for\n'
        'how to carry it forward. Nothing in this notebook needs editing: run it as it is.\n\n'
        'DYNAS was published on discrete supernets; every DYNAS run in the paper\n'
        'draws continuous widths (DYNAS-dagger). Here the sub-networks are the 16\n'
        'test widths, 0.25 to 1.00 in steps of 0.05: the free widths of the\n'
        'sandwich are drawn from the grid points between the ends, so nothing\n'
        'between them is ever trained. Four widths a step, as in every other row.\n'
        'Notebooks 127 to 131 together give three seeds in each of the three\n'
        'settings; they are independent and can run at the same time.\n\n'
        '| | |\n|---|---|\n{rows}\n'
        '**How to read it.** Each branch against DYNAS continuous at the same\n'
        'seed ("continuous"), then the three-seed means against:\n\n{means}\n'
        '{data}'
        '## Timing\n\n'
        '{timing}. When a session ends before training does, Save Version,\n'
        'attach that version\'s output (pick the version number, not Latest) and\n'
        'run again: the notebook finds the checkpoints.\n\n'
        '{sections}'
    ).format(n=number, count=count, rows=rows, means=means, data=data,
             timing=timing, sections=sections)
    nb['cells'][0]['source'] = lines(head + keep)
    cfg = ''.join(nb['cells'][1]['source'])
    cfg = swap(cfg, 'Notebook 88 of 88.', 'Notebook {}.'.format(number))
    cfg = swap(cfg, "['hc_solar_grid_r50', 'hd_dynas_grid_r50']",
               '[{}]'.format(', '.join("'{}'".format(g[1]) for g in group)))
    nb['cells'][1]['source'] = lines(cfg)
    nb['cells'][11]['source'] = lines(
        '## Results, against DYNAS continuous at the same seed\n\n'
        '"continuous" is DYNAS as in the paper\'s table at the same setting and '
        'seed; US-Net and SlimOT are at the same seed too.')
    assert 'shutil.copytree' in ''.join(nb['cells'][12]['source'])
    nb['cells'][12]['source'] = lines(references(group))
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'], cell['execution_count'] = [], None
    name = 'kaggle_kd_{}_{}.ipynb'.format(number, '_'.join(l.lower() for l in letters))
    path = os.path.join(ROOT, 'kaggle', name)
    with io.open(path, 'w', encoding='utf-8', newline='\n') as handle:
        json.dump(nb, handle, indent=1, ensure_ascii=False)
        handle.write('\n')
    print('wrote', path)
