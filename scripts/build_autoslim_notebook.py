"""kaggle/kaggle_kd_121_autoslim.ipynb from notebook 93

Notebook 93 (SOLAR as its authors train it) already runs s-net branches
of sixteen widths a step, one per card, over several sessions with
resume, and reads the last epoch's validation. AutoSlim's step 3 is the
same kind of run, so this copies it and changes the branches and the
introduction. The searched configurations are in the repository
(results/autoslim_search_r50.json), so the notebook needs no search.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, 'kaggle', 'kaggle_kd_93_he_hf.ipynb')
TARGET = os.path.join(ROOT, 'kaggle', 'kaggle_kd_121_autoslim.ipynb')
BRANCHES = ['kt_autoslim_r50_seed2026', 'ku_autoslim_r50_seed2006']

INTRO = '''# 121. AutoSlim, the searched configurations trained as one slimmable network

Two branches, one per card, over several sessions - see below for how to carry it
forward. Nothing in this notebook needs editing: run it as it is.

AutoSlim (Yu & Huang, arXiv 1903.11728) searches a channel count for every
layer at each computational budget, then trains the searched networks as
one slimmable network. Steps 1 and 2, the layer-wise supernet and the greedy
search, were run once (scripts/autoslim_search.py, seed 1995); their output,
one configuration at the MACs of each of our sixteen uniform widths, is
results/autoslim_search_r50.json in the repository. This notebook is step 3,
as the authors' code trains it: a private BN per configuration, one
classifier sliced to each configuration's feature size, all sixteen
configurations on every batch, the narrower ones distilled from the widest.
The recipe is the one every other row uses.

| | |
|---|---|
| KT | ResNet-50 on CIFAR-100, seed 2026 |
| KU | ResNet-50 on CIFAR-100, seed 2006 |

KS, seed 1995, runs on the lab machine. No BN calibration: every
configuration has its own BN, so the last epoch's validation is the result.

## Timing

Sixteen configurations a step instead of four: about 3.5 times a US-Net run
of the same setting, roughly 40 hours on a T4, so several sessions. When a
session ends, Save Version, attach that version's output (pick the version
number, not Latest) and run again: the notebook finds the checkpoints. A
session on another account works the same way once the output it attaches
is shared with that account. Attach the CIFAR-100 dataset as well.

The A and K columns of the last table are the ResNet-50 CIFAR-100
references at seed 1995, for orientation only.

## `kt_autoslim_r50_seed2026`

See apps/cifar100_kt_autoslim_r50_seed2026.yml for every setting and why.

## `ku_autoslim_r50_seed2006`

See apps/cifar100_ku_autoslim_r50_seed2006.yml for every setting and why.

'''


def main():
    with open(SOURCE, encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']

    intro = ''.join(cells[0]['source'])
    keep = intro[intro.index('## Before you start'):]
    cells[0]['source'] = (INTRO + keep).splitlines(True)

    code = ''.join(cells[1]['source'])
    code = code.replace('Notebook 93.', 'Notebook 121.')
    old = "BRANCHES = ['he_solar_native_r50', 'hf_solar_native_r50_seed2']"
    if old not in code:
        raise SystemExit('notebook 93 changed: BRANCHES line not found')
    code = code.replace(old, 'BRANCHES = {!r}'.format(BRANCHES))
    cells[1]['source'] = code.splitlines(True)

    for cell in cells:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None

    with open(TARGET, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print('wrote', TARGET)


if __name__ == '__main__':
    main()
