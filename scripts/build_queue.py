"""Write kaggle/QUEUE.md from the queue and the results store.

    python scripts/build_queue.py

Generated rather than edited, so running it twice is a no-op. An earlier
version patched the file in place, was run twice, and grew a second
result column.

This is also where DROPPED.md used to be. Two files said what is not
being run - one for branches held back, one for branches dropped - and
a reader had to find both to learn the same thing. They are one section
here.
"""
import ast
import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NOTEBOOK = re.compile(r'kaggle_kd_(\d+)_\w+\.ipynb$')
BRANCHES = re.compile(r"BRANCHES = (\[[^\]]*\])")


def notebooks():
    """(number, filename, branches, title) for every numbered notebook

    Read off the files rather than kept in a list beside them. The list
    this replaced numbered its rows by position, drifted from the files,
    and showed notebooks that had come back as free.
    """
    found = []
    folder = os.path.join(ROOT, 'kaggle')
    for name in sorted(os.listdir(folder)):
        match = NOTEBOOK.match(name)
        if not match:
            continue
        with io.open(os.path.join(folder, name), encoding='utf-8') as handle:
            cells = json.load(handle)['cells']
        header = ''.join(cells[0]['source']).split('\n')[0]
        title = header.split('. ', 1)[1] if '. ' in header else header
        config = ''.join(cells[1]['source'])
        branches = ast.literal_eval(BRANCHES.search(config).group(1))
        found.append((int(match.group(1)), name, branches, title))
    return found


HELD = [
    ('ax_var_ground, ay_inverse_ground', 'the same weighting by batch '
     'variance, and its reversal. Variance and activation disagree about '
     'a channel that is large and constant, so they are separate '
     'questions; ResNet-18, and the pair it followed up was never run.'),
    ('z_eps_100', 'a control built to lose: blur the plan away and see '
     'what is left. Y at eps 0.5 has since made most of its point for '
     'a quarter of the cost.'),
    ('j_feature_gram', 'a control for whether nesting is what makes K work.'),
    ('ap_logit_spread', 'a test of the flatness diagnosis on branch C, '
     'which came last of twelve and is not going to reach K.'),
    ('az_act_ground, ba_taylor_ground, bb_reorder_l1, bc_reorder_read, '
     'k_seed2, bj_chain_only, q_feature_mse, r_feature_mmd, bm_solo_full, '
     'bk_alpha_on_k, bl_gromov, ab_no_debias, bv_half_first_frozen, '
     'bw_narrow10_frozen', 'ResNet-18 branches whose notebooks (16, 17, 21 '
     'to 24, 29) were removed unrun. Every new run is on ResNet-50, so any '
     'of these that comes back comes back as a ResNet-50 twin of BX or BY '
     'in a new notebook at the end.'),
    ('cd_dsnet_r50, cf_scala_r50, cg_nasvit_r50', 'the published methods '
     'on ResNet-50, in the old notebooks 31 and 32 (removed unrun; the numbers now hold other runs). Needed for the '
     'paper once the ResNet-50 results settle.'),
    ('ch_a_heads4_r50, cj_a_heads2_r50, ck_a_heads8_r50, cl_a_heads16_r50',
     'band heads on A and the head-count sweep, notebooks 33 to 35, '
     'removed unrun. CI in notebook 31 asks first whether heads help K.'),
    ('cm_k_knots4_r50, cn_a_knots4_r50', 'BN scale and shift continuous in '
     'width, notebook 36, removed unrun.'),
    ('co_ce_only_r50', 'US-Net with KD off, the floor under CP; notebook '
     '37 removed unrun, and its other half, CP, is in notebook 31.'),
]

DROPPED = [
    ('n_jeffreys_narrow', 'L was the same reasoning applied to D. It '
     'projected to 73.58 and landed at 72.74, below the branch it was '
     'built from. Weighting a term by width does not keep the half of a '
     'per-width table that looked good.'),
    ('o_temp4, p_jeffreys_temp4', 'temperature was a diagnosis for six '
     'branches landing inside 0.8 points of each other. K then cleared A '
     'by 0.75 without it. AR has since tested the diagnosis directly on '
     'top of K and come out at 73.38, below A, so the remedy is ruled '
     'out at this tier rather than merely set aside.'),
]

MISTAKE = """## Two that were dropped and should not have been

`e_kl_pair` and `g_flat_cost` had already run. The reasoning for
dropping them was wrong anyway, and they turned out to be the two runs
that made the logit-tier story legible.

The mistake was to judge a control by its gap to A. E lands at 73.27, a
quarter point below A, which is why it looked like there was nothing
left to attribute. But a control is read against the branch it controls
for, not against the reference: E equals D exactly, and F sits 0.27
above both, which is what separates symmetry from the metric. Nothing
else in the set could have separated them.

G is the sharper case. It was dropped as a formality that would confirm
the classifier cost contributes nothing. It beats C by 0.51 instead: a
cost matrix with no geometry at all does better than the model's own.
The prediction was "contributes nothing" and the answer was "actively
hurts", which is a different claim and a stronger one.
"""


def main():
    with io.open(os.path.join(ROOT, 'results', 'runs.json'),
                 encoding='utf-8') as handle:
        store = json.load(handle)
    mean = {r['name']: sum(r['top1']) / len(r['top1'])
            for r in store['runs']}

    rows, free = [], 0
    for number, name, branches, title in notebooks():
        result = ' / '.join('{:.2f}'.format(mean[b]) if b in mean
                            else 'free' for b in branches)
        if 'free' in result:
            free += 1
        rows.append('| {} | `{}` | {} | {} | {} |'.format(
            number, name, ', '.join('`{}`'.format(b) for b in branches),
            title, result))

    out = [
        '# The queue',
        '',
        'Every notebook runs one branch per card. Rows up to 28 are'
        ' ResNet-18',
        'and have come back; from 30 on every run is ResNet-50, where A'
        ' takes',
        'about 10.7 hours on a T4 and K about 14, so a K branch needs a'
        ' second',
        'session. Take a row with a free slot, run the file as it is, and'
        ' say',
        'which number you took.',
        '',
        'On ResNet-50, K stands at 77.26 against 76.86 for A, ahead at 15'
        ' of 16',
        'widths, one seed each. Notebook 32 is the second seed.',
        '',
        '## How to run one',
        '',
        'Upload the `.ipynb` to Kaggle, then:',
        '',
        '1. Accelerator **GPU T4 x2**, Internet **On**',
        '2. Add the CIFAR-100 dataset as an input',
        '3. **Save Version -> Save & Run All (Commit)**, not the'
        ' interactive',
        '   run, which ends when the browser closes',
        '',
        'Nothing in a notebook needs editing. Each one clones this repo'
        ' when it',
        'starts, so it always picks up the current code.',
        '',
        'Do not skip the smoke step. The branch suite at the top mimics'
        ' the',
        'training loop rather than running it, so anything that lives'
        ' only in',
        '`train.py` is inert there; the smoke run is what exercises it.',
        '',
        '| # | notebook | branches | what it asks | result |',
        '|---|---|---|---|---|',
    ]
    out += rows
    out += ['', '## Held back', '',
            'These have configs in `apps/` and no notebook. To run one,'
            ' add a row',
            'for it at the end of QUEUE in scripts/build_notebooks.py.',
            '']
    for names, why in HELD:
        out.append('* `{}` - {}'.format(names, why))
    out += ['', '## Dropped on evidence', '']
    for names, why in DROPPED:
        out.append('* `{}` - {}'.format(names, why))
    out += ['', MISTAKE.rstrip('\n')]

    path = os.path.join(ROOT, 'kaggle', 'QUEUE.md')
    with io.open(path, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write('\n'.join(out) + '\n')
    print('wrote {} rows, {} with a free slot'.format(len(rows), free))
    return 0


if __name__ == '__main__':
    sys.exit(main())
