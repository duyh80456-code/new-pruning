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
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

QUEUE = [
    ('af_all_pairs', 'ae_k_plus_f',
     'Couple every pair of students, and put K and F together'),
    ('ah_everything', 'ag_five_widths',
     'All of it at once, and more students to pair'),
    ('ai_channel', 'ak_bures',
     'Transport along the shared channels, and the Gaussian closed form'),
    ('ao_spread', 'ar_k_temp4',
     'Give the metric some geometry, and the teacher something to say'),
    ('m_all_stages', 'aq_classwise',
     'Transport at every depth, and between class positions'),
    ('al_bures_diag', 'am_sliced_max',
     'Variances without directions, and the worst projection'),
    ('aj_channel_p1', 'an_sliced_p1',
     'An absolute gap instead of a squared one, on both'),
    ('v_unbalanced', 's_feature_sliced',
     'Let mass go unmatched, and solve it the cheap way'),
    ('ac_weight_half', 'ad_weight_double',
     'Is K on a plateau or on a peak'),
    ('y_eps_050', 'w_eps_002',
     'Blurrier and nearly hard, the two ends of the axis'),
    ('x_eps_010', 'aa_cosine_ground',
     'One step of blur, and direction instead of distance'),
    ('t_sliced_32', 'u_sliced_512',
     'How many directions stand in for a plan'),
    ('as_log_widths', 'at_macs_widths',
     'Where the sandwich rule spends its free samples'),
    ('au_entropy_kd', 'av_confidence_kd',
     'Which samples the teacher still has something to say about'),
    ('aw_teacher_chain', 'ak_bures',
     'A chain of teachers, and the Gaussian form that never ran'),
    ('az_act_ground', 'ba_taylor_ground',
     'Charge the channels for what they carry'),
]

# ak_bures died in linalg.eigh three minutes in and the fix landed after,
# so it has never been tried. It is in two rows and only the second is
# meant to be run; keyed by row so the table says which.
RERUN = {(3, 'ak_bures'): 'moved to 15',
         (15, 'ak_bures'): '**rerun**'}

HELD = [
    ('ax_var_ground, ay_inverse_ground', 'the same weighting by batch '
     'variance, and its reversal. Variance and activation disagree about '
     'a channel that is large and constant, so they are separate '
     'questions; this pair is the follow-up if row 16 moves anything.'),
    ('k_seed2, a_seed3', 'seeds. Six branches now sit within a quarter '
     'point of each other at the top and sigma is still unmeasured, so '
     'these have stopped being defensive and started being the thing '
     'that decides the order.'),
    ('z_eps_100, q_feature_mse', 'controls for K. They are built to lose: '
     'blur the plan away, or forbid rematching, and see what is left. Y '
     'at eps 0.5 has since made most of z_eps_100\'s point for a quarter '
     'of the cost.'),
    ('r_feature_mmd', 'a control from outside transport.'),
    ('j_feature_gram', 'a control for whether nesting is what makes K work.'),
    ('ab_no_debias', 'a control for whether the debiasing correction is '
     'load bearing.'),
    ('ap_logit_spread', 'a test of the flatness diagnosis on branch C, '
     'which came last of twelve and is not going to reach K.'),
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

    def cell(number, branch):
        if branch in mean:
            return '{:.2f}'.format(mean[branch])
        return RERUN.get((number, branch), 'free')

    rows, free = [], 0
    for number, (left, right, asks) in enumerate(QUEUE, start=1):
        result = '{} / {}'.format(cell(number, left), cell(number, right))
        if 'free' in result:
            free += 1
        rows.append(
            '| {} | `kaggle_kd_{:02d}_{}_{}.ipynb` | `{}`, `{}` | {} | {} |'
            .format(number, number, left.split('_')[0],
                    right.split('_')[0], left, right, asks, result))

    out = [
        '# The queue',
        '',
        'Every notebook in this directory runs two branches, one per card,'
        ' in a',
        'session of roughly three to five hours. Take a row, run the file'
        ' as it',
        'is, and say which number you took.',
        '',
        'Rows 1 to 12 all ask the same question - can a different'
        ' transport term',
        'beat K - and the answer has settled at no. Rows 13 to 15 leave'
        ' that',
        'axis: they change where the sandwich rule samples, which samples'
        ' KD',
        'attends to, and which width teaches which. None of them touches'
        ' the',
        'transport term, so they compose with K rather than competing'
        ' with it.',
        '',
        'K stands at 74.26 against 73.52 for A, ahead at all sixteen'
        ' widths.',
        'That is the number to pass. The result column carries each'
        ' branch\'s',
        'mean accuracy once it has come back, so take a row with a free'
        ' slot.',
        '`ak_bures` crashed three minutes into its session on a numerical'
        ' bug',
        'that is now fixed, so it has never actually been tried. It is'
        ' queued',
        'in row 15, not row 3.',
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
            'These have configs in `apps/` and are not queued. Any of'
            ' them runs',
            'from a notebook by editing `BRANCHES`.', '']
    for names, why in HELD:
        out.append('* `{}` - {}'.format(names, why))
    out += ['', '## Dropped on evidence', '']
    for names, why in DROPPED:
        out.append('* `{}` - {}'.format(names, why))
    out += ['', MISTAKE.rstrip('\n')]

    path = os.path.join(ROOT, 'kaggle', 'QUEUE.md')
    with io.open(path, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write('\n'.join(out) + '\n')
    print('wrote {} rows, {} with a free slot'.format(len(QUEUE), free))
    return 0


if __name__ == '__main__':
    sys.exit(main())
