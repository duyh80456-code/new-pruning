"""Regenerate the Kaggle notebooks from the branch banners.

    python scripts/build_notebooks.py

Each notebook is a template plus two branch names, and its prose is
lifted from the config banners at generation time, so a notebook
cannot describe a branch the config no longer is. Edit the banner,
run this, and the notebook follows.

Twelve notebooks asked one question: can a different transport term beat
K. Every one of the 33 runs sits on that axis, and the answer has settled
at no - the five best are all K with one knob moved.

Only rows 13 to 15 are listed. Rows 1 to 12 have all come back, and
regenerating them would rewrite files whose results are recorded.
"""
import glob
import io
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAGGLE = os.path.join(ROOT, 'kaggle')
HEADER_BAR = re.compile(r'^#\s*=+\s*Branch\s+\w+\s*=+\s*$')

OFF_AXIS = """K stands at 74.26 against 73.52 for **A**, US-Net as published.
Thirty three runs have not passed it, and the five closest are all K with
one knob moved: a different blur, different marginals, a different thing
transported, more pairs. That axis has been answered.

These branches are not on it. None of them changes the transport term,
so they compose with K rather than competing with it, and they sit on
choices US-Net made without arguing for them.
"""

ON_AXIS = """Forty one runs. **AW** leads at 74.32 and **K** is at 74.26, which is a
tie rather than an order: AW is ahead at eight widths of sixteen and
behind at eight. **A**, US-Net as published, is 73.52.

These two do change the transport term, which the twelve notebooks
before them all failed to improve. They change something else about it.
Every one of those tried a different distance or a different solver and
kept charging all 512 channels the same. Measured at the pooled tap, at
the middle widths this term runs between, the top half of the channels
carries 81 to 87 per cent of the Taylor score and only 44 to 66 per cent
are effective at all. So the cost being minimised is mostly distance
along channels that carry nothing, and that is the part nothing has
touched yet.

There is no reversed control in this pair, and it is not needed here:
weighting every channel the same **is** K, which has run and sits at
74.26. The two branches are read against it.
"""

REORDER = """US-Net decides at initialisation which channels a narrow subnet gets.
`weight[:k]`, and k counts from index zero, and nothing revisits it.
Once-for-All does revisit it: when it makes width elastic it permutes
the channels so the prefix holds the ones a criterion ranks highest.

Read on a finished K the prefix is already 96 to 98 per cent of the
best quarter, by every criterion tried, because it runs at every
sampled width and at 0.25 has to classify with nothing beside it. On a
freshly initialised model it is 29 per cent, which is the arbitrary
ordering OFA has to fix.

So the question is not whether the end state is sorted. It is when it
gets that way, and epoch 10 is a guess: early enough that the criterion
is not pure noise, late enough that there may still be something to
move. Nothing measured pins it, because pinning it needs a checkpoint
from the middle of a run and every one to hand is finished.

So both branches also print `prefix_sorted <epoch> <fraction>` every
epoch, before and after the permutation. One run draws the whole
trajectory from 0.29 to wherever it ends, which is the reading that
says whether epoch 10 was anywhere near the window - and if it was not,
it says which epoch to use instead without spending another session
guessing.
"""

WARM = """Sorting the channels only pays inside a window: late enough that the
criterion is not noise on near-random weights, early enough that the
prefix is not already sorted and there is training left to use it.

US-Net has no such window, and that is the point of these. What makes a
criterion meaningful here is the prefix taking gradient at every width
and having to classify alone at 0.25 - and that is the same thing that
sorts it. On a finished K the prefix already is 96 to 98 per cent of
the best quarter; on a fresh model it is 29. The signal and the problem
grow together. Once-for-All escapes that by making width elastic last,
so its weights mature while the ordering stays arbitrary.

So these make the window. The narrow end is held back for the first
epochs and only width 1.0 trains, which costs **less** than a sandwich
step rather than more, so the branch spends less total compute than K,
not more. The total stays at 100 epochs: the warm-up comes out of the
budget, because every schedule in common use rescales with the total
and 110 epochs would be a different run rather than a longer one.

The narrow widths get fewer epochs, which is a handicap, and it makes
this a one sided test. A win is clean, because it won despite the
handicap. A loss says nothing.
"""

SETTLE = """Fifty runs, and neither of the two things this table most needs to
know has been measured. Both are measured here, and neither is a new
idea - that is the point of this notebook.

**Is 74.26 real.** Ten branches are K with one knob moved in ten
mechanically unrelated directions: the free widths drawn narrower and
wider, the feature weight halved and doubled, and six warm-up variants.
They land between 73.71 and 73.98, mean 73.89, and K sits alone at
74.26. Ten unrelated nudges do not all land below a branch by accident.
Either every knob was already at its best setting, or K drew a high
seed and this family is worth about 73.9. Nothing in the table
separates those, because sigma has never been measured on K. A second
seed does, and it is the same config with `random_seed: 2026`.

**Does the teacher chain do anything by itself.** AW leads at 74.32 and
the page has been describing it as a second mechanism reaching K from
elsewhere. That was wrong, and diffing the configs says so: AW is K
plus one line, `teacher_chain: True`, with the transport term running
inside it untouched. So chaining has only ever been read on top of
transport, where it is worth +0.06. The two by two is missing a cell -
A at 73.52, K at 74.26, AW at 74.32, and nothing for chaining alone.
BJ is that cell: A with the chain and no transport anywhere.

Near 73.5 says the transport term is the only active ingredient and the
report has one clean claim. Near 74.3 says chaining gets there on its
own and the two are redundant rather than additive, which is a
different paper. Either answer is worth more than another term added to
the loss: every addition to K so far has cost accuracy except the one
that did nothing.
"""

QUEUE = [
    # 13 to 15 have come back; regenerating them would rewrite files
    # whose results are already recorded. OFF_AXIS is kept because it is
    # what they say.
    (16, 'az_act_ground', 'ba_taylor_ground',
     'Charge the channels for what they carry', ON_AXIS),
    (17, 'bb_reorder_l1', 'bc_reorder_read',
     'Which channels the narrow subnet gets', REORDER),
    (18, 'bd_warm10_sort', 'be_warm25_sort',
     'Make a window, then sort inside it', WARM),
    (19, 'bf_warm10_plain', 'bg_warm25_plain',
     'What the warm-up is worth on its own', WARM),
    (20, 'bh_warm10_taylor', 'bi_warm25_taylor',
     'The same window, sorted by what removal would cost', WARM),
    (21, 'k_seed2', 'bj_chain_only',
     'The noise on K, and the chain without the transport', SETTLE),
]

HEADER = """# {number}. {title}

Two branches, one per card, in a single session of roughly three to five
hours. Nothing in this notebook needs editing: run it as it is.

{preamble}
"""

FOOTER = """
## Before you start

* Accelerator **GPU T4 x2**, Internet **On**
* Add the CIFAR-100 dataset as an input
* **Save Version -> Save & Run All (Commit)**, not the interactive run.
  An interactive session ends when the browser closes.

The checks at the top run on the CPU and stop the session in about two
minutes if anything is wrong, before a card is touched. Note that the
branch suite mimics the training loop rather than running it, so for
these branches the smoke step below is what actually exercises the new
code. Do not skip it.

If the session times out partway, attach its output to a new copy and
name the logs directory in `RESUME_FROM`. Every epoch writes a
checkpoint, so at most one is lost.

## When it finishes

Send back the final table. Pasting the output of the last cell is enough.
"""

CONFIG = """# Fixed for this notebook. Notebook {number} of 21.
BRANCHES = {branches!r}

SMOKE_FIRST = True

REPO_URL = 'https://github.com/duyh80456-code/new-pruning.git'
REPO_BRANCH = 'nhan'

CIFAR_DIR = ('/kaggle/input/datasets/nlnk1607/cifar100/cifar-100-python')

# To carry a timed-out session forward, attach its output and name the
# logs directory. Leave empty to start fresh.
RESUME_FROM = ''
"""


def banner(branch):
    lines = []
    with io.open(os.path.join(ROOT, 'apps',
                              'cifar100_{}.yml'.format(branch)),
                 encoding='utf-8') as handle:
        for line in handle:
            line = line.rstrip('\n')
            if line.startswith('# machine') or not line.startswith('#'):
                if lines:
                    break
                continue
            if HEADER_BAR.match(line) or set(line) <= set('#= '):
                continue
            lines.append(line.lstrip('#').strip())
    text, para = [], []
    for line in lines:
        if line:
            para.append(line)
        elif para:
            text.append(' '.join(para))
            para = []
    if para:
        text.append(' '.join(para))
    return '\n\n'.join(text)


def source(text):
    lines = text.split('\n')
    while lines and not lines[-1].strip():
        lines.pop()
    return [line + '\n' for line in lines[:-1]] + [lines[-1]]


template_path = sorted(glob.glob(
    os.path.join(KAGGLE, 'kaggle_kd_[0-9][0-9]_*.ipynb')))[0]
with io.open(template_path, encoding='utf-8') as handle:
    template = json.load(handle)

# The template is notebook 01, whose results are recorded, so the extra
# suite is added here at generation time rather than edited into it.
SUITES_WAS = "          'tests/test_kd_variants.py']"
# Notebook 01's run_pinned drops every line that starts with two spaces,
# which is the whole body of a traceback: when bd and be died on their
# first real epoch the log kept "Traceback (most recent call last):" and
# the exception and threw away the file and the line between them. Its
# results are recorded, so the fix goes in here rather than into it.
QUIET_WAS = chr(10).join([
    "    lines = queue.Queue()",
    "    procs = {}"])
QUIET_NOW = chr(10).join([
    "    lines = queue.Queue()",
    "    procs = {}",
    "    failing = set()"])

FILTER_WAS = chr(10).join([
    "        if quiet and (line.startswith(('  ', ')', 'Model(', 'Total', 'Item'))",
    "                      or not line.strip()):"])
FILTER_NOW = chr(10).join([
    "        # Once a branch starts printing a traceback, stop filtering",
    "        # it. The body is indented, so the rule below would keep the",
    "        # exception and throw away where it came from.",
    "        if 'Traceback (most recent call last)' in line:",
    "            failing.add(label)",
    "        if quiet and label not in failing and (",
    "                line.startswith(('  ', ')', 'Model(', 'Total', 'Item'))",
    "                or not line.strip()):"])

SUITES_NOW = chr(10).join([
    "          'tests/test_kd_variants.py',",
    "          'tests/test_channel_reorder.py']"])

for number, left, right, title, preamble in QUEUE:
    nb = json.loads(json.dumps(template))
    for cell in nb['cells']:
        body = ''.join(cell['source'])
        for old, new in ((SUITES_WAS, SUITES_NOW),
                         (QUIET_WAS, QUIET_NOW),
                         (FILTER_WAS, FILTER_NOW)):
            body = body.replace(old, new)
        cell['source'] = source(body) if body.strip() else cell['source']
    body = HEADER.format(number=number, title=title,
                         preamble=preamble)
    for branch in (left, right):
        body += '## `{}`\n\n{}\n\n'.format(branch, banner(branch))
    nb['cells'][0]['source'] = source(body + FOOTER)
    nb['cells'][1]['source'] = source(CONFIG.format(
        number=number, branches=[left, right]))
    name = 'kaggle_kd_{:02d}_{}_{}.ipynb'.format(
        number, left.split('_')[0], right.split('_')[0])
    with io.open(os.path.join(KAGGLE, name), 'w',
                 encoding='utf-8', newline='\n') as handle:
        json.dump(nb, handle, indent=1, ensure_ascii=False)
        handle.write('\n')
    print('{:32} {:18} {}'.format(name, left, right))
