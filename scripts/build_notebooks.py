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
"""

QUEUE = [
    # 13 to 15 have come back; regenerating them would rewrite files
    # whose results are already recorded. OFF_AXIS is kept because it is
    # what they say.
    (16, 'ax_var_ground', 'ay_inverse_ground',
     'Charge the channels for what they carry', ON_AXIS),
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

CONFIG = """# Fixed for this notebook. Notebook {number} of 16.
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

for number, left, right, title, preamble in QUEUE:
    nb = json.loads(json.dumps(template))
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
