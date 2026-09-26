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

CONTROLS = """Fifty runs and the transport term has never had a control.

K is A plus entropic transport between the two middle widths' features,
and it is worth +0.75 over A at all sixteen widths. Thirty three branches
have since been stacked on it and exactly one came out above, by 0.06.
Every one of those thirty three changed the loss, its weight, the
sampler, the schedule or the channel order. So the axis is exhausted
without anyone having asked the prior question.

These two ask it. Both sit exactly where K's term sits - the feature
tier, between the same two sampled widths - and replace only the
divergence. **Q** uses squared distance, which pairs sample i with sample
i and nothing else, so it asks whether the freedom to rematch samples is
what transport buys. **R** uses maximum mean discrepancy, a control from
outside the transport family altogether.

Read the result as a fork. If either lands near 74.26, then +0.75 was
never about transport - it was about coupling the two middle widths at
the feature tier at all, and thirty four variants of the coupling were
always going to say nothing. If both land near A, transport is the active
ingredient and the paper has one clean claim.

Both configs have been in apps/ since the transport sweep and have never
been run. feature_weight is 0.3 and 9.5, from
scripts/calibrate_feature_weight.py, so the three terms apply comparable
gradient rather than comparable loss - otherwise "this divergence is
worse" and "this term was ten times weaker" would be the same reading.
"""

DIAGNOSTIC = """Two branches that are not ideas. One is a control this project should
have run first, the other is prior art it has been missing.

**BM** is one ResNet-18 at width 1.00 and nothing else. All sixty five
configs here train every width, so the table cannot say what multi-width
training costs the widest width - and that number gates a whole family of
work. A's width 1.00 reads 75.30. If BM lands well above it, the hard
one-hot label at the widest is being drowned by the narrow widths' much
larger losses, and the fix belongs at the label. If BM lands near 75.30,
that family is closed. The literature disagrees with itself on the sign:
the US-Net paper has MobileNet v1 *better* multi-width than solo by 0.9
and v2 worse by 0.3, while SlimCLR's supervised control is 76.6 solo
against 76.0 slimmable. A reviewer asks for this number either way.

BM also tests all sixteen widths afterwards, so the row shows what a
model trained at one width does when it is sliced anyway - the other
thing nothing here has measured.

**BK** is K with AlphaNet's alpha-divergence in place of the soft cross
entropy. Plain KL is zero-avoiding, so a student made to match a teacher
it cannot represent over-estimates the teacher's uncertainty; the
alpha-divergence penalises over- and under-estimation both, and the paper
applies it to slimmable networks over exactly this width range. The loss
has been in loss_ops since branch B and has never been put underneath K.

B is that loss on A and came out at 73.08, below A - but B has no
transport term, and every divergence in this project has depended on
which tier it sits at: KL, Jeffreys and Wasserstein all fail at the logit
tier while Wasserstein at the feature tier is worth +0.75. So B does not
settle what alpha does under K. Prior art either way: if it helps, the
baseline should have had it and the comparison has to be redrawn against
K plus alpha.
"""

NOTRUNC = """Two branches about the assumption K makes and never states.

K compares the two widths' features after truncating the wide one to the
narrow width. At 0.25 that discards 384 of 512 wide channels and asserts
narrow channel i is wide channel i - which is false in function space,
because narrow channel i is computed from k input channels and wide
channel i from all of them. They are different functions sharing an
index.

**BL** drops the assumption. Gromov transport compares how far sample i
is from sample k inside one cloud against the same pair inside the other,
so no correspondence between channels is needed and no channel is thrown
away. FeatureGromovLoss has been in loss_ops since the transport sweep
and had never been run until this notebook was built.

Read it as a prediction, not as variant 35. Both widths see the same
batch, so the sample correspondence is already known and trivial. If the
solver returns the identity coupling, the objective collapses to a
distance between the two clouds' own distance matrices - a three-line
relation loss, not transport at all. If it returns anything else it is
matching image i to image j, which is wrong supervision. Either outcome
says something the prefix branch cannot.

One thing to state plainly: at feature_weight 1.0 this term contributes a
gradient of 0.0088 against entropic transport's 3.6502, so the first
version of the config was K with no horizontal term and the branch suite
scored it at a_kl to four decimals. It now runs at 413.8, from
scripts/calibrate_feature_weight.py. A term needing 414x amplification to
be felt is a weak signal, and that calibration is one measurement on an
untrained model, so the ratio drifts as the widths converge.

**AB** is the cheapest control K never had: the same transport with the
debiasing correction switched off. Entropic blur makes the objective
positive even between a cloud and itself, and a horizontal term has to be
minimised at agreement, so the correction should be load bearing. Nobody
has checked.
"""

WHOTEACHES = """These two are the first branches in this project that add something to K
without adding a term to the loss.

Thirty three branches have been stacked on K and exactly one came out
above it, by 0.06. Grouped by what they touched: ten changed the loss,
eleven the loss and its weight, two the sampler, two the schedule, four
the channel order, one the teacher. Nothing has ever touched the
optimizer, the teacher's composition, the label, or added a parameter.
So the loss is not where the room is.

**BN** changes who teaches. US-Net trains the widest width on the hard
label alone and makes it the sole teacher for every other width. Two
ablations from different fields put that near the bottom of the choices
they tried: EED, on ResNet-18 and CIFAR-100, has the main classifier at
78.31 taught by itself against 79.25 taught by the ensemble of all exits
plus itself - and the *weakest* exit contributed more than the strongest.
CoQuant ranks six teacher choices for multi-bit training and puts
always-the-highest fifth of six. Here every sampled width learns from the
mean of what all four said, the widest included, which is the part with
no precedent in this setting: in US-Net the widest is the only width that
never receives a soft target, and forward_loss had no way to hand back
its logits until this branch needed them. K's transport term runs
untouched underneath.

**BP** adds 128 parameters at the one place per-width BN recalibration
provably cannot reach. body and shortcut each end in a BatchNorm, so
their scales are pinned by gamma - but the tensor they are added into has
no BatchNorm after it, and gamma and beta are shared across all sixteen
widths, so the branch-to-skip ratio on the residual stream drifts with
width and nothing corrects it. REPAIR states that on conv outputs alone
it is "mathematically equivalent to resetting the BatchNorm statistics",
and gets its extra mileage specifically from correcting residual-block
outputs. NeFL makes one learnable step size per residual block private
per submodel and reports +0.95 mean and +3.34 worst case on CIFAR-100
ResNet-18. One scalar per block per tested width, init 1.0, so epoch zero
is exactly K.

BP prints `branch_scale <epoch> min/mean/max` every epoch. Those scalars
get no weight decay - every one-dimensional parameter here gets zero - so
nothing pulls them back toward 1.0 and a scalar drifting to zero turns
its block into a bare identity. If this branch fails, that line is where
it will be visible; send it back with the table.
"""

GRADSCALE = """Two branches about a learning rate nobody set.

Prefix slicing makes the sandwich rule hand out unequal learning along
the channel index, by counting alone. Output channel i receives a
gradient from every sampled width whose channel count exceeds i, so with
{1.00, 0.25, w1, w2} and w uniform on [0.25, 1.00] the expected number of
updates per step is 4.00 for the first quarter of channels, drops to 3.00
just above 0.25, and falls linearly to 1.00 at the last channel. Nothing
in the loss intends that, and no branch here has ever addressed it.

**BQ** divides each output channel's gradient by the square root of how
many of that step's widths wrote to it. Gradient Equilibrium does exactly
this correction on the depth axis - a parameter several exits traverse
accumulates more terms, so divide by the count - and it has never been
done on the channel axis.

The exponent is the experiment, not the fix, because the literature
disagrees with itself. Weight decay plus BatchNorm scale-invariance says
the per-channel angular update converges to a value independent of
gradient magnitude, which would cancel this without help - but relaxation
takes about 1/(eta*lambda) steps, here roughly fifty epochs of a hundred
with cosine decay shrinking the target the whole way, so any cancelling
is partial. Against that, the one convergence theory covering
nested-mask training requires no per-coordinate rescaling at all. q = 0.5
is the midpoint; q = 0 is this branch's own control and is K exactly.

**BO** is US-Net's own Appendix A, which proposes dividing each conv
output by how many input channels are live, measures a slight gain on
US-MobileNet v1, and does not adopt it by default. The paper also notes
the constants "come for free since these constants can be merged into BN
statistics after training" - and that sentence is why this is on the same
notebook as BQ rather than with the normalization work. Every conv here
is followed by BatchNorm, and a per-output-channel constant in front of
BN is removed exactly by the per-width running statistics this project
already recalibrates, so the scale is invisible in the forward pass at
every point in training. What is left is the gradient: BN makes the
preceding weights scale-invariant, so a width-dependent constant in front
of it acts as a per-width effective learning rate. That reading is not in
the paper; it is the only mechanism left once the forward pass is ruled
out.

The flag was already in slimmable_ops, inherited from upstream, reading
an attribute USConv2d does not have - so switching it on raised
AttributeError and nothing in this project had ever run it. Fixed and
verified to give exactly C/k.

One warning that applies to both, and to BN and BP on the notebook
before. The branch suite cannot see any of these four: BQ and BO change
gradients and scales that BatchNorm absorbs, BP is identical to K at
initialisation, and the suite mimics the training loop rather than
running it, so it never reaches the ensemble term. All four scored 22.6405
or 22.6406 against K's 22.6405. Each was verified instead by running the
real train.py on one card and by a direct check - 21 tensors' gradients
changed with a largest divisor of exactly 2.000 for BQ, a measured ratio
of exactly C/k for BO, 32 scalars all moved off 1.0 for BP, and a printed
ens_loss for BN. The suite passing is not the evidence here.
"""

LONGER = """This notebook changes one number and nothing else: 100 epochs becomes
300, for **A** and for **K**. It is the first time this project has
questioned its own training budget.

Two reasons, and the second is the one that makes it more than a
formality.

**100 epochs is short for CIFAR-100.** CRD trains at 240, the standard
pytorch-cifar100 ResNet-18 baseline at 200. The protocol gap is not
small: WRN-40-1 scores 71.98 at 240 epochs against 69.11 at 200, the same
architecture either way. Every number in this table was read at 100.

**And a slimmable network dilutes its budget in a way a single network
does not.** make_divisible with divisor 8 collapses the uniform draw on
[0.25, 1.00] to exactly **90 distinct networks** - not a continuum, ninety
- and the sandwich rule activates four of them per step. So an
intermediate width sees on the order of 2/(90-2) of the total, about
**2.3 epochs out of 100**. Tripling the budget triples that.

Scala prices this axis directly: dropping from 13 granularities to 4 at a
fixed 300 epochs buys +1.6 at width 0.50 and +0.7 at 0.75. They stop at
13; this project sits at 90. HydraViT reports the same shape from the
other side - their many-subnet penalty is 0.7 points at 300 epochs and
gone by 800.

**Read the result against the 100-epoch rows, not against each other.**
If A and K both gain about the same, the budget was simply short and every
delta this project has measured survives with a better baseline underneath
it. If K's margin over A shrinks, then part of that +0.75 was K
*converging faster* rather than converging better - a different claim, and
one the current table cannot distinguish. Either answer changes what the
paper says.

## This one will not fit in a single session

Projected from the 100-epoch runs: **A about 8 hours, K about 15 hours.**
Both exceed one session, K by a factor of three.

Every epoch writes a checkpoint, so use the resume path rather than
starting over: when the session times out, attach its output to a fresh
copy of this notebook and put the logs directory in `RESUME_FROM`. Expect
roughly two rounds for A and three for K. The two branches run on
separate cards, so the rounds happen in parallel and the pair needs about
three sessions in total, not five.

If that is more than you want to spend, say so before starting - 200
epochs matches the standard CIFAR-100 protocol, halves K to about two
rounds, and still doubles the per-width budget. It is a weaker test of the
granularity hypothesis but a real one.
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
    (22, 'q_feature_mse', 'r_feature_mmd',
     'What the transport term actually bought', CONTROLS),
    (23, 'bm_solo_full', 'bk_alpha_on_k',
     'The control never run, and the loss never tried under K', DIAGNOSTIC),
    (24, 'bl_gromov', 'ab_no_debias',
     'Comparing the widths without truncating either', NOTRUNC),
    (25, 'bn_ensemble', 'bp_width_scalars',
     'Who teaches, and the one place recalibration cannot reach', WHOTEACHES),
    (26, 'bq_equalize_half', 'bo_conv_averaged',
     'The learning rate nobody set', GRADSCALE),
    (27, 'br_a300', 'bs_k300',
     'The same two branches, three times the schedule', LONGER),
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

CONFIG = """# Fixed for this notebook. Notebook {number} of 27.
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
