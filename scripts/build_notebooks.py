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

OTHERWAY = """The curriculum axis has one measured direction and it lost.

Six branches here train width 1.00 alone for the first 10 or 25 epochs
and then admit the narrow end. BF, BG, BD, BE, BH and BI average **-0.38
against K**. Large-first is answered.

This notebook is the other direction: **width 0.25 alone for the first 25
epochs**, then the full sandwich. The argument is about which subnet ends
up coherent. GrowTAS reports that a large subnet extended from a trained
small one inherits the small one's layer-wise structure, while a small one
cropped out of a trained large one shows "significantly lower cosine
similarity in the deeper layers" - cropping discards dependencies the
depth had learned. If that holds, this project has had the order
backwards for 51 runs.

**BU is the curriculum alone.** Nothing is frozen; the narrow end simply
goes first.

**BT adds the freeze.** After epoch 25 the gradient is zeroed over the
block the width-0.25 subnet owns - conv and linear weights, and the BN
scale and bias over the same channels, because otherwise the narrow
subnet keeps drifting one affine parameter at a time and the branch would
not test what it claims.

The freeze is lighter than it sounds: that block is **710,576 of
11,210,432 weights, 6.3 per cent**. The wider widths keep the rest. And
the frozen part is exactly the prefix every width reads, which is where
the widths disagree.

Read the pair against **BF and BG**, not against K alone. Those are the
same intervention pointed the other way, so the four rows together are
the curriculum axis rather than two more branches on the pile.

## Two things to be honest about

**The literature points the other way on who needs protecting.** Thirteen
measurements across representations, LLMs, ViTs, CNNs and quantization say
the small end is *rescued* by weight sharing and the large end pays for
it. If that is right here, settling the narrow end first protects
something that is not broken while constraining the end that is. BU
against BT is what separates "the order helped" from "the freezing
helped", and either could come out negative.

**This pair found two latent bugs, and both were unreachable until now.**
Every sampler in this project has returned the widest width first or
alone, so `chain_target` was always set by the time the narrower widths
read it. Returning [low] alone raised `UnboundLocalError` on the first
step of the real loop, and then the same assumption failed a second time
inside the branch suite's mimic. Both now fall back to the hard label when
no wider width has run, which is the correct semantics rather than a
patch. The suite reads 4.5859 for these two against K's 22.6405, so
unlike the last four branches it does see them.
"""

SETTLE_HOW_MUCH = """Notebook 28 asks whether the narrow end should go first. These two ask
how much of the network that phase should settle, and for how long.

The axis now has five rows. **BF** and **BG** train width 1.00 alone for
10 and 25 epochs and then admit the narrow end: -0.36 and -0.49 against
K, so large-first is answered. **BU** and **BT** are the same
intervention pointed the other way, without and with a freeze at 0.25.
These two move the freeze itself.

**BV settles half the network.** Width 0.50 alone for 25 epochs, then that
block is held still: 2,815,840 weights, 25.1 per cent, against BT's
710,576 at 6.3 per cent. Four times as much. If the narrow-first order
helps at all, this says whether it helps in proportion to how much is
settled before the rest may move, or whether 6.3 per cent was already the
whole effect.

The curriculum width and the freeze width have to match, which is why
this branch also changes what the first phase trains. Freezing 0.50 after
a phase that ran 0.25 alone would lock the channels between them at
their initialisation, and the row would be measuring a partly random
network rather than a settled one.

**BW shortens the phase.** Width 0.25 alone for 10 epochs rather than 25,
then frozen. On the large-first side this knob was worth almost nothing -
BF at 10 epochs reads 73.90 and BG at 25 reads 73.78, a spread of 0.13 -
but nothing there was ever held still. Under a freeze a phase that ends
too early locks a prefix that was not finished, which is a failure mode
the large-first branches cannot have.

## Same counterweight, and a third bug

Thirteen measurements in the literature say the small end is rescued by
weight sharing and the large end pays for it. If that holds here, every
row on this axis is protecting the half that was never in trouble, and
the honest outcome is that all four small-first branches land below K.

And this idea has now found three latent bugs, none of them reachable
before a sampler returned something other than the widest width. Two were
on notebook 28: `chain_target` read before assignment in the real loop,
then the same assumption inside the branch suite's mimic. The third is
BV's: the suite guarded its pair term with "mid_widths is not empty" and
then indexed `mids[0]` and `mids[1]` by hand, which is fine for every
sampler that came before and an IndexError for a phase that settles one
middle width. train.py asks `horizontal_pairs()` and gets an empty list in
that case, so only the mimic was wrong. It now checks for two.

The suite reads 4.5719 for BV and 4.5859 for BW against K's 22.6405, so
it does see both of these.
"""

DEEPER = """The same two branches, **A** and **K**, on a network with twice the
capacity. Everything else - 100 epochs, batch 256, learning rate 0.2,
sixteen widths, seed - is what every ResNet-18 row used.

**Why.** ResNet-18 is 11.2M parameters asked to be ninety different
networks at once, and the nesting tax this project measured lands at the
wide end: the full network is the one held back by having to contain all
the others. More capacity is the direct test of whether that is a
capacity limit. The slimmable line reports ResNet-50 on ImageNet at 100
epochs, the same budget as here.

**What changed in the code.** `depth: 50` swaps the basic block for the
bottleneck, [3, 4, 6, 3], 23.7M parameters, a 2048-wide final feature.
`depth` defaults to 18, and the ResNet-18 state dict is unchanged key for
key, so no earlier branch builds anything different.

K needed one more change to fit. It kept all four widths' graphs alive
until the end of the step, which on ResNet-50 at batch 256 is 19.3 GB, and
a T4 has 15. Only the pair term needs two widths at once, and its gradient
splits exactly into one part through each side, so under
`split_pair_backward` each width takes its part with the partner detached
and is freed. Checked through train.py's own loop on a fixed batch: the
gradient matches the old path to a relative 4e-8 on both depths, a
deliberately broken split reads 2e-1, and the peak falls to 6.7 GB. It
costs one forward without a graph per step. The flag is on in BY only.

**How to read it.** BY against BX, not against the ResNet-18 rows - the
"vs A" column in the final table is against ResNet-18's A, which is a
different network. Two questions: does K's lead over A at 100 epochs
survive the larger network, and does its NLL lead at every width.

## K will need a second session

The first session measured it on two T4s: **A took 643 minutes, about
10.7 hours, and K runs at about 8 minutes an epoch**, about 14 hours in
all. K reached epoch 77 when the session ended. Resume it exactly as in
notebook 27: attach that session's output, set `RESUME_FROM`, run again.
About 23 epochs of K are left, roughly three and a half hours. A's
checkpoint is already at its last epoch, so it will calibrate, print its
table and exit in a minute.
"""

PUBLISHED = """Every branch in this table so far is this project's own idea. A paper
needs the other column too: what the methods other people published in
the US-Net line score **on this protocol**, not what they reported on
theirs. **ResNet-50**, CIFAR-100, 100 epochs, batch 256, sixteen widths
read after BN post-statistics, seed 1995 - exactly what BX and BY, A and
K on ResNet-50 in notebook 30, were run under.

**Notebook 30 has to come back first.** These three are read against BX
and BY, and without them there is nothing on ResNet-50 to read them
against. The ResNet-18 rows, A and K included, are a different network.

Three methods across notebooks 31 and 32, each implemented from the
authors' **released code** rather than from the paper alone. Where the
two disagree the code wins, and each config lists what had to change to
fit a width-only ResNet on CIFAR:

| | method | from | what it adds to US-Net |
|---|---|---|---|
| CD | DS-Net | CVPR 2021, changlin31/DS-Net | students learn from an EMA of the network, decay 0.997 |
| CF | Scala | NeurIPS 2024, BeSpontaneous/Scala-pytorch | narrowest width on the last channels, stable sampling, progressive KD, label on every student, 10 warm-up epochs |
| CG | NASViT | ICLR 2022, facebookresearch/NASViT | AlphaNet's loss plus a boost to subnet gradients that conflict with the widest, from epoch 25 |

**Only peer-reviewed work, and only width.** Every method here was
accepted at the venue named, checked against the arXiv comment or
journal field or the authors' own repository. SortedNet and Slimmable
Pruned Networks are left out because they are still preprints. MutualNet
is left out because it trains every width at a random input resolution,
and this table holds the input fixed at 32. NASViT's own supernet also
samples resolution; only its gradient-conflict rule is taken here, at
32. Joslim (non-uniform widths), SlimCLR and US3L (self-supervised),
MatFormer (Transformer FFNs) and the NAS supernets are accepted but do
not measure what this table measures. AlphaNet (**B**) and IPKD-TA
(**BJ**) exist only on ResNet-18.

**What was checked before this reached you.** Every earlier branch still
gives a bit-identical gradient on a fixed batch, old code against new.
Each flag was shown to act inside train.py's own loop: DS-Net adds one
gradient-free forward of the EMA network, Scala trains only the widest
during warm-up and 1.00, 0.95, 0.50, 0.25 afterwards, and NASViT matches
B to 3e-8 before epoch 25 and departs from it after. NASViT's merge
matches the Constraint class from their repository to 5e-7 over 150
random tensors. Each config differs from its ResNet-18 twin only in
`depth: 50` and its log directory.

**Memory.** Peak allocated at batch 256 on ResNet-50, measured on a
16 GB card with Scala's warm-up and NASViT's conflict start moved to
epoch 0 so the expensive path is the one measured: DS-Net 9.55 GB,
Scala 9.38 GB, NASViT 9.55 GB, against 9.47 GB for A. A T4
has 15 GB.

**How to read it.** Against BX and BY. A method that beats BX but not BY
means K is ahead of published work on this protocol; one that beats BY
is the new reference. One seed each, so a gap under about 0.3 to either
says nothing on its own.

**Time.** Notebook 30 measured A on ResNet-50 at 10.7 hours on a T4.
Scala and NASViT cost about what A does. DS-Net adds one gradient-free
forward of the widest width every step and will finish close to the
12-hour limit or past it; if the session ends first, resume as in
notebook 30 - attach the output, set `RESUME_FROM`, run again. Notebook
32 has one method and leaves the second card idle.
"""

HEADS = """A and K on ResNet-50 again, each with **one classifier per band of
widths** instead of one classifier for every width. Everything else is BX
and BY from notebook 30, so **notebook 30 has to come back first**: CH is
read against BX and CI against BY.

**Where it comes from.** SOLAR (WACV 2026) gives every subnet of a
once-for-all network its own classifier over the shared backbone and
reports better accuracy and better calibration: the subnets stop fighting
over one set of logit weights. Its subnets are a fixed eight. Here the
width is continuous, so one head per width is impossible. `head_groups: 4`
cuts 0.25 to 1.00 into four equal bands - at the test widths 0.25-0.40,
0.45-0.60, 0.65-0.80, 0.85-1.00, four to a band - and each band has its
own head. The network still runs at any width. This is an adaptation of
SOLAR, not SOLAR, and should be written up that way.

**What else it touches.** Nothing. The widest band's head keeps the name
`classifier`, so the teacher, the class cost matrix and the state dict of
every earlier branch are unchanged; `head_groups` defaults to 1, which
builds the model it always did. The three extra heads add 0.6M parameters
to 23.7M. The two middle heads train only on steps where a free width
lands in their band, about 44 per cent of steps each.

**What was checked.** tests/test_head_groups.py: the sixteen test widths
fall four to a band, a backward at 0.25, 0.50, 0.70 and 1.00 reaches its
own head and no other, width 0.25 reads a different head from 1.00, and
the class cost matrix still reads the widest head. With the head selection
broken on purpose it fails. Peak memory at batch 256: CH 9.47 GB, CI
9.38 GB, both under the 15 GB of a T4.

**How to read it.** Two questions. Does a head per band help A - if so the
shared classifier was a bottleneck. And does it add to K, or does K's
transport already relieve the same thing. Look at NLL as well as top-1:
SOLAR's claim is calibration, and so is K's clearest one.

## K will need a second session

Notebook 30 measured A at 10.7 hours and K at about 8 minutes an epoch,
about 14 hours, so CH finishes in one session and CI will not. When the session ends, attach
its output, set `RESUME_FROM` and run again; CH will calibrate, print its
table and exit in a minute.
"""

HEAD_SWEEP = """How many classifier heads. Notebook 33 splits the width range into
four bands with a head each; these split it into **2, 8 and 16**, on A
only, so that with BX (one head) and CH (four) the number of heads is
swept at 1, 2, 4, 8, 16. **Notebook 30 has to come back first**, and
notebook 33 is the one that says whether heads help at all.

| | heads | test widths per head | a middle head trains on |
|---|---|---|---|
| BX | 1 | 16 | every step |
| CJ | 2 | 8 | every step |
| CH | 4 | 4 | about 44% of steps |
| CK | 8 | 2 | about 23% of steps |
| CL | 16 | 1 | about 12% of steps |

The trade is in the last two columns. More heads let each band of widths
set its own logit weights, and give each head fewer updates, because the
sandwich rule runs only two free widths a step. At 16 every test width
has its own head and a middle one is trained one step in eight - if that
starves it, CL will show it at the middle widths first. The network runs
at any width at every setting; only the bands move.

Why on A and not K: A is half the time of K, and the sweep only has to
find the number. The best one goes on K afterwards; CI in notebook 33 is
already K with four.

Each branch is A with one flag changed, so each takes about what A did
in notebook 30, 10.7 hours, inside one session. Notebook 35 runs CL
alone and leaves the second card idle.
"""

KNOTS = """K and A on ResNet-50 with **every BN scale and shift a continuous
function of the width**. The candidate for a second contribution next to
the transport term. **Notebook 30 has to come back first**: CM is read
against BY and CN against BX.

**Why.** US-Net shares one gamma and one beta across every width and
recalibrates only the running statistics after training. Whatever a width
needs from the affine half of BN, it cannot have. BP gave each of the
sixteen test widths one private scalar per residual branch, snapped to the
nearest slot, and is the top of the ResNet-18 table at 74.35 - 0.09 over K,
inside the noise, but the only branch above it.

**What.** `affine_knots: 4` puts an offset to gamma and to beta at widths
0.25, 0.50, 0.75 and 1.00 and interpolates linearly in between. Every width
in the range has its own affine, neighbouring widths have nearly the same
one, and nothing is snapped to a slot, so the network stays continuous in
width. It covers BP, since scaling the last BN of a residual body scales
the body. 0.21M parameters on 23.7M. The offsets start at zero, so epoch
zero is BX or BY exactly, and they stay out of weight decay like gamma and
beta.

**What was checked.** tests/test_affine_knots.py: the knot shares sum to
one and move continuously with the width; with zero offsets every width is
the plain model bit for bit; a backward at 0.30, 0.70, 0.80 and 1.00
reaches exactly the two knots around it; an offset at the first knot
changes 0.25 and 0.30 and leaves 0.50 and 1.00 exactly alone; the offsets
are out of weight decay; a channel permutation carries them along. Broken
on purpose in two places, it fails in both. Every earlier branch still
steps to the loss it did. Peak memory at batch 256: CM 9.38 GB, CN
9.46 GB, under the 15 GB of a T4.

**How to read it.** If CM beats BY at most widths and on NLL, this goes in
the paper next to K. If CN gains as much as CM, the effect does not need
the transport. One seed each: a gap under about 0.3 says nothing, and a
positive one earns a second seed before it is written up.

## K will need a second session

Notebook 30 measured A at 10.7 hours and K at about 14: CN finishes in
one session, CM will not. When the session
ends, attach its output, set `RESUME_FROM` and run again.
"""

NOKD = """**Can the Wasserstein terms stand without KD.** Every transport branch
in this project so far added to the logit KD that US-Net already has; the
code refused to run one without it. These two take it out, on ResNet-50,
read against BX and BY from notebook 30.

| | students learn from | ties the widths together |
|---|---|---|
| BX | the widest width, by KL | KL |
| CO | the label | nothing but the shared weights |
| BY | the widest width, by KL | KL and two feature transport terms |
| CP | the label | the two feature transport terms alone |

**CO** is US-Net with inplace distillation off - the floor. **CP** is K
with `student_kd_weight: 0` and `student_ce_weight: 1`: the students take
the label, the teacher still runs because the feature terms read it, and
the only link between widths is Wasserstein on the features. CP against
CO is the transport on its own; CP against BY is what the KL is worth
inside K. The nearest thing already measured is C, Wasserstein replacing
KL on the logits, which lost 0.8 to A on ResNet-18.

**What was checked.** tests/test_no_kd.py: at weight 0 a student loss is
the label cross entropy exactly and its gradient does not move when the
teacher changes; at the default it is the KD loss it always was. The
branch suite now applies both student weights the way train.py does - it
used to ignore student_ce_weight, which is why CF (Scala) reads 30.43
there now and 15.24 before; train.py itself did not change for CF.
Peak memory at batch 256: CP 9.38 GB; CO runs the same graph as A,
which measured 9.47 GB. Both under the 15 GB of a T4.

CO costs what A does, about 10.7 hours. CP costs what K does, about 14,
so it needs a second session: attach the output, set `RESUME_FROM`, run
again.
"""

NOKD_AND_HEADS = """Two changes to K on ResNet-50, one per card, each read against BY
(K on ResNet-50, notebook 30: **77.26** mean top-1, BX 76.86).

| | what changes from BY | the question |
|---|---|---|
| CP | the logit KD is taken out: students learn from the label, and the two Wasserstein terms on the features are the only link between widths | can the transport stand without KL |
| CI | the width range is cut into four bands, each with its own classifier over the shared backbone | do band heads add to the transport |

**CP.** `student_kd_weight: 0`, `student_ce_weight: 1`. The teacher still
runs, because the feature terms read it. Every earlier transport branch
added to KL; this is the first that replaces it. The closest thing
measured before, C, replaced KL with Wasserstein on the logits and lost
0.8 to A on ResNet-18. CP differs by where the transport sits: on the
features, where K's gain came from. Against BY it says what the KL is
worth inside K; against BX, whether transport alone beats US-Net with its
KD. tests/test_no_kd.py checks that at weight 0 a student loss is the
label cross entropy exactly and its gradient ignores the teacher.

**CI.** `head_groups: 4`: bands 0.25-0.40, 0.45-0.60, 0.65-0.80,
0.85-1.00 at the test widths, four test widths to a band, each band with
its own classifier. The network still runs at any width. Adapted from
SOLAR (WACV 2026), which gives each of a fixed set of subnets its own
head. The widest band keeps the name `classifier`, so the teacher and
the class cost matrix are unchanged; 0.6M extra parameters on 23.7M.
BY's gain sat at the middle widths, 0.35-0.70, which are the two middle
bands - the heads either add there or compete with the transport for the
same thing. tests/test_head_groups.py checks each width's backward
reaches its own head and no other.

Peak memory at batch 256: CP 9.38 GB, CI 9.38 GB, under the 15 GB of a
T4. Both smoke configs ran through train.py on a local GPU.

## Both need a second session

Both cost what K does: about 8 minutes an epoch on a T4, about 14 hours.
When the session ends, attach its output, set `RESUME_FROM` to the logs
directory inside it, and run again; both branches resume from their own
checkpoints. Notebook 30 reached epoch 77 of K in the first session, so
expect the resumed session to need about three and a half hours.
"""

SEED2_R50 = """A and K on ResNet-50 again, at **seed 2026** instead of 1995 - the
seed the ResNet-18 repeats used. Nothing else differs from BX and BY.

**Why this before anything new.** On ResNet-50, K beat A 77.26 against
76.86: +0.40 on the mean, ahead at 15 of 16 widths. That is one run of
each. The gap is about the size of the seed noise measured on ResNet-18,
so as it stands it is a single draw. A second pair turns it into a gap
with a spread, which is the first thing a reviewer asks for, and which
every other ResNet-50 comparison in this queue is read against.

**How to read it.** CR against CQ first: does K still lead, and at the
same middle widths (0.35-0.70) where BY led. Then all four together:
the mean gap over two seeds and how far the two draws sit apart.

## K will need a second session

As in notebook 30, where A took 10.7 hours and K about 14: CQ finishes in
one session, CR will not. When the session ends, attach its output, set
`RESUME_FROM` to the logs directory inside it, and run again; about three
and a half hours are left.
"""

PRE_RELU = """K again on ResNet-50, with one change: both feature transport terms
read the last block **before** its closing ReLU (`feature_pre_relu: True`).

**Why.** CR, K at seed 2026 in notebook 32, died. Opening its checkpoint:
every width from 0.25 to 0.60 gave an all-zero pooled feature and
constant logits, and the final BN shift on the channels the transport
reads had gone to -1.44 against -0.46 on the rest. After the ReLU those
channels are zero for every input. Two all-zero clouds transport to each
other at no cost, and the dead ReLU passes no gradient back, so K had a
trivial minimum it could fall into and not leave. A has no such term,
which is why CQ at the same seed trained normally. Before the ReLU the
same channels still vary with the input: on CR's own weights the
transport gradient into the last block goes from 0 to about 37 at width
0.25 with the flag on. tests/test_pre_relu.py rebuilds the dead state and
checks the flag gives it a gradient; the logits do not depend on it.

| | seed | read against |
|---|---|---|
| CT | 1995 | BY 77.26: what the change costs where nothing broke |
| CU | 2026 | CR (died) and CQ 76.46: whether it holds where K broke |

**Watch in the log:** width 0.25 must leave loss 4.605 (= ln 100) within
the first two epochs. If CU sits there through epoch 2, stop it.

## Both need a second session

About 8 minutes an epoch on a T4, 14 hours for each. When the session
ends, attach its output, set `RESUME_FROM` to the logs directory inside
it, and run again; about three hours are left. Attach the CIFAR-100
dataset as well, or 34 minutes go on downloading it.
"""

MULTI_SESSION = {27, 30, 31, 32, 33}

QUEUE = [
    # 13 to 15 have come back; regenerating them would rewrite files
    # whose results are already recorded. OFF_AXIS is kept because it is
    # what they say.
    (18, 'bd_warm10_sort', 'be_warm25_sort',
     'Make a window, then sort inside it', WARM),
    (19, 'bf_warm10_plain', 'bg_warm25_plain',
     'What the warm-up is worth on its own', WARM),
    (20, 'bh_warm10_taylor', 'bi_warm25_taylor',
     'The same window, sorted by what removal would cost', WARM),
    (25, 'bn_ensemble', 'bp_width_scalars',
     'Who teaches, and the one place recalibration cannot reach', WHOTEACHES),
    (26, 'bq_equalize_half', 'bo_conv_averaged',
     'The learning rate nobody set', GRADSCALE),
    (27, 'br_a300', 'bs_k300',
     'The same two branches, three times the schedule', LONGER),
    (28, 'bu_narrow_first', 'bt_narrow_first_frozen',
     'The narrow end first, and then held still', OTHERWAY),
    (30, 'bx_a_r50', 'by_k_r50',
     'A and K on ResNet-50', DEEPER),
    (31, 'cp_k_no_kd_r50', 'ci_k_heads4_r50',
     'K without KL, and K with a head per band of widths',
     NOKD_AND_HEADS),
    (32, 'cr_k_r50_seed2', 'cq_a_r50_seed2',
     'A and K on ResNet-50, second seed', SEED2_R50),
    (33, 'ct_k_prerelu_r50', 'cu_k_prerelu_r50_seed2',
     'K with the transport read before the last ReLU, two seeds',
     PRE_RELU),
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

If the session times out partway, carry it forward rather than starting
over. Every epoch writes a checkpoint, so at most one is lost, and the
learning-rate schedule is restored with it - resuming does not restart
the cosine.

1. Save Version on the timed-out session, so its output persists.
2. Make a fresh copy of this notebook.
3. Add data -> Notebook Output -> pick that session.
4. Set `RESUME_FROM` to the directory that *contains* `cifar100_<branch>`,
   then Save & Run All.

That directory is in one of two places depending on how the session
ended, because the copy at the bottom of this notebook only runs if the
session got that far:

    finished the last cell   /kaggle/input/<slug>/logs
    killed mid-training      /kaggle/input/<slug>/new-pruning/logs

If in doubt, run `!find /kaggle/input -name latest_checkpoint.pt` in a
scratch cell and take the parent of the parent. A correct path prints
`restored cifar100_<branch>` and then `Loaded checkpoint ... at epoch N`;
if you see `starting from scratch` instead, the path is wrong and the run
will silently begin again at epoch 1.

## When it finishes

Send back the final table. Pasting the output of the last cell is enough.
"""

CONFIG = """# Fixed for this notebook. Notebook {number} of {total}.
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

# The template carries on with zero GPUs and only fails at the first
# .cuda() in the smoke step, after the dataset has been fetched. Notebook
# 30 lost half an hour that way with Accelerator left at None.
GPU_WAS = "if n_gpu < len(BRANCHES):"
GPU_NOW = chr(10).join([
    "if n_gpu == 0:",
    "    raise SystemExit('No GPU in this session. Set Accelerator to '",
    "                     'GPU T4 x2 in the notebook settings and run again.')",
    "if n_gpu < len(BRANCHES):"])

SUITES_NOW = chr(10).join([
    "          'tests/test_kd_variants.py',",
    "          'tests/test_channel_reorder.py',",
    "          'tests/test_published_methods.py',",
    "          'tests/test_freeze.py',",
    "          'tests/test_head_groups.py',",
    "          'tests/test_affine_knots.py',",
    "          'tests/test_no_kd.py',",
    "          'tests/test_pre_relu.py']"])

for number, left, right, title, preamble in QUEUE:
    nb = json.loads(json.dumps(template))
    for cell in nb['cells']:
        body = ''.join(cell['source'])
        for old, new in ((SUITES_WAS, SUITES_NOW), (GPU_WAS, GPU_NOW),
                         (QUIET_WAS, QUIET_NOW),
                         (FILTER_WAS, FILTER_NOW)):
            body = body.replace(old, new)
        cell['source'] = source(body) if body.strip() else cell['source']
    body = HEADER.format(number=number, title=title,
                         preamble=preamble)
    # 27 and 30 do not fit in one session and say so below the header;
    # the header should not promise otherwise above it
    if number in MULTI_SESSION:
        body = body.replace(
            'in a single session of roughly three to five\nhours.',
            'over more than one session - see below for\nhow to carry it '
            'forward.')
    elif preamble is PUBLISHED or preamble is HEAD_SWEEP:
        body = body.replace('three to five\nhours', 'ten to twelve\nhours')
    # a notebook may hold one branch and leave the second card idle
    branches = [branch for branch in (left, right) if branch]
    if len(branches) == 1:
        body = body.replace('Two branches, one per card,',
                            'One branch, on one card,')
    for branch in branches:
        body += '## `{}`\n\n{}\n\n'.format(branch, banner(branch))
    nb['cells'][0]['source'] = source(body + FOOTER)
    nb['cells'][1]['source'] = source(CONFIG.format(
        number=number, total=max(entry[0] for entry in QUEUE),
        branches=branches))
    name = 'kaggle_kd_{:02d}_{}.ipynb'.format(
        number, '_'.join(branch.split('_')[0] for branch in branches))
    with io.open(os.path.join(KAGGLE, name), 'w',
                 encoding='utf-8', newline='\n') as handle:
        json.dump(nb, handle, indent=1, ensure_ascii=False)
        handle.write('\n')
    print('{:32} {:18} {}'.format(name, left, right or ''))
