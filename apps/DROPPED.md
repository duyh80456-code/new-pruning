Dropped from the queue, with what the evidence said at the time.

  n_jeffreys_narrow  L was the same reasoning applied to D. It projected
                     to 73.58 and landed at 72.74, below the branch it was
                     built from. Weighting a term by width does not keep
                     the half of a per-width table that looked good.

  o_temp4            Temperature was a diagnosis for six branches landing
  p_jeffreys_temp4   inside 0.8 points of each other. K then cleared A by
                     0.75 without it, so the diagnosis is no longer what
                     stands between the project and a result.

Two others were dropped and should not have been. e_kl_pair and
g_flat_cost had already run; the reasoning for dropping them was wrong
anyway, and they turned out to be the two runs that made the logit-tier
story legible.

The mistake was to judge a control by its gap to A. E lands at 73.27, a
quarter point below A, which is why it looked like there was nothing left
to attribute. But a control is read against the branch it controls for,
not against the reference: E equals D exactly, and F sits 0.27 above both,
which is what separates symmetry from the metric. Nothing else in the set
could have separated them.

G is the sharper case. It was dropped as a formality that would confirm
the classifier cost contributes nothing. It beats C by 0.51 instead: a
cost matrix with no geometry at all does better than the model's own. The
prediction was "contributes nothing" and the answer was "actively hurts",
which is a different claim and a stronger one.

The configs stay in apps/. Any of them runs from a notebook by editing
BRANCHES.
