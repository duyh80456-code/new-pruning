# The queue

Every notebook in this directory runs two branches, one per card, in a
session of roughly three to five hours. Take a row, run the file as it
is, and say which number you took.

Rows 1 to 12 all ask the same question - can a different transport term
beat K - and the answer has settled at no. Rows 13 to 15 leave that
axis: they change where the sandwich rule samples, which samples KD
attends to, and which width teaches which. None of them touches the
transport term, so they compose with K rather than competing with it.

K stands at 74.26 against 73.52 for A, ahead at all sixteen widths.
That is the number to pass. The result column carries each branch's
mean accuracy once it has come back, so take a row with a free slot.
`ak_bures` crashed three minutes into its session on a numerical bug
that is now fixed, so it has never actually been tried. It is queued
in row 15, not row 3.

## How to run one

Upload the `.ipynb` to Kaggle, then:

1. Accelerator **GPU T4 x2**, Internet **On**
2. Add the CIFAR-100 dataset as an input
3. **Save Version -> Save & Run All (Commit)**, not the interactive
   run, which ends when the browser closes

Nothing in a notebook needs editing. Each one clones this repo when it
starts, so it always picks up the current code.

Do not skip the smoke step. The branch suite at the top mimics the
training loop rather than running it, so anything that lives only in
`train.py` is inert there; the smoke run is what exercises it.

| # | notebook | branches | what it asks | result |
|---|---|---|---|---|
| 1 | `kaggle_kd_01_af_ae.ipynb` | `af_all_pairs`, `ae_k_plus_f` | Couple every pair of students, and put K and F together | 74.03 / 73.42 |
| 2 | `kaggle_kd_02_ah_ag.ipynb` | `ah_everything`, `ag_five_widths` | All of it at once, and more students to pair | 73.53 / 67.70 |
| 3 | `kaggle_kd_03_ai_ak.ipynb` | `ai_channel`, `ak_bures` | Transport along the shared channels, and the Gaussian closed form | 73.24 / 73.56 |
| 4 | `kaggle_kd_04_ao_ar.ipynb` | `ao_spread`, `ar_k_temp4` | Give the metric some geometry, and the teacher something to say | 74.14 / 73.38 |
| 5 | `kaggle_kd_05_m_aq.ipynb` | `m_all_stages`, `aq_classwise` | Transport at every depth, and between class positions | 73.53 / 74.10 |
| 6 | `kaggle_kd_06_al_am.ipynb` | `al_bures_diag`, `am_sliced_max` | Variances without directions, and the worst projection | 73.52 / 73.64 |
| 7 | `kaggle_kd_07_aj_an.ipynb` | `aj_channel_p1`, `an_sliced_p1` | An absolute gap instead of a squared one, on both | 37.47 / 20.27 |
| 8 | `kaggle_kd_08_v_s.ipynb` | `v_unbalanced`, `s_feature_sliced` | Let mass go unmatched, and solve it the cheap way | 74.18 / 73.22 |
| 9 | `kaggle_kd_09_ac_ad.ipynb` | `ac_weight_half`, `ad_weight_double` | Is K on a plateau or on a peak | 73.95 / 73.82 |
| 10 | `kaggle_kd_10_y_w.ipynb` | `y_eps_050`, `w_eps_002` | Blurrier and nearly hard, the two ends of the axis | 73.28 / 73.74 |
| 11 | `kaggle_kd_11_x_aa.ipynb` | `x_eps_010`, `aa_cosine_ground` | One step of blur, and direction instead of distance | 74.24 / 73.75 |
| 12 | `kaggle_kd_12_t_u.ipynb` | `t_sliced_32`, `u_sliced_512` | How many directions stand in for a plan | 73.19 / 72.99 |
| 13 | `kaggle_kd_13_as_at.ipynb` | `as_log_widths`, `at_macs_widths` | Where the sandwich rule spends its free samples | 73.93 / 73.96 |
| 14 | `kaggle_kd_14_au_av.ipynb` | `au_entropy_kd`, `av_confidence_kd` | Which samples the teacher still has something to say about | 73.00 / 30.46 |
| 15 | `kaggle_kd_15_aw_ak.ipynb` | `aw_teacher_chain`, `ak_bures` | A chain of teachers, and the Gaussian form that never ran | 74.32 / 73.56 |
| 16 | `kaggle_kd_16_az_ba.ipynb` | `az_act_ground`, `ba_taylor_ground` | Charge the channels for what they carry | _skipped_ |
| 17 | `kaggle_kd_17_bb_bc.ipynb` | `bb_reorder_l1`, `bc_reorder_read` | Which channels the narrow subnet gets | _skipped_ |
| 18 | `kaggle_kd_18_bd_be.ipynb` | `bd_warm10_sort`, `be_warm25_sort` | Make a window, then sort inside it | 73.96 / 73.98 |
| 19 | `kaggle_kd_19_bf_bg.ipynb` | `bf_warm10_plain`, `bg_warm25_plain` | What the warm-up is worth on its own | 73.90 / 73.78 |
| 20 | `kaggle_kd_20_bh_bi.ipynb` | `bh_warm10_taylor`, `bi_warm25_taylor` | The same window, sorted by what removal would cost | 73.71 / 73.95 |
| 21 | `kaggle_kd_21_k_bj.ipynb` | `k_seed2`, `bj_chain_only` | The noise on K, and the chain without the transport | free / free |
| 22 | `kaggle_kd_22_q_r.ipynb` | `q_feature_mse`, `r_feature_mmd` | What the transport term actually bought | free / free |
| 23 | `kaggle_kd_23_bm_bk.ipynb` | `bm_solo_full`, `bk_alpha_on_k` | The control never run, and the loss never tried under K | free / free |
| 24 | `kaggle_kd_24_bl_ab.ipynb` | `bl_gromov`, `ab_no_debias` | Comparing the widths without truncating either | free / free |

## Held back

These have configs in `apps/` and are not queued. Any of them runs
from a notebook by editing `BRANCHES`.

* `ax_var_ground, ay_inverse_ground` - the same weighting by batch variance, and its reversal. Variance and activation disagree about a channel that is large and constant, so they are separate questions; this pair is the follow-up if row 16 moves anything.
* `a_seed3` - a third seed of A. Held because the seed that matters is K, and that one has moved out of this list into row 21: ten unrelated perturbations of K now sit 0.28 to 0.55 below it, which is either ten knobs already optimal or one high draw, and only sigma tells them apart.
* `z_eps_100` - a control built to lose: blur the plan away and see what is left. Y at eps 0.5 has since made most of its point for a quarter of the cost.
* `j_feature_gram` - a control for whether nesting is what makes K work.
* `ap_logit_spread` - a test of the flatness diagnosis on branch C, which came last of twelve and is not going to reach K.

## Skipped

Still numbered, because renumbering would rename notebooks that are already pushed.

* **row 16** - the transport term charges every channel the same and these would have weighted it, by activation and by the Taylor score. Set aside for the channel-ordering axis rather than answered. The configs and the notebook are there if it comes back.
* **row 17** - superseded by 18. It permutes at epoch 10, an epoch nothing measured can justify, and the reason it cannot is that US-Net has no window to name: the prefix taking gradient at every width is what makes a criterion meaningful and also what sorts it. 18 makes the window instead. The one thing 17 alone would have produced, the prefix_sorted trajectory under an ordinary schedule, comes out of 19 from the end of its warm-up onward.

## Dropped on evidence

* `n_jeffreys_narrow` - L was the same reasoning applied to D. It projected to 73.58 and landed at 72.74, below the branch it was built from. Weighting a term by width does not keep the half of a per-width table that looked good.
* `o_temp4, p_jeffreys_temp4` - temperature was a diagnosis for six branches landing inside 0.8 points of each other. K then cleared A by 0.75 without it. AR has since tested the diagnosis directly on top of K and come out at 73.38, below A, so the remedy is ruled out at this tier rather than merely set aside.

## Two that were dropped and should not have been

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
