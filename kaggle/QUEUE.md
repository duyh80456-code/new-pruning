# The queue

Every notebook runs one branch per card. Rows up to 28 are ResNet-18
and have come back; from 30 on every run is ResNet-50, where A takes
about 10.7 hours on a T4 and K about 14, so a K branch needs a second
session. Take a row with a free slot, run the file as it is, and say
which number you took.

On ResNet-50, K stands at 77.26 against 76.86 for A, ahead at 15 of 16
widths, one seed each. Notebook 32 is the second seed.

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
| 18 | `kaggle_kd_18_bd_be.ipynb` | `bd_warm10_sort`, `be_warm25_sort` | Make a window, then sort inside it | 73.96 / 73.98 |
| 19 | `kaggle_kd_19_bf_bg.ipynb` | `bf_warm10_plain`, `bg_warm25_plain` | What the warm-up is worth on its own | 73.90 / 73.78 |
| 20 | `kaggle_kd_20_bh_bi.ipynb` | `bh_warm10_taylor`, `bi_warm25_taylor` | The same window, sorted by what removal would cost | 73.71 / 73.95 |
| 25 | `kaggle_kd_25_bn_bp.ipynb` | `bn_ensemble`, `bp_width_scalars` | Who teaches, and the one place recalibration cannot reach | 73.93 / 74.35 |
| 26 | `kaggle_kd_26_bq_bo.ipynb` | `bq_equalize_half`, `bo_conv_averaged` | The learning rate nobody set | 73.63 / 74.16 |
| 27 | `kaggle_kd_27_br_bs.ipynb` | `br_a300`, `bs_k300` | The same two branches, three times the schedule | 74.13 / 73.91 |
| 28 | `kaggle_kd_28_bu_bt.ipynb` | `bu_narrow_first`, `bt_narrow_first_frozen` | The narrow end first, and then held still | 72.81 / 64.39 |
| 30 | `kaggle_kd_30_bx_by.ipynb` | `bx_a_r50`, `by_k_r50` | A and K on ResNet-50 | 76.86 / 77.26 |
| 31 | `kaggle_kd_31_cp_ci.ipynb` | `cp_k_no_kd_r50`, `ci_k_heads4_r50` | K without KL, and K with a head per band of widths | free / free |
| 32 | `kaggle_kd_32_cr_cq.ipynb` | `cr_k_r50_seed2`, `cq_a_r50_seed2` | A and K on ResNet-50, second seed | free / 76.46 |
| 33 | `kaggle_kd_33_ct_cu.ipynb` | `ct_k_prerelu_r50`, `cu_k_prerelu_r50_seed2` | K with the transport read before the last ReLU, two seeds | free / free |

## Held back

These have configs in `apps/` and no notebook. To run one, add a row
for it at the end of QUEUE in scripts/build_notebooks.py.

* `ax_var_ground, ay_inverse_ground` - the same weighting by batch variance, and its reversal. Variance and activation disagree about a channel that is large and constant, so they are separate questions; ResNet-18, and the pair it followed up was never run.
* `a_seed3` - a third seed of A on ResNet-18. The seed question has moved to ResNet-50, notebook 32.
* `z_eps_100` - a control built to lose: blur the plan away and see what is left. Y at eps 0.5 has since made most of its point for a quarter of the cost.
* `j_feature_gram` - a control for whether nesting is what makes K work.
* `ap_logit_spread` - a test of the flatness diagnosis on branch C, which came last of twelve and is not going to reach K.
* `az_act_ground, ba_taylor_ground, bb_reorder_l1, bc_reorder_read, k_seed2, bj_chain_only, q_feature_mse, r_feature_mmd, bm_solo_full, bk_alpha_on_k, bl_gromov, ab_no_debias, bv_half_first_frozen, bw_narrow10_frozen` - ResNet-18 branches whose notebooks (16, 17, 21 to 24, 29) were removed unrun. Every new run is on ResNet-50, so any of these that comes back comes back as a ResNet-50 twin of BX or BY in a new notebook at the end.
* `cd_dsnet_r50, cf_scala_r50, cg_nasvit_r50` - the published methods on ResNet-50, in the old notebooks 31 and 32 (removed unrun; the numbers now hold other runs). Needed for the paper once the ResNet-50 results settle.
* `ch_a_heads4_r50, cj_a_heads2_r50, ck_a_heads8_r50, cl_a_heads16_r50` - band heads on A and the head-count sweep, notebooks 33 to 35, removed unrun. CI in notebook 31 asks first whether heads help K.
* `cm_k_knots4_r50, cn_a_knots4_r50` - BN scale and shift continuous in width, notebook 36, removed unrun.
* `co_ce_only_r50` - US-Net with KD off, the floor under CP; notebook 37 removed unrun, and its other half, CP, is in notebook 31.

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
