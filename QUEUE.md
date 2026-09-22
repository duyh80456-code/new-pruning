# The queue

Twelve notebooks, two branches each, one per card. Take a row, run
the file as it is, and say which number you took.

K stands at 74.26 against 73.52 for A, ahead at all sixteen widths.
That is the number to pass. The result column carries the mean
accuracy of each branch once it has come back, so take a row with a
free slot in it. `ak_bures` says rerun rather than free: it crashed
three minutes into its session on a numerical bug, and that bug is
fixed, so the branch has never actually been tried.

| # | notebook | branches | what it asks | result |
|---|---|---|---|---|
| 1 | `kaggle_kd_01_af_ae.ipynb` | `af_all_pairs`, `ae_k_plus_f` | Couple every pair of students, and put K and F together | 74.03 / 73.42 |
| 2 | `kaggle_kd_02_ah_ag.ipynb` | `ah_everything`, `ag_five_widths` | All of it at once, and more students to pair | 73.53 / 67.70 |
| 3 | `kaggle_kd_03_ai_ak.ipynb` | `ai_channel`, `ak_bures` | Transport along the shared channels, and the Gaussian closed form | 73.24 / **rerun** |
| 4 | `kaggle_kd_04_ao_ar.ipynb` | `ao_spread`, `ar_k_temp4` | Give the metric some geometry, and the teacher something to say | 74.14 / 73.38 |
| 5 | `kaggle_kd_05_m_aq.ipynb` | `m_all_stages`, `aq_classwise` | Transport at every depth, and between class positions | 73.53 / 74.10 |
| 6 | `kaggle_kd_06_al_am.ipynb` | `al_bures_diag`, `am_sliced_max` | Variances without directions, and the worst projection | 73.52 / 73.64 |
| 7 | `kaggle_kd_07_aj_an.ipynb` | `aj_channel_p1`, `an_sliced_p1` | An absolute gap instead of a squared one, on both | 37.47 / 20.27 |
| 8 | `kaggle_kd_08_v_s.ipynb` | `v_unbalanced`, `s_feature_sliced` | Let mass go unmatched, and solve it the cheap way | 74.18 / 73.22 |
| 9 | `kaggle_kd_09_ac_ad.ipynb` | `ac_weight_half`, `ad_weight_double` | Is K on a plateau or on a peak | free / free |
| 10 | `kaggle_kd_10_y_w.ipynb` | `y_eps_050`, `w_eps_002` | Blurrier and nearly hard, the two ends of the axis | 73.28 / 73.74 |
| 11 | `kaggle_kd_11_x_aa.ipynb` | `x_eps_010`, `aa_cosine_ground` | One step of blur, and direction instead of distance | 74.24 / 73.75 |
| 12 | `kaggle_kd_12_t_u.ipynb` | `t_sliced_32`, `u_sliced_512` | How many directions stand in for a plan | 73.19 / 72.99 |

## Held back

These have configs and are not in the queue.

* `k_seed2, a_seed3` - seeds. Five branches now sit within a quarter point of each other at the top and sigma is still unmeasured, so these have stopped being defensive and started being the thing that decides the order.
* `z_eps_100, q_feature_mse` - controls for K. They are built to lose: blur the plan away, or forbid rematching, and see what is left.
* `r_feature_mmd` - a control from outside transport.
* `j_feature_gram` - a control for whether nesting is what makes K work.
* `ab_no_debias` - a control for whether the debiasing correction is load bearing.
* `ap_logit_spread` - a test of the flatness diagnosis on branch C, which came last of twelve and is not going to reach K.
* `n_jeffreys_narrow, o_temp4, p_jeffreys_temp4` - dropped earlier on evidence, with the reasons in `apps/DROPPED.md`.

Nothing in a notebook needs editing. Set the accelerator to GPU
T4 x2, attach the CIFAR-100 dataset, and use Save & Run All
(Commit) so the session survives the browser closing.
