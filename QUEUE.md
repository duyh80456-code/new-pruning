# The queue

Twelve notebooks, two branches each, one per card. Every one of
them is a branch that could beat K. Take a row, run the file as it
is, and say which number you took.

K stands at 74.26 against 73.52 for A, ahead at all sixteen widths.
That is the number to pass.

| # | notebook | branches | what it asks |
|---|---|---|---|
| 1 | `kaggle_kd_01_af_ae.ipynb` | `af_all_pairs`, `ae_k_plus_f` | Couple every pair of students, and put K and F together |
| 2 | `kaggle_kd_02_ah_ag.ipynb` | `ah_everything`, `ag_five_widths` | All of it at once, and more students to pair |
| 3 | `kaggle_kd_03_ai_ak.ipynb` | `ai_channel`, `ak_bures` | Transport along the shared channels, and the Gaussian closed form |
| 4 | `kaggle_kd_04_ao_ar.ipynb` | `ao_spread`, `ar_k_temp4` | Give the metric some geometry, and the teacher something to say |
| 5 | `kaggle_kd_05_m_aq.ipynb` | `m_all_stages`, `aq_classwise` | Transport at every depth, and between class positions |
| 6 | `kaggle_kd_06_al_am.ipynb` | `al_bures_diag`, `am_sliced_max` | Variances without directions, and the worst projection |
| 7 | `kaggle_kd_07_aj_an.ipynb` | `aj_channel_p1`, `an_sliced_p1` | An absolute gap instead of a squared one, on both |
| 8 | `kaggle_kd_08_v_s.ipynb` | `v_unbalanced`, `s_feature_sliced` | Let mass go unmatched, and solve it the cheap way |
| 9 | `kaggle_kd_09_ac_ad.ipynb` | `ac_weight_half`, `ad_weight_double` | Is K on a plateau or on a peak |
| 10 | `kaggle_kd_10_y_w.ipynb` | `y_eps_050`, `w_eps_002` | Blurrier and nearly hard, the two ends of the axis |
| 11 | `kaggle_kd_11_x_aa.ipynb` | `x_eps_010`, `aa_cosine_ground` | One step of blur, and direction instead of distance |
| 12 | `kaggle_kd_12_t_u.ipynb` | `t_sliced_32`, `u_sliced_512` | How many directions stand in for a plan |

## Held back

These have configs and are not in the queue. None of them is
built to win; they are what a result needs once there is one.

* `k_seed2, a_seed3` - seeds. What a result needs to be believed, not what finds one.
* `z_eps_100, q_feature_mse` - controls for K. They are built to lose: blur the plan away, or forbid rematching, and see what is left.
* `r_feature_mmd` - a control from outside transport.
* `j_feature_gram` - a control for whether nesting is what makes K work.
* `ab_no_debias` - a control for whether the debiasing correction is load bearing.
* `ap_logit_spread` - a test of the flatness diagnosis on branch C, which came last of twelve and is not going to reach K.

* `n_jeffreys_narrow, o_temp4, p_jeffreys_temp4` - dropped earlier on
  evidence, with the reasons in `apps/DROPPED.md`.

Running them now would spend a session each on a number that
cannot improve anything. They come back when something has beaten
K and needs defending.

Nothing in a notebook needs editing. Set the accelerator to GPU
T4 x2, attach the CIFAR-100 dataset, and use Save & Run All
(Commit) so the session survives the browser closing.
