# The queue

One notebook per pair, two branches a session, one per card. The
name carries the number, which is the priority, and the two branch
prefixes, so the file says what it runs without opening it.

Rows one to four could move the result. The rest explain or defend
it. Take a row, run the file as it is, and say which number you
took.

| # | notebook | branches | what it asks |
|---|---|---|---|
| 1 | `kaggle_kd_01_af_ae.ipynb` | `af_all_pairs`, `ae_k_plus_f` | Couple every pair of students, and put K and F together |
| 2 | `kaggle_kd_02_ai_ak.ipynb` | `ai_channel`, `ak_bures` | Transport along the shared channels, and the Gaussian closed form |
| 3 | `kaggle_kd_03_ap_ao.ipynb` | `ap_logit_spread`, `ao_spread` | Push the classifier rows apart: does the flatness explanation hold |
| 4 | `kaggle_kd_04_ah_ag.ipynb` | `ah_everything`, `ag_five_widths` | All of it at once, and more students to pair |
| 5 | `kaggle_kd_05_z_q.ipynb` | `z_eps_100`, `q_feature_mse` | Does the matching matter, asked two ways |
| 6 | `kaggle_kd_06_al_am.ipynb` | `al_bures_diag`, `am_sliced_max` | Variances without directions, and the worst projection |
| 7 | `kaggle_kd_07_aj_an.ipynb` | `aj_channel_p1`, `an_sliced_p1` | An absolute gap instead of a squared one, on both |
| 8 | `kaggle_kd_08_k_a.ipynb` | `k_seed2`, `a_seed3` | The third seed, which finally puts a number on sigma |
| 9 | `kaggle_kd_09_j_m.ipynb` | `j_feature_gram`, `m_all_stages` | Does nesting matter, and transport at every depth |
| 10 | `kaggle_kd_10_s_v.ipynb` | `s_feature_sliced`, `v_unbalanced` | The cheap solver, and letting mass go unmatched |
| 11 | `kaggle_kd_11_ac_ad.ipynb` | `ac_weight_half`, `ad_weight_double` | Is K on a plateau or on a peak |
| 12 | `kaggle_kd_12_w_y.ipynb` | `w_eps_002`, `y_eps_050` | The two ends of the blur axis |
| 13 | `kaggle_kd_13_ab_x.ipynb` | `ab_no_debias`, `x_eps_010` | Is the debiasing correction load bearing |
| 14 | `kaggle_kd_14_t_u.ipynb` | `t_sliced_32`, `u_sliced_512` | How many directions stand in for a plan |
| 15 | `kaggle_kd_15_aa_r.ipynb` | `aa_cosine_ground`, `r_feature_mmd` | Direction instead of distance, and a control outside transport |
| 16 | `kaggle_kd_16_aq.ipynb` | `aq_classwise` | Transport between class positions rather than samples |

Nothing in a notebook needs editing. Set the accelerator to
GPU T4 x2, attach the CIFAR-100 dataset, and use Save & Run All
(Commit) so the session survives the browser closing.
