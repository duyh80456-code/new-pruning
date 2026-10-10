# Results

Universally slimmable ResNet-18 on CIFAR-100, sixteen widths from 0.25 to 1.00, accuracy after BN post-statistics. One seed each unless the row says otherwise. Generated from `results/runs.json` by `scripts/collect_results.py`.

## Where it stands

| | branch | mean top-1 | vs A | worst width | mean NLL | vs A |
|---|---|---|---|---|---|---|
| **ED** | DO (K with four heads, seed 1995) trained with Sharpness-Aware Minimization, rho 0.1. 78.16: 0.32 above DO, inside the noise; appendix only, SAM is not part of the method | **78.16** | +4.64 | 76.61 | 0.852 | -0.509 |
| **DM** | K as DK but the block of 20 widths is plain uniform draws, sorted and dealt spread: one free width from each half, steps shuffled | **78.16** | +4.64 | 75.89 | 0.939 | -0.422 |
| **EC** | DO (K with four heads, seed 1995) trained with Sharpness-Aware Minimization, rho 0.05. 78.02: 0.18 above DO, inside the noise; appendix only, SAM is not part of the method | **78.02** | +4.50 | 76.34 | 0.903 | -0.458 |
| **DO** | K as DD with a classifier head per band of widths, 4 bands (SOLAR's idea on K) | **77.84** | +4.32 | 76.12 | 0.900 | -0.461 |
| **EE** | K with four classifier heads (DO) and far free-width pairs (DM), seed 1995: SlimOT + SPS + WBH. 77.83, level with DO (77.84) and 0.33 under DM, so the two do not add | **77.83** | +4.31 | 75.87 | 0.901 | -0.460 |
| **EI** | DO (K with four heads, SlimOT + WBH) at seed 2026. 77.76: +0.49, above at 15/16 against DC (K at 2026, 77.27), +1.30, above at 16/16 against CQ (A at 2026) | **77.76** | +4.24 | 76.00 | 0.905 | -0.456 |
| **DQ** | K as DD with a classifier head per band of widths, 16 bands (one per test width) | **77.67** | +4.15 | 75.64 | 0.932 | -0.429 |
| **DP** | K as DD with a classifier head per band of widths, 8 bands | **77.66** | +4.14 | 75.79 | 0.911 | -0.450 |
| **FI** | DD (the method without WBH: one shared classifier) at seed 2006 (notebook 70), over two sessions. 77.66: 0.28 ABOVE the method with WBH at the same seed (FC 77.38), above it at 15/16 widths; +0.93 over US-Net (FB 76.73). DD 77.50 and DC 77.27 at the other seeds, so SlimOT without WBH is 77.47 +- 0.20 against 77.66 +- 0.24 with it: WBH on SlimOT is +0.34, +0.49, -0.28 by seed (+0.19 +- 0.41), while on US-Net it is +0.35, +0.40, +0.44. The 2x2 interaction, about 0 at 1995 and 2026, is -0.72 at 2006 | **77.66** | +4.14 | 76.03 | 0.981 | -0.380 |
| **FL** | CI (the method without Delayed Transport) at seed 2006 (notebook 69), over two sessions. 77.62: no collapse; 0.23 ABOVE the method at the same seed (FC 77.38), above it at 15/16 widths, as FI (without WBH, 77.66) was: at 2006 FC sits under both of its ablations. Row 'w/o Delayed Transport' over three seeds: CI 77.27, FH 77.55, FL 77.62, mean 77.48 +- 0.19 | **77.62** | +4.10 | 75.84 | 0.899 | -0.462 |
| **DN** | K as DD with a classifier head per band of widths, 2 bands (SOLAR's idea on K) | **77.60** | +4.08 | 75.91 | 0.953 | -0.408 |
| **EN** | DO (SlimOT + WBH, seed 1995) without Teacher Transport: feature_kd off, Peer Transport, logit KD, Delayed Transport and the four heads kept (notebook 55, resumed at epoch 89). 77.60: 0.24 under DO (77.84), above it at 6/16 widths, inside one seed's noise; +0.39 over the heads alone (CH 77.21) at 15/16. Peer Transport alone keeps most of the gain | **77.60** | +4.08 | 75.16 | 0.922 | -0.438 |
| **FH** | CI (the method without Delayed Transport: both transport terms from epoch 1) at seed 2026 (notebook 69), over two sessions. 77.55: no collapse, at the seed where K without Delayed Transport and without heads collapsed (CR, widths up to 0.60 at chance); 0.22 under the method at the same seed (EI 77.76), above it at 1/16 widths | **77.55** | +4.03 | 75.51 | 0.930 | -0.431 |
| **DT** | DD's final weights frozen, a rank-4 LoRA update per conv at 4 width knots added and trained alone for 20 epochs on K's loss (TAS-LoRA's recipe, interpolation in width): level with DD | **77.53** | +4.01 | 75.74 | 1.008 | -0.353 |
| **DD** | K on ResNet-50 as BY with both feature transport terms held off until epoch 6, at BY's seed 1995: what the late start costs where nothing broke | **77.50** | +3.98 | 75.63 | 1.010 | -0.351 |
| **DU** | DD's final weights frozen, BN scale/shift offsets at 4 width knots added and trained alone for 20 epochs on K's loss: level with DD | **77.48** | +3.96 | 75.76 | 1.005 | -0.356 |
| **FF** | EN (the method without Teacher Transport: feature_kd off) at seed 2026 (notebook 67), over two sessions. 77.44: 0.33 under the method at the same seed (EI 77.76), above it at 2/16 widths; +0.97 over US-Net (CQ 76.46) at 16/16. No collapse at the seed where K without Delayed Transport collapsed (CR). With EN (-0.24 at 1995) the row 'w/o Teacher Transport' is a consistent small loss | **77.44** | +3.92 | 75.20 | 0.933 | -0.428 |
| **FC** | The method (SlimOT + WBH, as DO) at seed 2006 (notebook 60), over two sessions. 77.38: +0.66 over US-Net at the same seed (FB 76.73) at 16/16 widths, +0.22 over SOLAR (FM 77.16) at 14/16, +0.72 over Scala (FN 76.66) at 16/16; DO 77.84 and EI 77.76 at the other seeds, so the method's three-seed mean is 77.66 ± 0.24 | **77.38** | +3.86 | 75.39 | 0.926 | -0.435 |
| **DK** | K as DD with the same block draws dealt out spread: one free width from each half (about 0.375 apart), steps shuffled | **77.31** | +3.79 | 75.26 | 1.024 | -0.337 |
| **DR** | K as DD with a rank-4 LoRA update per conv at 4 width knots, mixed by distance, trained from the start (1.27M stored, merged at inference) | **77.28** | +3.76 | 75.22 | 1.015 | -0.345 |
| **DC** | K on ResNet-50 as BY (post-ReLU read) with both feature transport terms held off until epoch 6, at seed 2026 where CR collapsed: survives | **77.27** | +3.75 | 75.47 | 1.020 | -0.341 |
| **CI** | K on ResNet-50 with four band heads (0.25-0.40, 0.45-0.60, 0.65-0.80, 0.85-1.00) over the shared backbone | **77.27** | +3.75 | 75.60 | 0.928 | -0.433 |
| **BY** | K on ResNet-50, the pair for BX: the same feature transport, 100 epochs | **77.26** | +3.74 | 75.12 | 1.018 | -0.343 |
| **FG** | EO (the method without Peer Transport: horizontal_kd off) at seed 2026 (notebook 67). 77.21: 0.55 under the method at the same seed (EI 77.76) at 16/16 widths; +0.75 over US-Net (CQ 76.46) at 16/16. With EO (-0.81 at 1995, 16/16) Peer Transport is the term the gain rests on | **77.21** | +3.69 | 75.51 | 1.084 | -0.277 |
| **CH** | SOLAR (WACV 2026) adapted to US-Net on ResNet-50: A with a classifier head per band of widths (four bands of four test widths), 0.6M extra parameters; seed 1995 | **77.21** | +3.69 | 75.42 | 1.189 | -0.172 |
| **FM** | SOLAR (WACV 2026) adapted to US-Net on ResNet-50, as CH (four band heads, no transport) at seed 2006 (notebook 61). 77.16: +0.30 over BX and 0.34 under DD at seed 1995's references; CH 77.21 and FD 76.86 at the other seeds, so SOLAR's three-seed mean is 77.08 ± 0.19. Read against FC (the method at 2006, notebook 60) | **77.17** | +3.65 | 75.37 | 1.172 | -0.189 |
| **DS** | K as DD with a rank-8 LoRA update per conv at 2 width knots (the range ends), trained from the start (1.27M stored) | **77.06** | +3.54 | 75.53 | 1.019 | -0.342 |
| **EO** | DO (SlimOT + WBH, seed 1995) without Peer Transport: horizontal_kd off, Teacher Transport, logit KD, Delayed Transport and the four heads kept (notebook 55, resumed at epoch 92). 77.03: 0.81 under DO (77.84) at 16/16 widths, and 0.18 under the heads alone (CH 77.21), above it at 1/16. Teacher Transport alone adds nothing on top of the heads; it helps only beside Peer Transport (DO against EN, +0.24) | **77.03** | +3.51 | 75.35 | 1.103 | -0.258 |
| **BX** | A on ResNet-50: US-Net as published, bottleneck blocks, 100 epochs | **76.86** | +3.34 | 75.26 | 1.331 | -0.030 |
| **FD** | SOLAR (WACV 2026) adapted to US-Net on ResNet-50, as CH (four band heads, no transport) at seed 2026 (notebook 63). 76.86: 0.91 under the method at the same seed (EI 77.76) at 16/16 widths, +0.40 over US-Net (CQ 76.46) at 15/16; CH gave 77.21 at 1995, so SOLAR's two-seed mean is 77.03 against the method's 77.80 | **76.86** | +3.34 | 75.00 | 1.172 | -0.189 |
| **CF** | Scala (NeurIPS 2024) on ResNet-50, ported from the authors' code: isolated narrowest width, stable sampling, teacher chain, the label for every width, 10 epochs of the full width alone; published baseline, seed 1995 | **76.80** | +3.28 | 75.33 | 1.264 | -0.097 |
| **FE** | Scala (NeurIPS 2024) on ResNet-50, as CF at seed 2026 (notebook 63). 76.73: 1.03 under the method at the same seed (EI 77.76) at 16/16 widths, +0.27 over US-Net (CQ 76.46) at 14/16; CF gave 76.80 at 1995, so Scala's two-seed mean is 76.77 against the method's 77.80 | **76.73** | +3.22 | 74.83 | 1.286 | -0.075 |
| **FB** | A on ResNet-50 (US-Net as published, BX) at seed 2006 (notebook 60), the third seed of the main table. 76.73: BX 76.86 and CQ 76.46 at the other seeds, so US-Net's three-seed mean is 76.68 ± 0.20. Read against FC | **76.73** | +3.21 | 75.38 | 1.321 | -0.040 |
| **FN** | Scala (NeurIPS 2024) on ResNet-50, as CF at seed 2006 (notebook 61). 76.66: 0.50 under SOLAR at the same seed (FM 77.16) at 16/16 widths; CF 76.80 and FE 76.73 at the other seeds, so Scala's three-seed mean is 76.73 ± 0.07. Read against FC (the method at 2006, notebook 60) | **76.66** | +3.15 | 74.96 | 1.299 | -0.062 |
| **DL** | K as DJ but the block of 20 widths is plain uniform draws, sorted and dealt adjacent: close pairs, narrow to wide | **76.62** | +3.10 | 75.14 | 1.153 | -0.208 |
| **DJ** | K as DD with the free widths drawn ten steps at a time, one per twentieth of the range, sorted and paired as neighbours (about 0.04 apart), each block narrow to wide | **76.56** | +3.04 | 74.91 | 1.169 | -0.192 |
| **EF** | DM (far free-width pairs, SlimOT + SPS) at seed 2026. 76.51: 0.76 under K at the same seed (DC 77.27) at 16/16 widths and level with A at that seed (CQ 76.46), so DM's 78.16 did not repeat | **76.51** | +2.99 | 75.17 | 1.060 | -0.301 |
| **CP** | K on ResNet-50 with the logit KD taken out: students learn from the label, the two feature transport terms are the only link between widths | **76.50** | +2.98 | 75.39 | 1.143 | -0.218 |
| **CQ** | A on ResNet-50 again at seed 2026, the second draw of BX | **76.46** | +2.94 | 74.62 | 1.362 | +0.001 |
| **GX** | LCS (WACV 2023) from the authors' code with BatchNorm, as EB (lcs_l_bn) at seed 2026 through the patch's --seed (notebook 65). 76.42: 1.34 under the method at the same seed (EI 77.76) and 0.04 under US-Net (CQ 76.46); EB gave 75.24 at 1995, so LCS's two-seed mean is 75.83 | **76.42** | +2.91 | 74.49 | 1.091 | -0.269 |
| **DE** | A on ResNet-50 exactly as BX with amp: True (fp16 training forward, fp32 losses, validation and calibration): what mixed precision does to A | **76.22** | +2.70 | 74.60 | 1.364 | +0.003 |
| **CV** | AlphaNet (ICML 2021) on ResNet-50: BX with the inplace distillation KL replaced by the adaptive alpha-divergence, alpha in [-1, 1], ratios clipped at 5; published baseline, seed 1995 | **76.18** | +2.66 | 74.16 | 1.490 | +0.129 |
| **FZ** | AlphaNet (ICML 2021) on ResNet-50, as CV at seed 2006 (notebook 62). 75.94: 1.22 under SOLAR at the same seed (FM 77.16); CV 76.18 and FY 75.91 at the other seeds, so AlphaNet's three-seed mean is 76.01 ± 0.15. Read against FC (the method at 2006, notebook 60) | **75.94** | +2.42 | 74.30 | 1.514 | +0.153 |
| **FY** | AlphaNet (ICML 2021) on ResNet-50, as CV at seed 2026 (notebook 62). 75.91: 1.86 under the method at the same seed (EI 77.76) and 0.55 under US-Net (CQ 76.46), both at 16/16 widths; CV 76.18 and FZ 75.94 at the other seeds, so AlphaNet's three-seed mean is 76.01 ± 0.15 | **75.91** | +2.39 | 74.41 | 1.479 | +0.118 |
| **CU** | K on ResNet-50 with the transport read before the last ReLU, seed 2026, the seed CR collapsed at: whether the fix holds where K broke; against CQ | **75.49** | +1.97 | 73.76 | 1.077 | -0.284 |
| **CT** | K on ResNet-50 with both feature transport terms read before the last ReLU (feature_pre_relu), seed 1995: what the collapse fix costs where K did not collapse; against BY | **75.41** | +1.89 | 73.52 | 1.055 | -0.306 |
| **EB** | LCS (WACV 2023) from the authors' code (apple/learning-compressible-subspaces@e6d3924 + third_party/patches/lcs.patch) on ResNet-50: lcs_l line subspace with BatchNorm recalibrated per width on 20 batches, our recipe | **75.24** | +1.73 | 73.81 | 1.179 | -0.182 |
| **GZ** | LCS (WACV 2023) from the authors' code with BatchNorm, as EB at seed 2006 through the patch's --seed (notebook 66). 75.17: 1.56 under US-Net (FB 76.73) and 2.21 under the method (FC 77.38) at the same seed; EB 75.24 and GX 76.42, so LCS's three-seed mean is 75.61 +- 0.70, the widest spread in the table | **75.17** | +1.65 | 73.62 | 1.155 | -0.206 |
| **DG** | WKD-L (NeurIPS 2024) as published in place of US-Net's inplace KL on ResNet-50: target term, non-target transport at T 8 under the fc class cost, weight 600 cosine-decayed over the last 37.5%, CE for every student; seed 1995 | **74.72** | +1.20 | 73.68 | 2.217 | +0.856 |
| **GC** | WKD-L (NeurIPS 2024) on ResNet-50, as DG at seed 2026 (notebook 64). 74.72, the same mean as DG by coincidence (the per-width values differ): 3.05 under the method at the same seed (EI 77.76) and 1.74 under US-Net (CQ 76.46); DG 74.72 and GD 73.94 at the other seeds, so WKD-L's three-seed mean is 74.46 ± 0.45 | **74.72** | +1.20 | 73.66 | 2.339 | +0.978 |
| **BP** | K plus one learnable scalar per width on each residual branch, 128 parameters | **74.35** | +0.83 | 70.90 | 1.148 | -0.213 |
| **AW** | K with each width taught by the next larger one instead of by the widest | **74.32** | +0.80 | 71.37 | 1.091 | -0.270 |
| **GA** | DYNAS (CVPR 2025) on ResNet-50, as DB at seed 2026 (notebook 63). 74.32: 3.44 under the method at the same seed (EI 77.76) and 2.14 under US-Net (CQ 76.46); the narrow end is where it loses, 67.14 at 0.25 against 77.10 at 1.00, as DB did. DB 73.92 and GB 74.23 at the other seeds, so DYNAS's three-seed mean is 74.16 ± 0.21 | **74.32** | +0.80 | 67.14 | 1.111 | -0.250 |
| **K** | I plus transport between the two middle widths, at the feature tier | **74.26** | +0.75 | 70.84 | 1.127 | -0.234 |
| **X** | K at eps 0.1 instead of 0.2 | **74.24** | +0.72 | 71.38 | 1.170 | -0.191 |
| **GB** | DYNAS (CVPR 2025) on ResNet-50, as DB at seed 2006 (notebook 63). 74.23: 2.93 under SOLAR at the same seed (FM 77.16), 67.58 at 0.25 against 76.79 at 1.00; DB 73.92 and GA 74.32 at the other seeds, so DYNAS's three-seed mean is 74.16 ± 0.21. Read against FC (the method at 2006, notebook 60) | **74.23** | +0.71 | 67.58 | 1.127 | -0.234 |
| **V** | K with transport that may leave mass unmatched, at tau 1.0 | **74.18** | +0.66 | 71.11 | 1.083 | -0.278 |
| **EW** | EP (SlimOT + WBH) on MobileNetV2 with the last layer slimmed (slim_last), seed 1995, notebook 59. 74.18: +0.77 against EV (same backbone), +0.16 against ER on the standard backbone. 384 min. | **74.18** | +0.66 | 69.96 | 0.979 | -0.382 |
| **BO** | K with the conv output divided by how many input channels are live, US-Net Appendix A | **74.16** | +0.64 | 70.98 | 1.112 | -0.249 |
| **AO** | K plus a repulsion pushing the classifier rows apart | **74.14** | +0.62 | 71.00 | 1.135 | -0.226 |
| **BR** | A at 300 epochs, US-Net as published with three times the schedule | **74.13** | +0.61 | 70.87 | 1.518 | +0.157 |
| **AQ** | K transporting the two widths' class means rather than their samples | **74.10** | +0.58 | 70.79 | 1.187 | -0.173 |
| **EU** | EP without the band heads (head_groups 1): SlimOT alone on MobileNetV2, CIFAR-100, seed 1995 (notebook 58). 74.06: +0.04 against ER, +0.38 against EP and above EP at 16/16 widths, so on this backbone the heads cost every width, the reverse of ResNet-50. Against ER the narrow end is still down (0.25-0.35: -0.56 to -0.93) and the wide end up (0.85-1.0: +0.3 to +0.5). Its partner ET is recorded separately, recalibrated by hand. | **74.06** | +0.54 | 70.36 | 1.038 | -0.323 |
| **AF** | K coupling all three student pairs instead of one | **74.03** | +0.51 | 70.95 | 1.127 | -0.234 |
| **EA** | Joslim (ECML-PKDD 2021) from the authors' code (enyac-group/Joslim@9b743d9 + third_party/patches/joslim.patch) on ResNet-50: per-layer widths searched by Bayesian optimisation, inplace distillation from the full width, our recipe, read at the 16 uniform widths after 20-batch BN recalibration; architectures scored on 5k train images, not test | **74.02** | +0.50 | 72.82 | 1.368 | +0.007 |
| **ER** | US-Net as published (BX recipe) on MobileNetV2, CIFAR-100, seed 1995: the baseline for EP. models/us_mobilenet_v2_cifar.py, CIFAR strides, last 1x1 to 1280 unslimmed as in US-Net. 74.02. | **74.02** | +0.50 | 70.92 | 1.115 | -0.246 |
| **BE** | K, narrow end held back 25 epochs, then channels permuted by L1 | **73.98** | +0.47 | 70.45 | 1.124 | -0.237 |
| **BD** | K, narrow end held back 10 epochs, then channels permuted by L1 | **73.96** | +0.44 | 71.23 | 1.092 | -0.269 |
| **AT** | K with the same free widths drawn flat in compute instead | **73.96** | +0.44 | 70.71 | 1.113 | -0.248 |
| **BI** | K, narrow end held back 25 epochs, then channels permuted by the Taylor score | **73.95** | +0.44 | 70.60 | 1.097 | -0.264 |
| **AC** | K at half the weight on the feature terms | **73.95** | +0.43 | 70.37 | 1.109 | -0.252 |
| **GD** | WKD-L (NeurIPS 2024) on ResNet-50, as DG at seed 2006 (notebook 64). 73.94: 3.22 under SOLAR at the same seed (FM 77.16); DG 74.72 and GC 74.72 at the other seeds, so WKD-L's three-seed mean is 74.46 ± 0.45. Read against FC (the method at 2006, notebook 60) | **73.94** | +0.42 | 72.79 | 2.546 | +1.185 |
| **GU** | SOLAR (one head per band of widths, four bands) on MobileNetV2 with the last layer slimmed, CIFAR-100, seed 2026 | **73.94** | +0.42 | 70.22 | 1.050 | -0.311 |
| **BN** | K, and every sampled width learns from the mean of what all of them said | **73.93** | +0.41 | 71.01 | 1.034 | -0.327 |
| **AS** | K with the sandwich rule's free widths drawn flat in log width | **73.93** | +0.41 | 70.94 | 1.120 | -0.241 |
| **DB** | DYNAS (CVPR 2025) on ResNet-50: per-width learning rate (1 - t/T)^p, p 4 at width 0.25 to 1/4 at 1.0, momentum per band of widths, each width stepped on its own; seed 1995 | **73.92** | +0.41 | 67.27 | 1.107 | -0.254 |
| **FV** | SlimOT on MobileNetV2 with the last layer slimmed, CIFAR-100, seed 2006 | **73.92** | +0.40 | 69.80 | 0.986 | -0.374 |
| **BS** | K at 300 epochs, the pair for BR | **73.91** | +0.39 | 71.23 | 1.178 | -0.183 |
| **GY** | Joslim (ECML-PKDD 2021) from the authors' code, as EA a third time (notebook 66); its code takes no seed. 73.91; EA 74.02 and GW 73.70, so Joslim's three-run mean is 73.88 +- 0.16. Its own Pareto set reaches 75.44 at 74% of the MACs | **73.91** | +0.39 | 72.47 | 1.432 | +0.071 |
| **BF** | K, narrow end held back 10 epochs, nothing permuted: the control for BD and BH | **73.90** | +0.39 | 70.74 | 1.104 | -0.257 |
| **GT** | SOLAR (one head per band of widths, four bands) on MobileNetV2 with the last layer slimmed, CIFAR-100, seed 1995 | **73.88** | +0.36 | 69.87 | 1.075 | -0.286 |
| **AD** | K at twice the weight, the other side of the same question | **73.82** | +0.31 | 70.80 | 1.152 | -0.209 |
| **BG** | K, narrow end held back 25 epochs, nothing permuted: the control for BE and BI | **73.78** | +0.26 | 70.21 | 1.099 | -0.262 |
| **AA** | K charging transport by direction rather than distance | **73.75** | +0.23 | 70.88 | 1.266 | -0.095 |
| **W** | K at eps 0.02, a plan collapsed onto a permutation | **73.74** | +0.23 | 70.79 | 1.230 | -0.131 |
| **GV** | SOLAR† (US-Net + four band heads) on MobileNetV2 with the last layer slimmed, CIFAR-100, seed 2006 | **73.74** | +0.22 | 70.12 | 1.068 | -0.293 |
| **BH** | K, narrow end held back 10 epochs, then channels permuted by the Taylor score | **73.71** | +0.20 | 70.65 | 1.110 | -0.251 |
| **GW** | Joslim (ECML-PKDD 2021) from the authors' code, as EA (joslim_r50) a second time (notebook 65); Joslim's code takes no seed, so this is an independent draw. 73.70 at the sixteen uniform widths, flat from 73.3 at 0.25 to 75.4 at 1.00; its own Pareto set reaches 74.55 at 27% of the MACs. EA gave 74.02, so Joslim's two-run mean is 73.86 | **73.70** | +0.18 | 72.58 | 1.363 | +0.002 |
| **EP** | SlimOT + WBH (DO recipe) on MobileNetV2, CIFAR-100, seed 1995. 73.68: -0.34 against ER, above at 2/16 widths; the loss is at the narrow end (0.25-0.40: -0.5 to -1.4), the wide half level. The transport reads the unslimmed 1280-channel feature, so prefix alignment is the whole feature at every width, unlike ResNet where it reads the slimmed last block. | **73.68** | +0.16 | 70.02 | 1.028 | -0.333 |
| **I** | A plus transport between the student and teacher feature clouds | **73.66** | +0.14 | 70.78 | 1.311 | -0.050 |
| **AM** | K with sliced transport keeping the worst direction rather than the average | **73.64** | +0.12 | 70.68 | 1.337 | -0.024 |
| **BQ** | K with each channel gradient divided by the square root of how many widths wrote to it | **73.63** | +0.11 | 70.38 | 1.136 | -0.225 |
| **AK** | K with the closed-form Gaussian transport, mean gap plus a covariance term | **73.56** | +0.04 | 70.22 | 1.325 | -0.036 |
| **F** | A plus a symmetrized KL between the two middle widths | **73.54** | +0.03 | 70.30 | 1.255 | -0.106 |
| **A'** | the same branch again, to measure the noise | **73.53** | +0.01 | 69.80 | 1.377 | +0.016 |
| **M** | K with transport at every stage, not the last one alone | **73.53** | +0.01 | 70.19 | 1.203 | -0.158 |
| **AH** | K, F and all three pairs at once, every addition together | **73.53** | +0.01 | 71.45 | 1.135 | -0.225 |
| **AL** | K with Gaussian transport on the per channel variances alone | **73.52** | +0.00 | 69.60 | 1.393 | +0.032 |
| **A** | US-Net as published: inplace distillation with KL | **73.52** | +0.00 | 70.10 | 1.361 | +0.000 |
| **AE** | K and F at once, transport on features and Jeffreys on logits | **73.42** | -0.10 | 70.53 | 1.175 | -0.186 |
| **EV** | ER (US-Net) on MobileNetV2 with the last 1x1 to 1280 slimmed (slim_last), with its BN and the classifier input, seed 1995, notebook 59. 73.41: slimming the last layer costs US-Net 0.61 against ER (74.02). The baseline EW is read against. | **73.41** | -0.11 | 69.49 | 1.099 | -0.262 |
| **AR** | K with the logit teacher softened to temperature 4 | **73.38** | -0.14 | 70.80 | 1.522 | +0.161 |
| **Y** | K at eps 0.5, mass spread over many partners | **73.28** | -0.24 | 70.13 | 1.354 | -0.007 |
| **FU** | US-Net on MobileNetV2 with the last layer slimmed, CIFAR-100, seed 2006 | **73.28** | -0.24 | 69.69 | 1.106 | -0.255 |
| **D** | C plus Wasserstein between the two middle widths | **73.27** | -0.25 | 70.30 | 1.361 | +0.000 |
| **E** | A plus plain KL between the two middle widths, neither symmetric nor metric aware | **73.27** | -0.25 | 69.80 | 1.276 | -0.085 |
| **G** | C with every pair of classes equally far apart, so transport has no geometry to use | **73.25** | -0.27 | 69.50 | 1.379 | +0.019 |
| **AI** | K transporting one shared channel at a time, exactly rather than by projection | **73.24** | -0.28 | 69.63 | 1.393 | +0.032 |
| **S** | K with sliced transport on 128 projections | **73.22** | -0.30 | 70.16 | 1.399 | +0.039 |
| **T** | K with sliced transport on 32 random projections rather than the entropic plan | **73.19** | -0.33 | 70.34 | 1.377 | +0.016 |
| **H** | C with the class metric taken from what the teacher confuses, rather than from the classifier rows | **73.16** | -0.36 | 69.90 | 1.343 | -0.018 |
| **GN** | AlphaNet on MobileNetV2 with the last layer slimmed, CIFAR-100, seed 1995 | **73.15** | -0.37 | 69.40 | 1.186 | -0.175 |
| **GP** | AlphaNet on MobileNetV2 with the last layer slimmed, CIFAR-100, seed 2006 | **73.08** | -0.44 | 69.51 | 1.174 | -0.187 |
| **B** | adaptive alpha-divergence, AlphaNet, bit-for-bit against the reference implementation | **73.08** | -0.44 | 70.40 | 1.573 | +0.212 |
| **AU** | K with KD weighted per sample by the teacher's entropy | **73.00** | -0.52 | 67.76 | 1.063 | -0.298 |
| **U** | the same on 512 projections, four times the cost of 128 | **72.99** | -0.53 | 69.87 | 1.409 | +0.049 |
| **GO** | AlphaNet on MobileNetV2 with the last layer slimmed, CIFAR-100, seed 2026 | **72.95** | -0.57 | 69.55 | 1.188 | -0.173 |
| **ET** | EP with the transport reading the last slimmed layer (feature_read last_slimmed, the 320 x width linear bottleneck), notebook 58. The log reads 1% at every width: the 1x1 to 1280 is unslimmed and its BN is a plain BatchNorm2d with one set of running statistics for all widths. Nothing here holds the widths' 1280 features together, and they ended nearly constant within a width (channel sd 0.006-0.02) with means 0.1 apart across widths, so statistics averaged over the sixteen widths are many sd off for each; train.py calibrates every width on one model. Recalibrated one width at a time from the checkpoint, locally: 72.94, below EP (73.68), ER (74.02) and EU (74.06), though that protocol only favours it. Not a train.py result: top1 from the notebook 58 checkpoint, each width calibrated alone on 20 train batches, no NLL. | **72.94** | -0.58 | 69.14 | nan | +nan |
| **BU** | K, width 0.25 alone for the first 25 epochs, then the full sandwich | **72.81** | -0.71 | 70.12 | 1.172 | -0.189 |
| **C** | Wasserstein on the logits, vertical only | **72.74** | -0.77 | 69.40 | 1.415 | +0.054 |
| **L** | D with the horizontal term weighted toward the narrow widths | **72.74** | -0.78 | 70.00 | 1.438 | +0.077 |
| **EK** | LCS (WACV 2023) in the authors' published configuration: lcs_l line subspace with their InstanceNorm (LinesAdaptiveIN), no running statistics and so no recalibration; authors' code at e6d3924 + third_party/patches/lcs.patch. 71.85, 3.4 under our BatchNorm version (EB 75.24) at every width, so the BN class we wrote is not what holds LCS down; EB stays as LCS's row | **71.85** | -1.67 | 69.14 | 1.054 | -0.307 |
| **DI** | US-Net's inplace KL kept and WKD-L's non-target transport (weight 600) added beside it, no student CE, ResNet-50, seed 1995: the full width sat at chance until epoch 12 and recovered only as the weight decayed | **71.42** | -2.10 | 69.66 | 1.618 | +0.257 |
| **GS** | DYNAS on MobileNetV2 with the last layer slimmed, CIFAR-100, seed 2006 | **70.18** | -3.33 | 57.13 | 1.177 | -0.184 |
| **GR** | DYNAS on MobileNetV2 with the last layer slimmed, CIFAR-100, seed 2026 | **69.95** | -3.57 | 55.75 | 1.179 | -0.182 |
| **GQ** | DYNAS on MobileNetV2 with the last layer slimmed, CIFAR-100, seed 1995 | **69.58** | -3.94 | 55.57 | 1.209 | -0.152 |
| **AG** | K with five sampled widths and every pair coupled: the narrowest never trained | **67.70** | -5.82 | 1.00 | 1.446 | +0.086 |
| **BT** | INVALID: BU plus a freeze that did not hold - weight decay and momentum shrank the frozen block to zero, width 0.25 ended at chance. Not a measurement of freezing | **64.39** | -9.13 | 1.18 | 1.417 | +0.056 |
| **DF** | K on ResNet-50 exactly as BY with amp: True: collapsed at seed 1995, where BY in fp32 did not; widths up to 0.55 near 50% error | **60.60** | -12.92 | 45.13 | 1.561 | +0.200 |
| **EQ** | SlimOT + WBH (DO recipe) on ResNet-18, Tiny ImageNet, seed 1995, notebook 57: the second dataset. 59.27: +1.14 against ES, above at 15/16 widths (0.25 level, -0.05; 0.40-0.60 +1.5 to +2.0; 0.80-1.0 +1.2 to +1.5), beside DO against BX on CIFAR-100 (+0.98). One seed. 630 min on a T4, one session. | **59.27** | -14.25 | 54.93 | 1.813 | +0.452 |
| **GE** | AlphaNet on ResNet-18, Tiny ImageNet, seed 1995 | **59.23** | -14.29 | 55.62 | 2.209 | +0.848 |
| **GF** | AlphaNet on ResNet-18, Tiny ImageNet, seed 2026 | **58.99** | -14.53 | 54.75 | 2.216 | +0.855 |
| **GG** | AlphaNet on ResNet-18, Tiny ImageNet, seed 2006 | **58.94** | -14.58 | 55.32 | 2.253 | +0.892 |
| **GM** | SOLAR† (US-Net + four band heads) on ResNet-18, Tiny ImageNet, seed 2006 | **58.85** | -14.67 | 55.35 | 1.996 | +0.635 |
| **GK** | SOLAR (one head per band of widths, four bands) on ResNet-18, Tiny ImageNet, seed 1995 | **58.79** | -14.73 | 54.24 | 2.018 | +0.657 |
| **GL** | SOLAR (one head per band of widths, four bands) on ResNet-18, Tiny ImageNet, seed 2026 | **58.39** | -15.12 | 54.13 | 2.025 | +0.664 |
| **ES** | US-Net as published (BX recipe) on ResNet-18, Tiny ImageNet (200 classes, 64x64, official val split as test), stem_stride 2, seed 1995, notebook 57: the baseline EQ is read against. 58.13. Width 1.0 ends at 0.15% train error against 39.5% on val. | **58.13** | -15.39 | 54.98 | 2.149 | +0.788 |
| **GI** | DYNAS on ResNet-18, Tiny ImageNet, seed 2026 | **50.08** | -23.43 | 30.21 | 2.447 | +1.086 |
| **GH** | DYNAS on ResNet-18, Tiny ImageNet, seed 1995 | **49.74** | -23.78 | 29.88 | 2.475 | +1.114 |
| **GJ** | DYNAS on ResNet-18, Tiny ImageNet, seed 2006 | **49.56** | -23.96 | 29.16 | 2.469 | +1.109 |
| **AJ** | K with per channel transport at an absolute gap: the narrow half of the range collapsed | **37.47** | -36.05 | 1.00 | 2.951 | +1.590 |
| **AV** | the same weighting reversed: collapsed to chance at every width below 0.70 | **30.46** | -43.06 | 1.00 | 3.230 | +1.869 |
| **AN** | K with sliced transport at an absolute gap: collapsed below width 0.80 | **20.27** | -53.25 | 1.00 | 3.669 | +2.308 |

## What the numbers can carry

Accuracy was logged to three decimals of error, so every value is a multiple of 0.10 and each one carries up to 0.05 of rounding. A difference between two of them carries up to **0.10**, and a difference of means over sixteen widths roughly 0.01 to 0.03.

A and A' are the same branch under different seeds. Their means differ by **0.01**, which is at that floor: what the pair establishes is that seed noise on the mean sits under it, not that it equals 0.01. Per width the same pair differs by as much as **0.40**, so the worst-width column is noise and should not be ranked.

So the gaps worth reading are the ones far outside that floor: A over C by 0.77, over B by 0.44, over D by 0.25. F against A, at +0.03, is not one of them.

NLL is printed to three decimals on a scale near 1.3, so it resolves to 0.001 and none of this applies to it.

## Accuracy by width

| width | MACs (M) | ED | DM | EC | DO | EE | EI | DQ | DP | FI | FL | DN | EN | FH | DT | DD | DU | FF | FC | DK | DR | DC | CI | BY | FG | CH | FM | DS | EO | BX | FD | CF | FE | FB | FN | DL | DJ | EF | CP | CQ | GX | DE | CV | FZ | FY | CU | CT | EB | GZ | DG | GC | BP | AW | GA | K | X | GB | V | EW | BO | AO | BR | AQ | EU | AF | EA | ER | BE | BD | AT | BI | AC | GD | GU | BN | AS | DB | FV | BS | GY | BF | GT | AD | BG | AA | W | GV | BH | GW | EP | I | AM | BQ | AK | F | A' | M | AH | AL | A | AE | EV | AR | Y | FU | D | E | G | AI | S | T | H | GN | GP | B | AU | U | GO | ET | BU | C | L | EK | DI | GS | GR | GQ | AG | BT | DF | EQ | GE | GF | GG | GM | GK | GL | ES | GI | GH | GJ | AJ | AV | AN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 35.1 | **76.61** | 75.89 | 76.34 | 76.12 | 75.87 | 76.00 | 75.64 | 75.79 | 76.05 | 75.84 | 75.91 | 75.16 | 75.51 | 75.74 | 75.63 | 75.76 | 75.20 | 75.39 | 75.26 | 75.22 | 75.47 | 75.60 | 75.12 | 75.51 | 75.42 | 75.37 | 75.53 | 75.35 | 75.26 | 75.20 | 75.33 | 74.83 | 75.38 | 74.96 | 75.20 | 74.91 | 75.17 | 75.41 | 74.62 | 74.49 | 74.60 | 74.16 | 74.65 | 74.41 | 73.76 | 73.52 | 73.89 | 73.70 | 73.68 | 73.66 | 70.90 | 71.37 | 67.14 | 70.84 | 71.38 | 67.58 | 71.11 | 69.96 | 70.98 | 71.00 | 70.87 | 70.79 | 70.36 | 70.95 | 73.38 | 70.92 | 70.45 | 71.23 | 70.71 | 70.60 | 70.37 | 72.79 | 70.22 | 71.01 | 70.94 | 67.27 | 69.80 | 71.23 | 73.28 | 70.74 | 69.87 | 70.80 | 70.21 | 70.88 | 70.79 | 70.12 | 70.65 | 73.30 | 70.02 | 70.78 | 70.68 | 70.38 | 70.22 | 70.30 | 69.80 | 70.19 | 71.45 | 69.60 | 70.10 | 70.53 | 69.49 | 70.80 | 70.13 | 69.69 | 70.30 | 69.80 | 69.50 | 69.63 | 70.16 | 70.34 | 69.90 | 69.40 | 69.51 | 70.40 | 67.76 | 69.87 | 69.55 | 74.65 | 70.12 | 69.40 | 70.00 | 69.14 | 69.66 | 57.13 | 55.75 | 55.57 | 1.00 | 1.18 | 45.13 | 54.93 | 55.62 | 54.75 | 55.32 | 55.35 | 54.24 | 54.13 | 54.98 | 30.21 | 29.88 | 29.16 | 1.00 | 1.00 | 1.00 |
| 0.30 | 60.5 | **77.06** | 76.62 | 76.68 | 76.63 | 76.33 | 76.57 | 76.24 | 76.10 | 76.03 | 76.23 | 76.03 | 75.93 | 76.08 | 76.26 | 76.29 | 76.24 | 75.58 | 76.08 | 75.98 | 75.88 | 75.76 | 75.77 | 75.75 | 75.92 | 75.71 | 75.71 | 75.77 | 75.59 | 75.35 | 75.00 | 75.83 | 75.50 | 75.73 | 75.40 | 75.14 | 75.04 | 75.55 | 75.39 | 75.29 | 74.55 | 75.17 | 74.60 | 74.30 | 74.71 | 74.02 | 74.19 | 73.81 | 73.62 | 74.02 | 73.72 | 71.97 | 71.53 | 68.34 | 71.73 | 71.95 | 68.66 | 71.74 | 71.95 | 71.64 | 71.25 | 71.92 | 71.44 | 72.01 | 71.84 | 73.20 | 72.94 | 71.06 | 71.44 | 71.19 | 71.05 | 71.08 | 72.93 | 71.60 | 71.61 | 71.72 | 68.75 | 71.55 | 71.90 | 73.22 | 71.34 | 71.27 | 71.15 | 70.74 | 71.67 | 71.07 | 71.39 | 71.49 | 73.09 | 71.57 | 71.29 | 70.87 | 71.15 | 70.82 | 71.30 | 70.70 | 71.21 | 71.78 | 70.54 | 70.80 | 71.60 | 71.34 | 71.67 | 70.81 | 71.62 | 71.30 | 70.90 | 70.60 | 70.31 | 70.75 | 71.02 | 70.60 | 71.06 | 71.28 | 71.10 | 68.96 | 70.46 | 71.22 | 74.37 | 70.62 | 70.40 | 70.80 | 69.58 | 69.80 | 62.15 | 60.95 | 60.63 | 58.76 | 57.17 | 45.57 | 56.03 | 56.03 | 55.70 | 56.16 | 56.04 | 55.73 | 55.23 | 55.50 | 33.80 | 32.92 | 33.54 | 1.00 | 1.00 | 1.00 |
| 0.35 | 72.7 | **77.41** | 76.99 | 77.15 | 77.07 | 76.83 | 76.79 | 76.70 | 76.30 | 76.76 | 76.72 | 76.49 | 76.21 | 76.53 | 76.54 | 76.54 | 76.55 | 75.82 | 76.38 | 76.05 | 76.35 | 76.56 | 76.31 | 76.34 | 76.36 | 76.14 | 76.03 | 76.08 | 76.09 | 75.54 | 75.60 | 76.49 | 75.93 | 75.75 | 75.81 | 75.94 | 75.52 | 75.57 | 75.73 | 75.44 | 75.19 | 75.19 | 74.94 | 74.95 | 75.07 | 74.45 | 74.48 | 74.04 | 73.83 | 74.24 | 73.76 | 72.89 | 72.21 | 69.87 | 72.36 | 72.67 | 70.41 | 72.33 | 72.14 | 72.37 | 71.75 | 72.36 | 72.05 | 72.26 | 72.56 | 72.98 | 73.01 | 72.00 | 71.99 | 71.73 | 72.07 | 71.64 | 73.35 | 71.75 | 71.86 | 72.07 | 70.74 | 72.01 | 72.57 | 73.16 | 71.76 | 71.87 | 71.91 | 71.72 | 72.23 | 71.92 | 71.47 | 72.17 | 72.58 | 71.96 | 71.58 | 71.41 | 71.51 | 71.51 | 72.00 | 71.50 | 71.43 | 72.09 | 71.36 | 71.50 | 72.18 | 72.15 | 71.63 | 71.40 | 71.44 | 71.60 | 71.60 | 71.20 | 70.99 | 71.09 | 71.57 | 70.80 | 71.05 | 71.62 | 71.40 | 70.00 | 71.15 | 71.18 | 74.29 | 70.91 | 70.90 | 70.80 | 70.19 | 70.23 | 62.99 | 61.81 | 61.99 | 66.74 | 64.76 | 45.72 | 56.56 | 56.96 | 56.55 | 56.91 | 56.26 | 56.37 | 56.08 | 55.96 | 37.06 | 35.98 | 35.88 | 1.00 | 1.00 | 1.00 |
| 0.40 | 84.8 | **77.62** | 77.59 | 77.51 | 77.39 | 77.08 | 76.86 | 77.05 | 76.97 | 76.82 | 76.91 | 76.80 | 76.77 | 76.86 | 77.04 | 76.97 | 77.13 | 76.59 | 76.79 | 76.38 | 76.92 | 76.87 | 76.56 | 76.75 | 76.61 | 76.53 | 76.64 | 76.49 | 76.62 | 76.01 | 76.10 | 76.82 | 76.33 | 76.30 | 76.19 | 76.11 | 75.90 | 75.99 | 75.84 | 75.79 | 75.51 | 75.66 | 75.55 | 75.12 | 75.12 | 75.02 | 74.92 | 74.41 | 74.00 | 74.28 | 74.19 | 72.85 | 73.11 | 71.93 | 72.69 | 73.24 | 71.93 | 72.92 | 72.65 | 72.92 | 72.93 | 73.01 | 72.76 | 73.55 | 72.69 | 72.82 | 73.23 | 72.46 | 72.97 | 72.52 | 72.58 | 72.66 | 73.49 | 72.77 | 73.10 | 72.91 | 71.97 | 72.33 | 73.16 | 72.79 | 72.32 | 72.70 | 72.51 | 72.59 | 72.45 | 72.37 | 72.62 | 72.30 | 73.02 | 72.70 | 72.22 | 72.21 | 72.58 | 72.54 | 72.40 | 72.60 | 72.15 | 72.56 | 72.12 | 72.20 | 72.58 | 72.35 | 72.21 | 72.18 | 72.16 | 72.20 | 72.00 | 71.90 | 71.92 | 71.46 | 71.91 | 71.50 | 72.13 | 72.27 | 71.80 | 71.07 | 71.94 | 72.06 | 74.15 | 71.76 | 71.30 | 71.30 | 70.81 | 70.26 | 66.39 | 65.26 | 65.78 | 69.34 | 67.22 | 46.35 | 57.92 | 57.42 | 57.31 | 57.21 | 57.43 | 57.28 | 56.53 | 56.43 | 40.65 | 40.11 | 38.72 | 1.00 | 1.00 | 1.00 |
| 0.45 | 118.0 | **77.76** | 77.53 | 77.72 | 77.73 | 77.36 | 77.11 | 77.25 | 77.43 | 77.34 | 77.07 | 77.06 | 77.02 | 76.93 | 77.75 | 77.67 | 77.65 | 77.35 | 77.27 | 76.93 | 77.00 | 76.97 | 76.81 | 76.80 | 76.80 | 76.98 | 76.86 | 76.87 | 76.81 | 76.30 | 76.22 | 76.62 | 76.75 | 76.77 | 76.68 | 76.28 | 75.97 | 76.09 | 76.51 | 76.03 | 75.71 | 75.81 | 75.61 | 75.47 | 75.68 | 75.32 | 74.89 | 74.81 | 74.31 | 74.49 | 74.35 | 73.83 | 74.00 | 73.82 | 73.29 | 73.95 | 73.38 | 73.46 | 73.22 | 73.13 | 73.45 | 73.61 | 73.59 | 73.55 | 73.30 | 73.06 | 73.45 | 72.95 | 73.30 | 73.24 | 73.30 | 73.28 | 73.74 | 73.19 | 73.35 | 73.31 | 73.39 | 73.06 | 73.68 | 72.47 | 73.34 | 73.04 | 73.18 | 73.20 | 73.14 | 72.73 | 72.94 | 72.85 | 73.21 | 73.34 | 73.09 | 72.82 | 73.06 | 73.01 | 72.80 | 72.70 | 72.78 | 73.15 | 72.77 | 72.80 | 72.82 | 73.04 | 72.66 | 72.41 | 72.57 | 72.60 | 72.80 | 72.40 | 72.36 | 71.85 | 72.50 | 72.40 | 72.85 | 72.74 | 72.20 | 71.71 | 72.09 | 72.52 | 74.28 | 71.99 | 72.20 | 72.00 | 71.18 | 70.83 | 68.25 | 68.05 | 67.57 | 71.37 | 68.44 | 47.45 | 58.43 | 58.16 | 58.40 | 58.29 | 57.68 | 57.54 | 57.06 | 57.48 | 45.31 | 45.56 | 44.79 | 1.00 | 1.00 | 1.00 |
| 0.50 | 139.3 | 78.08 | 78.10 | **78.25** | 77.87 | 77.69 | 77.71 | 77.80 | 77.49 | 77.68 | 77.74 | 77.56 | 77.51 | 77.51 | 77.46 | 77.43 | 77.48 | 77.28 | 77.49 | 77.23 | 77.26 | 77.10 | 77.50 | 77.16 | 76.96 | 77.03 | 77.04 | 77.00 | 76.88 | 76.77 | 76.79 | 76.73 | 76.85 | 76.93 | 76.64 | 76.49 | 76.19 | 76.68 | 76.67 | 76.40 | 75.99 | 76.04 | 75.84 | 75.90 | 75.97 | 75.52 | 75.52 | 75.07 | 74.45 | 74.56 | 74.78 | 73.96 | 74.10 | 74.32 | 74.00 | 74.24 | 74.31 | 73.88 | 74.07 | 73.78 | 73.85 | 74.17 | 74.08 | 73.81 | 73.63 | 73.36 | 73.88 | 73.44 | 73.48 | 73.76 | 73.76 | 73.77 | 73.98 | 73.50 | 73.64 | 73.63 | 73.87 | 73.78 | 73.91 | 73.01 | 73.77 | 73.75 | 73.74 | 73.49 | 73.64 | 73.35 | 73.67 | 73.27 | 72.94 | 73.68 | 73.58 | 73.22 | 73.48 | 73.49 | 73.50 | 73.10 | 73.14 | 73.36 | 73.11 | 73.20 | 73.03 | 73.22 | 73.19 | 73.06 | 72.71 | 72.80 | 73.30 | 73.20 | 73.06 | 72.75 | 72.77 | 72.70 | 73.29 | 73.00 | 72.60 | 72.74 | 72.66 | 72.90 | 73.99 | 72.41 | 72.30 | 72.30 | 71.29 | 70.85 | 69.75 | 69.42 | 69.31 | 72.45 | 69.36 | 48.61 | 59.18 | 58.99 | 59.09 | 58.89 | 58.60 | 58.18 | 57.95 | 57.26 | 48.56 | 48.50 | 48.04 | 1.00 | 1.00 | 1.00 |
| 0.55 | 163.2 | 78.20 | 78.24 | **78.34** | 77.88 | 77.98 | 77.96 | 77.97 | 77.71 | 77.64 | 77.98 | 77.90 | 77.85 | 77.48 | 77.50 | 77.58 | 77.56 | 77.74 | 77.59 | 77.23 | 77.54 | 77.33 | 77.57 | 77.35 | 77.30 | 77.31 | 77.15 | 77.29 | 77.13 | 76.63 | 76.92 | 76.63 | 76.95 | 76.66 | 76.85 | 76.76 | 76.92 | 76.46 | 76.71 | 76.63 | 76.20 | 76.13 | 76.24 | 75.93 | 75.88 | 75.47 | 75.70 | 75.26 | 74.76 | 74.99 | 74.77 | 74.50 | 74.70 | 75.04 | 74.60 | 74.32 | 75.05 | 74.04 | 74.42 | 74.32 | 74.39 | 74.42 | 74.38 | 73.99 | 73.97 | 73.52 | 74.20 | 74.26 | 74.10 | 74.16 | 74.01 | 73.92 | 73.99 | 73.98 | 74.01 | 73.85 | 74.33 | 74.29 | 74.11 | 73.29 | 74.05 | 73.39 | 74.27 | 73.85 | 73.75 | 73.60 | 73.87 | 73.33 | 73.15 | 73.87 | 73.92 | 73.95 | 73.74 | 73.45 | 73.60 | 73.40 | 73.44 | 73.63 | 73.85 | 73.40 | 73.43 | 73.35 | 73.62 | 73.27 | 73.42 | 73.30 | 73.60 | 73.40 | 73.66 | 73.09 | 73.17 | 73.20 | 73.14 | 73.32 | 72.90 | 73.43 | 72.89 | 73.12 | 73.52 | 72.84 | 72.60 | 72.50 | 71.68 | 71.40 | 70.39 | 70.43 | 70.02 | 72.99 | 69.33 | 50.61 | 59.72 | 59.83 | 59.38 | 59.58 | 58.55 | 58.61 | 58.16 | 57.69 | 51.02 | 50.91 | 50.19 | 1.00 | 1.00 | 1.00 |
| 0.60 | 207.6 | 78.29 | **78.47** | 78.44 | 78.26 | 78.18 | 78.10 | 77.65 | 78.07 | 78.06 | 77.94 | 77.93 | 77.87 | 78.18 | 77.65 | 77.62 | 77.59 | 77.82 | 77.45 | 77.43 | 77.60 | 77.45 | 77.55 | 77.70 | 77.54 | 77.37 | 77.49 | 77.38 | 76.98 | 77.03 | 77.20 | 76.85 | 77.00 | 76.90 | 77.13 | 76.88 | 76.89 | 76.71 | 76.85 | 76.63 | 76.51 | 76.48 | 76.27 | 75.95 | 76.09 | 75.83 | 75.85 | 75.37 | 75.00 | 74.96 | 74.92 | 74.94 | 74.57 | 75.27 | 74.81 | 74.68 | 75.20 | 74.74 | 75.14 | 74.58 | 75.06 | 74.45 | 74.76 | 74.44 | 74.35 | 73.91 | 74.19 | 74.60 | 74.38 | 74.61 | 74.40 | 74.51 | 74.25 | 74.39 | 74.32 | 74.40 | 74.95 | 74.61 | 74.21 | 73.59 | 74.30 | 74.52 | 74.44 | 73.99 | 74.12 | 74.23 | 74.01 | 73.99 | 73.61 | 74.23 | 74.10 | 73.86 | 74.05 | 73.89 | 73.60 | 73.80 | 74.10 | 73.61 | 74.30 | 73.80 | 73.78 | 73.79 | 73.92 | 73.60 | 73.68 | 73.70 | 73.80 | 73.40 | 73.69 | 73.64 | 73.59 | 73.50 | 73.89 | 73.34 | 73.30 | 73.72 | 73.30 | 73.65 | 73.53 | 73.21 | 73.00 | 73.00 | 72.00 | 71.54 | 72.12 | 72.66 | 71.87 | 73.52 | 69.59 | 66.52 | 59.80 | 59.54 | 59.75 | 59.33 | 59.17 | 58.93 | 59.24 | 58.27 | 53.35 | 53.40 | 53.12 | 1.00 | 1.00 | 1.00 |
| 0.65 | 227.7 | 78.53 | **78.74** | 78.32 | 77.96 | 78.18 | 78.09 | 77.98 | 77.94 | 78.11 | 77.83 | 78.10 | 78.18 | 77.94 | 77.92 | 77.83 | 77.86 | 77.94 | 77.62 | 77.91 | 77.74 | 77.76 | 77.50 | 77.68 | 77.34 | 77.44 | 77.47 | 77.48 | 77.40 | 77.13 | 77.17 | 76.72 | 77.18 | 76.89 | 77.05 | 76.75 | 77.13 | 76.96 | 76.79 | 76.69 | 76.69 | 76.38 | 76.42 | 76.36 | 76.09 | 75.85 | 75.78 | 75.58 | 75.38 | 75.09 | 74.83 | 75.06 | 74.86 | 75.94 | 74.83 | 74.85 | 75.96 | 74.76 | 75.01 | 74.84 | 74.96 | 74.62 | 74.71 | 74.77 | 74.26 | 74.01 | 74.46 | 74.88 | 74.73 | 74.97 | 74.77 | 74.52 | 74.30 | 74.33 | 74.38 | 74.50 | 75.48 | 74.71 | 74.44 | 73.78 | 74.84 | 74.64 | 74.49 | 74.48 | 74.58 | 74.61 | 74.16 | 74.34 | 73.65 | 74.26 | 74.30 | 74.23 | 74.39 | 73.86 | 74.00 | 74.20 | 74.56 | 73.99 | 74.44 | 73.90 | 74.09 | 74.41 | 73.73 | 73.61 | 73.74 | 74.00 | 73.80 | 73.80 | 74.00 | 74.17 | 73.62 | 73.80 | 73.47 | 73.66 | 73.60 | 74.05 | 73.67 | 73.64 | 73.36 | 73.37 | 73.40 | 73.20 | 72.31 | 71.55 | 72.71 | 73.12 | 72.53 | 73.84 | 69.73 | 70.00 | 59.47 | 60.18 | 59.90 | 59.46 | 59.43 | 59.26 | 58.97 | 58.79 | 55.01 | 54.24 | 54.50 | 66.62 | 1.00 | 1.00 |
| 0.70 | 280.2 | 78.61 | **78.74** | 78.33 | 78.34 | 78.32 | 78.13 | 78.09 | 78.22 | 78.05 | 78.13 | 78.27 | 78.22 | 78.13 | 78.02 | 78.02 | 77.99 | 78.31 | 77.89 | 77.98 | 77.76 | 77.56 | 77.79 | 77.72 | 77.60 | 77.85 | 77.51 | 77.46 | 77.55 | 77.20 | 77.31 | 76.96 | 77.10 | 77.03 | 77.16 | 76.87 | 76.99 | 76.75 | 76.84 | 76.70 | 76.75 | 76.51 | 76.64 | 76.27 | 76.21 | 75.97 | 75.62 | 75.57 | 75.65 | 74.93 | 75.15 | 75.34 | 75.20 | 76.38 | 75.14 | 74.87 | 76.02 | 75.04 | 75.28 | 75.32 | 75.16 | 74.87 | 74.68 | 74.70 | 74.80 | 74.37 | 74.62 | 74.77 | 74.80 | 75.09 | 74.92 | 74.75 | 74.17 | 74.86 | 74.39 | 74.61 | 75.36 | 74.86 | 74.45 | 74.33 | 74.99 | 75.02 | 74.58 | 74.59 | 74.58 | 74.83 | 74.55 | 74.38 | 73.60 | 74.60 | 74.31 | 74.55 | 74.69 | 74.29 | 74.20 | 74.50 | 74.43 | 74.08 | 74.60 | 74.30 | 74.10 | 74.18 | 74.09 | 73.91 | 73.89 | 74.10 | 73.90 | 74.10 | 74.05 | 74.20 | 74.01 | 74.00 | 73.90 | 73.87 | 73.60 | 74.36 | 73.63 | 73.61 | 73.08 | 73.51 | 73.30 | 73.40 | 72.35 | 71.84 | 73.28 | 73.26 | 72.92 | 74.26 | 70.19 | 70.81 | 60.05 | 60.50 | 59.74 | 59.79 | 59.34 | 59.92 | 59.50 | 59.18 | 56.08 | 55.42 | 56.00 | 73.05 | 52.72 | 1.09 |
| 0.75 | 312.8 | 78.68 | **78.78** | 78.55 | 78.32 | 78.62 | 78.28 | 78.29 | 78.49 | 78.20 | 78.20 | 78.16 | 78.49 | 78.15 | 78.11 | 78.00 | 77.95 | 78.19 | 77.95 | 77.86 | 77.78 | 77.75 | 77.79 | 77.70 | 77.77 | 77.71 | 77.55 | 77.70 | 77.70 | 77.39 | 77.53 | 77.08 | 77.10 | 76.89 | 77.07 | 77.11 | 76.82 | 76.74 | 76.87 | 76.92 | 77.26 | 76.50 | 76.88 | 76.46 | 76.17 | 76.06 | 76.02 | 75.74 | 76.00 | 74.86 | 75.08 | 75.43 | 75.49 | 76.60 | 75.23 | 75.02 | 76.23 | 75.08 | 75.57 | 75.07 | 75.04 | 74.90 | 75.15 | 74.80 | 75.10 | 74.25 | 74.61 | 75.23 | 74.94 | 75.00 | 75.03 | 74.90 | 74.34 | 74.80 | 74.74 | 74.73 | 75.97 | 74.98 | 74.61 | 74.58 | 74.95 | 74.95 | 74.80 | 74.86 | 74.77 | 74.78 | 74.85 | 74.57 | 73.87 | 74.69 | 74.64 | 74.69 | 74.62 | 74.60 | 74.60 | 74.60 | 74.65 | 74.21 | 74.67 | 74.60 | 74.24 | 74.32 | 74.23 | 74.18 | 74.12 | 74.10 | 74.10 | 74.30 | 74.16 | 74.51 | 73.85 | 74.20 | 74.29 | 73.97 | 74.10 | 74.77 | 73.93 | 73.88 | 72.78 | 73.55 | 73.60 | 73.60 | 72.48 | 71.90 | 73.79 | 73.94 | 73.62 | 74.44 | 70.19 | 71.49 | 60.41 | 60.67 | 60.04 | 59.63 | 60.06 | 59.99 | 59.66 | 59.41 | 56.86 | 55.89 | 56.39 | 74.01 | 68.06 | 1.08 |
| 0.80 | 347.9 | 78.77 | **78.86** | 78.67 | 78.43 | 78.60 | 78.29 | 78.43 | 78.48 | 78.41 | 78.10 | 78.35 | 78.35 | 78.14 | 78.01 | 78.01 | 77.80 | 78.27 | 77.85 | 78.19 | 77.84 | 77.87 | 78.07 | 77.91 | 77.81 | 78.00 | 77.83 | 77.54 | 77.61 | 77.56 | 77.68 | 77.17 | 77.09 | 77.26 | 77.00 | 77.27 | 77.18 | 77.02 | 76.92 | 77.06 | 77.51 | 76.77 | 77.08 | 76.40 | 76.25 | 75.99 | 75.98 | 75.84 | 76.22 | 75.08 | 75.33 | 75.28 | 75.43 | 76.39 | 75.44 | 75.25 | 76.31 | 75.31 | 75.37 | 75.41 | 75.34 | 75.06 | 75.21 | 74.95 | 74.92 | 74.59 | 74.75 | 75.19 | 75.02 | 75.06 | 75.16 | 75.31 | 74.29 | 75.10 | 75.20 | 75.06 | 75.88 | 74.84 | 74.79 | 74.71 | 75.10 | 75.07 | 75.01 | 74.95 | 74.55 | 74.93 | 74.88 | 74.72 | 74.06 | 74.62 | 74.81 | 74.88 | 74.62 | 74.84 | 74.60 | 74.70 | 74.56 | 74.34 | 74.80 | 74.80 | 74.42 | 74.48 | 74.45 | 74.30 | 74.02 | 74.40 | 74.20 | 74.50 | 74.42 | 74.71 | 74.26 | 74.30 | 74.04 | 73.82 | 74.20 | 74.87 | 74.19 | 73.91 | 72.37 | 73.81 | 73.90 | 73.70 | 73.07 | 71.84 | 73.99 | 74.37 | 73.98 | 74.57 | 70.62 | 71.89 | 60.90 | 60.44 | 60.29 | 60.00 | 60.44 | 60.76 | 60.32 | 59.39 | 57.95 | 57.58 | 57.50 | 74.85 | 70.14 | 39.78 |
| 0.85 | 411.6 | 78.56 | **78.86** | 78.45 | 78.37 | 78.45 | 78.57 | 78.45 | 78.38 | 78.27 | 78.32 | 78.28 | 78.42 | 78.35 | 78.06 | 78.09 | 77.98 | 78.15 | 77.98 | 78.04 | 77.85 | 77.77 | 77.82 | 77.87 | 77.98 | 78.02 | 77.80 | 77.40 | 77.76 | 77.60 | 77.60 | 77.34 | 77.35 | 77.24 | 77.01 | 77.32 | 77.23 | 76.97 | 76.86 | 77.21 | 77.43 | 76.91 | 77.09 | 76.53 | 76.55 | 76.17 | 76.02 | 75.85 | 76.30 | 75.10 | 75.20 | 75.41 | 75.46 | 77.00 | 75.72 | 75.33 | 76.52 | 75.27 | 75.36 | 75.55 | 75.32 | 75.29 | 75.44 | 75.45 | 75.41 | 74.95 | 74.91 | 75.65 | 75.02 | 75.27 | 75.18 | 75.34 | 74.26 | 75.34 | 75.21 | 75.10 | 76.20 | 75.37 | 74.80 | 74.56 | 75.07 | 75.28 | 74.94 | 75.36 | 74.78 | 74.95 | 75.10 | 74.97 | 74.16 | 74.87 | 74.83 | 74.99 | 74.79 | 75.01 | 74.80 | 74.70 | 74.74 | 74.44 | 74.90 | 75.10 | 74.35 | 74.46 | 74.38 | 74.55 | 74.60 | 74.30 | 74.40 | 74.70 | 74.66 | 74.50 | 74.37 | 74.60 | 74.72 | 74.23 | 74.20 | 74.92 | 74.28 | 74.06 | 71.81 | 73.95 | 74.10 | 73.90 | 73.26 | 72.69 | 74.68 | 74.82 | 74.15 | 74.86 | 70.36 | 72.13 | 61.18 | 60.80 | 60.53 | 60.58 | 60.49 | 60.55 | 59.91 | 59.72 | 58.52 | 58.36 | 58.19 | 75.24 | 71.08 | 64.47 |
| 0.90 | 439.8 | 78.77 | **79.05** | 78.48 | 78.23 | 78.55 | 78.57 | 78.34 | 78.35 | 78.33 | 78.22 | 78.35 | 78.53 | 78.32 | 78.11 | 78.23 | 78.08 | 78.34 | 78.17 | 78.15 | 77.88 | 78.03 | 77.88 | 78.06 | 77.82 | 77.95 | 78.00 | 77.58 | 77.79 | 77.99 | 77.69 | 77.59 | 77.24 | 77.14 | 77.15 | 77.19 | 77.33 | 77.18 | 76.81 | 77.26 | 77.67 | 77.04 | 77.12 | 76.75 | 76.66 | 76.21 | 76.03 | 76.19 | 76.46 | 75.07 | 75.21 | 75.73 | 75.64 | 76.85 | 75.73 | 75.30 | 76.57 | 75.49 | 75.61 | 75.56 | 75.46 | 75.43 | 75.33 | 75.48 | 75.47 | 75.09 | 75.03 | 75.48 | 75.14 | 75.28 | 75.51 | 75.50 | 74.33 | 75.49 | 75.31 | 75.36 | 76.21 | 75.45 | 74.90 | 75.07 | 75.31 | 75.44 | 74.99 | 75.45 | 74.88 | 75.20 | 75.07 | 75.46 | 74.72 | 74.70 | 74.86 | 75.22 | 74.81 | 75.13 | 74.90 | 75.20 | 74.85 | 74.47 | 74.95 | 75.10 | 74.53 | 74.57 | 74.58 | 74.80 | 74.84 | 74.60 | 74.50 | 74.80 | 74.95 | 74.77 | 74.76 | 74.80 | 74.61 | 74.22 | 74.50 | 75.19 | 74.39 | 74.01 | 71.07 | 74.16 | 74.20 | 74.30 | 73.27 | 72.60 | 74.92 | 75.23 | 74.43 | 74.92 | 70.39 | 72.16 | 60.90 | 60.92 | 60.61 | 60.61 | 60.51 | 60.95 | 60.23 | 59.54 | 58.50 | 58.70 | 58.64 | 75.72 | 71.94 | 68.58 |
| 0.95 | 511.6 | 78.84 | **79.12** | 78.46 | 78.51 | 78.57 | 78.67 | 78.52 | 78.52 | 78.27 | 78.27 | 78.27 | 78.52 | 78.35 | 78.23 | 78.08 | 78.11 | 78.15 | 78.14 | 78.22 | 77.88 | 78.01 | 77.84 | 78.14 | 77.99 | 77.94 | 77.95 | 77.76 | 77.65 | 78.01 | 77.76 | 76.90 | 76.94 | 77.26 | 77.12 | 77.18 | 77.45 | 77.08 | 76.91 | 77.19 | 77.74 | 77.11 | 77.23 | 77.05 | 76.80 | 76.16 | 75.97 | 76.30 | 76.46 | 75.11 | 75.28 | 75.65 | 75.73 | 77.16 | 75.82 | 75.39 | 76.79 | 75.84 | 75.52 | 75.44 | 75.60 | 75.36 | 75.63 | 75.45 | 75.71 | 75.46 | 75.02 | 75.66 | 75.36 | 75.37 | 75.43 | 75.75 | 74.52 | 75.88 | 75.34 | 75.39 | 76.23 | 75.48 | 74.97 | 75.15 | 75.23 | 75.58 | 75.18 | 75.41 | 74.92 | 75.28 | 75.40 | 75.49 | 74.82 | 74.79 | 74.96 | 75.36 | 75.15 | 75.06 | 74.90 | 75.40 | 75.13 | 74.47 | 75.08 | 75.40 | 74.54 | 74.75 | 74.60 | 75.13 | 75.02 | 74.50 | 74.70 | 74.90 | 74.89 | 74.80 | 74.58 | 75.00 | 74.36 | 74.34 | 74.70 | 75.17 | 74.59 | 73.81 | 70.61 | 74.37 | 74.60 | 74.50 | 73.50 | 72.74 | 75.31 | 75.29 | 74.32 | 74.97 | 70.93 | 72.46 | 61.23 | 60.75 | 60.87 | 60.53 | 60.89 | 60.94 | 60.44 | 59.98 | 59.16 | 59.04 | 59.14 | 75.98 | 72.08 | 69.98 |
| 1.00 | 555.5 | 78.80 | **78.92** | 78.55 | 78.29 | 78.64 | 78.49 | 78.32 | 78.30 | 78.51 | 78.35 | 78.14 | 78.52 | 78.29 | 78.03 | 77.95 | 77.94 | 78.23 | 78.08 | 78.12 | 77.96 | 78.04 | 77.90 | 78.06 | 78.06 | 77.95 | 78.24 | 77.57 | 77.59 | 78.04 | 77.94 | 77.75 | 77.61 | 77.48 | 77.41 | 77.36 | 77.45 | 77.19 | 76.84 | 77.51 | 77.60 | 77.16 | 77.26 | 76.96 | 76.85 | 75.99 | 76.00 | 76.17 | 76.58 | 75.12 | 75.24 | 75.82 | 75.77 | 77.10 | 76.00 | 75.42 | 76.79 | 75.83 | 75.54 | 75.58 | 75.68 | 75.70 | 75.60 | 75.40 | 75.56 | 75.39 | 75.10 | 75.66 | 75.46 | 75.36 | 75.49 | 75.88 | 74.37 | 75.80 | 75.39 | 75.25 | 76.18 | 75.54 | 74.87 | 75.53 | 75.36 | 75.69 | 75.20 | 75.57 | 75.05 | 75.27 | 75.73 | 75.45 | 75.42 | 74.93 | 75.21 | 75.34 | 75.10 | 75.27 | 75.20 | 75.60 | 75.13 | 74.84 | 75.22 | 75.30 | 74.50 | 74.63 | 74.38 | 75.10 | 74.89 | 74.50 | 74.90 | 75.30 | 75.14 | 75.01 | 74.75 | 75.30 | 74.21 | 74.13 | 74.70 | 75.33 | 74.73 | 74.11 | 69.14 | 74.40 | 74.70 | 74.50 | 73.51 | 72.93 | 75.09 | 74.84 | 74.57 | 75.11 | 70.72 | 72.64 | 61.62 | 60.86 | 60.98 | 60.72 | 61.35 | 61.34 | 60.89 | 60.46 | 59.30 | 59.28 | 59.20 | 76.05 | 72.39 | 70.28 |
| **mean** | 248.0 | **78.16** | 78.16 | 78.02 | 77.84 | 77.83 | 77.76 | 77.67 | 77.66 | 77.66 | 77.62 | 77.60 | 77.60 | 77.55 | 77.53 | 77.50 | 77.48 | 77.44 | 77.38 | 77.31 | 77.28 | 77.27 | 77.27 | 77.26 | 77.21 | 77.21 | 77.17 | 77.06 | 77.03 | 76.86 | 76.86 | 76.80 | 76.73 | 76.73 | 76.66 | 76.62 | 76.56 | 76.51 | 76.50 | 76.46 | 76.42 | 76.22 | 76.18 | 75.94 | 75.91 | 75.49 | 75.41 | 75.24 | 75.17 | 74.72 | 74.72 | 74.35 | 74.32 | 74.32 | 74.26 | 74.24 | 74.23 | 74.18 | 74.18 | 74.16 | 74.14 | 74.13 | 74.10 | 74.06 | 74.03 | 74.02 | 74.02 | 73.98 | 73.96 | 73.96 | 73.95 | 73.95 | 73.94 | 73.94 | 73.93 | 73.93 | 73.92 | 73.92 | 73.91 | 73.91 | 73.90 | 73.88 | 73.82 | 73.78 | 73.75 | 73.74 | 73.74 | 73.71 | 73.70 | 73.68 | 73.66 | 73.64 | 73.63 | 73.56 | 73.54 | 73.53 | 73.53 | 73.53 | 73.52 | 73.52 | 73.42 | 73.41 | 73.38 | 73.28 | 73.28 | 73.27 | 73.27 | 73.25 | 73.24 | 73.22 | 73.19 | 73.16 | 73.15 | 73.08 | 73.08 | 73.00 | 72.99 | 72.95 | 72.94 | 72.81 | 72.74 | 72.74 | 71.85 | 71.42 | 70.18 | 69.95 | 69.58 | 67.70 | 64.39 | 60.60 | 59.27 | 59.23 | 58.99 | 58.94 | 58.85 | 58.79 | 58.39 | 58.13 | 50.08 | 49.74 | 49.56 | 37.47 | 30.46 | 20.27 |

The mean row is the column each branch is ranked by, which the table above it could not be read off before.

## Against A, by width

Every branch that has moved at all has moved the same way: ahead at the narrow widths, behind at the wide ones. Five times here, and three more in the literature.

| width | ED | DM | EC | DO | EE | EI | DQ | DP | FI | FL | DN | EN | FH | DT | DD | DU | FF | FC | DK | DR | DC | CI | BY | FG | CH | FM | DS | EO | BX | FD | CF | FE | FB | FN | DL | DJ | EF | CP | CQ | GX | DE | CV | FZ | FY | CU | CT | EB | GZ | DG | GC | BP | AW | GA | K | X | GB | V | EW | BO | AO | BR | AQ | EU | AF | EA | ER | BE | BD | AT | BI | AC | GD | GU | BN | AS | DB | FV | BS | GY | BF | GT | AD | BG | AA | W | GV | BH | GW | EP | I | AM | BQ | AK | F | A' | M | AH | AL | AE | EV | AR | Y | FU | D | E | G | AI | S | T | H | GN | GP | B | AU | U | GO | ET | BU | C | L | EK | DI | GS | GR | GQ | AG | BT | DF | EQ | GE | GF | GG | GM | GK | GL | ES | GI | GH | GJ | AJ | AV | AN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | +6.51 | +5.79 | +6.24 | +6.02 | +5.77 | +5.90 | +5.54 | +5.69 | +5.95 | +5.74 | +5.81 | +5.06 | +5.41 | +5.64 | +5.53 | +5.66 | +5.10 | +5.29 | +5.16 | +5.12 | +5.37 | +5.50 | +5.02 | +5.41 | +5.32 | +5.27 | +5.43 | +5.25 | +5.16 | +5.10 | +5.23 | +4.73 | +5.28 | +4.86 | +5.10 | +4.81 | +5.07 | +5.31 | +4.52 | +4.39 | +4.50 | +4.06 | +4.55 | +4.31 | +3.66 | +3.42 | +3.79 | +3.60 | +3.58 | +3.56 | +0.80 | +1.27 | -2.96 | +0.74 | +1.28 | -2.52 | +1.01 | -0.14 | +0.88 | +0.90 | +0.77 | +0.69 | +0.26 | +0.85 | +3.28 | +0.82 | +0.35 | +1.13 | +0.61 | +0.50 | +0.27 | +2.69 | +0.12 | +0.91 | +0.84 | -2.83 | -0.30 | +1.13 | +3.18 | +0.64 | -0.23 | +0.70 | +0.11 | +0.78 | +0.69 | · | +0.55 | +3.20 | -0.08 | +0.68 | +0.58 | +0.28 | +0.12 | +0.20 | -0.30 | +0.09 | +1.35 | -0.50 | +0.43 | -0.61 | +0.70 | · | -0.41 | +0.20 | -0.30 | -0.60 | -0.47 | +0.06 | +0.24 | -0.20 | -0.70 | -0.59 | +0.30 | -2.34 | -0.23 | -0.55 | +4.55 | · | -0.70 | -0.10 | -0.96 | -0.44 | -12.97 | -14.35 | -14.53 | -69.10 | -68.92 | -24.97 | -15.17 | -14.48 | -15.35 | -14.78 | -14.75 | -15.86 | -15.97 | -15.12 | -39.89 | -40.22 | -40.94 | -69.10 | -69.10 | -69.10 |
| 0.30 | +6.26 | +5.82 | +5.88 | +5.83 | +5.53 | +5.77 | +5.44 | +5.30 | +5.23 | +5.43 | +5.23 | +5.13 | +5.28 | +5.46 | +5.49 | +5.44 | +4.78 | +5.28 | +5.18 | +5.08 | +4.96 | +4.97 | +4.95 | +5.12 | +4.91 | +4.91 | +4.97 | +4.79 | +4.55 | +4.20 | +5.03 | +4.70 | +4.93 | +4.60 | +4.34 | +4.24 | +4.75 | +4.59 | +4.49 | +3.75 | +4.37 | +3.80 | +3.50 | +3.91 | +3.22 | +3.39 | +3.01 | +2.82 | +3.22 | +2.92 | +1.17 | +0.73 | -2.46 | +0.93 | +1.15 | -2.14 | +0.94 | +1.15 | +0.84 | +0.45 | +1.12 | +0.64 | +1.21 | +1.04 | +2.40 | +2.14 | +0.26 | +0.64 | +0.39 | +0.25 | +0.28 | +2.13 | +0.80 | +0.81 | +0.92 | -2.05 | +0.75 | +1.10 | +2.42 | +0.54 | +0.47 | +0.35 | -0.06 | +0.87 | +0.27 | +0.59 | +0.69 | +2.29 | +0.77 | +0.49 | +0.07 | +0.35 | · | +0.50 | -0.10 | +0.41 | +0.98 | -0.26 | +0.80 | +0.54 | +0.87 | · | +0.82 | +0.50 | +0.10 | -0.20 | -0.49 | · | +0.22 | -0.20 | +0.26 | +0.48 | +0.30 | -1.84 | -0.34 | +0.42 | +3.57 | -0.18 | -0.40 | · | -1.22 | -1.00 | -8.65 | -9.85 | -10.17 | -12.04 | -13.63 | -25.23 | -14.77 | -14.77 | -15.10 | -14.64 | -14.76 | -15.07 | -15.57 | -15.30 | -37.00 | -37.88 | -37.26 | -69.80 | -69.80 | -69.80 |
| 0.35 | +5.91 | +5.49 | +5.65 | +5.57 | +5.33 | +5.29 | +5.20 | +4.80 | +5.26 | +5.22 | +4.99 | +4.71 | +5.03 | +5.04 | +5.04 | +5.05 | +4.32 | +4.88 | +4.55 | +4.85 | +5.06 | +4.81 | +4.84 | +4.86 | +4.64 | +4.53 | +4.58 | +4.59 | +4.04 | +4.10 | +4.99 | +4.43 | +4.25 | +4.31 | +4.44 | +4.02 | +4.07 | +4.23 | +3.94 | +3.69 | +3.69 | +3.44 | +3.45 | +3.57 | +2.95 | +2.98 | +2.54 | +2.33 | +2.74 | +2.26 | +1.39 | +0.71 | -1.63 | +0.86 | +1.17 | -1.09 | +0.83 | +0.64 | +0.87 | +0.25 | +0.86 | +0.55 | +0.76 | +1.06 | +1.48 | +1.51 | +0.50 | +0.49 | +0.23 | +0.57 | +0.14 | +1.85 | +0.25 | +0.36 | +0.57 | -0.76 | +0.51 | +1.07 | +1.66 | +0.26 | +0.37 | +0.41 | +0.22 | +0.73 | +0.42 | · | +0.67 | +1.08 | +0.46 | +0.08 | -0.09 | · | · | +0.50 | · | -0.07 | +0.59 | -0.14 | +0.68 | +0.65 | +0.13 | -0.10 | -0.06 | +0.10 | +0.10 | -0.30 | -0.51 | -0.41 | +0.07 | -0.70 | -0.45 | +0.12 | -0.10 | -1.50 | -0.35 | -0.32 | +2.79 | -0.59 | -0.60 | -0.70 | -1.31 | -1.27 | -8.51 | -9.69 | -9.51 | -4.76 | -6.74 | -25.78 | -14.94 | -14.54 | -14.95 | -14.59 | -15.24 | -15.13 | -15.42 | -15.54 | -34.44 | -35.52 | -35.62 | -70.50 | -70.50 | -70.50 |
| 0.40 | +5.42 | +5.39 | +5.31 | +5.19 | +4.88 | +4.66 | +4.85 | +4.77 | +4.62 | +4.71 | +4.60 | +4.57 | +4.66 | +4.84 | +4.77 | +4.93 | +4.39 | +4.59 | +4.18 | +4.72 | +4.67 | +4.36 | +4.55 | +4.41 | +4.33 | +4.44 | +4.29 | +4.42 | +3.81 | +3.90 | +4.62 | +4.13 | +4.10 | +3.99 | +3.91 | +3.70 | +3.79 | +3.64 | +3.59 | +3.31 | +3.46 | +3.35 | +2.92 | +2.92 | +2.82 | +2.72 | +2.21 | +1.80 | +2.08 | +1.99 | +0.65 | +0.91 | -0.27 | +0.49 | +1.04 | -0.27 | +0.72 | +0.45 | +0.72 | +0.73 | +0.81 | +0.56 | +1.35 | +0.49 | +0.62 | +1.03 | +0.26 | +0.77 | +0.32 | +0.38 | +0.46 | +1.29 | +0.57 | +0.90 | +0.71 | -0.23 | +0.13 | +0.96 | +0.59 | +0.12 | +0.50 | +0.31 | +0.39 | +0.25 | +0.17 | +0.42 | +0.10 | +0.82 | +0.50 | · | · | +0.38 | +0.34 | +0.20 | +0.40 | · | +0.36 | -0.08 | +0.38 | +0.15 | · | · | · | · | -0.20 | -0.30 | -0.28 | -0.74 | -0.29 | -0.70 | -0.07 | +0.07 | -0.40 | -1.13 | -0.26 | -0.14 | +1.95 | -0.44 | -0.90 | -0.90 | -1.39 | -1.94 | -5.81 | -6.94 | -6.42 | -2.86 | -4.98 | -25.85 | -14.28 | -14.78 | -14.89 | -14.99 | -14.77 | -14.92 | -15.67 | -15.77 | -31.55 | -32.09 | -33.48 | -71.20 | -71.20 | -71.20 |
| 0.45 | +4.96 | +4.73 | +4.92 | +4.93 | +4.56 | +4.31 | +4.45 | +4.63 | +4.54 | +4.27 | +4.26 | +4.22 | +4.13 | +4.95 | +4.87 | +4.85 | +4.55 | +4.47 | +4.13 | +4.20 | +4.17 | +4.01 | +4.00 | +4.00 | +4.18 | +4.06 | +4.07 | +4.01 | +3.50 | +3.42 | +3.82 | +3.95 | +3.97 | +3.88 | +3.48 | +3.17 | +3.29 | +3.71 | +3.23 | +2.91 | +3.01 | +2.81 | +2.67 | +2.88 | +2.52 | +2.09 | +2.01 | +1.51 | +1.69 | +1.55 | +1.03 | +1.20 | +1.02 | +0.49 | +1.15 | +0.58 | +0.66 | +0.42 | +0.33 | +0.65 | +0.81 | +0.79 | +0.75 | +0.50 | +0.26 | +0.65 | +0.15 | +0.50 | +0.44 | +0.50 | +0.48 | +0.94 | +0.39 | +0.55 | +0.51 | +0.59 | +0.26 | +0.88 | -0.33 | +0.54 | +0.24 | +0.38 | +0.40 | +0.34 | -0.07 | +0.14 | · | +0.41 | +0.54 | +0.29 | · | +0.26 | +0.21 | · | -0.10 | · | +0.35 | · | · | +0.24 | -0.14 | -0.39 | -0.23 | -0.20 | · | -0.40 | -0.44 | -0.95 | -0.30 | -0.40 | · | -0.06 | -0.60 | -1.09 | -0.71 | -0.28 | +1.48 | -0.81 | -0.60 | -0.80 | -1.62 | -1.97 | -4.55 | -4.75 | -5.23 | -1.43 | -4.36 | -25.35 | -14.37 | -14.64 | -14.40 | -14.51 | -15.12 | -15.26 | -15.74 | -15.32 | -27.49 | -27.24 | -28.01 | -71.80 | -71.80 | -71.80 |
| 0.50 | +4.88 | +4.90 | +5.05 | +4.67 | +4.49 | +4.51 | +4.60 | +4.29 | +4.48 | +4.54 | +4.36 | +4.31 | +4.31 | +4.26 | +4.23 | +4.28 | +4.08 | +4.29 | +4.03 | +4.06 | +3.90 | +4.30 | +3.96 | +3.76 | +3.83 | +3.84 | +3.80 | +3.68 | +3.57 | +3.59 | +3.53 | +3.65 | +3.73 | +3.44 | +3.29 | +2.99 | +3.48 | +3.47 | +3.20 | +2.79 | +2.84 | +2.64 | +2.70 | +2.77 | +2.32 | +2.32 | +1.87 | +1.25 | +1.36 | +1.58 | +0.76 | +0.90 | +1.12 | +0.80 | +1.04 | +1.11 | +0.68 | +0.87 | +0.58 | +0.65 | +0.97 | +0.88 | +0.61 | +0.43 | +0.16 | +0.68 | +0.24 | +0.28 | +0.56 | +0.56 | +0.57 | +0.78 | +0.30 | +0.44 | +0.43 | +0.67 | +0.58 | +0.71 | -0.19 | +0.57 | +0.55 | +0.54 | +0.29 | +0.44 | +0.15 | +0.47 | +0.07 | -0.26 | +0.48 | +0.38 | · | +0.28 | +0.29 | +0.30 | -0.10 | -0.06 | +0.16 | -0.09 | -0.17 | · | · | -0.14 | -0.49 | -0.40 | +0.10 | · | -0.14 | -0.45 | -0.43 | -0.50 | +0.09 | -0.20 | -0.60 | -0.46 | -0.54 | -0.30 | +0.79 | -0.79 | -0.90 | -0.90 | -1.91 | -2.35 | -3.45 | -3.78 | -3.89 | -0.75 | -3.84 | -24.59 | -14.02 | -14.21 | -14.11 | -14.31 | -14.60 | -15.02 | -15.25 | -15.94 | -24.64 | -24.70 | -25.16 | -72.20 | -72.20 | -72.20 |
| 0.55 | +4.80 | +4.84 | +4.94 | +4.48 | +4.58 | +4.56 | +4.57 | +4.31 | +4.24 | +4.58 | +4.50 | +4.45 | +4.08 | +4.10 | +4.18 | +4.16 | +4.34 | +4.19 | +3.83 | +4.14 | +3.93 | +4.17 | +3.95 | +3.90 | +3.91 | +3.75 | +3.89 | +3.73 | +3.23 | +3.52 | +3.23 | +3.55 | +3.26 | +3.45 | +3.36 | +3.52 | +3.06 | +3.31 | +3.23 | +2.80 | +2.73 | +2.84 | +2.53 | +2.48 | +2.07 | +2.30 | +1.86 | +1.36 | +1.59 | +1.37 | +1.10 | +1.30 | +1.64 | +1.20 | +0.92 | +1.65 | +0.64 | +1.02 | +0.92 | +0.99 | +1.02 | +0.98 | +0.59 | +0.57 | +0.12 | +0.80 | +0.86 | +0.70 | +0.76 | +0.61 | +0.52 | +0.59 | +0.58 | +0.61 | +0.45 | +0.93 | +0.89 | +0.71 | -0.11 | +0.65 | · | +0.87 | +0.45 | +0.35 | +0.20 | +0.47 | -0.07 | -0.25 | +0.47 | +0.52 | +0.55 | +0.34 | · | +0.20 | · | · | +0.23 | +0.45 | · | -0.05 | +0.22 | -0.13 | · | -0.10 | +0.20 | · | +0.26 | -0.31 | -0.23 | -0.20 | -0.26 | -0.08 | -0.50 | · | -0.51 | -0.28 | +0.12 | -0.56 | -0.80 | -0.90 | -1.72 | -2.00 | -3.01 | -2.97 | -3.38 | -0.41 | -4.07 | -22.79 | -13.68 | -13.57 | -14.02 | -13.82 | -14.85 | -14.79 | -15.24 | -15.71 | -22.38 | -22.49 | -23.21 | -72.40 | -72.40 | -72.40 |
| 0.60 | +4.49 | +4.67 | +4.64 | +4.46 | +4.38 | +4.30 | +3.85 | +4.27 | +4.26 | +4.14 | +4.13 | +4.07 | +4.38 | +3.85 | +3.82 | +3.79 | +4.02 | +3.65 | +3.63 | +3.80 | +3.65 | +3.75 | +3.90 | +3.74 | +3.57 | +3.69 | +3.58 | +3.18 | +3.23 | +3.40 | +3.05 | +3.20 | +3.10 | +3.33 | +3.08 | +3.09 | +2.91 | +3.05 | +2.83 | +2.71 | +2.68 | +2.47 | +2.15 | +2.29 | +2.03 | +2.05 | +1.57 | +1.20 | +1.16 | +1.12 | +1.14 | +0.77 | +1.47 | +1.01 | +0.88 | +1.40 | +0.94 | +1.34 | +0.78 | +1.26 | +0.65 | +0.96 | +0.64 | +0.55 | +0.11 | +0.39 | +0.80 | +0.58 | +0.81 | +0.60 | +0.71 | +0.45 | +0.59 | +0.52 | +0.60 | +1.15 | +0.81 | +0.41 | -0.21 | +0.50 | +0.72 | +0.64 | +0.19 | +0.32 | +0.43 | +0.21 | +0.19 | -0.19 | +0.43 | +0.30 | +0.06 | +0.25 | +0.09 | -0.20 | · | +0.30 | -0.19 | +0.50 | · | · | +0.12 | -0.20 | -0.12 | -0.10 | · | -0.40 | -0.11 | -0.16 | -0.21 | -0.30 | +0.09 | -0.46 | -0.50 | -0.08 | -0.50 | -0.15 | -0.27 | -0.59 | -0.80 | -0.80 | -1.80 | -2.26 | -1.68 | -1.14 | -1.93 | -0.28 | -4.21 | -7.28 | -14.00 | -14.26 | -14.05 | -14.47 | -14.63 | -14.87 | -14.56 | -15.53 | -20.45 | -20.40 | -20.68 | -72.80 | -72.80 | -72.80 |
| 0.65 | +4.63 | +4.84 | +4.42 | +4.06 | +4.28 | +4.19 | +4.08 | +4.04 | +4.21 | +3.93 | +4.20 | +4.28 | +4.04 | +4.02 | +3.93 | +3.96 | +4.04 | +3.72 | +4.01 | +3.84 | +3.86 | +3.60 | +3.78 | +3.44 | +3.54 | +3.57 | +3.58 | +3.50 | +3.23 | +3.27 | +2.82 | +3.28 | +2.99 | +3.15 | +2.85 | +3.23 | +3.06 | +2.89 | +2.79 | +2.79 | +2.48 | +2.52 | +2.46 | +2.19 | +1.95 | +1.88 | +1.68 | +1.48 | +1.19 | +0.93 | +1.16 | +0.96 | +2.04 | +0.93 | +0.95 | +2.06 | +0.86 | +1.11 | +0.94 | +1.06 | +0.72 | +0.81 | +0.87 | +0.36 | +0.11 | +0.56 | +0.98 | +0.83 | +1.07 | +0.87 | +0.62 | +0.40 | +0.43 | +0.48 | +0.60 | +1.58 | +0.81 | +0.54 | -0.12 | +0.94 | +0.74 | +0.59 | +0.58 | +0.68 | +0.71 | +0.26 | +0.44 | -0.25 | +0.36 | +0.40 | +0.33 | +0.49 | · | +0.10 | +0.30 | +0.66 | +0.09 | +0.54 | +0.19 | +0.51 | -0.17 | -0.29 | -0.16 | +0.10 | -0.10 | -0.10 | +0.10 | +0.27 | -0.28 | -0.10 | -0.43 | -0.24 | -0.30 | +0.15 | -0.23 | -0.26 | -0.54 | -0.53 | -0.50 | -0.70 | -1.59 | -2.35 | -1.19 | -0.78 | -1.37 | -0.06 | -4.17 | -3.90 | -14.43 | -13.72 | -14.00 | -14.44 | -14.47 | -14.64 | -14.93 | -15.11 | -18.89 | -19.66 | -19.40 | -7.28 | -72.90 | -72.90 |
| 0.70 | +4.31 | +4.44 | +4.03 | +4.04 | +4.02 | +3.83 | +3.79 | +3.92 | +3.75 | +3.83 | +3.97 | +3.92 | +3.83 | +3.72 | +3.72 | +3.69 | +4.01 | +3.59 | +3.68 | +3.46 | +3.26 | +3.49 | +3.42 | +3.30 | +3.55 | +3.21 | +3.16 | +3.25 | +2.90 | +3.01 | +2.66 | +2.80 | +2.73 | +2.86 | +2.57 | +2.69 | +2.45 | +2.54 | +2.40 | +2.45 | +2.21 | +2.34 | +1.97 | +1.91 | +1.67 | +1.32 | +1.27 | +1.35 | +0.63 | +0.85 | +1.04 | +0.90 | +2.08 | +0.84 | +0.57 | +1.72 | +0.74 | +0.98 | +1.02 | +0.86 | +0.57 | +0.38 | +0.40 | +0.50 | +0.07 | +0.32 | +0.47 | +0.50 | +0.79 | +0.62 | +0.45 | -0.13 | +0.56 | +0.09 | +0.31 | +1.06 | +0.56 | +0.15 | · | +0.69 | +0.72 | +0.28 | +0.29 | +0.28 | +0.53 | +0.25 | +0.08 | -0.70 | +0.30 | · | +0.25 | +0.39 | · | -0.10 | +0.20 | +0.13 | -0.22 | +0.30 | -0.20 | -0.12 | -0.21 | -0.39 | -0.41 | -0.20 | -0.40 | -0.20 | -0.25 | -0.10 | -0.29 | -0.30 | -0.40 | -0.43 | -0.70 | +0.06 | -0.67 | -0.69 | -1.22 | -0.79 | -1.00 | -0.90 | -1.95 | -2.46 | -1.02 | -1.04 | -1.38 | · | -4.11 | -3.49 | -14.25 | -13.80 | -14.56 | -14.51 | -14.96 | -14.38 | -14.80 | -15.12 | -18.22 | -18.88 | -18.30 | -1.25 | -21.58 | -73.21 |
| 0.75 | +4.08 | +4.18 | +3.95 | +3.72 | +4.02 | +3.68 | +3.69 | +3.89 | +3.60 | +3.60 | +3.56 | +3.89 | +3.55 | +3.51 | +3.40 | +3.35 | +3.59 | +3.35 | +3.26 | +3.18 | +3.15 | +3.19 | +3.10 | +3.17 | +3.11 | +2.95 | +3.10 | +3.10 | +2.79 | +2.93 | +2.48 | +2.50 | +2.29 | +2.47 | +2.51 | +2.22 | +2.14 | +2.27 | +2.32 | +2.66 | +1.90 | +2.28 | +1.86 | +1.57 | +1.46 | +1.42 | +1.14 | +1.40 | +0.26 | +0.48 | +0.83 | +0.89 | +2.00 | +0.63 | +0.42 | +1.63 | +0.48 | +0.97 | +0.47 | +0.44 | +0.30 | +0.55 | +0.20 | +0.50 | -0.35 | · | +0.63 | +0.34 | +0.40 | +0.43 | +0.30 | -0.26 | +0.20 | +0.14 | +0.13 | +1.37 | +0.38 | · | · | +0.35 | +0.35 | +0.20 | +0.26 | +0.17 | +0.18 | +0.25 | · | -0.73 | +0.09 | · | +0.09 | · | · | · | · | +0.05 | -0.39 | +0.07 | -0.36 | -0.28 | -0.37 | -0.42 | -0.48 | -0.50 | -0.50 | -0.30 | -0.44 | -0.09 | -0.75 | -0.40 | -0.31 | -0.63 | -0.50 | +0.17 | -0.67 | -0.72 | -1.82 | -1.05 | -1.00 | -1.00 | -2.12 | -2.70 | -0.81 | -0.66 | -0.98 | -0.16 | -4.41 | -3.11 | -14.19 | -13.93 | -14.56 | -14.97 | -14.54 | -14.61 | -14.94 | -15.19 | -17.74 | -18.71 | -18.21 | -0.59 | -6.54 | -73.52 |
| 0.80 | +3.97 | +4.06 | +3.87 | +3.63 | +3.80 | +3.49 | +3.63 | +3.68 | +3.61 | +3.30 | +3.55 | +3.55 | +3.34 | +3.21 | +3.21 | +3.00 | +3.47 | +3.05 | +3.39 | +3.04 | +3.07 | +3.27 | +3.11 | +3.01 | +3.20 | +3.03 | +2.74 | +2.81 | +2.76 | +2.88 | +2.37 | +2.29 | +2.46 | +2.20 | +2.47 | +2.38 | +2.22 | +2.12 | +2.26 | +2.71 | +1.97 | +2.28 | +1.60 | +1.45 | +1.19 | +1.18 | +1.04 | +1.42 | +0.28 | +0.53 | +0.48 | +0.63 | +1.59 | +0.64 | +0.45 | +1.51 | +0.51 | +0.57 | +0.61 | +0.54 | +0.26 | +0.41 | +0.15 | +0.12 | -0.21 | · | +0.39 | +0.22 | +0.26 | +0.36 | +0.51 | -0.51 | +0.30 | +0.40 | +0.26 | +1.08 | · | · | -0.09 | +0.30 | +0.27 | +0.21 | +0.15 | -0.25 | +0.13 | +0.08 | -0.08 | -0.74 | -0.18 | · | +0.08 | -0.18 | · | -0.20 | -0.10 | -0.24 | -0.46 | · | -0.38 | -0.32 | -0.35 | -0.50 | -0.78 | -0.40 | -0.60 | -0.30 | -0.38 | -0.09 | -0.54 | -0.50 | -0.76 | -0.98 | -0.60 | +0.07 | -0.61 | -0.89 | -2.43 | -0.99 | -0.90 | -1.10 | -1.73 | -2.96 | -0.81 | -0.43 | -0.82 | -0.23 | -4.18 | -2.91 | -13.90 | -14.36 | -14.51 | -14.80 | -14.36 | -14.04 | -14.48 | -15.41 | -16.85 | -17.22 | -17.30 | · | -4.66 | -35.02 |
| 0.85 | +3.46 | +3.76 | +3.35 | +3.27 | +3.35 | +3.47 | +3.35 | +3.28 | +3.17 | +3.22 | +3.18 | +3.32 | +3.25 | +2.96 | +2.99 | +2.88 | +3.05 | +2.88 | +2.94 | +2.75 | +2.67 | +2.72 | +2.77 | +2.88 | +2.92 | +2.70 | +2.30 | +2.66 | +2.50 | +2.50 | +2.24 | +2.25 | +2.14 | +1.91 | +2.22 | +2.13 | +1.87 | +1.76 | +2.11 | +2.33 | +1.81 | +1.99 | +1.43 | +1.45 | +1.07 | +0.92 | +0.75 | +1.20 | · | +0.10 | +0.31 | +0.36 | +1.90 | +0.62 | +0.23 | +1.42 | +0.17 | +0.26 | +0.45 | +0.22 | +0.19 | +0.34 | +0.35 | +0.31 | -0.15 | -0.19 | +0.55 | -0.08 | +0.17 | +0.08 | +0.24 | -0.84 | +0.24 | +0.11 | · | +1.10 | +0.27 | -0.30 | -0.54 | · | +0.18 | -0.16 | +0.26 | -0.32 | -0.15 | · | -0.13 | -0.94 | -0.23 | -0.27 | -0.11 | -0.31 | -0.09 | -0.30 | -0.40 | -0.36 | -0.66 | -0.20 | -0.75 | -0.64 | -0.72 | -0.55 | -0.50 | -0.80 | -0.70 | -0.40 | -0.44 | -0.60 | -0.73 | -0.50 | -0.38 | -0.87 | -0.90 | -0.18 | -0.82 | -1.04 | -3.29 | -1.15 | -1.00 | -1.20 | -1.84 | -2.41 | -0.42 | -0.28 | -0.95 | -0.24 | -4.74 | -2.97 | -13.92 | -14.30 | -14.57 | -14.52 | -14.61 | -14.55 | -15.19 | -15.38 | -16.58 | -16.74 | -16.91 | +0.14 | -4.02 | -10.63 |
| 0.90 | +3.67 | +3.95 | +3.38 | +3.13 | +3.45 | +3.47 | +3.24 | +3.25 | +3.23 | +3.12 | +3.25 | +3.43 | +3.22 | +3.01 | +3.13 | +2.98 | +3.24 | +3.07 | +3.05 | +2.78 | +2.93 | +2.78 | +2.96 | +2.72 | +2.85 | +2.90 | +2.48 | +2.69 | +2.89 | +2.59 | +2.49 | +2.14 | +2.04 | +2.05 | +2.09 | +2.23 | +2.08 | +1.71 | +2.16 | +2.57 | +1.94 | +2.02 | +1.65 | +1.56 | +1.11 | +0.93 | +1.09 | +1.36 | · | +0.11 | +0.63 | +0.54 | +1.75 | +0.63 | +0.20 | +1.47 | +0.39 | +0.51 | +0.46 | +0.36 | +0.33 | +0.23 | +0.38 | +0.37 | · | -0.07 | +0.38 | · | +0.18 | +0.41 | +0.40 | -0.77 | +0.39 | +0.21 | +0.26 | +1.11 | +0.35 | -0.20 | · | +0.21 | +0.34 | -0.11 | +0.35 | -0.22 | +0.10 | · | +0.36 | -0.38 | -0.40 | -0.24 | +0.12 | -0.29 | · | -0.20 | +0.10 | -0.25 | -0.63 | -0.15 | -0.57 | -0.53 | -0.52 | -0.30 | -0.26 | -0.50 | -0.60 | -0.30 | -0.15 | -0.33 | -0.34 | -0.30 | -0.49 | -0.88 | -0.60 | +0.09 | -0.71 | -1.09 | -4.03 | -0.94 | -0.90 | -0.80 | -1.83 | -2.50 | -0.18 | +0.13 | -0.67 | -0.18 | -4.71 | -2.94 | -14.20 | -14.18 | -14.49 | -14.49 | -14.59 | -14.15 | -14.87 | -15.56 | -16.60 | -16.40 | -16.46 | +0.62 | -3.16 | -6.52 |
| 0.95 | +3.44 | +3.72 | +3.06 | +3.11 | +3.17 | +3.27 | +3.12 | +3.12 | +2.87 | +2.87 | +2.87 | +3.12 | +2.95 | +2.83 | +2.68 | +2.71 | +2.75 | +2.74 | +2.82 | +2.48 | +2.61 | +2.44 | +2.74 | +2.59 | +2.54 | +2.55 | +2.36 | +2.25 | +2.61 | +2.36 | +1.50 | +1.54 | +1.86 | +1.72 | +1.78 | +2.05 | +1.68 | +1.51 | +1.79 | +2.34 | +1.71 | +1.83 | +1.65 | +1.40 | +0.76 | +0.57 | +0.90 | +1.06 | -0.29 | -0.12 | +0.25 | +0.33 | +1.76 | +0.42 | · | +1.39 | +0.44 | +0.12 | · | +0.20 | · | +0.23 | · | +0.31 | +0.06 | -0.38 | +0.26 | · | · | · | +0.35 | -0.88 | +0.48 | -0.06 | · | +0.83 | +0.08 | -0.43 | -0.25 | -0.17 | +0.18 | -0.22 | · | -0.48 | -0.12 | · | +0.09 | -0.58 | -0.61 | -0.44 | · | -0.25 | -0.34 | -0.50 | · | -0.27 | -0.93 | -0.32 | -0.86 | -0.65 | -0.80 | -0.27 | -0.38 | -0.90 | -0.70 | -0.50 | -0.51 | -0.60 | -0.82 | -0.40 | -1.04 | -1.06 | -0.70 | -0.23 | -0.81 | -1.59 | -4.79 | -1.03 | -0.80 | -0.90 | -1.90 | -2.66 | -0.09 | -0.11 | -1.08 | -0.43 | -4.47 | -2.94 | -14.17 | -14.65 | -14.53 | -14.87 | -14.51 | -14.46 | -14.96 | -15.42 | -16.24 | -16.36 | -16.26 | +0.58 | -3.32 | -5.42 |
| 1.00 | +3.50 | +3.62 | +3.25 | +2.99 | +3.34 | +3.19 | +3.02 | +3.00 | +3.21 | +3.05 | +2.84 | +3.22 | +2.99 | +2.73 | +2.65 | +2.64 | +2.93 | +2.78 | +2.82 | +2.66 | +2.74 | +2.60 | +2.76 | +2.76 | +2.65 | +2.94 | +2.27 | +2.29 | +2.74 | +2.64 | +2.45 | +2.31 | +2.18 | +2.11 | +2.06 | +2.15 | +1.89 | +1.54 | +2.21 | +2.30 | +1.86 | +1.96 | +1.66 | +1.55 | +0.69 | +0.70 | +0.87 | +1.28 | -0.18 | -0.06 | +0.52 | +0.47 | +1.80 | +0.70 | +0.12 | +1.49 | +0.53 | +0.24 | +0.28 | +0.38 | +0.40 | +0.30 | +0.10 | +0.26 | +0.09 | -0.20 | +0.36 | +0.16 | +0.06 | +0.19 | +0.58 | -0.93 | +0.50 | +0.09 | · | +0.88 | +0.24 | -0.43 | +0.23 | +0.06 | +0.39 | -0.10 | +0.27 | -0.25 | · | +0.43 | +0.15 | +0.12 | -0.37 | -0.09 | · | -0.20 | · | -0.10 | +0.30 | -0.17 | -0.46 | -0.08 | -0.80 | -0.67 | -0.92 | -0.20 | -0.41 | -0.80 | -0.40 | · | -0.16 | -0.29 | -0.55 | · | -1.09 | -1.17 | -0.60 | · | -0.57 | -1.19 | -6.16 | -0.90 | -0.60 | -0.80 | -1.79 | -2.37 | -0.21 | -0.46 | -0.73 | -0.19 | -4.58 | -2.66 | -13.68 | -14.44 | -14.32 | -14.58 | -13.95 | -13.96 | -14.41 | -14.84 | -16.00 | -16.02 | -16.10 | +0.75 | -2.91 | -5.02 |

A dot is a difference at or inside the rounding floor.

## NLL by width

The one place a branch has separated from A by more than the measurement can be blamed for.

| width | ED | DM | EC | DO | EE | EI | DQ | DP | FI | FL | DN | EN | FH | DT | DD | DU | FF | FC | DK | DR | DC | CI | BY | FG | CH | FM | DS | EO | BX | FD | CF | FE | FB | FN | DL | DJ | EF | CP | CQ | GX | DE | CV | FZ | FY | CU | CT | EB | GZ | DG | GC | BP | AW | GA | K | X | GB | V | EW | BO | AO | BR | AQ | EU | AF | EA | ER | BE | BD | AT | BI | AC | GD | GU | BN | AS | DB | FV | BS | GY | BF | GT | AD | BG | AA | W | GV | BH | GW | EP | I | AM | BQ | AK | F | M | AH | AL | A | AE | EV | AR | Y | FU | D | E | G | AI | S | T | H | GN | GP | B | AU | U | GO | ET | BU | C | L | EK | DI | GS | GR | GQ | AG | BT | DF | EQ | GE | GF | GG | GM | GK | GL | ES | GI | GH | GJ | AJ | AV | AN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | **0.834** | 0.972 | 0.894 | 0.912 | 0.945 | 0.942 | 0.960 | 0.943 | 0.979 | 0.927 | 0.949 | 1.111 | 0.968 | 1.009 | 1.012 | 1.011 | 1.131 | 0.949 | 1.048 | 1.026 | 1.019 | 0.941 | 1.027 | 1.086 | 1.141 | 1.143 | 1.001 | 1.096 | 1.166 | 1.151 | 1.097 | 1.136 | 1.148 | 1.124 | 1.073 | 1.100 | 1.059 | 1.065 | 1.198 | 1.129 | 1.220 | 1.383 | 1.388 | 1.376 | 1.031 | 1.021 | 1.195 | 1.158 | 2.204 | 2.343 | 1.183 | 1.117 | 1.179 | 1.148 | 1.180 | 1.194 | 1.143 | 1.093 | 1.152 | 1.179 | 1.445 | 1.189 | 1.061 | 1.146 | 1.230 | 1.011 | 1.152 | 1.124 | 1.161 | 1.165 | 1.189 | 2.525 | 1.106 | 1.156 | 1.154 | 1.178 | 1.074 | 1.233 | 1.262 | 1.143 | 1.110 | 1.162 | 1.139 | 1.230 | 1.239 | 1.115 | 1.161 | 1.208 | 1.084 | 1.239 | 1.307 | 1.184 | 1.281 | 1.296 | 1.206 | 1.222 | 1.343 | 1.272 | 1.212 | 1.123 | 1.670 | 1.286 | 1.118 | 1.400 | 1.317 | 1.439 | 1.346 | 1.329 | 1.301 | 1.377 | 1.202 | 1.204 | 1.465 | 1.119 | 1.347 | 1.198 | nan | 1.214 | 1.453 | 1.448 | 1.099 | 1.586 | 1.589 | 1.633 | 1.648 | 4.605 | 4.612 | 2.061 | 1.883 | 2.088 | 2.121 | 2.103 | 1.935 | 1.959 | 1.954 | 1.979 | 3.186 | 3.228 | 3.230 | 4.605 | 4.605 | 4.605 |
| 0.30 | **0.820** | 0.915 | 0.875 | 0.887 | 0.907 | 0.909 | 0.907 | 0.917 | 0.953 | 0.907 | 0.923 | 0.953 | 0.934 | 0.966 | 0.969 | 0.966 | 0.950 | 0.921 | 0.975 | 0.981 | 0.986 | 0.924 | 0.980 | 1.102 | 1.167 | 1.174 | 0.969 | 1.113 | 1.205 | 1.175 | 1.081 | 1.086 | 1.170 | 1.111 | 1.062 | 1.103 | 1.010 | 1.080 | 1.232 | 1.135 | 1.254 | 1.421 | 1.433 | 1.399 | 1.017 | 1.005 | 1.202 | 1.166 | 2.220 | 2.347 | 1.098 | 1.110 | 1.158 | 1.130 | 1.153 | 1.159 | 1.116 | 1.003 | 1.117 | 1.158 | 1.453 | 1.171 | 0.997 | 1.134 | 1.250 | 0.961 | 1.133 | 1.102 | 1.138 | 1.131 | 1.150 | 2.542 | 1.056 | 1.117 | 1.125 | 1.136 | 1.012 | 1.192 | 1.301 | 1.130 | 1.075 | 1.140 | 1.121 | 1.233 | 1.228 | 1.080 | 1.137 | 1.234 | 1.000 | 1.238 | 1.315 | 1.167 | 1.288 | 1.291 | 1.190 | 1.198 | 1.334 | 1.287 | 1.202 | 1.077 | 1.637 | 1.283 | 1.067 | 1.395 | 1.313 | 1.419 | 1.336 | 1.330 | 1.306 | 1.364 | 1.161 | 1.156 | 1.482 | 1.087 | 1.345 | 1.174 | nan | 1.185 | 1.429 | 1.448 | 1.101 | 1.595 | 1.411 | 1.436 | 1.455 | 1.458 | 1.809 | 2.065 | 1.831 | 2.080 | 2.104 | 2.102 | 1.932 | 1.946 | 1.944 | 1.982 | 3.023 | 3.094 | 3.014 | 4.605 | 4.605 | 4.605 |
| 0.35 | **0.817** | 0.910 | 0.866 | 0.873 | 0.888 | 0.900 | 0.901 | 0.883 | 0.940 | 0.895 | 0.920 | 0.938 | 0.916 | 0.960 | 0.963 | 0.960 | 0.931 | 0.916 | 0.974 | 0.976 | 0.974 | 0.914 | 0.963 | 1.113 | 1.200 | 1.191 | 0.966 | 1.131 | 1.242 | 1.188 | 1.114 | 1.136 | 1.203 | 1.145 | 1.066 | 1.109 | 1.010 | 1.093 | 1.256 | 1.118 | 1.271 | 1.445 | 1.447 | 1.427 | 1.013 | 0.998 | 1.200 | 1.168 | 2.203 | 2.361 | 1.089 | 1.094 | 1.148 | 1.127 | 1.152 | 1.139 | 1.088 | 1.001 | 1.106 | 1.142 | 1.465 | 1.153 | 0.970 | 1.133 | 1.273 | 0.967 | 1.115 | 1.090 | 1.128 | 1.107 | 1.122 | 2.558 | 1.074 | 1.103 | 1.105 | 1.097 | 1.008 | 1.184 | 1.342 | 1.120 | 1.078 | 1.143 | 1.092 | 1.235 | 1.222 | 1.084 | 1.116 | 1.265 | 0.981 | 1.247 | 1.311 | 1.156 | 1.292 | 1.301 | 1.174 | 1.185 | 1.338 | 1.286 | 1.199 | 1.080 | 1.627 | 1.287 | 1.083 | 1.381 | 1.316 | 1.404 | 1.340 | 1.338 | 1.326 | 1.352 | 1.200 | 1.167 | 1.476 | 1.065 | 1.349 | 1.205 | nan | 1.181 | 1.433 | 1.459 | 1.083 | 1.584 | 1.390 | 1.420 | 1.418 | 1.208 | 1.369 | 2.061 | 1.815 | 2.066 | 2.097 | 2.109 | 1.921 | 1.934 | 1.935 | 1.979 | 2.910 | 2.963 | 2.929 | 4.605 | 4.605 | 4.605 |
| 0.40 | **0.811** | 0.909 | 0.868 | 0.867 | 0.885 | 0.893 | 0.886 | 0.880 | 0.936 | 0.886 | 0.918 | 0.926 | 0.911 | 0.958 | 0.961 | 0.956 | 0.924 | 0.910 | 0.975 | 0.969 | 0.971 | 0.905 | 0.962 | 1.122 | 1.220 | 1.209 | 0.973 | 1.137 | 1.271 | 1.210 | 1.144 | 1.183 | 1.225 | 1.184 | 1.077 | 1.113 | 1.010 | 1.100 | 1.279 | 1.120 | 1.299 | 1.455 | 1.471 | 1.454 | 1.003 | 0.998 | 1.193 | 1.162 | 2.203 | 2.353 | 1.099 | 1.080 | 1.088 | 1.114 | 1.155 | 1.092 | 1.078 | 0.982 | 1.111 | 1.120 | 1.484 | 1.167 | 0.939 | 1.142 | 1.305 | 0.973 | 1.107 | 1.077 | 1.112 | 1.098 | 1.101 | 2.552 | 1.054 | 1.075 | 1.098 | 1.062 | 0.991 | 1.170 | 1.371 | 1.104 | 1.069 | 1.132 | 1.077 | 1.241 | 1.228 | 1.063 | 1.116 | 1.272 | 0.955 | 1.259 | 1.320 | 1.128 | 1.296 | 1.291 | 1.182 | 1.169 | 1.345 | 1.295 | 1.187 | 1.071 | 1.615 | 1.308 | 1.084 | 1.384 | 1.323 | 1.395 | 1.357 | 1.361 | 1.341 | 1.354 | 1.194 | 1.160 | 1.518 | 1.045 | 1.358 | 1.202 | nan | 1.177 | 1.443 | 1.466 | 1.065 | 1.596 | 1.273 | 1.273 | 1.294 | 1.164 | 1.229 | 2.032 | 1.795 | 2.101 | 2.141 | 2.157 | 1.912 | 1.942 | 1.941 | 2.003 | 2.764 | 2.827 | 2.856 | 4.605 | 4.605 | 4.605 |
| 0.45 | **0.799** | 0.910 | 0.840 | 0.833 | 0.851 | 0.864 | 0.891 | 0.871 | 0.936 | 0.849 | 0.924 | 0.876 | 0.870 | 0.960 | 0.961 | 0.957 | 0.880 | 0.863 | 0.989 | 0.974 | 0.980 | 0.856 | 0.961 | 1.029 | 1.110 | 1.108 | 0.972 | 1.042 | 1.295 | 1.110 | 1.172 | 1.193 | 1.257 | 1.214 | 1.095 | 1.120 | 1.019 | 1.113 | 1.312 | 1.109 | 1.314 | 1.490 | 1.493 | 1.475 | 1.018 | 1.004 | 1.184 | 1.157 | 2.216 | 2.347 | 1.086 | 1.078 | 1.043 | 1.108 | 1.150 | 1.061 | 1.075 | 0.958 | 1.105 | 1.116 | 1.487 | 1.158 | 0.943 | 1.124 | 1.302 | 0.986 | 1.103 | 1.072 | 1.099 | 1.089 | 1.093 | 2.556 | 1.022 | 1.053 | 1.103 | 1.029 | 0.963 | 1.165 | 1.398 | 1.094 | 1.031 | 1.131 | 1.081 | 1.244 | 1.224 | 1.024 | 1.102 | 1.290 | 0.950 | 1.255 | 1.316 | 1.114 | 1.294 | 1.286 | 1.191 | 1.156 | 1.363 | 1.318 | 1.183 | 1.069 | 1.558 | 1.321 | 1.088 | 1.374 | 1.310 | 1.391 | 1.373 | 1.357 | 1.352 | 1.345 | 1.191 | 1.173 | 1.539 | 1.016 | 1.374 | 1.202 | nan | 1.166 | 1.427 | 1.465 | 1.054 | 1.593 | 1.209 | 1.201 | 1.229 | 1.160 | 1.182 | 1.998 | 1.775 | 2.107 | 2.124 | 2.140 | 1.912 | 1.932 | 1.947 | 2.010 | 2.503 | 2.539 | 2.545 | 4.605 | 4.605 | 4.605 |
| 0.50 | **0.818** | 0.916 | 0.858 | 0.856 | 0.860 | 0.871 | 0.895 | 0.878 | 0.942 | 0.858 | 0.931 | 0.888 | 0.894 | 0.971 | 0.973 | 0.968 | 0.893 | 0.880 | 0.996 | 0.983 | 0.983 | 0.882 | 0.974 | 1.071 | 1.157 | 1.154 | 0.984 | 1.091 | 1.310 | 1.156 | 1.185 | 1.206 | 1.280 | 1.228 | 1.118 | 1.133 | 1.026 | 1.127 | 1.333 | 1.101 | 1.328 | 1.504 | 1.513 | 1.496 | 1.019 | 1.016 | 1.183 | 1.156 | 2.224 | 2.339 | 1.093 | 1.073 | 1.036 | 1.107 | 1.150 | 1.056 | 1.064 | 0.963 | 1.097 | 1.110 | 1.503 | 1.165 | 0.948 | 1.115 | 1.313 | 1.006 | 1.100 | 1.068 | 1.091 | 1.084 | 1.082 | 2.565 | 1.024 | 1.033 | 1.095 | 1.032 | 0.962 | 1.166 | 1.422 | 1.081 | 1.047 | 1.131 | 1.076 | 1.248 | 1.222 | 1.033 | 1.098 | 1.315 | 0.953 | 1.261 | 1.329 | 1.106 | 1.304 | 1.275 | 1.175 | 1.146 | 1.367 | 1.337 | 1.172 | 1.084 | 1.535 | 1.329 | 1.097 | 1.364 | 1.301 | 1.373 | 1.366 | 1.373 | 1.352 | 1.349 | 1.193 | 1.179 | 1.562 | 1.003 | 1.382 | 1.203 | nan | 1.149 | 1.413 | 1.451 | 1.046 | 1.600 | 1.178 | 1.161 | 1.201 | 1.151 | 1.165 | 1.933 | 1.769 | 2.123 | 2.154 | 2.166 | 1.930 | 1.950 | 1.940 | 2.052 | 2.394 | 2.413 | 2.427 | 4.605 | 4.605 | 4.605 |
| 0.55 | **0.833** | 0.911 | 0.879 | 0.870 | 0.867 | 0.882 | 0.891 | 0.878 | 0.954 | 0.865 | 0.936 | 0.894 | 0.908 | 0.978 | 0.979 | 0.974 | 0.901 | 0.896 | 1.008 | 0.992 | 0.991 | 0.900 | 0.980 | 1.085 | 1.195 | 1.188 | 0.991 | 1.111 | 1.327 | 1.181 | 1.211 | 1.225 | 1.316 | 1.241 | 1.138 | 1.137 | 1.037 | 1.136 | 1.356 | 1.101 | 1.366 | 1.518 | 1.543 | 1.505 | 1.038 | 1.025 | 1.178 | 1.161 | 2.232 | 2.342 | 1.103 | 1.069 | 1.038 | 1.105 | 1.152 | 1.050 | 1.058 | 0.959 | 1.093 | 1.106 | 1.506 | 1.168 | 0.974 | 1.099 | 1.336 | 1.048 | 1.094 | 1.069 | 1.093 | 1.071 | 1.084 | 2.567 | 1.038 | 1.020 | 1.096 | 1.032 | 0.964 | 1.166 | 1.435 | 1.078 | 1.068 | 1.129 | 1.080 | 1.250 | 1.210 | 1.046 | 1.095 | 1.337 | 0.958 | 1.278 | 1.329 | 1.115 | 1.321 | 1.274 | 1.186 | 1.131 | 1.378 | 1.337 | 1.166 | 1.100 | 1.520 | 1.342 | 1.103 | 1.362 | 1.292 | 1.369 | 1.370 | 1.387 | 1.369 | 1.343 | 1.197 | 1.182 | 1.563 | 1.009 | 1.389 | 1.203 | nan | 1.146 | 1.403 | 1.446 | 1.039 | 1.611 | 1.170 | 1.146 | 1.186 | 1.157 | 1.162 | 1.847 | 1.756 | 2.161 | 2.187 | 2.204 | 1.955 | 1.975 | 1.968 | 2.102 | 2.328 | 2.346 | 2.356 | 4.605 | 4.605 | 4.605 |
| 0.60 | **0.839** | 0.909 | 0.886 | 0.878 | 0.875 | 0.885 | 0.913 | 0.895 | 0.964 | 0.876 | 0.938 | 0.899 | 0.911 | 0.989 | 0.990 | 0.985 | 0.910 | 0.910 | 1.020 | 0.992 | 1.001 | 0.911 | 0.993 | 1.103 | 1.222 | 1.221 | 1.001 | 1.128 | 1.340 | 1.210 | 1.226 | 1.236 | 1.333 | 1.257 | 1.152 | 1.151 | 1.042 | 1.151 | 1.374 | 1.090 | 1.385 | 1.524 | 1.561 | 1.519 | 1.055 | 1.030 | 1.181 | 1.162 | 2.236 | 2.346 | 1.116 | 1.073 | 1.045 | 1.107 | 1.157 | 1.048 | 1.056 | 0.945 | 1.088 | 1.106 | 1.510 | 1.175 | 0.977 | 1.101 | 1.349 | 1.063 | 1.101 | 1.073 | 1.093 | 1.067 | 1.081 | 2.558 | 1.040 | 1.011 | 1.091 | 1.041 | 0.956 | 1.169 | 1.454 | 1.081 | 1.060 | 1.134 | 1.077 | 1.258 | 1.212 | 1.052 | 1.082 | 1.351 | 0.976 | 1.295 | 1.339 | 1.113 | 1.321 | 1.265 | 1.181 | 1.123 | 1.398 | 1.354 | 1.162 | 1.081 | 1.495 | 1.353 | 1.095 | 1.340 | 1.283 | 1.358 | 1.386 | 1.399 | 1.382 | 1.345 | 1.178 | 1.176 | 1.587 | 1.015 | 1.398 | 1.182 | nan | 1.146 | 1.392 | 1.427 | 1.038 | 1.602 | 1.095 | 1.081 | 1.135 | 1.182 | 1.145 | 1.225 | 1.772 | 2.181 | 2.200 | 2.250 | 1.957 | 1.980 | 1.990 | 2.125 | 2.258 | 2.276 | 2.244 | 4.605 | 4.605 | 4.605 |
| 0.65 | **0.838** | 0.911 | 0.886 | 0.874 | 0.866 | 0.870 | 0.914 | 0.896 | 0.964 | 0.863 | 0.879 | 0.877 | 0.898 | 0.993 | 0.994 | 0.988 | 0.886 | 0.899 | 0.985 | 1.001 | 1.006 | 0.895 | 1.002 | 1.047 | 1.167 | 1.144 | 1.006 | 1.063 | 1.365 | 1.141 | 1.238 | 1.257 | 1.359 | 1.263 | 1.162 | 1.169 | 1.035 | 1.157 | 1.403 | 1.084 | 1.405 | 1.525 | 1.556 | 1.519 | 1.066 | 1.040 | 1.180 | 1.163 | 2.234 | 2.340 | 1.133 | 1.078 | 1.041 | 1.114 | 1.167 | 1.052 | 1.060 | 0.932 | 1.102 | 1.112 | 1.539 | 1.185 | 1.004 | 1.106 | 1.368 | 1.103 | 1.107 | 1.077 | 1.097 | 1.075 | 1.089 | 2.555 | 1.031 | 1.008 | 1.098 | 1.038 | 0.939 | 1.173 | 1.460 | 1.080 | 1.053 | 1.139 | 1.080 | 1.266 | 1.221 | 1.044 | 1.088 | 1.374 | 0.982 | 1.316 | 1.345 | 1.117 | 1.332 | 1.259 | 1.190 | 1.113 | 1.400 | 1.389 | 1.159 | 1.091 | 1.501 | 1.370 | 1.103 | 1.347 | 1.278 | 1.366 | 1.413 | 1.420 | 1.396 | 1.338 | 1.185 | 1.181 | 1.611 | 1.028 | 1.426 | 1.187 | nan | 1.148 | 1.412 | 1.423 | 1.036 | 1.625 | 1.078 | 1.078 | 1.119 | 1.207 | 1.141 | 1.157 | 1.800 | 2.224 | 2.244 | 2.296 | 2.000 | 2.021 | 2.034 | 2.164 | 2.209 | 2.260 | 2.233 | 1.328 | 4.605 | 4.605 |
| 0.70 | **0.860** | 0.915 | 0.914 | 0.901 | 0.888 | 0.896 | 0.930 | 0.910 | 0.974 | 0.887 | 0.912 | 0.896 | 0.922 | 1.002 | 1.003 | 0.998 | 0.908 | 0.925 | 0.986 | 1.009 | 1.012 | 0.924 | 1.016 | 1.069 | 1.190 | 1.172 | 1.023 | 1.091 | 1.374 | 1.172 | 1.245 | 1.260 | 1.373 | 1.270 | 1.177 | 1.182 | 1.049 | 1.161 | 1.416 | 1.078 | 1.403 | 1.536 | 1.565 | 1.524 | 1.077 | 1.058 | 1.177 | 1.161 | 2.235 | 2.343 | 1.140 | 1.079 | 1.064 | 1.115 | 1.167 | 1.073 | 1.060 | 0.946 | 1.098 | 1.118 | 1.547 | 1.194 | 1.032 | 1.112 | 1.383 | 1.133 | 1.116 | 1.075 | 1.097 | 1.072 | 1.087 | 2.551 | 1.033 | 1.005 | 1.105 | 1.056 | 0.955 | 1.168 | 1.454 | 1.084 | 1.057 | 1.139 | 1.083 | 1.275 | 1.220 | 1.048 | 1.091 | 1.401 | 0.999 | 1.337 | 1.347 | 1.119 | 1.342 | 1.242 | 1.199 | 1.104 | 1.410 | 1.395 | 1.156 | 1.101 | 1.494 | 1.381 | 1.109 | 1.347 | 1.263 | 1.356 | 1.412 | 1.436 | 1.412 | 1.335 | 1.188 | 1.170 | 1.622 | 1.035 | 1.438 | 1.185 | nan | 1.154 | 1.406 | 1.430 | 1.035 | 1.639 | 1.067 | 1.076 | 1.115 | 1.224 | 1.137 | 1.167 | 1.806 | 2.262 | 2.246 | 2.312 | 2.021 | 2.032 | 2.062 | 2.186 | 2.195 | 2.257 | 2.222 | 1.216 | 1.837 | 4.605 |
| 0.75 | **0.876** | 0.925 | 0.927 | 0.919 | 0.903 | 0.915 | 0.941 | 0.909 | 0.986 | 0.911 | 0.939 | 0.904 | 0.941 | 1.013 | 1.014 | 1.009 | 0.920 | 0.943 | 1.004 | 1.021 | 1.023 | 0.944 | 1.027 | 1.085 | 1.214 | 1.189 | 1.026 | 1.106 | 1.385 | 1.189 | 1.247 | 1.261 | 1.391 | 1.277 | 1.188 | 1.195 | 1.056 | 1.171 | 1.425 | 1.079 | 1.419 | 1.540 | 1.564 | 1.520 | 1.099 | 1.067 | 1.173 | 1.155 | 2.231 | 2.342 | 1.153 | 1.081 | 1.079 | 1.120 | 1.170 | 1.100 | 1.066 | 0.947 | 1.099 | 1.123 | 1.550 | 1.197 | 1.060 | 1.115 | 1.412 | 1.163 | 1.116 | 1.085 | 1.097 | 1.074 | 1.091 | 2.555 | 1.028 | 0.999 | 1.107 | 1.083 | 0.959 | 1.172 | 1.468 | 1.090 | 1.061 | 1.156 | 1.085 | 1.274 | 1.225 | 1.050 | 1.094 | 1.415 | 1.010 | 1.347 | 1.350 | 1.124 | 1.343 | 1.222 | 1.208 | 1.093 | 1.416 | 1.391 | 1.152 | 1.104 | 1.475 | 1.388 | 1.113 | 1.348 | 1.248 | 1.370 | 1.416 | 1.442 | 1.408 | 1.334 | 1.179 | 1.182 | 1.627 | 1.046 | 1.449 | 1.177 | nan | 1.158 | 1.409 | 1.419 | 1.040 | 1.643 | 1.056 | 1.061 | 1.089 | 1.236 | 1.132 | 1.179 | 1.809 | 2.275 | 2.269 | 2.320 | 2.038 | 2.051 | 2.078 | 2.209 | 2.192 | 2.250 | 2.211 | 1.240 | 1.323 | 4.605 |
| 0.80 | **0.882** | 0.938 | 0.933 | 0.929 | 0.909 | 0.920 | 0.954 | 0.925 | 0.996 | 0.919 | 0.963 | 0.906 | 0.947 | 1.031 | 1.032 | 1.026 | 0.921 | 0.950 | 1.027 | 1.033 | 1.038 | 0.954 | 1.042 | 1.092 | 1.217 | 1.189 | 1.045 | 1.114 | 1.397 | 1.186 | 1.300 | 1.309 | 1.396 | 1.329 | 1.198 | 1.210 | 1.066 | 1.175 | 1.437 | 1.072 | 1.433 | 1.529 | 1.567 | 1.508 | 1.120 | 1.082 | 1.174 | 1.147 | 2.229 | 2.343 | 1.171 | 1.083 | 1.100 | 1.123 | 1.174 | 1.124 | 1.071 | 0.951 | 1.103 | 1.129 | 1.553 | 1.203 | 1.081 | 1.120 | 1.430 | 1.220 | 1.128 | 1.092 | 1.106 | 1.079 | 1.090 | 2.554 | 1.023 | 0.994 | 1.114 | 1.111 | 0.964 | 1.172 | 1.487 | 1.092 | 1.055 | 1.159 | 1.091 | 1.285 | 1.232 | 1.053 | 1.095 | 1.436 | 1.027 | 1.359 | 1.346 | 1.125 | 1.350 | 1.205 | 1.211 | 1.086 | 1.428 | 1.400 | 1.152 | 1.116 | 1.460 | 1.393 | 1.120 | 1.343 | 1.228 | 1.371 | 1.421 | 1.439 | 1.408 | 1.332 | 1.188 | 1.175 | 1.629 | 1.064 | 1.448 | 1.184 | nan | 1.163 | 1.409 | 1.423 | 1.037 | 1.642 | 1.057 | 1.060 | 1.084 | 1.247 | 1.121 | 1.192 | 1.801 | 2.310 | 2.275 | 2.351 | 2.038 | 2.052 | 2.067 | 2.256 | 2.190 | 2.225 | 2.213 | 1.274 | 1.337 | 2.232 |
| 0.85 | **0.861** | 0.950 | 0.910 | 0.903 | 0.906 | 0.897 | 0.971 | 0.932 | 1.008 | 0.894 | 0.990 | 0.877 | 0.921 | 1.043 | 1.044 | 1.038 | 0.893 | 0.921 | 1.053 | 1.040 | 1.051 | 0.929 | 1.056 | 1.052 | 1.180 | 1.147 | 1.052 | 1.073 | 1.410 | 1.144 | 1.350 | 1.361 | 1.407 | 1.387 | 1.214 | 1.221 | 1.085 | 1.179 | 1.434 | 1.069 | 1.434 | 1.523 | 1.558 | 1.508 | 1.129 | 1.099 | 1.167 | 1.146 | 2.226 | 2.341 | 1.185 | 1.093 | 1.120 | 1.129 | 1.180 | 1.148 | 1.077 | 0.967 | 1.109 | 1.133 | 1.556 | 1.202 | 1.105 | 1.127 | 1.463 | 1.254 | 1.132 | 1.097 | 1.109 | 1.085 | 1.097 | 2.552 | 1.059 | 0.993 | 1.121 | 1.143 | 0.977 | 1.173 | 1.507 | 1.099 | 1.101 | 1.167 | 1.099 | 1.298 | 1.233 | 1.097 | 1.101 | 1.452 | 1.105 | 1.368 | 1.347 | 1.134 | 1.350 | 1.192 | 1.218 | 1.078 | 1.435 | 1.413 | 1.153 | 1.115 | 1.462 | 1.393 | 1.118 | 1.347 | 1.205 | 1.371 | 1.423 | 1.446 | 1.412 | 1.331 | 1.179 | 1.168 | 1.636 | 1.077 | 1.456 | 1.178 | nan | 1.170 | 1.409 | 1.420 | 1.040 | 1.647 | 1.042 | 1.032 | 1.067 | 1.262 | 1.120 | 1.214 | 1.800 | 2.322 | 2.302 | 2.358 | 2.058 | 2.087 | 2.111 | 2.292 | 2.195 | 2.207 | 2.211 | 1.294 | 1.378 | 1.389 |
| 0.90 | **0.885** | 0.972 | 0.936 | 0.934 | 0.927 | 0.919 | 0.979 | 0.947 | 1.025 | 0.921 | 1.011 | 0.890 | 0.948 | 1.054 | 1.055 | 1.050 | 0.908 | 0.948 | 1.079 | 1.053 | 1.065 | 0.956 | 1.070 | 1.073 | 1.194 | 1.156 | 1.067 | 1.095 | 1.398 | 1.164 | 1.417 | 1.456 | 1.414 | 1.454 | 1.230 | 1.236 | 1.108 | 1.184 | 1.441 | 1.062 | 1.425 | 1.492 | 1.535 | 1.492 | 1.151 | 1.114 | 1.164 | 1.144 | 2.221 | 2.337 | 1.207 | 1.105 | 1.133 | 1.144 | 1.191 | 1.182 | 1.085 | 0.987 | 1.119 | 1.149 | 1.561 | 1.214 | 1.134 | 1.137 | 1.475 | 1.289 | 1.144 | 1.108 | 1.119 | 1.101 | 1.104 | 2.548 | 1.069 | 0.992 | 1.138 | 1.166 | 0.997 | 1.175 | 1.510 | 1.113 | 1.106 | 1.179 | 1.114 | 1.298 | 1.247 | 1.101 | 1.110 | 1.470 | 1.129 | 1.385 | 1.359 | 1.142 | 1.357 | 1.184 | 1.226 | 1.080 | 1.435 | 1.427 | 1.155 | 1.119 | 1.439 | 1.407 | 1.126 | 1.342 | 1.201 | 1.374 | 1.435 | 1.442 | 1.417 | 1.329 | 1.187 | 1.168 | 1.628 | 1.099 | 1.463 | 1.175 | nan | 1.184 | 1.409 | 1.425 | 1.044 | 1.645 | 1.039 | 1.028 | 1.064 | 1.277 | 1.118 | 1.241 | 1.842 | 2.342 | 2.330 | 2.390 | 2.092 | 2.128 | 2.132 | 2.329 | 2.229 | 2.206 | 2.232 | 1.326 | 1.420 | 1.416 |
| 0.95 | 0.913 | 0.996 | 0.969 | 0.964 | 0.953 | 0.943 | 0.993 | 0.937 | 1.048 | 0.947 | 1.038 | **0.902** | 0.975 | 1.079 | 1.080 | 1.075 | 0.922 | 0.976 | 1.105 | 1.075 | 1.087 | 0.991 | 1.092 | 1.101 | 1.215 | 1.170 | 1.089 | 1.119 | 1.404 | 1.177 | 1.538 | 1.572 | 1.428 | 1.572 | 1.240 | 1.254 | 1.140 | 1.190 | 1.448 | 1.060 | 1.430 | 1.480 | 1.521 | 1.474 | 1.177 | 1.136 | 1.162 | 1.138 | 2.216 | 2.333 | 1.225 | 1.114 | 1.159 | 1.158 | 1.201 | 1.210 | 1.097 | 1.004 | 1.132 | 1.165 | 1.567 | 1.224 | 1.170 | 1.151 | 1.489 | 1.322 | 1.157 | 1.121 | 1.126 | 1.114 | 1.120 | 2.538 | 1.071 | 0.992 | 1.163 | 1.169 | 1.021 | 1.178 | 1.518 | 1.128 | 1.112 | 1.189 | 1.129 | 1.305 | 1.254 | 1.103 | 1.125 | 1.483 | 1.159 | 1.396 | 1.363 | 1.157 | 1.362 | 1.199 | 1.241 | 1.095 | 1.450 | 1.436 | 1.169 | 1.124 | 1.427 | 1.410 | 1.134 | 1.354 | 1.211 | 1.363 | 1.444 | 1.444 | 1.425 | 1.327 | 1.178 | 1.173 | 1.619 | 1.128 | 1.465 | 1.175 | nan | 1.198 | 1.399 | 1.426 | 1.051 | 1.664 | 1.047 | 1.042 | 1.072 | 1.295 | 1.111 | 1.275 | 1.863 | 2.351 | 2.336 | 2.393 | 2.112 | 2.148 | 2.146 | 2.356 | 2.201 | 2.186 | 2.232 | 1.343 | 1.452 | 1.475 |
| 1.00 | **0.948** | 1.058 | 1.004 | 1.004 | 0.990 | 0.978 | 0.989 | 0.974 | 1.092 | 0.977 | 1.076 | 1.021 | 1.014 | 1.128 | 1.129 | 1.124 | 1.048 | 1.012 | 1.160 | 1.122 | 1.130 | 1.025 | 1.141 | 1.117 | 1.238 | 1.188 | 1.141 | 1.141 | 1.409 | 1.194 | 1.655 | 1.703 | 1.437 | 1.724 | 1.262 | 1.274 | 1.208 | 1.200 | 1.452 | 1.056 | 1.437 | 1.472 | 1.509 | 1.461 | 1.223 | 1.186 | 1.153 | 1.133 | 2.143 | 2.262 | 1.282 | 1.131 | 1.343 | 1.179 | 1.214 | 1.345 | 1.128 | 1.023 | 1.154 | 1.187 | 1.566 | 1.236 | 1.212 | 1.173 | 1.510 | 1.343 | 1.181 | 1.142 | 1.141 | 1.141 | 1.163 | 2.464 | 1.071 | 0.997 | 1.202 | 1.336 | 1.043 | 1.190 | 1.530 | 1.145 | 1.115 | 1.205 | 1.160 | 1.308 | 1.260 | 1.097 | 1.145 | 1.501 | 1.180 | 1.397 | 1.373 | 1.177 | 1.367 | 1.304 | 1.262 | 1.188 | 1.450 | 1.437 | 1.216 | 1.130 | 1.437 | 1.416 | 1.137 | 1.351 | 1.331 | 1.352 | 1.449 | 1.447 | 1.426 | 1.333 | 1.179 | 1.171 | 1.596 | 1.163 | 1.463 | 1.174 | nan | 1.217 | 1.398 | 1.426 | 1.056 | 1.613 | 1.134 | 1.137 | 1.165 | 1.310 | 1.112 | 1.325 | 1.896 | 2.346 | 2.329 | 2.394 | 2.121 | 2.155 | 2.153 | 2.360 | 2.367 | 2.318 | 2.358 | 1.353 | 1.489 | 1.529 |
| **mean** | **0.852** | 0.939 | 0.903 | 0.900 | 0.901 | 0.905 | 0.932 | 0.911 | 0.981 | 0.899 | 0.953 | 0.922 | 0.930 | 1.008 | 1.010 | 1.005 | 0.933 | 0.926 | 1.024 | 1.015 | 1.020 | 0.928 | 1.018 | 1.084 | 1.189 | 1.172 | 1.019 | 1.103 | 1.331 | 1.172 | 1.264 | 1.286 | 1.321 | 1.299 | 1.153 | 1.169 | 1.060 | 1.143 | 1.362 | 1.091 | 1.364 | 1.490 | 1.514 | 1.479 | 1.077 | 1.055 | 1.179 | 1.155 | 2.217 | 2.339 | 1.148 | 1.091 | 1.111 | 1.127 | 1.170 | 1.127 | 1.083 | 0.979 | 1.112 | 1.135 | 1.518 | 1.187 | 1.038 | 1.127 | 1.368 | 1.115 | 1.124 | 1.092 | 1.113 | 1.097 | 1.109 | 2.546 | 1.050 | 1.034 | 1.120 | 1.107 | 0.986 | 1.178 | 1.432 | 1.104 | 1.075 | 1.152 | 1.099 | 1.266 | 1.230 | 1.068 | 1.110 | 1.363 | 1.028 | 1.311 | 1.337 | 1.136 | 1.325 | 1.255 | 1.203 | 1.135 | 1.393 | 1.361 | 1.175 | 1.099 | 1.522 | 1.354 | 1.106 | 1.361 | 1.276 | 1.379 | 1.393 | 1.399 | 1.377 | 1.343 | 1.186 | 1.174 | 1.573 | 1.063 | 1.409 | 1.188 | nan | 1.172 | 1.415 | 1.438 | 1.054 | 1.618 | 1.177 | 1.179 | 1.209 | 1.446 | 1.417 | 1.561 | 1.813 | 2.209 | 2.216 | 2.253 | 1.996 | 2.018 | 2.025 | 2.149 | 2.447 | 2.475 | 2.469 | 2.951 | 3.230 | 3.669 |

A's NLL climbs from 1.272 at width 0.25 to 1.437 at 1.00: it is least calibrated where it is most accurate, which is what a training error of 0.000 at the widest width predicts. F does not do that.

## How each branch is built

Each row names the closest branch above it and lists only what differs, so the line that makes a branch itself is the line you read. Diffed against A, most of this table would be K repeated ten times with the distinguishing setting arriving last.

Read out of `apps/cifar100_<name>.yml` when this page was written, so a branch cannot be described here as something its config has stopped being.

Some branches are only readable in pairs, one leaning each way from K: W against Y on blur; AS against AT on where the free widths are drawn; AC against AD on the weight on the feature terms. A bracket where both ends win says the axis does not matter, which is an answer the winning end alone cannot give.

One bracket does not close: AU against AV on which samples KD attends to, where AV collapsed. A collapsed end is not a losing end, so it cannot stand as the control its pair needed, and the axis stays open.

**ED** - DO (K with four heads, seed 1995) trained with Sharpness-Aware Minimization, rho 0.1. 78.16: 0.32 above DO, inside the noise; appendix only, SAM is not part of the method

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 6`
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `sam_rho: 0.1` - Sharpness-Aware Minimization: each step runs twice, at the weights and at the worst nearby point rho away along the gradient, and steps with the second gradient
* `split_pair_backward: True`

**DM** - K as DK but the block of 20 widths is plain uniform draws, sorted and dealt spread: one free width from each half, steps shuffled

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 6`
* `split_pair_backward: True`
* `stratified_block: 10` - the number of steps drawn together
* `stratified_draw: uniform` - the block's widths are plain uniform draws, sorted, instead of one per slice
* `stratified_pairing: spread` - how a block's sorted draws are dealt out: `adjacent` gives a step two neighbouring widths and runs the block narrow to wide, `spread` gives it one from each half and shuffles the steps
* `width_sampling: stratified` - the free widths drawn a block of steps at a time, one per equal slice of the range, so the block covers it evenly

**EC** - DO (K with four heads, seed 1995) trained with Sharpness-Aware Minimization, rho 0.05. 78.02: 0.18 above DO, inside the noise; appendix only, SAM is not part of the method

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 6`
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `sam_rho: 0.05` - Sharpness-Aware Minimization: each step runs twice, at the weights and at the worst nearby point rho away along the gradient, and steps with the second gradient
* `split_pair_backward: True`

**DO** - K as DD with a classifier head per band of widths, 4 bands (SOLAR's idea on K)

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 6`
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `split_pair_backward: True`

**EE** - K with four classifier heads (DO) and far free-width pairs (DM), seed 1995: SlimOT + SPS + WBH. 77.83, level with DO (77.84) and 0.33 under DM, so the two do not add

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 6`
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `split_pair_backward: True`
* `stratified_block: 10` - the number of steps drawn together
* `stratified_draw: uniform` - the block's widths are plain uniform draws, sorted, instead of one per slice
* `stratified_pairing: spread` - how a block's sorted draws are dealt out: `adjacent` gives a step two neighbouring widths and runs the block narrow to wide, `spread` gives it one from each half and shuffles the steps
* `width_sampling: stratified` - the free widths drawn a block of steps at a time, one per equal slice of the range, so the block covers it evenly

**EI** - DO (K with four heads, SlimOT + WBH) at seed 2026. 77.76: +0.49, above at 15/16 against DC (K at 2026, 77.27), +1.30, above at 16/16 against CQ (A at 2026)

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 6`
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `random_seed: 2026` - the seed, for a repeat of a branch already run
* `split_pair_backward: True`

**DQ** - K as DD with a classifier head per band of widths, 16 bands (one per test width)

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 6`
* `head_groups: 16` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `split_pair_backward: True`

**DP** - K as DD with a classifier head per band of widths, 8 bands

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 6`
* `head_groups: 8` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `split_pair_backward: True`

**FI** - DD (the method without WBH: one shared classifier) at seed 2006 (notebook 70), over two sessions. 77.66: 0.28 ABOVE the method with WBH at the same seed (FC 77.38), above it at 15/16 widths; +0.93 over US-Net (FB 76.73). DD 77.50 and DC 77.27 at the other seeds, so SlimOT without WBH is 77.47 +- 0.20 against 77.66 +- 0.24 with it: WBH on SlimOT is +0.34, +0.49, -0.28 by seed (+0.19 +- 0.41), while on US-Net it is +0.35, +0.40, +0.44. The 2x2 interaction, about 0 at 1995 and 2026, is -0.72 at 2006

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 6`
* `random_seed: 2006` - the seed, for a repeat of a branch already run
* `split_pair_backward: True`

**FL** - CI (the method without Delayed Transport) at seed 2006 (notebook 69), over two sessions. 77.62: no collapse; 0.23 ABOVE the method at the same seed (FC 77.38), above it at 15/16 widths, as FI (without WBH, 77.66) was: at 2006 FC sits under both of its ablations. Row 'w/o Delayed Transport' over three seeds: CI 77.27, FH 77.55, FL 77.62, mean 77.48 +- 0.19

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `random_seed: 2006` - the seed, for a repeat of a branch already run
* `split_pair_backward: True`

**DN** - K as DD with a classifier head per band of widths, 2 bands (SOLAR's idea on K)

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 6`
* `head_groups: 2` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `split_pair_backward: True`

**EN** - DO (SlimOT + WBH, seed 1995) without Teacher Transport: feature_kd off, Peer Transport, logit KD, Delayed Transport and the four heads kept (notebook 55, resumed at epoch 89). 77.60: 0.24 under DO (77.84), above it at 6/16 widths, inside one seed's noise; +0.39 over the heads alone (CH 77.21) at 15/16. Peer Transport alone keeps most of the gain

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_kd: False` - adds a vertical term between the student's pooled features and the teacher's, on top of the logit KD
* `feature_start_epoch: 6`
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `split_pair_backward: True`

**FH** - CI (the method without Delayed Transport: both transport terms from epoch 1) at seed 2026 (notebook 69), over two sessions. 77.55: no collapse, at the seed where K without Delayed Transport and without heads collapsed (CR, widths up to 0.60 at chance); 0.22 under the method at the same seed (EI 77.76), above it at 1/16 widths

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `random_seed: 2026` - the seed, for a repeat of a branch already run
* `split_pair_backward: True`

**DT** - DD's final weights frozen, a rank-4 LoRA update per conv at 4 width knots added and trained alone for 20 epochs on K's loss (TAS-LoRA's recipe, interpolation in width): level with DD

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 0`
* `lora_knots: 4` - every conv kernel gets a low-rank update at this many evenly spaced widths, mixed by distance in between, so each width has a little private weight
* `lora_rank: 4` - the rank of each knot's update
* `lr: 0.02`
* `lr_warmup_epochs: 1`
* `num_epochs: 20` - the training budget; every row without this line trained for 100 epochs
* `pretrained: pretrained/dd_k_late_r50.pt` - the run starts from these weights; keys a newer model adds keep their initial values
* `split_pair_backward: True`
* `train_only: [lora_]` - only parameters whose names contain these fragments train; the rest is loaded from pretrained, frozen and out of the optimizer

**DD** - K on ResNet-50 as BY with both feature transport terms held off until epoch 6, at BY's seed 1995: what the late start costs where nothing broke

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 6`
* `split_pair_backward: True`

**DU** - DD's final weights frozen, BN scale/shift offsets at 4 width knots added and trained alone for 20 epochs on K's loss: level with DD

K, with:

* `affine_knots: 4` - every BN scale and shift is a continuous function of the width: an offset at each of this many evenly spaced knots, interpolated linearly between them, zero at the start
* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 0`
* `lr: 0.02`
* `lr_warmup_epochs: 1`
* `num_epochs: 20` - the training budget; every row without this line trained for 100 epochs
* `pretrained: pretrained/dd_k_late_r50.pt` - the run starts from these weights; keys a newer model adds keep their initial values
* `split_pair_backward: True`
* `train_only: [knot_]` - only parameters whose names contain these fragments train; the rest is loaded from pretrained, frozen and out of the optimizer

**FF** - EN (the method without Teacher Transport: feature_kd off) at seed 2026 (notebook 67), over two sessions. 77.44: 0.33 under the method at the same seed (EI 77.76), above it at 2/16 widths; +0.97 over US-Net (CQ 76.46) at 16/16. No collapse at the seed where K without Delayed Transport collapsed (CR). With EN (-0.24 at 1995) the row 'w/o Teacher Transport' is a consistent small loss

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_kd: False` - adds a vertical term between the student's pooled features and the teacher's, on top of the logit KD
* `feature_start_epoch: 6`
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `random_seed: 2026` - the seed, for a repeat of a branch already run
* `split_pair_backward: True`

**FC** - The method (SlimOT + WBH, as DO) at seed 2006 (notebook 60), over two sessions. 77.38: +0.66 over US-Net at the same seed (FB 76.73) at 16/16 widths, +0.22 over SOLAR (FM 77.16) at 14/16, +0.72 over Scala (FN 76.66) at 16/16; DO 77.84 and EI 77.76 at the other seeds, so the method's three-seed mean is 77.66 ± 0.24

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 6`
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `random_seed: 2006` - the seed, for a repeat of a branch already run
* `split_pair_backward: True`

**DK** - K as DD with the same block draws dealt out spread: one free width from each half (about 0.375 apart), steps shuffled

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 6`
* `split_pair_backward: True`
* `stratified_block: 10` - the number of steps drawn together
* `stratified_pairing: spread` - how a block's sorted draws are dealt out: `adjacent` gives a step two neighbouring widths and runs the block narrow to wide, `spread` gives it one from each half and shuffles the steps
* `width_sampling: stratified` - the free widths drawn a block of steps at a time, one per equal slice of the range, so the block covers it evenly

**DR** - K as DD with a rank-4 LoRA update per conv at 4 width knots, mixed by distance, trained from the start (1.27M stored, merged at inference)

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 6`
* `lora_knots: 4` - every conv kernel gets a low-rank update at this many evenly spaced widths, mixed by distance in between, so each width has a little private weight
* `lora_rank: 4` - the rank of each knot's update
* `split_pair_backward: True`

**DC** - K on ResNet-50 as BY (post-ReLU read) with both feature transport terms held off until epoch 6, at seed 2026 where CR collapsed: survives

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 6`
* `random_seed: 2026` - the seed, for a repeat of a branch already run
* `split_pair_backward: True`

**CI** - K on ResNet-50 with four band heads (0.25-0.40, 0.45-0.60, 0.65-0.80, 0.85-1.00) over the shared backbone

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `split_pair_backward: True`

**BY** - K on ResNet-50, the pair for BX: the same feature transport, 100 epochs

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `split_pair_backward: True`

**FG** - EO (the method without Peer Transport: horizontal_kd off) at seed 2026 (notebook 67). 77.21: 0.55 under the method at the same seed (EI 77.76) at 16/16 widths; +0.75 over US-Net (CQ 76.46) at 16/16. With EO (-0.81 at 1995, 16/16) Peer Transport is the term the gain rests on

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 6`
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `horizontal_kd: False` - adds a term between two co-sampled widths that stand in no teacher relation to each other
* `random_seed: 2026` - the seed, for a repeat of a branch already run
* `split_pair_backward: True`

**CH** - SOLAR (WACV 2026) adapted to US-Net on ResNet-50: A with a classifier head per band of widths (four bands of four test widths), 0.6M extra parameters; seed 1995

A, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone

**FM** - SOLAR (WACV 2026) adapted to US-Net on ResNet-50, as CH (four band heads, no transport) at seed 2006 (notebook 61). 77.16: +0.30 over BX and 0.34 under DD at seed 1995's references; CH 77.21 and FD 76.86 at the other seeds, so SOLAR's three-seed mean is 77.08 ± 0.19. Read against FC (the method at 2006, notebook 60)

A, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `random_seed: 2006` - the seed, for a repeat of a branch already run

**DS** - K as DD with a rank-8 LoRA update per conv at 2 width knots (the range ends), trained from the start (1.27M stored)

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 6`
* `lora_knots: 2` - every conv kernel gets a low-rank update at this many evenly spaced widths, mixed by distance in between, so each width has a little private weight
* `lora_rank: 8` - the rank of each knot's update
* `split_pair_backward: True`

**EO** - DO (SlimOT + WBH, seed 1995) without Peer Transport: horizontal_kd off, Teacher Transport, logit KD, Delayed Transport and the four heads kept (notebook 55, resumed at epoch 92). 77.03: 0.81 under DO (77.84) at 16/16 widths, and 0.18 under the heads alone (CH 77.21), above it at 1/16. Teacher Transport alone adds nothing on top of the heads; it helps only beside Peer Transport (DO against EN, +0.24)

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 6`
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `horizontal_kd: False` - adds a term between two co-sampled widths that stand in no teacher relation to each other
* `split_pair_backward: True`

**BX** - A on ResNet-50: US-Net as published, bottleneck blocks, 100 epochs

A, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width

**FD** - SOLAR (WACV 2026) adapted to US-Net on ResNet-50, as CH (four band heads, no transport) at seed 2026 (notebook 63). 76.86: 0.91 under the method at the same seed (EI 77.76) at 16/16 widths, +0.40 over US-Net (CQ 76.46) at 15/16; CH gave 77.21 at 1995, so SOLAR's two-seed mean is 77.03 against the method's 77.80

A, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `random_seed: 2026` - the seed, for a repeat of a branch already run

**CF** - Scala (NeurIPS 2024) on ResNet-50, ported from the authors' code: isolated narrowest width, stable sampling, teacher chain, the label for every width, 10 epochs of the full width alone; published baseline, seed 1995

A, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `isolate_smallest: True` - Scala: the narrowest width takes the last channels of every layer instead of the first
* `narrow_start_epoch: 10`
* `stable_granularity: 0.05` - the grid step for stable sampling
* `student_ce_weight: 1.0` - Scala: every student also takes the hard label at this weight
* `teacher_chain: True` - each width learns from the next larger one in the batch instead of every width learning from the widest
* `width_sampling: stable` - Scala: one free width from each half of the range, on a grid

**FE** - Scala (NeurIPS 2024) on ResNet-50, as CF at seed 2026 (notebook 63). 76.73: 1.03 under the method at the same seed (EI 77.76) at 16/16 widths, +0.27 over US-Net (CQ 76.46) at 14/16; CF gave 76.80 at 1995, so Scala's two-seed mean is 76.77 against the method's 77.80

A, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `isolate_smallest: True` - Scala: the narrowest width takes the last channels of every layer instead of the first
* `narrow_start_epoch: 10`
* `random_seed: 2026` - the seed, for a repeat of a branch already run
* `stable_granularity: 0.05` - the grid step for stable sampling
* `student_ce_weight: 1.0` - Scala: every student also takes the hard label at this weight
* `teacher_chain: True` - each width learns from the next larger one in the batch instead of every width learning from the widest
* `width_sampling: stable` - Scala: one free width from each half of the range, on a grid

**FB** - A on ResNet-50 (US-Net as published, BX) at seed 2006 (notebook 60), the third seed of the main table. 76.73: BX 76.86 and CQ 76.46 at the other seeds, so US-Net's three-seed mean is 76.68 ± 0.20. Read against FC

A, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `random_seed: 2006` - the seed, for a repeat of a branch already run

**FN** - Scala (NeurIPS 2024) on ResNet-50, as CF at seed 2006 (notebook 61). 76.66: 0.50 under SOLAR at the same seed (FM 77.16) at 16/16 widths; CF 76.80 and FE 76.73 at the other seeds, so Scala's three-seed mean is 76.73 ± 0.07. Read against FC (the method at 2006, notebook 60)

A, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `isolate_smallest: True` - Scala: the narrowest width takes the last channels of every layer instead of the first
* `narrow_start_epoch: 10`
* `random_seed: 2006` - the seed, for a repeat of a branch already run
* `stable_granularity: 0.05` - the grid step for stable sampling
* `student_ce_weight: 1.0` - Scala: every student also takes the hard label at this weight
* `teacher_chain: True` - each width learns from the next larger one in the batch instead of every width learning from the widest
* `width_sampling: stable` - Scala: one free width from each half of the range, on a grid

**DL** - K as DJ but the block of 20 widths is plain uniform draws, sorted and dealt adjacent: close pairs, narrow to wide

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 6`
* `split_pair_backward: True`
* `stratified_block: 10` - the number of steps drawn together
* `stratified_draw: uniform` - the block's widths are plain uniform draws, sorted, instead of one per slice
* `stratified_pairing: adjacent` - how a block's sorted draws are dealt out: `adjacent` gives a step two neighbouring widths and runs the block narrow to wide, `spread` gives it one from each half and shuffles the steps
* `width_sampling: stratified` - the free widths drawn a block of steps at a time, one per equal slice of the range, so the block covers it evenly

**DJ** - K as DD with the free widths drawn ten steps at a time, one per twentieth of the range, sorted and paired as neighbours (about 0.04 apart), each block narrow to wide

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 6`
* `split_pair_backward: True`
* `stratified_block: 10` - the number of steps drawn together
* `stratified_pairing: adjacent` - how a block's sorted draws are dealt out: `adjacent` gives a step two neighbouring widths and runs the block narrow to wide, `spread` gives it one from each half and shuffles the steps
* `width_sampling: stratified` - the free widths drawn a block of steps at a time, one per equal slice of the range, so the block covers it evenly

**EF** - DM (far free-width pairs, SlimOT + SPS) at seed 2026. 76.51: 0.76 under K at the same seed (DC 77.27) at 16/16 widths and level with A at that seed (CQ 76.46), so DM's 78.16 did not repeat

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_start_epoch: 6`
* `random_seed: 2026` - the seed, for a repeat of a branch already run
* `split_pair_backward: True`
* `stratified_block: 10` - the number of steps drawn together
* `stratified_draw: uniform` - the block's widths are plain uniform draws, sorted, instead of one per slice
* `stratified_pairing: spread` - how a block's sorted draws are dealt out: `adjacent` gives a step two neighbouring widths and runs the block narrow to wide, `spread` gives it one from each half and shuffles the steps
* `width_sampling: stratified` - the free widths drawn a block of steps at a time, one per equal slice of the range, so the block covers it evenly

**CP** - K on ResNet-50 with the logit KD taken out: students learn from the label, the two feature transport terms are the only link between widths

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `split_pair_backward: True`
* `student_ce_weight: 1.0` - Scala: every student also takes the hard label at this weight
* `student_kd_weight: 0.0` - the weight on logit KD for every student; 0 leaves them learning from the label and the transport terms alone

**CQ** - A on ResNet-50 again at seed 2026, the second draw of BX

A, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `random_seed: 2026` - the seed, for a repeat of a branch already run

**DE** - A on ResNet-50 exactly as BX with amp: True (fp16 training forward, fp32 losses, validation and calibration): what mixed precision does to A

A, with:

* `amp: True`
* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width

**CV** - AlphaNet (ICML 2021) on ResNet-50: BX with the inplace distillation KL replaced by the adaptive alpha-divergence, alpha in [-1, 1], ratios clipped at 5; published baseline, seed 1995

A, with:

* `alpha_iw_clip: 5.0` - importance weight clip in the alpha-divergence
* `alpha_max: 1.0` - upper end of the adaptive alpha range
* `alpha_min: -1.0` - lower end of the adaptive alpha range
* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `kd_loss: alpha` - the vertical KD term becomes an adaptive alpha-divergence, AlphaNet as published

**FZ** - AlphaNet (ICML 2021) on ResNet-50, as CV at seed 2006 (notebook 62). 75.94: 1.22 under SOLAR at the same seed (FM 77.16); CV 76.18 and FY 75.91 at the other seeds, so AlphaNet's three-seed mean is 76.01 ± 0.15. Read against FC (the method at 2006, notebook 60)

A, with:

* `alpha_iw_clip: 5.0` - importance weight clip in the alpha-divergence
* `alpha_max: 1.0` - upper end of the adaptive alpha range
* `alpha_min: -1.0` - lower end of the adaptive alpha range
* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `kd_loss: alpha` - the vertical KD term becomes an adaptive alpha-divergence, AlphaNet as published
* `random_seed: 2006` - the seed, for a repeat of a branch already run

**FY** - AlphaNet (ICML 2021) on ResNet-50, as CV at seed 2026 (notebook 62). 75.91: 1.86 under the method at the same seed (EI 77.76) and 0.55 under US-Net (CQ 76.46), both at 16/16 widths; CV 76.18 and FZ 75.94 at the other seeds, so AlphaNet's three-seed mean is 76.01 ± 0.15

A, with:

* `alpha_iw_clip: 5.0` - importance weight clip in the alpha-divergence
* `alpha_max: 1.0` - upper end of the adaptive alpha range
* `alpha_min: -1.0` - lower end of the adaptive alpha range
* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `kd_loss: alpha` - the vertical KD term becomes an adaptive alpha-divergence, AlphaNet as published
* `random_seed: 2026` - the seed, for a repeat of a branch already run

**CU** - K on ResNet-50 with the transport read before the last ReLU, seed 2026, the seed CR collapsed at: whether the fix holds where K broke; against CQ

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_pre_relu: True`
* `random_seed: 2026` - the seed, for a repeat of a branch already run
* `split_pair_backward: True`

**CT** - K on ResNet-50 with both feature transport terms read before the last ReLU (feature_pre_relu), seed 1995: what the collapse fix costs where K did not collapse; against BY

K, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `feature_pre_relu: True`
* `split_pair_backward: True`

**DG** - WKD-L (NeurIPS 2024) as published in place of US-Net's inplace KL on ResNet-50: target term, non-target transport at T 8 under the fc class cost, weight 600 cosine-decayed over the last 37.5%, CE for every student; seed 1995

A, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `kd_loss: wkd`
* `student_ce_weight: 1.0` - Scala: every student also takes the hard label at this weight
* `wkd_decay_from: 0.625`
* `wkd_iters: 10`
* `wkd_reg: 0.05`
* `wkd_temperature: 8.0`
* `wkd_weight: 600.0`

**GC** - WKD-L (NeurIPS 2024) on ResNet-50, as DG at seed 2026 (notebook 64). 74.72, the same mean as DG by coincidence (the per-width values differ): 3.05 under the method at the same seed (EI 77.76) and 1.74 under US-Net (CQ 76.46); DG 74.72 and GD 73.94 at the other seeds, so WKD-L's three-seed mean is 74.46 ± 0.45

A, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `kd_loss: wkd`
* `random_seed: 2026` - the seed, for a repeat of a branch already run
* `student_ce_weight: 1.0` - Scala: every student also takes the hard label at this weight
* `wkd_decay_from: 0.625`
* `wkd_iters: 10`
* `wkd_reg: 0.05`
* `wkd_temperature: 8.0`
* `wkd_weight: 600.0`

**BP** - K plus one learnable scalar per width on each residual branch, 128 parameters

K, with:

* `width_scalars: True`

**AW** - K with each width taught by the next larger one instead of by the widest

K, with:

* `teacher_chain: True` - each width learns from the next larger one in the batch instead of every width learning from the widest

**GA** - DYNAS (CVPR 2025) on ResNet-50, as DB at seed 2026 (notebook 63). 74.32: 3.44 under the method at the same seed (EI 77.76) and 2.14 under US-Net (CQ 76.46); the narrow end is where it loses, 67.14 at 0.25 against 77.10 at 1.00, as DB did. DB 73.92 and GB 74.23 at the other seeds, so DYNAS's three-seed mean is 74.16 ± 0.21

A, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `dynas: True`
* `dynas_groups: 4`
* `dynas_max_coeff: 4.0`
* `random_seed: 2026` - the seed, for a repeat of a branch already run

**K** - I plus transport between the two middle widths, at the feature tier

I, with:

* `horizontal_kd: True` - adds a term between two co-sampled widths that stand in no teacher relation to each other
* `horizontal_weight: 1.0` - the weight on the horizontal term
* `horizontal_where: feature` - which tier the horizontal term sits at: `logit`, `feature`, or `both`

**X** - K at eps 0.1 instead of 0.2

K, with:

* `sinkhorn_eps: 0.1` - the entropic blur. Small collapses the plan onto a permutation, large spreads mass over many partners

**GB** - DYNAS (CVPR 2025) on ResNet-50, as DB at seed 2006 (notebook 63). 74.23: 2.93 under SOLAR at the same seed (FM 77.16), 67.58 at 0.25 against 76.79 at 1.00; DB 73.92 and GA 74.32 at the other seeds, so DYNAS's three-seed mean is 74.16 ± 0.21. Read against FC (the method at 2006, notebook 60)

A, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `dynas: True`
* `dynas_groups: 4`
* `dynas_max_coeff: 4.0`
* `random_seed: 2006` - the seed, for a repeat of a branch already run

**V** - K with transport that may leave mass unmatched, at tau 1.0

K, with:

* `feature_loss: unbalanced` - transport that may leave mass unmatched, the marginals penalised rather than enforced
* `feature_weight: 0.8` - the weight on the feature terms, set by matching gradient norms rather than loss values
* `unbalanced_tau: 1.0` - the price of leaving mass unmatched. Large recovers the balanced plan

**EW** - EP (SlimOT + WBH) on MobileNetV2 with the last layer slimmed (slim_last), seed 1995, notebook 59. 74.18: +0.77 against EV (same backbone), +0.16 against ER on the standard backbone. 384 min.

K, with:

* `feature_start_epoch: 6`
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `model: models.us_mobilenet_v2_cifar`
* `slim_last: True`
* `split_pair_backward: True`

**BO** - K with the conv output divided by how many input channels are live, US-Net Appendix A

K, with:

* `conv_averaged: True`

**AO** - K plus a repulsion pushing the classifier rows apart

K, with:

* `spread_neighbours: 5` - how many nearest rows the repulsion charges for, which makes it a minimum margin rather than a mean spread
* `spread_weight: 1.0` - a repulsion on the classifier rows, the one term here with the opposite sign to the rest

**BR** - A at 300 epochs, US-Net as published with three times the schedule

A, with:

* `num_epochs: 300` - the training budget; every row without this line trained for 100 epochs

**AQ** - K transporting the two widths' class means rather than their samples

K, with:

* `feature_classwise: True` - transport runs between class means rather than between samples
* `feature_weight: 0.7` - the weight on the feature terms, set by matching gradient norms rather than loss values

**EU** - EP without the band heads (head_groups 1): SlimOT alone on MobileNetV2, CIFAR-100, seed 1995 (notebook 58). 74.06: +0.04 against ER, +0.38 against EP and above EP at 16/16 widths, so on this backbone the heads cost every width, the reverse of ResNet-50. Against ER the narrow end is still down (0.25-0.35: -0.56 to -0.93) and the wide end up (0.85-1.0: +0.3 to +0.5). Its partner ET is recorded separately, recalibrated by hand.

K, with:

* `feature_start_epoch: 6`
* `head_groups: 1` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `model: models.us_mobilenet_v2_cifar`
* `split_pair_backward: True`

**AF** - K coupling all three student pairs instead of one

K, with:

* `horizontal_pairs: all` - every pair among the co-sampled students is coupled, not just the two middle ones

**ER** - US-Net as published (BX recipe) on MobileNetV2, CIFAR-100, seed 1995: the baseline for EP. models/us_mobilenet_v2_cifar.py, CIFAR strides, last 1x1 to 1280 unslimmed as in US-Net. 74.02.

A, with:

* `model: models.us_mobilenet_v2_cifar`

**BE** - K, narrow end held back 25 epochs, then channels permuted by L1

K, with:

* `narrow_start_epoch: 25`
* `reorder_by: l1`
* `reorder_epoch: 25`
* `reorder_report: True`

**BD** - K, narrow end held back 10 epochs, then channels permuted by L1

K, with:

* `narrow_start_epoch: 10`
* `reorder_by: l1`
* `reorder_epoch: 10`
* `reorder_report: True`

**AT** - K with the same free widths drawn flat in compute instead

K, with:

* `width_sampling: macs` - the same draw made flat in compute. MACs measured at width^1.965 here, so this leans toward the wide end

**BI** - K, narrow end held back 25 epochs, then channels permuted by the Taylor score

K, with:

* `narrow_start_epoch: 25`
* `reorder_by: taylor`
* `reorder_epoch: 25`
* `reorder_report: True`
* `taylor_batches: 8`

**AC** - K at half the weight on the feature terms

K, with:

* `feature_weight: 0.5` - the weight on the feature terms, set by matching gradient norms rather than loss values

**GD** - WKD-L (NeurIPS 2024) on ResNet-50, as DG at seed 2006 (notebook 64). 73.94: 3.22 under SOLAR at the same seed (FM 77.16); DG 74.72 and GC 74.72 at the other seeds, so WKD-L's three-seed mean is 74.46 ± 0.45. Read against FC (the method at 2006, notebook 60)

A, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `kd_loss: wkd`
* `random_seed: 2006` - the seed, for a repeat of a branch already run
* `student_ce_weight: 1.0` - Scala: every student also takes the hard label at this weight
* `wkd_decay_from: 0.625`
* `wkd_iters: 10`
* `wkd_reg: 0.05`
* `wkd_temperature: 8.0`
* `wkd_weight: 600.0`

**GU** - SOLAR (one head per band of widths, four bands) on MobileNetV2 with the last layer slimmed, CIFAR-100, seed 2026

A, with:

* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `model: models.us_mobilenet_v2_cifar`
* `random_seed: 2026` - the seed, for a repeat of a branch already run
* `slim_last: True`

**BN** - K, and every sampled width learns from the mean of what all of them said

K, with:

* `ensemble_teacher: True`
* `ensemble_weight: 1.0`

**AS** - K with the sandwich rule's free widths drawn flat in log width

K, with:

* `width_sampling: log` - the sandwich rule draws its free widths flat in log width, which puts half of them below 0.50 against a third for uniform

**DB** - DYNAS (CVPR 2025) on ResNet-50: per-width learning rate (1 - t/T)^p, p 4 at width 0.25 to 1/4 at 1.0, momentum per band of widths, each width stepped on its own; seed 1995

A, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `dynas: True`
* `dynas_groups: 4`
* `dynas_max_coeff: 4.0`

**FV** - SlimOT on MobileNetV2 with the last layer slimmed, CIFAR-100, seed 2006

K, with:

* `feature_start_epoch: 6`
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `model: models.us_mobilenet_v2_cifar`
* `random_seed: 2006` - the seed, for a repeat of a branch already run
* `slim_last: True`
* `split_pair_backward: True`

**BS** - K at 300 epochs, the pair for BR

K, with:

* `num_epochs: 300` - the training budget; every row without this line trained for 100 epochs

**BF** - K, narrow end held back 10 epochs, nothing permuted: the control for BD and BH

K, with:

* `narrow_start_epoch: 10`
* `reorder_report: True`

**GT** - SOLAR (one head per band of widths, four bands) on MobileNetV2 with the last layer slimmed, CIFAR-100, seed 1995

A, with:

* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `model: models.us_mobilenet_v2_cifar`
* `slim_last: True`

**AD** - K at twice the weight, the other side of the same question

K, with:

* `feature_weight: 2.0` - the weight on the feature terms, set by matching gradient norms rather than loss values

**BG** - K, narrow end held back 25 epochs, nothing permuted: the control for BE and BI

K, with:

* `narrow_start_epoch: 25`
* `reorder_report: True`

**AA** - K charging transport by direction rather than distance

K, with:

* `feature_ground: cosine` - the ground cost is angle rather than distance
* `feature_weight: 78.0` - the weight on the feature terms, set by matching gradient norms rather than loss values

**W** - K at eps 0.02, a plan collapsed onto a permutation

K, with:

* `sinkhorn_eps: 0.02` - the entropic blur. Small collapses the plan onto a permutation, large spreads mass over many partners

**GV** - SOLAR† (US-Net + four band heads) on MobileNetV2 with the last layer slimmed, CIFAR-100, seed 2006

A, with:

* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `model: models.us_mobilenet_v2_cifar`
* `random_seed: 2006` - the seed, for a repeat of a branch already run
* `slim_last: True`

**BH** - K, narrow end held back 10 epochs, then channels permuted by the Taylor score

K, with:

* `narrow_start_epoch: 10`
* `reorder_by: taylor`
* `reorder_epoch: 10`
* `reorder_report: True`
* `taylor_batches: 8`

**EP** - SlimOT + WBH (DO recipe) on MobileNetV2, CIFAR-100, seed 1995. 73.68: -0.34 against ER, above at 2/16 widths; the loss is at the narrow end (0.25-0.40: -0.5 to -1.4), the wide half level. The transport reads the unslimmed 1280-channel feature, so prefix alignment is the whole feature at every width, unlike ResNet where it reads the slimmed last block.

K, with:

* `feature_start_epoch: 6`
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `model: models.us_mobilenet_v2_cifar`
* `split_pair_backward: True`

**I** - A plus transport between the student and teacher feature clouds

K, with:

* `horizontal_kd: False` - adds a term between two co-sampled widths that stand in no teacher relation to each other
* `horizontal_weight` not set, against `1.0` in K
* `horizontal_where` not set, against `feature` in K

**AM** - K with sliced transport keeping the worst direction rather than the average

K, with:

* `feature_loss: sliced` - transport along random one dimensional projections instead of a full plan
* `feature_weight: 0.09` - the weight on the feature terms, set by matching gradient norms rather than loss values
* `sliced_reduce: max` - keeps the worst projection rather than the average of them

**BQ** - K with each channel gradient divided by the square root of how many widths wrote to it

K, with:

* `width_equalize_q: 0.5`

**AK** - K with the closed-form Gaussian transport, mean gap plus a covariance term

K, with:

* `feature_loss: bures` - the closed form between two Gaussians fitted to the clouds: mean gap plus a covariance term
* `feature_weight: 0.4` - the weight on the feature terms, set by matching gradient norms rather than loss values

**F** - A plus a symmetrized KL between the two middle widths

C, with:

* `horizontal_kd: True` - adds a term between two co-sampled widths that stand in no teacher relation to each other
* `horizontal_loss: jeffreys` - what compares the two widths at the logit tier: `wasserstein`, `jeffreys`, or `kl`
* `horizontal_where: logit` - which tier the horizontal term sits at: `logit`, `feature`, or `both`
* `kd_loss: soft_ce`

**A'** - the same branch again, to measure the noise

A, with:

* `random_seed: 2026` - the seed, for a repeat of a branch already run

**M** - K with transport at every stage, not the last one alone

K, with:

* `feature_layers: [stage2, stage3, stage4, final]` - the depths the feature term is applied at, instead of the final pooled tap alone

**AH** - K, F and all three pairs at once, every addition together

K, with:

* `horizontal_loss: jeffreys` - what compares the two widths at the logit tier: `wasserstein`, `jeffreys`, or `kl`
* `horizontal_pairs: all` - every pair among the co-sampled students is coupled, not just the two middle ones
* `horizontal_where: both` - which tier the horizontal term sits at: `logit`, `feature`, or `both`

**AL** - K with Gaussian transport on the per channel variances alone

K, with:

* `bures_diagonal: True` - keeps only the per channel variances and drops every cross channel term
* `feature_loss: bures` - the closed form between two Gaussians fitted to the clouds: mean gap plus a covariance term
* `feature_weight: 2.7` - the weight on the feature terms, set by matching gradient norms rather than loss values

**A** - US-Net as published: inplace distillation with KL. The reference every row below is measured against.

**AE** - K and F at once, transport on features and Jeffreys on logits

K, with:

* `horizontal_loss: jeffreys` - what compares the two widths at the logit tier: `wasserstein`, `jeffreys`, or `kl`
* `horizontal_where: both` - which tier the horizontal term sits at: `logit`, `feature`, or `both`

**EV** - ER (US-Net) on MobileNetV2 with the last 1x1 to 1280 slimmed (slim_last), with its BN and the classifier input, seed 1995, notebook 59. 73.41: slimming the last layer costs US-Net 0.61 against ER (74.02). The baseline EW is read against.

A, with:

* `model: models.us_mobilenet_v2_cifar`
* `slim_last: True`

**AR** - K with the logit teacher softened to temperature 4

K, with:

* `kd_temperature: 4.0` - the logit teacher is softened by this factor before the student is matched to it, with a T^2 rescale so the gradient stays comparable

**Y** - K at eps 0.5, mass spread over many partners

K, with:

* `sinkhorn_eps: 0.5` - the entropic blur. Small collapses the plan onto a permutation, large spreads mass over many partners

**FU** - US-Net on MobileNetV2 with the last layer slimmed, CIFAR-100, seed 2006

A, with:

* `model: models.us_mobilenet_v2_cifar`
* `random_seed: 2006` - the seed, for a repeat of a branch already run
* `slim_last: True`

**D** - C plus Wasserstein between the two middle widths

C, with:

* `horizontal_kd: True` - adds a term between two co-sampled widths that stand in no teacher relation to each other
* `horizontal_weight: 1.0` - the weight on the horizontal term

**E** - A plus plain KL between the two middle widths, neither symmetric nor metric aware

C, with:

* `horizontal_kd: True` - adds a term between two co-sampled widths that stand in no teacher relation to each other
* `horizontal_loss: kl` - what compares the two widths at the logit tier: `wasserstein`, `jeffreys`, or `kl`
* `horizontal_where: logit` - which tier the horizontal term sits at: `logit`, `feature`, or `both`
* `kd_loss: soft_ce`

**G** - C with every pair of classes equally far apart, so transport has no geometry to use

C, with:

* `cost_source: identity` - where the class cost matrix comes from. `fc` is the distance between classifier rows, `confusion` is what the teacher mistakes for what, `identity` is every pair equally far apart

**AI** - K transporting one shared channel at a time, exactly rather than by projection

K, with:

* `feature_loss: channel` - exact transport along each shared channel, one dimension at a time, which the nesting makes meaningful
* `feature_weight: 1.7` - the weight on the feature terms, set by matching gradient norms rather than loss values

**S** - K with sliced transport on 128 projections

K, with:

* `feature_loss: sliced` - transport along random one dimensional projections instead of a full plan
* `feature_weight: 1.6` - the weight on the feature terms, set by matching gradient norms rather than loss values
* `sliced_projections: 128` - how many random directions stand in for the full plan

**T** - K with sliced transport on 32 random projections rather than the entropic plan

K, with:

* `feature_loss: sliced` - transport along random one dimensional projections instead of a full plan
* `feature_weight: 0.8` - the weight on the feature terms, set by matching gradient norms rather than loss values
* `sliced_projections: 32` - how many random directions stand in for the full plan

**H** - C with the class metric taken from what the teacher confuses, rather than from the classifier rows

C, with:

* `confusion_momentum: 0.01` - how fast the confusion embedding updates
* `confusion_warmup: 50` - steps before the confusion cost is trusted
* `cost_source: confusion` - where the class cost matrix comes from. `fc` is the distance between classifier rows, `confusion` is what the teacher mistakes for what, `identity` is every pair equally far apart

**GN** - AlphaNet on MobileNetV2 with the last layer slimmed, CIFAR-100, seed 1995

A, with:

* `alpha_iw_clip: 5.0` - importance weight clip in the alpha-divergence
* `alpha_max: 1.0` - upper end of the adaptive alpha range
* `alpha_min: -1.0` - lower end of the adaptive alpha range
* `kd_loss: alpha` - the vertical KD term becomes an adaptive alpha-divergence, AlphaNet as published
* `model: models.us_mobilenet_v2_cifar`
* `slim_last: True`

**GP** - AlphaNet on MobileNetV2 with the last layer slimmed, CIFAR-100, seed 2006

A, with:

* `alpha_iw_clip: 5.0` - importance weight clip in the alpha-divergence
* `alpha_max: 1.0` - upper end of the adaptive alpha range
* `alpha_min: -1.0` - lower end of the adaptive alpha range
* `kd_loss: alpha` - the vertical KD term becomes an adaptive alpha-divergence, AlphaNet as published
* `model: models.us_mobilenet_v2_cifar`
* `random_seed: 2006` - the seed, for a repeat of a branch already run
* `slim_last: True`

**B** - adaptive alpha-divergence, AlphaNet, bit-for-bit against the reference implementation

A, with:

* `alpha_iw_clip: 5.0` - importance weight clip in the alpha-divergence
* `alpha_max: 1.0` - upper end of the adaptive alpha range
* `alpha_min: -1.0` - lower end of the adaptive alpha range
* `kd_loss: alpha` - the vertical KD term becomes an adaptive alpha-divergence, AlphaNet as published

**AU** - K with KD weighted per sample by the teacher's entropy

K, with:

* `kd_weighting: entropy` - each sample's KD loss is scaled by the teacher's entropy on it, normalised to mean one

**U** - the same on 512 projections, four times the cost of 128

K, with:

* `feature_loss: sliced` - transport along random one dimensional projections instead of a full plan
* `feature_weight: 2.7` - the weight on the feature terms, set by matching gradient norms rather than loss values
* `sliced_projections: 512` - how many random directions stand in for the full plan

**GO** - AlphaNet on MobileNetV2 with the last layer slimmed, CIFAR-100, seed 2026

A, with:

* `alpha_iw_clip: 5.0` - importance weight clip in the alpha-divergence
* `alpha_max: 1.0` - upper end of the adaptive alpha range
* `alpha_min: -1.0` - lower end of the adaptive alpha range
* `kd_loss: alpha` - the vertical KD term becomes an adaptive alpha-divergence, AlphaNet as published
* `model: models.us_mobilenet_v2_cifar`
* `random_seed: 2026` - the seed, for a repeat of a branch already run
* `slim_last: True`

**ET** - EP with the transport reading the last slimmed layer (feature_read last_slimmed, the 320 x width linear bottleneck), notebook 58. The log reads 1% at every width: the 1x1 to 1280 is unslimmed and its BN is a plain BatchNorm2d with one set of running statistics for all widths. Nothing here holds the widths' 1280 features together, and they ended nearly constant within a width (channel sd 0.006-0.02) with means 0.1 apart across widths, so statistics averaged over the sixteen widths are many sd off for each; train.py calibrates every width on one model. Recalibrated one width at a time from the checkpoint, locally: 72.94, below EP (73.68), ER (74.02) and EU (74.06), though that protocol only favours it. Not a train.py result: top1 from the notebook 58 checkpoint, each width calibrated alone on 20 train batches, no NLL.

K, with:

* `feature_read: last_slimmed`
* `feature_start_epoch: 6`
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `model: models.us_mobilenet_v2_cifar`
* `split_pair_backward: True`

**BU** - K, width 0.25 alone for the first 25 epochs, then the full sandwich

K, with:

* `narrow_first_epochs: 25`

**C** - Wasserstein on the logits, vertical only

I, with:

* `kd_loss: wasserstein` - the vertical KD term becomes entropic transport over a class cost matrix, instead of soft cross entropy
* `feature_align` not set, against `prefix` in I
* `feature_kd` not set, against `True` in I
* `feature_weight` not set, against `1.0` in I

**L** - D with the horizontal term weighted toward the narrow widths

C, with:

* `horizontal_kd: True` - adds a term between two co-sampled widths that stand in no teacher relation to each other
* `horizontal_weight: 1.0` - the weight on the horizontal term
* `weight_schedule: narrow` - the extra term is faded out toward the wide widths, where it was measured to hurt

**DI** - US-Net's inplace KL kept and WKD-L's non-target transport (weight 600) added beside it, no student CE, ResNet-50, seed 1995: the full width sat at chance until epoch 12 and recovered only as the weight decayed

A, with:

* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `kd_loss: kl_wkd`
* `wkd_decay_from: 0.625`
* `wkd_iters: 10`
* `wkd_reg: 0.05`
* `wkd_temperature: 8.0`
* `wkd_weight: 600.0`

**GS** - DYNAS on MobileNetV2 with the last layer slimmed, CIFAR-100, seed 2006

A, with:

* `dynas: True`
* `dynas_groups: 4`
* `dynas_max_coeff: 4.0`
* `model: models.us_mobilenet_v2_cifar`
* `random_seed: 2006` - the seed, for a repeat of a branch already run
* `slim_last: True`

**GR** - DYNAS on MobileNetV2 with the last layer slimmed, CIFAR-100, seed 2026

A, with:

* `dynas: True`
* `dynas_groups: 4`
* `dynas_max_coeff: 4.0`
* `model: models.us_mobilenet_v2_cifar`
* `random_seed: 2026` - the seed, for a repeat of a branch already run
* `slim_last: True`

**GQ** - DYNAS on MobileNetV2 with the last layer slimmed, CIFAR-100, seed 1995

A, with:

* `dynas: True`
* `dynas_groups: 4`
* `dynas_max_coeff: 4.0`
* `model: models.us_mobilenet_v2_cifar`
* `slim_last: True`

**AG** - K with five sampled widths and every pair coupled: the narrowest never trained

K, with:

* `horizontal_pairs: all` - every pair among the co-sampled students is coupled, not just the two middle ones
* `num_sample_training: 5` - how many widths are run per step, the sandwich rule's two ends included

**BT** - INVALID: BU plus a freeze that did not hold - weight decay and momentum shrank the frozen block to zero, width 0.25 ended at chance. Not a measurement of freezing

K, with:

* `freeze_prefix_at: 0.25`
* `narrow_first_epochs: 25`

**DF** - K on ResNet-50 exactly as BY with amp: True: collapsed at seed 1995, where BY in fp32 did not; widths up to 0.55 near 50% error

K, with:

* `amp: True`
* `depth: 50` - ResNet-50: bottleneck blocks [3, 4, 6, 3], 23.7M parameters and 2.34 times the MACs of ResNet-18 at every width
* `split_pair_backward: True`

**EQ** - SlimOT + WBH (DO recipe) on ResNet-18, Tiny ImageNet, seed 1995, notebook 57: the second dataset. 59.27: +1.14 against ES, above at 15/16 widths (0.25 level, -0.05; 0.40-0.60 +1.5 to +2.0; 0.80-1.0 +1.2 to +1.5), beside DO against BX on CIFAR-100 (+0.98). One seed. 630 min on a T4, one session.

K, with:

* `data_loader: data.tinyimagenet`
* `data_transforms: data.tinyimagenet`
* `dataset: data.tinyimagenet`
* `depth: 18`
* `feature_start_epoch: 6`
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `image_size: 64`
* `num_classes: 200`
* `split_pair_backward: True`
* `stem_stride: 2`

**GE** - AlphaNet on ResNet-18, Tiny ImageNet, seed 1995

A, with:

* `alpha_iw_clip: 5.0` - importance weight clip in the alpha-divergence
* `alpha_max: 1.0` - upper end of the adaptive alpha range
* `alpha_min: -1.0` - lower end of the adaptive alpha range
* `data_loader: data.tinyimagenet`
* `data_transforms: data.tinyimagenet`
* `dataset: data.tinyimagenet`
* `depth: 18`
* `image_size: 64`
* `kd_loss: alpha` - the vertical KD term becomes an adaptive alpha-divergence, AlphaNet as published
* `num_classes: 200`
* `stem_stride: 2`

**GF** - AlphaNet on ResNet-18, Tiny ImageNet, seed 2026

A, with:

* `alpha_iw_clip: 5.0` - importance weight clip in the alpha-divergence
* `alpha_max: 1.0` - upper end of the adaptive alpha range
* `alpha_min: -1.0` - lower end of the adaptive alpha range
* `data_loader: data.tinyimagenet`
* `data_transforms: data.tinyimagenet`
* `dataset: data.tinyimagenet`
* `depth: 18`
* `image_size: 64`
* `kd_loss: alpha` - the vertical KD term becomes an adaptive alpha-divergence, AlphaNet as published
* `num_classes: 200`
* `random_seed: 2026` - the seed, for a repeat of a branch already run
* `stem_stride: 2`

**GG** - AlphaNet on ResNet-18, Tiny ImageNet, seed 2006

A, with:

* `alpha_iw_clip: 5.0` - importance weight clip in the alpha-divergence
* `alpha_max: 1.0` - upper end of the adaptive alpha range
* `alpha_min: -1.0` - lower end of the adaptive alpha range
* `data_loader: data.tinyimagenet`
* `data_transforms: data.tinyimagenet`
* `dataset: data.tinyimagenet`
* `depth: 18`
* `image_size: 64`
* `kd_loss: alpha` - the vertical KD term becomes an adaptive alpha-divergence, AlphaNet as published
* `num_classes: 200`
* `random_seed: 2006` - the seed, for a repeat of a branch already run
* `stem_stride: 2`

**GM** - SOLAR† (US-Net + four band heads) on ResNet-18, Tiny ImageNet, seed 2006

A, with:

* `data_loader: data.tinyimagenet`
* `data_transforms: data.tinyimagenet`
* `dataset: data.tinyimagenet`
* `depth: 18`
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `image_size: 64`
* `num_classes: 200`
* `random_seed: 2006` - the seed, for a repeat of a branch already run
* `stem_stride: 2`

**GK** - SOLAR (one head per band of widths, four bands) on ResNet-18, Tiny ImageNet, seed 1995

A, with:

* `data_loader: data.tinyimagenet`
* `data_transforms: data.tinyimagenet`
* `dataset: data.tinyimagenet`
* `depth: 18`
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `image_size: 64`
* `num_classes: 200`
* `stem_stride: 2`

**GL** - SOLAR (one head per band of widths, four bands) on ResNet-18, Tiny ImageNet, seed 2026

A, with:

* `data_loader: data.tinyimagenet`
* `data_transforms: data.tinyimagenet`
* `dataset: data.tinyimagenet`
* `depth: 18`
* `head_groups: 4` - SOLAR, made continuous: the width range is cut into this many equal bands and each band has its own classifier over the shared backbone
* `image_size: 64`
* `num_classes: 200`
* `random_seed: 2026` - the seed, for a repeat of a branch already run
* `stem_stride: 2`

**ES** - US-Net as published (BX recipe) on ResNet-18, Tiny ImageNet (200 classes, 64x64, official val split as test), stem_stride 2, seed 1995, notebook 57: the baseline EQ is read against. 58.13. Width 1.0 ends at 0.15% train error against 39.5% on val.

A, with:

* `data_loader: data.tinyimagenet`
* `data_transforms: data.tinyimagenet`
* `dataset: data.tinyimagenet`
* `depth: 18`
* `image_size: 64`
* `num_classes: 200`
* `stem_stride: 2`

**GI** - DYNAS on ResNet-18, Tiny ImageNet, seed 2026

A, with:

* `data_loader: data.tinyimagenet`
* `data_transforms: data.tinyimagenet`
* `dataset: data.tinyimagenet`
* `depth: 18`
* `dynas: True`
* `dynas_groups: 4`
* `dynas_max_coeff: 4.0`
* `image_size: 64`
* `num_classes: 200`
* `random_seed: 2026` - the seed, for a repeat of a branch already run
* `stem_stride: 2`

**GH** - DYNAS on ResNet-18, Tiny ImageNet, seed 1995

A, with:

* `data_loader: data.tinyimagenet`
* `data_transforms: data.tinyimagenet`
* `dataset: data.tinyimagenet`
* `depth: 18`
* `dynas: True`
* `dynas_groups: 4`
* `dynas_max_coeff: 4.0`
* `image_size: 64`
* `num_classes: 200`
* `stem_stride: 2`

**GJ** - DYNAS on ResNet-18, Tiny ImageNet, seed 2006

A, with:

* `data_loader: data.tinyimagenet`
* `data_transforms: data.tinyimagenet`
* `dataset: data.tinyimagenet`
* `depth: 18`
* `dynas: True`
* `dynas_groups: 4`
* `dynas_max_coeff: 4.0`
* `image_size: 64`
* `num_classes: 200`
* `random_seed: 2006` - the seed, for a repeat of a branch already run
* `stem_stride: 2`

**AJ** - K with per channel transport at an absolute gap: the narrow half of the range collapsed

K, with:

* `feature_loss: channel` - exact transport along each shared channel, one dimension at a time, which the nesting makes meaningful
* `feature_weight: 0.05` - the weight on the feature terms, set by matching gradient norms rather than loss values
* `wasserstein_p: 1.0` - an absolute ground cost instead of a squared one, so the gradient does not decay as the widths converge

**AV** - the same weighting reversed: collapsed to chance at every width below 0.70

K, with:

* `kd_weighting: confidence` - the same scaling, reversed: strongest where the teacher is most certain

**AN** - K with sliced transport at an absolute gap: collapsed below width 0.80

K, with:

* `feature_loss: sliced` - transport along random one dimensional projections instead of a full plan
* `feature_weight: 0.04` - the weight on the feature terms, set by matching gradient norms rather than loss values
* `wasserstein_p: 1.0` - an absolute ground cost instead of a squared one, so the gradient does not decay as the widths converge

## Which channels the prefix holds

US-Net slices channels as `weight[:k]`, so which channels a narrow width gets is decided by initialisation and never revisited. Sorting them by importance, the way Once-for-All does for elastic width, is the obvious thing to try, and it would have cost a session.

`scripts/channel_order.py` reads a checkpoint and compares the importance mass the first k channels hold against the mass the best k would hold. The control is the same checkpoint with each layer's output channels shuffled, so the shapes and the values are identical and only the order is destroyed. Read on K, at width 0.25:

| criterion | shuffled | as trained | best possible | recovered |
|---|---|---|---|---|
| L1 of the conv filter | 0.247 | 0.468 | 0.472 | 99% |
| L2 of the conv filter | 0.247 | 0.520 | 0.522 | 99% |
| absolute batch-norm gain | 0.248 | 0.303 | 0.357 | 51% |
| read by the next layer | 0.247 | 0.474 | 0.475 | 100% |
| distance from the other filters | 0.249 | 0.388 | 0.389 | 99% |
| Fisher at width 1.00 | 0.250 | 0.798 | 0.835 | 94% |
| Taylor at width 1.00 | 0.259 | 0.603 | 0.626 | 94% |
| Taylor at width 0.50 | 0.248 | 0.753 | 0.770 | 97% |

Seven of the eight say the sorting is done. The one that does not is the batch-norm gain, and it is also the one that agrees least with the others: ranked against the Taylor score, every other criterion correlates between 0.89 and 0.95 and the gain manages 0.603, falling to 0.127 in one layer. Being both the outlier in the answer and the outlier in the ordering is what a bad measure looks like, not a revealing one.

The L1 of a conv filter is not durable in principle: scale a filter by c, divide the gain of that channel by c, and the function is unchanged, because the normalisation removes the scale. For one commit this page treated that as a reason to disbelieve the L1 row and believe the gain. It is not. Training does not wander through that freedom, because `train.py:267` decays the conv weights and `train.py:271` leaves every one-dimensional parameter alone, so the conv scale is pinned and the gain is the one left to drift. Which is why L1 tracks Taylor at 0.892 here and the gain does not, and why Once-for-All can sort by L1 and get a usable order out of it.

There is a second reason, and it is in this repository rather than in the measurement. Ranking channels by the batch-norm gain is Network Slimming, and that method trains with an L1 penalty on the gains, which is what drives them apart and makes the small ones mean something. `train.py:271` gives every one-dimensional parameter a weight decay of zero, so the gains here are trained under no penalty at all. The criterion is being read outside the regime it was built for.

The row above it says the rest. The gain records how far a channel is turned up and nothing about whether anything downstream reads it; how much the next layer reads each channel is a separate quantity, and that one is sorted to 100%. A channel can be quiet and still matter. The Taylor score multiplies the gain by the gradient reaching it and so carries both halves, which is why it lands with the majority.

One of the eight is not asking about importance at all. Distance from the other filters is the FPGM criterion, which ranks a channel by how much of a duplicate it is, and the sandwich rule presses directly for a prefix that is good and only indirectly for one that is varied. It was the likeliest place to find headroom and it reads 99%.

The widths agree on the order too, which was the objection that would have closed the idea for every criterion at once. Rank correlation between the Taylor orderings is 0.905 between widths 1.00 and 0.50, 0.882 between 1.00 and 0.25, and 0.981 between 0.50 and 0.25, with no layer below 0.74. That is three of the sixteen widths, not all of them, though the three include both ends of the range and the 1.00 against 0.25 pair is the one with the most room to disagree.

Every row above is read on a finished checkpoint, which cannot say when the prefix got that way. Six branches print the same overlap at the top of every epoch and hold the narrow end back for the first 10 or 25, so between them they measure it. Under L1, at the last warm-up epoch, the epoch after, and the end:

| | last warm-up epoch | the epoch after | 99 |
|---|---|---|---|
| BF, warm 10, nothing permuted | 0.238 | 0.744 | 0.987 |
| BH, warm 10, permuted by taylor | 0.231 | 0.844 | 0.991 |
| BD, warm 10, permuted by L1 | 0.231 | 0.992 | 1.000 |
| BG, warm 25, nothing permuted | 0.211 | 0.731 | 0.971 |
| BI, warm 25, permuted by taylor | 0.231 | 0.830 | 0.994 |
| BE, warm 25, permuted by L1 | 0.244 | 0.994 | 1.000 |

The two L1 rows are partly tautological and are here for one reason. A branch sorted by L1 and read by L1 must read about 1.000 the moment after it is sorted, so that column measures nothing about training. What it does fix is the size of the lever: the permutation can place 99 per cent of the best quarter in the prefix against the 74 per cent the network reaches unaided, and the accuracy below says what that bought.

Two things, and the second closes the idea.

Twenty five epochs at full width leave the ordering where initialisation put it. It does not drift up and it does not drift down: all 105 readings taken during a warm-up, across six branches, sit between 0.195 and 0.262, against a floor of 0.25 at this fraction. So the 96 to 98 per cent a finished K reads is not what training does to a network, it is what the narrow widths do to it.

And they do it in one epoch, without being asked. BF and BG permute nothing, and the epoch after the narrow widths arrive they read 0.744 and 0.731. The sandwich rule sorts its own prefix immediately: weight[:k] takes gradient at every width and has to classify alone at 0.25, and one epoch of that is enough. A permutation at that moment buys 0.10 of overlap, which is one epoch of head start, and by the end the permuted and unpermuted runs are both at 0.97 to 0.99.

Which is why sorting is worth nothing here, and the accuracy agrees. Each permutation has its own control at the same warm-up, which makes four comparisons. Taylor is -0.19 at a warm-up of 10 and +0.17 at 25, opposite signs averaging -0.01. L1 is +0.06 and +0.20, averaging +0.13. Every one of the four is inside the noise, and the L1 pair is the informative half: it drives the overlap from 0.23 to 0.99, four times the head start the taylor permutation gives and a near-perfect prefix by construction, and buys a tenth of a point. The quantity the method is built to maximise can be maximised, and the accuracy does not follow. This is also the difference between US-Net and Once-for-All that the idea came from. OFA makes width elastic last, so its weights mature while no position is asked for anything the others are not, and the ordering stays arbitrary until something sorts it. US-Net asks from step one, so there is nothing left to sort.

Nothing here is sorted, and saying the sandwich rule sorts the channels gets the mechanism backwards. The index mapping is fixed before the first step and never moves: width 0.25 is channels 0 to 127 at the start and at the end. What changes is what those positions learn. A channel in the prefix runs at every sampled width, and at 0.25 the prefix has to classify with nothing else, while a channel at index 400 is only ever alive above width 0.78 and is never asked to stand alone. The prefix does not collect the important channels. It grows them.

Which is why the idea inverts here and not at Once-for-All. There, width is the last axis to be made elastic, so until that stage every channel lives at every step and no position is asked for anything the others are not. The ordering really is arbitrary, the shuffled floor of 25 per cent is what it looks like, and a permutation has to supply what training did not. Here there is nothing to move, and moving anyway would take a channel that grew up in the tail, trained only at wide widths, and ask it to work inside a subnet it has never run in.

Six per cent of the available ordering is left at width 1.00 and three at 0.50, and no criterion tried here finds more. This section said closed, then open, then closed again, and the middle reading rested on the single measure that the other seven contradict.

## What the epoch budget decides

Every other number on this page was read at 100 epochs. BR and BS are A and K at 300, the only pair measured at a second budget, and they do not agree with the rest of the table about what K is.

| | top-1 at 100 | at 300 | change | NLL at 100 | at 300 | minutes |
|---|---|---|---|---|---|---|
| A | 73.52 | 74.13 | **+0.61** | 1.361 | 1.518 | 165 to 495 |
| K | 74.26 | 73.91 | **-0.35** | 1.127 | 1.178 | 299 to 849 |

On top-1, K is ahead of A at 16/16 widths at 100 epochs and at 4/16 at 300, and the mean gap goes from +0.75 to -0.22. That much inverts when the schedule is tripled.

On NLL it does not invert, and reading this pair on accuracy alone is what makes K look beaten. K is better calibrated than A at 16/16 widths at 100 epochs and at 16/16 at 300, and the margin **widens** with the longer schedule, -0.234 to -0.340. So at 300 epochs K does not lose to A, it trades: 0.22 of top-1 for 0.340 of NLL. Given that accuracy here resolves to 0.10 and NLL to 0.001, that is not an obviously good trade for A.

What A buys the extra accuracy with is visible in the same column. A at 300 epochs reads 1.518 of NLL against 1.361 at 100: its calibration gets substantially worse while its accuracy improves, which is the signature of a network that has memorised the training set - train error at the widest width ends at 0.02 per cent - and is now confidently wrong on what it misses. The transport term is not a classification term, so it constrains how far the logits can saturate, and that is the most plausible reason K resists this while A does not.

The shape is what makes it mechanical rather than a bad seed. What the extra 200 epochs bought each branch, by width:

| width | A | K |
|---|---|---|
| 0.25 | +0.77 | +0.39 |
| 0.40 | +0.81 | +0.47 |
| 0.55 | +1.02 | -0.49 |
| 0.70 | +0.57 | -0.69 |
| 0.85 | +0.19 | -0.92 |
| 1.00 | +0.40 | -1.13 |

A gains at fifteen of sixteen widths. K gains below 0.50 and loses above it, monotonically, all the way to -1.13 at the widest width, where it drops from 76.00 to 74.87. Training the same branch three times longer costs it more than a point exactly where the network has the most capacity.

The training log says why, and it is visible without another run. The transport term does not decay. Over the last fifty-three epochs of BS the logged `pair_loss` stays between 0.135 and 0.162 while both task losses collapse: cross-entropy at the widest width falls 0.0111 to 0.0026, and at the narrowest 0.2000 to 0.0574. The share of the objective carried by the transport term therefore goes from 42 per cent to 70 per cent without the term itself changing. Late in a long schedule K is mostly matching features between two widths with almost no classification signal left to hold it in place, and the widest width, which carries no transport term of its own and only pays for the others, is where that costs the most.

Three things this does not say. It does not retract K at 100 epochs: 74.26 against 73.52, ahead at all sixteen widths, stands as measured. It does not say longer is better in general, because neither 300-epoch run is the best branch at a single one of the sixteen widths, and adding both leaves the per-width envelope unchanged. And it does not say the loss work was wasted compute, because that comparison runs the other way, and on both metrics at once. K at 100 epochs beats A at 300 on top-1, 74.26 against 74.13; on NLL, 1.127 against 1.518, better at all sixteen widths; and on wall-clock, 299 minutes against 495, or 60 per cent. There is no axis on which A at 300 epochs wins that comparison, so the transport term buys more than tripling the budget of the baseline does. What it does not do is compound: K at 300 costs 849 minutes to fall back to 73.91 on top-1, though it keeps its advantage on NLL.

One caveat on that wall-clock reading, because it is the most favourable sentence in this section. A was never run at 200 epochs, so there is no measurement of A at the 299-minute budget K used; the comparison above is against A at 300, which spent more. That direction is the honest one, but the matched point is missing and it is a cheap run.

The repair is already in the repository and has never been pointed at K. `width_gate` in `utils/loss_ops.py` implements `weight_schedule: narrow`, which fades an extra term out toward the wide widths, and its docstring was written to predict exactly this shape: that every intervention here helps the narrow end and costs something at the wide one. The flag appears in two configs, L and N, both at 100 epochs, where the cost at the wide end had not yet grown large enough to see. K at 300 with the term scheduled off above the middle is the run this section asks for.

## The curriculum axis, both directions

Six branches train width 1.00 alone first and then admit the narrow end; they sit between -0.55 and -0.28 against K, mean -0.38. BU is the other direction, width 0.25 alone for the first 25 epochs, and it reads 72.81: -1.45 against K, -0.71 against A, and behind K at 0 of the sixteen widths. It does not even help the width it trained first, -0.72 against K at 0.25, and it costs most at the wide end. GrowTAS predicted the opposite; the weight-sharing literature, which says the narrow end is not the part that needs protecting, predicted this. Both directions of the curriculum lose, the narrow-first one by about three times as much, so the axis is closed.

BT is in the table and is not a result. It was meant to be BU with the width-0.25 block held still after epoch 25, and the freeze only zeroed the gradient. SGD adds weight decay to the gradient inside the step and momentum carries it, so the block shrank toward zero for 75 epochs: width 0.25 ended at chance, 1.18 per cent with loss ln(100), and every wider width lost about four points because the prefix they all read was being erased. The check that shipped with it measured the gradient, not the weights. The freeze now writes the block back after the step, and tests/test_freeze.py checks the weights are bit-identical after three real steps, having first shown they move without it. BT has not been rerun; with BU already -1.45 against K, a freeze on top of it is not a good use of a session.

## Not settled

- Whether F is ahead of A at all. The accuracy gap is at the rounding floor; only the NLL gap is outside it.
- What F is doing, exactly. It has symmetry and it has a coupling between the two middle widths. E, the same term as plain KL, is what separates those.
- Sigma, properly. Two seeds put it under the floor; three would make it a number worth quoting. It has stopped being a precaution: the four branches below decide their own reading on it.
- Why AV collapsed. Everything at width 0.70 and above trained; everything at 0.65 and below sits at exactly chance, 1.00 accuracy and NLL ln(100). A cliff, not a slope, which is the shape AJ and AN already showed. What has been measured is that the weight itself does not explode: across teacher sharpness from uniform to memorised the per-sample weight stays between 0 and about 4 with mean 1, so a blown-up KD term is ruled out rather than merely unlikely. What has not been found is the path from that weighting to dead leading channels. It is undiagnosed, not explained.
- One defect the probe did find, which is not yet shown to be the cause. AV measures confidence as `entropy.max() - entropy`, taking its zero point from a batch order statistic that itself drifts to zero as the teacher memorises. When every entropy in a batch underflows to exactly zero both AU and AV divide zero by zero and hand KD a weight of exactly zero for every sample. The fixed reference the quantity actually has, log(C) minus entropy, survives that case and does not move with the batch. A rerun of this axis should use it.
- Whether AW is ahead of K, and the table should not be read as saying it is. AW is first at 74.32 against 74.26, a gap of 0.06 where a difference of means over sixteen widths carries 0.01 to 0.03 of rounding alone. The number that settles it is not the mean: AW is ahead of K at eight widths out of sixteen, behind at eight, scattered from -0.26 to +0.71. A real gain does not look like that. K against A is ahead at all sixteen, which is what one does look like - at 100 epochs, and the budget section above is where that stops being true. On this evidence AW ties K and the ordering between them is a coin.
- What AW is still worth, and a correction to what this page said about it. AW is not a second route to K. Diff the two configs and they differ by one line, `teacher_chain: True`, and that flag only changes which soft target the logits are matched against; the transport term runs inside AW untouched, with train.py restoring the width order so the pairing is the same one K uses. The ladder is additive: A, plus transport, is K at +0.75 and ahead at all sixteen widths; K, plus chaining, is AW at +0.06. So chaining on top of transport buys nothing in accuracy. What it does move is NLL, 1.127 to 1.091, the largest such gain of anything stacked on K, and the mechanism is plain enough: a target from the adjacent width is closer than one from the widest, so the student is less over-confident. Better calibration, not better accuracy.
- Whether chaining does anything on its own, which has never been measured. `teacher_chain` appears in exactly two configs and both are AW, so every reading of it sits on top of the transport term. That leaves one cell of a two by two empty: widest-teacher and chained-teacher, crossed with transport and no transport, has A at 73.52, K at 74.26, AW at 74.32 and nothing at all for chaining alone. BJ is that cell - A with teacher_chain and no transport anywhere. Near 73.5 and the transport term is the only active ingredient. Near 74.3 and the two are redundant rather than additive, which changes what this report claims.
- That K is saturated, which is now four branches deep. Every addition to K has cost accuracy except the one that did nothing: AE at -0.84, AH at -0.73, AF at -0.23, AW at +0.06. The next idea worth running is unlikely to be another term added to this loss.
- Whether K sits on a peak or on a high draw. AS, AT, AC and AD are K with one knob moved in four different directions - the free widths drawn narrower, the free widths drawn wider, the feature weight halved, the feature weight doubled - and all four land between 73.82 and 73.96, which is 0.30 to 0.44 below K. Four perturbations that share nothing mechanically do not usually agree by accident, so either 1.0 and a uniform draw are both genuinely best, or the 74.26 for K is a high seed and the branch is really worth about 73.9. Both readings fit this table. Only sigma separates them, and it is the same measurement the bullet above asks for.

- Whether K sits on a peak, and the shape of that question has changed. Fourteen branches are now K with one knob moved in directions that share nothing mechanically: the free widths drawn narrower and wider, the feature weight halved and doubled, six warm-up variants, and the four that touched the optimizer, the teacher and the architecture. They run from 73.63 to 74.35 with a mean of 73.93.
- What changed with BP is that K is no longer clear of them. Ten perturbations used to sit entirely below K, which was the strongest argument in this report that 74.26 was a high draw. BP reads 74.35 and AW 74.32, so the top of the table is three branches inside 0.09 of each other with nothing to separate them. That is not evidence K was lucky and it is not evidence it was not; it is the same unmeasured sigma, now sitting between the first three rows instead of under one of them. A second seed of K remains the single most valuable run on the list, and it is now more valuable than before, because three branches depend on it rather than one.
- That the ranking at the top is currently unreadable. BP, AW, K and X span 0.11 across four branches whose mechanisms have nothing to do with each other - a per-width residual scalar, a chained teacher, a transport term, and a blur setting. Four unrelated mechanisms landing inside a tenth of a point is what a saturated axis looks like, and reporting an order among them would be reporting noise.
