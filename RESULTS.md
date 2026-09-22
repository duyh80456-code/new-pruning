# Results

Universally slimmable ResNet-18 on CIFAR-100, sixteen widths from 0.25 to 1.00, accuracy after BN post-statistics. One seed each unless the row says otherwise. Generated from `results/runs.json` by `scripts/collect_results.py`.

## Where it stands

| | branch | mean top-1 | vs A | worst width | mean NLL | vs A |
|---|---|---|---|---|---|---|
| **AW** | K with each width taught by the next larger one instead of by the widest | **74.32** | +0.80 | 71.37 | 1.091 | -0.270 |
| **K** | I plus transport between the two middle widths, at the feature tier | **74.26** | +0.75 | 70.84 | 1.127 | -0.234 |
| **X** | K at eps 0.1 instead of 0.2 | **74.24** | +0.72 | 71.38 | 1.170 | -0.191 |
| **V** | K with transport that may leave mass unmatched, at tau 1.0 | **74.18** | +0.66 | 71.11 | 1.083 | -0.278 |
| **AO** | K plus a repulsion pushing the classifier rows apart | **74.14** | +0.62 | 71.00 | 1.135 | -0.226 |
| **AQ** | K transporting the two widths' class means rather than their samples | **74.10** | +0.58 | 70.79 | 1.187 | -0.173 |
| **AF** | K coupling all three student pairs instead of one | **74.03** | +0.51 | 70.95 | 1.127 | -0.234 |
| **AT** | K with the same free widths drawn flat in compute instead | **73.96** | +0.44 | 70.71 | 1.113 | -0.248 |
| **AC** | K at half the weight on the feature terms | **73.95** | +0.43 | 70.37 | 1.109 | -0.252 |
| **AS** | K with the sandwich rule's free widths drawn flat in log width | **73.93** | +0.41 | 70.94 | 1.120 | -0.241 |
| **AD** | K at twice the weight, the other side of the same question | **73.82** | +0.31 | 70.80 | 1.152 | -0.209 |
| **AA** | K charging transport by direction rather than distance | **73.75** | +0.23 | 70.88 | 1.266 | -0.095 |
| **W** | K at eps 0.02, a plan collapsed onto a permutation | **73.74** | +0.23 | 70.79 | 1.230 | -0.131 |
| **I** | A plus transport between the student and teacher feature clouds | **73.66** | +0.14 | 70.78 | 1.311 | -0.050 |
| **AM** | K with sliced transport keeping the worst direction rather than the average | **73.64** | +0.12 | 70.68 | 1.337 | -0.024 |
| **AK** | K with the closed-form Gaussian transport, mean gap plus a covariance term | **73.56** | +0.04 | 70.22 | 1.325 | -0.036 |
| **F** | A plus a symmetrized KL between the two middle widths | **73.54** | +0.03 | 70.30 | 1.255 | -0.106 |
| **A'** | the same branch again, to measure the noise | **73.53** | +0.01 | 69.80 | 1.377 | +0.016 |
| **M** | K with transport at every stage, not the last one alone | **73.53** | +0.01 | 70.19 | 1.203 | -0.158 |
| **AH** | K, F and all three pairs at once, every addition together | **73.53** | +0.01 | 71.45 | 1.135 | -0.225 |
| **AL** | K with Gaussian transport on the per channel variances alone | **73.52** | +0.00 | 69.60 | 1.393 | +0.032 |
| **A** | US-Net as published: inplace distillation with KL | **73.52** | +0.00 | 70.10 | 1.361 | +0.000 |
| **AE** | K and F at once, transport on features and Jeffreys on logits | **73.42** | -0.10 | 70.53 | 1.175 | -0.186 |
| **AR** | K with the logit teacher softened to temperature 4 | **73.38** | -0.14 | 70.80 | 1.522 | +0.161 |
| **Y** | K at eps 0.5, mass spread over many partners | **73.28** | -0.24 | 70.13 | 1.354 | -0.007 |
| **D** | C plus Wasserstein between the two middle widths | **73.27** | -0.25 | 70.30 | 1.361 | +0.000 |
| **E** | A plus plain KL between the two middle widths, neither symmetric nor metric aware | **73.27** | -0.25 | 69.80 | 1.276 | -0.085 |
| **G** | C with every pair of classes equally far apart, so transport has no geometry to use | **73.25** | -0.27 | 69.50 | 1.379 | +0.019 |
| **AI** | K transporting one shared channel at a time, exactly rather than by projection | **73.24** | -0.28 | 69.63 | 1.393 | +0.032 |
| **S** | K with sliced transport on 128 projections | **73.22** | -0.30 | 70.16 | 1.399 | +0.039 |
| **T** | K with sliced transport on 32 random projections rather than the entropic plan | **73.19** | -0.33 | 70.34 | 1.377 | +0.016 |
| **H** | C with the class metric taken from what the teacher confuses, rather than from the classifier rows | **73.16** | -0.36 | 69.90 | 1.343 | -0.018 |
| **B** | adaptive alpha-divergence, AlphaNet, bit-for-bit against the reference implementation | **73.08** | -0.44 | 70.40 | 1.573 | +0.212 |
| **AU** | K with KD weighted per sample by the teacher's entropy | **73.00** | -0.52 | 67.76 | 1.063 | -0.298 |
| **U** | the same on 512 projections, four times the cost of 128 | **72.99** | -0.53 | 69.87 | 1.409 | +0.049 |
| **C** | Wasserstein on the logits, vertical only | **72.74** | -0.77 | 69.40 | 1.415 | +0.054 |
| **L** | D with the horizontal term weighted toward the narrow widths | **72.74** | -0.78 | 70.00 | 1.438 | +0.077 |
| **AG** | K with five sampled widths and every pair coupled: the narrowest never trained | **67.70** | -5.82 | 1.00 | 1.446 | +0.086 |
| **AJ** | K with per channel transport at an absolute gap: the narrow half of the range collapsed | **37.47** | -36.05 | 1.00 | 2.951 | +1.590 |
| **AV** | the same weighting reversed: collapsed to chance at every width below 0.70 | **30.46** | -43.06 | 1.00 | 3.230 | +1.869 |
| **AN** | K with sliced transport at an absolute gap: collapsed below width 0.80 | **20.27** | -53.25 | 1.00 | 3.669 | +2.308 |

## What the numbers can carry

Accuracy was logged to three decimals of error, so every value is a multiple of 0.10 and each one carries up to 0.05 of rounding. A difference between two of them carries up to **0.10**, and a difference of means over sixteen widths roughly 0.01 to 0.03.

A and A' are the same branch under different seeds. Their means differ by **0.01**, which is at that floor: what the pair establishes is that seed noise on the mean sits under it, not that it equals 0.01. Per width the same pair differs by as much as **0.40**, so the worst-width column is noise and should not be ranked.

So the gaps worth reading are the ones far outside that floor: A over C by 0.77, over B by 0.44, over D by 0.25. F against A, at +0.03, is not one of them.

NLL is printed to three decimals on a scale near 1.3, so it resolves to 0.001 and none of this applies to it.

## Accuracy by width

| width | MACs (M) | AW | K | X | V | AO | AQ | AF | AT | AC | AS | AD | AA | W | I | AM | AK | F | A' | M | AH | AL | A | AE | AR | Y | D | E | G | AI | S | T | H | B | AU | U | C | L | AG | AJ | AV | AN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 35.1 | 71.37 | 70.84 | 71.38 | 71.11 | 71.00 | 70.79 | 70.95 | 70.71 | 70.37 | 70.94 | 70.80 | 70.88 | 70.79 | 70.78 | 70.68 | 70.22 | 70.30 | 69.80 | 70.19 | **71.45** | 69.60 | 70.10 | 70.53 | 70.80 | 70.13 | 70.30 | 69.80 | 69.50 | 69.63 | 70.16 | 70.34 | 69.90 | 70.40 | 67.76 | 69.87 | 69.40 | 70.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.30 | 60.5 | 71.53 | 71.73 | **71.95** | 71.74 | 71.25 | 71.44 | 71.84 | 71.19 | 71.08 | 71.72 | 71.15 | 71.67 | 71.07 | 71.29 | 70.87 | 70.82 | 71.30 | 70.70 | 71.21 | 71.78 | 70.54 | 70.80 | 71.60 | 71.67 | 70.81 | 71.30 | 70.90 | 70.60 | 70.31 | 70.75 | 71.02 | 70.60 | 71.10 | 68.96 | 70.46 | 70.40 | 70.80 | 58.76 | 1.00 | 1.00 | 1.00 |
| 0.35 | 72.7 | 72.21 | 72.36 | **72.67** | 72.33 | 71.75 | 72.05 | 72.56 | 71.73 | 71.64 | 72.07 | 71.91 | 72.23 | 71.92 | 71.58 | 71.41 | 71.51 | 72.00 | 71.50 | 71.43 | 72.09 | 71.36 | 71.50 | 72.18 | 71.63 | 71.40 | 71.60 | 71.60 | 71.20 | 70.99 | 71.09 | 71.57 | 70.80 | 71.40 | 70.00 | 71.15 | 70.90 | 70.80 | 66.74 | 1.00 | 1.00 | 1.00 |
| 0.40 | 84.8 | 73.11 | 72.69 | **73.24** | 72.92 | 72.93 | 72.76 | 72.69 | 72.52 | 72.66 | 72.91 | 72.51 | 72.45 | 72.37 | 72.22 | 72.21 | 72.54 | 72.40 | 72.60 | 72.15 | 72.56 | 72.12 | 72.20 | 72.58 | 72.21 | 72.18 | 72.20 | 72.00 | 71.90 | 71.92 | 71.46 | 71.91 | 71.50 | 71.80 | 71.07 | 71.94 | 71.30 | 71.30 | 69.34 | 1.00 | 1.00 | 1.00 |
| 0.45 | 118.0 | **74.00** | 73.29 | 73.95 | 73.46 | 73.45 | 73.59 | 73.30 | 73.24 | 73.28 | 73.31 | 73.18 | 73.14 | 72.73 | 73.09 | 72.82 | 73.01 | 72.80 | 72.70 | 72.78 | 73.15 | 72.77 | 72.80 | 72.82 | 72.66 | 72.41 | 72.60 | 72.80 | 72.40 | 72.36 | 71.85 | 72.50 | 72.40 | 72.20 | 71.71 | 72.09 | 72.20 | 72.00 | 71.37 | 1.00 | 1.00 | 1.00 |
| 0.50 | 139.3 | 74.10 | 74.00 | **74.24** | 73.88 | 73.85 | 74.08 | 73.63 | 73.76 | 73.77 | 73.63 | 73.74 | 73.64 | 73.35 | 73.58 | 73.22 | 73.49 | 73.50 | 73.10 | 73.14 | 73.36 | 73.11 | 73.20 | 73.03 | 73.19 | 73.06 | 72.80 | 73.30 | 73.20 | 73.06 | 72.75 | 72.77 | 72.70 | 72.60 | 72.74 | 72.66 | 72.30 | 72.30 | 72.45 | 1.00 | 1.00 | 1.00 |
| 0.55 | 163.2 | **74.70** | 74.60 | 74.32 | 74.04 | 74.39 | 74.38 | 73.97 | 74.16 | 73.92 | 73.85 | 74.27 | 73.75 | 73.60 | 73.92 | 73.95 | 73.45 | 73.60 | 73.40 | 73.44 | 73.63 | 73.85 | 73.40 | 73.43 | 73.62 | 73.27 | 73.30 | 73.60 | 73.40 | 73.66 | 73.09 | 73.17 | 73.20 | 72.90 | 73.43 | 72.89 | 72.60 | 72.50 | 72.99 | 1.00 | 1.00 | 1.00 |
| 0.60 | 207.6 | 74.57 | 74.81 | 74.68 | 74.74 | **75.06** | 74.76 | 74.35 | 74.61 | 74.51 | 74.40 | 74.44 | 74.12 | 74.23 | 74.10 | 73.86 | 73.89 | 73.60 | 73.80 | 74.10 | 73.61 | 74.30 | 73.80 | 73.78 | 73.92 | 73.60 | 73.70 | 73.80 | 73.40 | 73.69 | 73.64 | 73.59 | 73.50 | 73.30 | 73.72 | 73.30 | 73.00 | 73.00 | 73.52 | 1.00 | 1.00 | 1.00 |
| 0.65 | 227.7 | 74.86 | 74.83 | 74.85 | 74.76 | 74.96 | 74.71 | 74.26 | **74.97** | 74.52 | 74.50 | 74.49 | 74.58 | 74.61 | 74.30 | 74.23 | 73.86 | 74.00 | 74.20 | 74.56 | 73.99 | 74.44 | 73.90 | 74.09 | 73.73 | 73.61 | 74.00 | 73.80 | 73.80 | 74.00 | 74.17 | 73.62 | 73.80 | 73.60 | 74.05 | 73.67 | 73.40 | 73.20 | 73.84 | 66.62 | 1.00 | 1.00 |
| 0.70 | 280.2 | **75.20** | 75.14 | 74.87 | 75.04 | 75.16 | 74.68 | 74.80 | 75.09 | 74.75 | 74.61 | 74.58 | 74.58 | 74.83 | 74.31 | 74.55 | 74.29 | 74.20 | 74.50 | 74.43 | 74.08 | 74.60 | 74.30 | 74.10 | 74.09 | 73.91 | 74.10 | 73.90 | 74.10 | 74.05 | 74.20 | 74.01 | 74.00 | 73.60 | 74.36 | 73.63 | 73.30 | 73.40 | 74.26 | 73.05 | 52.72 | 1.09 |
| 0.75 | 312.8 | **75.49** | 75.23 | 75.02 | 75.08 | 75.04 | 75.15 | 75.10 | 75.00 | 74.90 | 74.73 | 74.80 | 74.77 | 74.78 | 74.64 | 74.69 | 74.60 | 74.60 | 74.60 | 74.65 | 74.21 | 74.67 | 74.60 | 74.24 | 74.23 | 74.18 | 74.10 | 74.10 | 74.30 | 74.16 | 74.51 | 73.85 | 74.20 | 74.10 | 74.77 | 73.93 | 73.60 | 73.60 | 74.44 | 74.01 | 68.06 | 1.08 |
| 0.80 | 347.9 | 75.43 | **75.44** | 75.25 | 75.31 | 75.34 | 75.21 | 74.92 | 75.06 | 75.31 | 75.06 | 75.01 | 74.55 | 74.93 | 74.81 | 74.88 | 74.84 | 74.60 | 74.70 | 74.56 | 74.34 | 74.80 | 74.80 | 74.42 | 74.45 | 74.30 | 74.40 | 74.20 | 74.50 | 74.42 | 74.71 | 74.26 | 74.30 | 74.20 | 74.87 | 74.19 | 73.90 | 73.70 | 74.57 | 74.85 | 70.14 | 39.78 |
| 0.85 | 411.6 | 75.46 | **75.72** | 75.33 | 75.27 | 75.32 | 75.44 | 75.41 | 75.27 | 75.34 | 75.10 | 74.94 | 74.78 | 74.95 | 74.83 | 74.99 | 75.01 | 74.80 | 74.70 | 74.74 | 74.44 | 74.90 | 75.10 | 74.35 | 74.38 | 74.55 | 74.30 | 74.40 | 74.70 | 74.66 | 74.50 | 74.37 | 74.60 | 74.20 | 74.92 | 74.28 | 74.10 | 73.90 | 74.86 | 75.24 | 71.08 | 64.47 |
| 0.90 | 439.8 | 75.64 | **75.73** | 75.30 | 75.49 | 75.46 | 75.33 | 75.47 | 75.28 | 75.50 | 75.36 | 74.99 | 74.88 | 75.20 | 74.86 | 75.22 | 75.13 | 74.90 | 75.20 | 74.85 | 74.47 | 74.95 | 75.10 | 74.53 | 74.58 | 74.80 | 74.60 | 74.50 | 74.80 | 74.95 | 74.77 | 74.76 | 74.80 | 74.50 | 75.19 | 74.39 | 74.20 | 74.30 | 74.92 | 75.72 | 71.94 | 68.58 |
| 0.95 | 511.6 | 75.73 | 75.82 | 75.39 | 75.84 | 75.60 | 75.63 | 75.71 | 75.37 | 75.75 | 75.39 | 75.18 | 74.92 | 75.28 | 74.96 | 75.36 | 75.06 | 74.90 | 75.40 | 75.13 | 74.47 | 75.08 | 75.40 | 74.54 | 74.60 | 75.13 | 74.50 | 74.70 | 74.90 | 74.89 | 74.80 | 74.58 | 75.00 | 74.70 | 75.17 | 74.59 | 74.60 | 74.50 | 74.97 | **75.98** | 72.08 | 69.98 |
| 1.00 | 555.5 | 75.77 | 76.00 | 75.42 | 75.83 | 75.68 | 75.60 | 75.56 | 75.36 | 75.88 | 75.25 | 75.20 | 75.05 | 75.27 | 75.21 | 75.34 | 75.27 | 75.20 | 75.60 | 75.13 | 74.84 | 75.22 | 75.30 | 74.50 | 74.38 | 75.10 | 74.50 | 74.90 | 75.30 | 75.14 | 75.01 | 74.75 | 75.30 | 74.70 | 75.33 | 74.73 | 74.70 | 74.50 | 75.11 | **76.05** | 72.39 | 70.28 |
| **mean** | 248.0 | **74.32** | 74.26 | 74.24 | 74.18 | 74.14 | 74.10 | 74.03 | 73.96 | 73.95 | 73.93 | 73.82 | 73.75 | 73.74 | 73.66 | 73.64 | 73.56 | 73.54 | 73.53 | 73.53 | 73.53 | 73.52 | 73.52 | 73.42 | 73.38 | 73.28 | 73.27 | 73.27 | 73.25 | 73.24 | 73.22 | 73.19 | 73.16 | 73.08 | 73.00 | 72.99 | 72.74 | 72.74 | 67.70 | 37.47 | 30.46 | 20.27 |

The mean row is the column each branch is ranked by, which the table above it could not be read off before.

## Against A, by width

Every branch that has moved at all has moved the same way: ahead at the narrow widths, behind at the wide ones. Five times here, and three more in the literature.

| width | AW | K | X | V | AO | AQ | AF | AT | AC | AS | AD | AA | W | I | AM | AK | F | A' | M | AH | AL | AE | AR | Y | D | E | G | AI | S | T | H | B | AU | U | C | L | AG | AJ | AV | AN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | +1.27 | +0.74 | +1.28 | +1.01 | +0.90 | +0.69 | +0.85 | +0.61 | +0.27 | +0.84 | +0.70 | +0.78 | +0.69 | +0.68 | +0.58 | +0.12 | +0.20 | -0.30 | +0.09 | +1.35 | -0.50 | +0.43 | +0.70 | · | +0.20 | -0.30 | -0.60 | -0.47 | +0.06 | +0.24 | -0.20 | +0.30 | -2.34 | -0.23 | -0.70 | -0.10 | -69.10 | -69.10 | -69.10 | -69.10 |
| 0.30 | +0.73 | +0.93 | +1.15 | +0.94 | +0.45 | +0.64 | +1.04 | +0.39 | +0.28 | +0.92 | +0.35 | +0.87 | +0.27 | +0.49 | +0.07 | · | +0.50 | -0.10 | +0.41 | +0.98 | -0.26 | +0.80 | +0.87 | · | +0.50 | +0.10 | -0.20 | -0.49 | · | +0.22 | -0.20 | +0.30 | -1.84 | -0.34 | -0.40 | · | -12.04 | -69.80 | -69.80 | -69.80 |
| 0.35 | +0.71 | +0.86 | +1.17 | +0.83 | +0.25 | +0.55 | +1.06 | +0.23 | +0.14 | +0.57 | +0.41 | +0.73 | +0.42 | +0.08 | -0.09 | · | +0.50 | · | -0.07 | +0.59 | -0.14 | +0.68 | +0.13 | -0.10 | +0.10 | +0.10 | -0.30 | -0.51 | -0.41 | +0.07 | -0.70 | -0.10 | -1.50 | -0.35 | -0.60 | -0.70 | -4.76 | -70.50 | -70.50 | -70.50 |
| 0.40 | +0.91 | +0.49 | +1.04 | +0.72 | +0.73 | +0.56 | +0.49 | +0.32 | +0.46 | +0.71 | +0.31 | +0.25 | +0.17 | · | · | +0.34 | +0.20 | +0.40 | · | +0.36 | -0.08 | +0.38 | · | · | · | -0.20 | -0.30 | -0.28 | -0.74 | -0.29 | -0.70 | -0.40 | -1.13 | -0.26 | -0.90 | -0.90 | -2.86 | -71.20 | -71.20 | -71.20 |
| 0.45 | +1.20 | +0.49 | +1.15 | +0.66 | +0.65 | +0.79 | +0.50 | +0.44 | +0.48 | +0.51 | +0.38 | +0.34 | -0.07 | +0.29 | · | +0.21 | · | -0.10 | · | +0.35 | · | · | -0.14 | -0.39 | -0.20 | · | -0.40 | -0.44 | -0.95 | -0.30 | -0.40 | -0.60 | -1.09 | -0.71 | -0.60 | -0.80 | -1.43 | -71.80 | -71.80 | -71.80 |
| 0.50 | +0.90 | +0.80 | +1.04 | +0.68 | +0.65 | +0.88 | +0.43 | +0.56 | +0.57 | +0.43 | +0.54 | +0.44 | +0.15 | +0.38 | · | +0.29 | +0.30 | -0.10 | -0.06 | +0.16 | -0.09 | -0.17 | · | -0.14 | -0.40 | +0.10 | · | -0.14 | -0.45 | -0.43 | -0.50 | -0.60 | -0.46 | -0.54 | -0.90 | -0.90 | -0.75 | -72.20 | -72.20 | -72.20 |
| 0.55 | +1.30 | +1.20 | +0.92 | +0.64 | +0.99 | +0.98 | +0.57 | +0.76 | +0.52 | +0.45 | +0.87 | +0.35 | +0.20 | +0.52 | +0.55 | · | +0.20 | · | · | +0.23 | +0.45 | · | +0.22 | -0.13 | -0.10 | +0.20 | · | +0.26 | -0.31 | -0.23 | -0.20 | -0.50 | · | -0.51 | -0.80 | -0.90 | -0.41 | -72.40 | -72.40 | -72.40 |
| 0.60 | +0.77 | +1.01 | +0.88 | +0.94 | +1.26 | +0.96 | +0.55 | +0.81 | +0.71 | +0.60 | +0.64 | +0.32 | +0.43 | +0.30 | +0.06 | +0.09 | -0.20 | · | +0.30 | -0.19 | +0.50 | · | +0.12 | -0.20 | -0.10 | · | -0.40 | -0.11 | -0.16 | -0.21 | -0.30 | -0.50 | -0.08 | -0.50 | -0.80 | -0.80 | -0.28 | -72.80 | -72.80 | -72.80 |
| 0.65 | +0.96 | +0.93 | +0.95 | +0.86 | +1.06 | +0.81 | +0.36 | +1.07 | +0.62 | +0.60 | +0.59 | +0.68 | +0.71 | +0.40 | +0.33 | · | +0.10 | +0.30 | +0.66 | +0.09 | +0.54 | +0.19 | -0.17 | -0.29 | +0.10 | -0.10 | -0.10 | +0.10 | +0.27 | -0.28 | -0.10 | -0.30 | +0.15 | -0.23 | -0.50 | -0.70 | -0.06 | -7.28 | -72.90 | -72.90 |
| 0.70 | +0.90 | +0.84 | +0.57 | +0.74 | +0.86 | +0.38 | +0.50 | +0.79 | +0.45 | +0.31 | +0.28 | +0.28 | +0.53 | · | +0.25 | · | -0.10 | +0.20 | +0.13 | -0.22 | +0.30 | -0.20 | -0.21 | -0.39 | -0.20 | -0.40 | -0.20 | -0.25 | -0.10 | -0.29 | -0.30 | -0.70 | +0.06 | -0.67 | -1.00 | -0.90 | · | -1.25 | -21.58 | -73.21 |
| 0.75 | +0.89 | +0.63 | +0.42 | +0.48 | +0.44 | +0.55 | +0.50 | +0.40 | +0.30 | +0.13 | +0.20 | +0.17 | +0.18 | · | +0.09 | · | · | · | +0.05 | -0.39 | +0.07 | -0.36 | -0.37 | -0.42 | -0.50 | -0.50 | -0.30 | -0.44 | -0.09 | -0.75 | -0.40 | -0.50 | +0.17 | -0.67 | -1.00 | -1.00 | -0.16 | -0.59 | -6.54 | -73.52 |
| 0.80 | +0.63 | +0.64 | +0.45 | +0.51 | +0.54 | +0.41 | +0.12 | +0.26 | +0.51 | +0.26 | +0.21 | -0.25 | +0.13 | · | +0.08 | · | -0.20 | -0.10 | -0.24 | -0.46 | · | -0.38 | -0.35 | -0.50 | -0.40 | -0.60 | -0.30 | -0.38 | -0.09 | -0.54 | -0.50 | -0.60 | +0.07 | -0.61 | -0.90 | -1.10 | -0.23 | · | -4.66 | -35.02 |
| 0.85 | +0.36 | +0.62 | +0.23 | +0.17 | +0.22 | +0.34 | +0.31 | +0.17 | +0.24 | · | -0.16 | -0.32 | -0.15 | -0.27 | -0.11 | -0.09 | -0.30 | -0.40 | -0.36 | -0.66 | -0.20 | -0.75 | -0.72 | -0.55 | -0.80 | -0.70 | -0.40 | -0.44 | -0.60 | -0.73 | -0.50 | -0.90 | -0.18 | -0.82 | -1.00 | -1.20 | -0.24 | +0.14 | -4.02 | -10.63 |
| 0.90 | +0.54 | +0.63 | +0.20 | +0.39 | +0.36 | +0.23 | +0.37 | +0.18 | +0.40 | +0.26 | -0.11 | -0.22 | +0.10 | -0.24 | +0.12 | · | -0.20 | +0.10 | -0.25 | -0.63 | -0.15 | -0.57 | -0.52 | -0.30 | -0.50 | -0.60 | -0.30 | -0.15 | -0.33 | -0.34 | -0.30 | -0.60 | +0.09 | -0.71 | -0.90 | -0.80 | -0.18 | +0.62 | -3.16 | -6.52 |
| 0.95 | +0.33 | +0.42 | · | +0.44 | +0.20 | +0.23 | +0.31 | · | +0.35 | · | -0.22 | -0.48 | -0.12 | -0.44 | · | -0.34 | -0.50 | · | -0.27 | -0.93 | -0.32 | -0.86 | -0.80 | -0.27 | -0.90 | -0.70 | -0.50 | -0.51 | -0.60 | -0.82 | -0.40 | -0.70 | -0.23 | -0.81 | -0.80 | -0.90 | -0.43 | +0.58 | -3.32 | -5.42 |
| 1.00 | +0.47 | +0.70 | +0.12 | +0.53 | +0.38 | +0.30 | +0.26 | +0.06 | +0.58 | · | -0.10 | -0.25 | · | -0.09 | · | · | -0.10 | +0.30 | -0.17 | -0.46 | -0.08 | -0.80 | -0.92 | -0.20 | -0.80 | -0.40 | · | -0.16 | -0.29 | -0.55 | · | -0.60 | · | -0.57 | -0.60 | -0.80 | -0.19 | +0.75 | -2.91 | -5.02 |

A dot is a difference at or inside the rounding floor.

## NLL by width

The one place a branch has separated from A by more than the measurement can be blamed for.

| width | AW | K | X | V | AO | AQ | AF | AT | AC | AS | AD | AA | W | I | AM | AK | F | M | AH | AL | A | AE | AR | Y | D | E | G | AI | S | T | H | B | AU | U | C | L | AG | AJ | AV | AN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | **1.117** | 1.148 | 1.180 | 1.143 | 1.179 | 1.189 | 1.146 | 1.161 | 1.189 | 1.154 | 1.162 | 1.230 | 1.239 | 1.239 | 1.307 | 1.281 | 1.296 | 1.206 | 1.222 | 1.343 | 1.272 | 1.212 | 1.670 | 1.286 | 1.400 | 1.317 | 1.439 | 1.346 | 1.329 | 1.301 | 1.377 | 1.465 | 1.119 | 1.347 | 1.453 | 1.448 | 4.605 | 4.605 | 4.605 | 4.605 |
| 0.30 | 1.110 | 1.130 | 1.153 | 1.116 | 1.158 | 1.171 | 1.134 | 1.138 | 1.150 | 1.125 | 1.140 | 1.233 | 1.228 | 1.238 | 1.315 | 1.288 | 1.291 | 1.190 | 1.198 | 1.334 | 1.287 | 1.202 | 1.637 | 1.283 | 1.395 | 1.313 | 1.419 | 1.336 | 1.330 | 1.306 | 1.364 | 1.482 | **1.087** | 1.345 | 1.429 | 1.448 | 1.458 | 4.605 | 4.605 | 4.605 |
| 0.35 | 1.094 | 1.127 | 1.152 | 1.088 | 1.142 | 1.153 | 1.133 | 1.128 | 1.122 | 1.105 | 1.143 | 1.235 | 1.222 | 1.247 | 1.311 | 1.292 | 1.301 | 1.174 | 1.185 | 1.338 | 1.286 | 1.199 | 1.627 | 1.287 | 1.381 | 1.316 | 1.404 | 1.340 | 1.338 | 1.326 | 1.352 | 1.476 | **1.065** | 1.349 | 1.433 | 1.459 | 1.208 | 4.605 | 4.605 | 4.605 |
| 0.40 | 1.080 | 1.114 | 1.155 | 1.078 | 1.120 | 1.167 | 1.142 | 1.112 | 1.101 | 1.098 | 1.132 | 1.241 | 1.228 | 1.259 | 1.320 | 1.296 | 1.291 | 1.182 | 1.169 | 1.345 | 1.295 | 1.187 | 1.615 | 1.308 | 1.384 | 1.323 | 1.395 | 1.357 | 1.361 | 1.341 | 1.354 | 1.518 | **1.045** | 1.358 | 1.443 | 1.466 | 1.164 | 4.605 | 4.605 | 4.605 |
| 0.45 | 1.078 | 1.108 | 1.150 | 1.075 | 1.116 | 1.158 | 1.124 | 1.099 | 1.093 | 1.103 | 1.131 | 1.244 | 1.224 | 1.255 | 1.316 | 1.294 | 1.286 | 1.191 | 1.156 | 1.363 | 1.318 | 1.183 | 1.558 | 1.321 | 1.374 | 1.310 | 1.391 | 1.373 | 1.357 | 1.352 | 1.345 | 1.539 | **1.016** | 1.374 | 1.427 | 1.465 | 1.160 | 4.605 | 4.605 | 4.605 |
| 0.50 | 1.073 | 1.107 | 1.150 | 1.064 | 1.110 | 1.165 | 1.115 | 1.091 | 1.082 | 1.095 | 1.131 | 1.248 | 1.222 | 1.261 | 1.329 | 1.304 | 1.275 | 1.175 | 1.146 | 1.367 | 1.337 | 1.172 | 1.535 | 1.329 | 1.364 | 1.301 | 1.373 | 1.366 | 1.373 | 1.352 | 1.349 | 1.562 | **1.003** | 1.382 | 1.413 | 1.451 | 1.151 | 4.605 | 4.605 | 4.605 |
| 0.55 | 1.069 | 1.105 | 1.152 | 1.058 | 1.106 | 1.168 | 1.099 | 1.093 | 1.084 | 1.096 | 1.129 | 1.250 | 1.210 | 1.278 | 1.329 | 1.321 | 1.274 | 1.186 | 1.131 | 1.378 | 1.337 | 1.166 | 1.520 | 1.342 | 1.362 | 1.292 | 1.369 | 1.370 | 1.387 | 1.369 | 1.343 | 1.563 | **1.009** | 1.389 | 1.403 | 1.446 | 1.157 | 4.605 | 4.605 | 4.605 |
| 0.60 | 1.073 | 1.107 | 1.157 | 1.056 | 1.106 | 1.175 | 1.101 | 1.093 | 1.081 | 1.091 | 1.134 | 1.258 | 1.212 | 1.295 | 1.339 | 1.321 | 1.265 | 1.181 | 1.123 | 1.398 | 1.354 | 1.162 | 1.495 | 1.353 | 1.340 | 1.283 | 1.358 | 1.386 | 1.399 | 1.382 | 1.345 | 1.587 | **1.015** | 1.398 | 1.392 | 1.427 | 1.182 | 4.605 | 4.605 | 4.605 |
| 0.65 | 1.078 | 1.114 | 1.167 | 1.060 | 1.112 | 1.185 | 1.106 | 1.097 | 1.089 | 1.098 | 1.139 | 1.266 | 1.221 | 1.316 | 1.345 | 1.332 | 1.259 | 1.190 | 1.113 | 1.400 | 1.389 | 1.159 | 1.501 | 1.370 | 1.347 | 1.278 | 1.366 | 1.413 | 1.420 | 1.396 | 1.338 | 1.611 | **1.028** | 1.426 | 1.412 | 1.423 | 1.207 | 1.328 | 4.605 | 4.605 |
| 0.70 | 1.079 | 1.115 | 1.167 | 1.060 | 1.118 | 1.194 | 1.112 | 1.097 | 1.087 | 1.105 | 1.139 | 1.275 | 1.220 | 1.337 | 1.347 | 1.342 | 1.242 | 1.199 | 1.104 | 1.410 | 1.395 | 1.156 | 1.494 | 1.381 | 1.347 | 1.263 | 1.356 | 1.412 | 1.436 | 1.412 | 1.335 | 1.622 | **1.035** | 1.438 | 1.406 | 1.430 | 1.224 | 1.216 | 1.837 | 4.605 |
| 0.75 | 1.081 | 1.120 | 1.170 | 1.066 | 1.123 | 1.197 | 1.115 | 1.097 | 1.091 | 1.107 | 1.156 | 1.274 | 1.225 | 1.347 | 1.350 | 1.343 | 1.222 | 1.208 | 1.093 | 1.416 | 1.391 | 1.152 | 1.475 | 1.388 | 1.348 | 1.248 | 1.370 | 1.416 | 1.442 | 1.408 | 1.334 | 1.627 | **1.046** | 1.449 | 1.409 | 1.419 | 1.236 | 1.240 | 1.323 | 4.605 |
| 0.80 | 1.083 | 1.123 | 1.174 | 1.071 | 1.129 | 1.203 | 1.120 | 1.106 | 1.090 | 1.114 | 1.159 | 1.285 | 1.232 | 1.359 | 1.346 | 1.350 | 1.205 | 1.211 | 1.086 | 1.428 | 1.400 | 1.152 | 1.460 | 1.393 | 1.343 | 1.228 | 1.371 | 1.421 | 1.439 | 1.408 | 1.332 | 1.629 | **1.064** | 1.448 | 1.409 | 1.423 | 1.247 | 1.274 | 1.337 | 2.232 |
| 0.85 | 1.093 | 1.129 | 1.180 | 1.077 | 1.133 | 1.202 | 1.127 | 1.109 | 1.097 | 1.121 | 1.167 | 1.298 | 1.233 | 1.368 | 1.347 | 1.350 | 1.192 | 1.218 | 1.078 | 1.435 | 1.413 | 1.153 | 1.462 | 1.393 | 1.347 | 1.205 | 1.371 | 1.423 | 1.446 | 1.412 | 1.331 | 1.636 | **1.077** | 1.456 | 1.409 | 1.420 | 1.262 | 1.294 | 1.378 | 1.389 |
| 0.90 | 1.105 | 1.144 | 1.191 | 1.085 | 1.149 | 1.214 | 1.137 | 1.119 | 1.104 | 1.138 | 1.179 | 1.298 | 1.247 | 1.385 | 1.359 | 1.357 | 1.184 | 1.226 | **1.080** | 1.435 | 1.427 | 1.155 | 1.439 | 1.407 | 1.342 | 1.201 | 1.374 | 1.435 | 1.442 | 1.417 | 1.329 | 1.628 | 1.099 | 1.463 | 1.409 | 1.425 | 1.277 | 1.326 | 1.420 | 1.416 |
| 0.95 | 1.114 | 1.158 | 1.201 | 1.097 | 1.165 | 1.224 | 1.151 | 1.126 | 1.120 | 1.163 | 1.189 | 1.305 | 1.254 | 1.396 | 1.363 | 1.362 | 1.199 | 1.241 | **1.095** | 1.450 | 1.436 | 1.169 | 1.427 | 1.410 | 1.354 | 1.211 | 1.363 | 1.444 | 1.444 | 1.425 | 1.327 | 1.619 | 1.128 | 1.465 | 1.399 | 1.426 | 1.295 | 1.343 | 1.452 | 1.475 |
| 1.00 | 1.131 | 1.179 | 1.214 | **1.128** | 1.187 | 1.236 | 1.173 | 1.141 | 1.163 | 1.202 | 1.205 | 1.308 | 1.260 | 1.397 | 1.373 | 1.367 | 1.304 | 1.262 | 1.188 | 1.450 | 1.437 | 1.216 | 1.437 | 1.416 | 1.351 | 1.331 | 1.352 | 1.449 | 1.447 | 1.426 | 1.333 | 1.596 | 1.163 | 1.463 | 1.398 | 1.426 | 1.310 | 1.353 | 1.489 | 1.529 |
| **mean** | 1.091 | 1.127 | 1.170 | 1.083 | 1.135 | 1.187 | 1.127 | 1.113 | 1.109 | 1.120 | 1.152 | 1.266 | 1.230 | 1.311 | 1.337 | 1.325 | 1.255 | 1.203 | 1.135 | 1.393 | 1.361 | 1.175 | 1.522 | 1.354 | 1.361 | 1.276 | 1.379 | 1.393 | 1.399 | 1.377 | 1.343 | 1.573 | **1.063** | 1.409 | 1.415 | 1.438 | 1.446 | 2.951 | 3.230 | 3.669 |

A's NLL climbs from 1.272 at width 0.25 to 1.437 at 1.00: it is least calibrated where it is most accurate, which is what a training error of 0.000 at the widest width predicts. F does not do that.

## How each branch is built

Each row names the closest branch above it and lists only what differs, so the line that makes a branch itself is the line you read. Diffed against A, most of this table would be K repeated ten times with the distinguishing setting arriving last.

Read out of `apps/cifar100_<name>.yml` when this page was written, so a branch cannot be described here as something its config has stopped being.

Some branches are only readable in pairs, one leaning each way from K: W against Y on blur; AS against AT on where the free widths are drawn; AC against AD on the weight on the feature terms. A bracket where both ends win says the axis does not matter, which is an answer the winning end alone cannot give.

One bracket does not close: AU against AV on which samples KD attends to, where AV collapsed. A collapsed end is not a losing end, so it cannot stand as the control its pair needed, and the axis stays open.

**AW** - K with each width taught by the next larger one instead of by the widest

K, with:

* `teacher_chain: True` - each width learns from the next larger one in the batch instead of every width learning from the widest

**K** - I plus transport between the two middle widths, at the feature tier

I, with:

* `horizontal_kd: True` - adds a term between two co-sampled widths that stand in no teacher relation to each other
* `horizontal_weight: 1.0` - the weight on the horizontal term
* `horizontal_where: feature` - which tier the horizontal term sits at: `logit`, `feature`, or `both`

**X** - K at eps 0.1 instead of 0.2

K, with:

* `sinkhorn_eps: 0.1` - the entropic blur. Small collapses the plan onto a permutation, large spreads mass over many partners

**V** - K with transport that may leave mass unmatched, at tau 1.0

K, with:

* `feature_loss: unbalanced` - transport that may leave mass unmatched, the marginals penalised rather than enforced
* `feature_weight: 0.8` - the weight on the feature terms, set by matching gradient norms rather than loss values
* `unbalanced_tau: 1.0` - the price of leaving mass unmatched. Large recovers the balanced plan

**AO** - K plus a repulsion pushing the classifier rows apart

K, with:

* `spread_neighbours: 5` - how many nearest rows the repulsion charges for, which makes it a minimum margin rather than a mean spread
* `spread_weight: 1.0` - a repulsion on the classifier rows, the one term here with the opposite sign to the rest

**AQ** - K transporting the two widths' class means rather than their samples

K, with:

* `feature_classwise: True` - transport runs between class means rather than between samples
* `feature_weight: 0.7` - the weight on the feature terms, set by matching gradient norms rather than loss values

**AF** - K coupling all three student pairs instead of one

K, with:

* `horizontal_pairs: all` - every pair among the co-sampled students is coupled, not just the two middle ones

**AT** - K with the same free widths drawn flat in compute instead

K, with:

* `width_sampling: macs` - the same draw made flat in compute. MACs measured at width^1.965 here, so this leans toward the wide end

**AC** - K at half the weight on the feature terms

K, with:

* `feature_weight: 0.5` - the weight on the feature terms, set by matching gradient norms rather than loss values

**AS** - K with the sandwich rule's free widths drawn flat in log width

K, with:

* `width_sampling: log` - the sandwich rule draws its free widths flat in log width, which puts half of them below 0.50 against a third for uniform

**AD** - K at twice the weight, the other side of the same question

K, with:

* `feature_weight: 2.0` - the weight on the feature terms, set by matching gradient norms rather than loss values

**AA** - K charging transport by direction rather than distance

K, with:

* `feature_ground: cosine` - the ground cost is angle rather than distance
* `feature_weight: 78.0` - the weight on the feature terms, set by matching gradient norms rather than loss values

**W** - K at eps 0.02, a plan collapsed onto a permutation

K, with:

* `sinkhorn_eps: 0.02` - the entropic blur. Small collapses the plan onto a permutation, large spreads mass over many partners

**I** - A plus transport between the student and teacher feature clouds

K, with:

* `horizontal_kd: False` - adds a term between two co-sampled widths that stand in no teacher relation to each other

**AM** - K with sliced transport keeping the worst direction rather than the average

K, with:

* `feature_loss: sliced` - transport along random one dimensional projections instead of a full plan
* `feature_weight: 0.09` - the weight on the feature terms, set by matching gradient norms rather than loss values
* `sliced_reduce: max` - keeps the worst projection rather than the average of them

**AK** - K with the closed-form Gaussian transport, mean gap plus a covariance term

K, with:

* `feature_loss: bures` - the closed form between two Gaussians fitted to the clouds: mean gap plus a covariance term
* `feature_weight: 0.4` - the weight on the feature terms, set by matching gradient norms rather than loss values

**F** - A plus a symmetrized KL between the two middle widths

K, with:

* `horizontal_loss: jeffreys` - what compares the two widths at the logit tier: `wasserstein`, `jeffreys`, or `kl`
* `horizontal_where: logit` - which tier the horizontal term sits at: `logit`, `feature`, or `both`

**A'** - the same branch again, to measure the noise

I, with:

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

**A** - US-Net as published: inplace distillation with KL. Identical to I in config; they differ by seed.

**AE** - K and F at once, transport on features and Jeffreys on logits

K, with:

* `horizontal_loss: jeffreys` - what compares the two widths at the logit tier: `wasserstein`, `jeffreys`, or `kl`
* `horizontal_where: both` - which tier the horizontal term sits at: `logit`, `feature`, or `both`

**AR** - K with the logit teacher softened to temperature 4

K, with:

* `kd_temperature: 4.0` - the logit teacher is softened by this factor before the student is matched to it, with a T^2 rescale so the gradient stays comparable

**Y** - K at eps 0.5, mass spread over many partners

K, with:

* `sinkhorn_eps: 0.5` - the entropic blur. Small collapses the plan onto a permutation, large spreads mass over many partners

**D** - C plus Wasserstein between the two middle widths

K, with:

* `kd_loss: wasserstein` - the vertical KD term becomes entropic transport over a class cost matrix, instead of soft cross entropy

**E** - A plus plain KL between the two middle widths, neither symmetric nor metric aware

K, with:

* `horizontal_loss: kl` - what compares the two widths at the logit tier: `wasserstein`, `jeffreys`, or `kl`
* `horizontal_where: logit` - which tier the horizontal term sits at: `logit`, `feature`, or `both`

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

**B** - adaptive alpha-divergence, AlphaNet, bit-for-bit against the reference implementation

I, with:

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

**C** - Wasserstein on the logits, vertical only

I, with:

* `kd_loss: wasserstein` - the vertical KD term becomes entropic transport over a class cost matrix, instead of soft cross entropy

**L** - D with the horizontal term weighted toward the narrow widths

K, with:

* `kd_loss: wasserstein` - the vertical KD term becomes entropic transport over a class cost matrix, instead of soft cross entropy
* `weight_schedule: narrow` - the extra term is faded out toward the wide widths, where it was measured to hurt

**AG** - K with five sampled widths and every pair coupled: the narrowest never trained

K, with:

* `horizontal_pairs: all` - every pair among the co-sampled students is coupled, not just the two middle ones
* `num_sample_training: 5` - how many widths are run per step, the sandwich rule's two ends included

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

There is a second reason, and it is in this repository rather than in the measurement. Ranking channels by the batch-norm gain is Network Slimming, and that method trains with an L1 penalty on the gains, which is what drives them apart and makes the small ones mean something. `train.py:271` gives every one-dimensional parameter a weight decay of zero, so the gains here are trained under no penalty at all. The criterion is being read outside the regime it was built for.

The row above it says the rest. The gain records how far a channel is turned up and nothing about whether anything downstream reads it; how much the next layer reads each channel is a separate quantity, and that one is sorted to 100%. A channel can be quiet and still matter. The Taylor score multiplies the gain by the gradient reaching it and so carries both halves, which is why it lands with the majority.

One of the eight is not asking about importance at all. Distance from the other filters is the FPGM criterion, which ranks a channel by how much of a duplicate it is, and the sandwich rule presses directly for a prefix that is good and only indirectly for one that is varied. It was the likeliest place to find headroom and it reads 99%.

The widths agree on the order too, which was the objection that would have closed the idea for every criterion at once. Rank correlation between the Taylor orderings is 0.905 between widths 1.00 and 0.50, 0.882 between 1.00 and 0.25, and 0.981 between 0.50 and 0.25, with no layer below 0.74. That is three of the sixteen widths, not all of them, though the three include both ends of the range and the 1.00 against 0.25 pair is the one with the most room to disagree.

So the sandwich rule sorts the channels, and the reason is the objective: the prefix trains at every sampled width and at 0.25 has to classify alone. Six per cent of the available sorting is left at width 1.00 and three at 0.50, and no criterion tried here finds more. This section said closed, then open, then closed again, and the middle reading rested on the single measure that the other seven contradict.

## Not settled

- Whether F is ahead of A at all. The accuracy gap is at the rounding floor; only the NLL gap is outside it.
- What F is doing, exactly. It has symmetry and it has a coupling between the two middle widths. E, the same term as plain KL, is what separates those.
- Sigma, properly. Two seeds put it under the floor; three would make it a number worth quoting. It has stopped being a precaution: the four branches below decide their own reading on it.
- Why AV collapsed. Everything at width 0.70 and above trained; everything at 0.65 and below sits at exactly chance, 1.00 accuracy and NLL ln(100). A cliff, not a slope, which is the shape AJ and AN already showed. What has been measured is that the weight itself does not explode: across teacher sharpness from uniform to memorised the per-sample weight stays between 0 and about 4 with mean 1, so a blown-up KD term is ruled out rather than merely unlikely. What has not been found is the path from that weighting to dead leading channels. It is undiagnosed, not explained.
- One defect the probe did find, which is not yet shown to be the cause. AV measures confidence as `entropy.max() - entropy`, taking its zero point from a batch order statistic that itself drifts to zero as the teacher memorises. When every entropy in a batch underflows to exactly zero both AU and AV divide zero by zero and hand KD a weight of exactly zero for every sample. The fixed reference the quantity actually has, log(C) minus entropy, survives that case and does not move with the batch. A rerun of this axis should use it.
- Whether AW is ahead of K, and the table should not be read as saying it is. AW is first at 74.32 against 74.26, a gap of 0.06 where a difference of means over sixteen widths carries 0.01 to 0.03 of rounding alone. The number that settles it is not the mean: AW is ahead of K at eight widths out of sixteen, behind at eight, scattered from -0.26 to +0.71. A real gain does not look like that. K against A is ahead at all sixteen, which is what one does look like. On this evidence AW ties K and the ordering between them is a coin.
- What AW is still worth. It reaches K from somewhere else - it changes which width teaches which and never touches the transport term - and it is the first branch off that axis to get there. Its NLL is 1.091 against 1.127, about twice the gap two seeds of A showed, which is weak but points the same way. Two mechanisms arriving at the same place is worth a second seed on both, which is the run to do next.
- Whether K sits on a peak or on a high draw. AS, AT, AC and AD are K with one knob moved in four different directions - the free widths drawn narrower, the free widths drawn wider, the feature weight halved, the feature weight doubled - and all four land between 73.82 and 73.96, which is 0.30 to 0.44 below K. Four perturbations that share nothing mechanically do not usually agree by accident, so either 1.0 and a uniform draw are both genuinely best, or the 74.26 for K is a high seed and the branch is really worth about 73.9. Both readings fit this table. Only sigma separates them, and it is the same measurement the bullet above asks for.
