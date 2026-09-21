# Results

Universally slimmable ResNet-18 on CIFAR-100, sixteen widths from 0.25 to 1.00, accuracy after BN post-statistics. One seed each unless the row says otherwise. Generated from `results/runs.json` by `scripts/collect_results.py`.

## Where it stands

| | branch | mean top-1 | vs A | worst width | mean NLL | vs A |
|---|---|---|---|---|---|---|
| **F** | A plus a symmetrized KL between the two middle widths | **73.54** | +0.03 | 70.30 | 1.255 | -0.105 |
| **A'** | the same branch again, to measure the noise | **73.53** | +0.01 | 69.80 | 1.377 | +0.016 |
| **A** | US-Net as published: inplace distillation with KL | **73.52** | +0.00 | 70.10 | 1.361 | +0.000 |
| **D** | C plus Wasserstein between the two middle widths | **73.27** | -0.25 | 70.30 | 1.361 | +0.000 |
| **B** | adaptive alpha-divergence, AlphaNet, bit-for-bit against the reference implementation | **73.08** | -0.44 | 70.40 | 1.573 | +0.212 |
| **C** | Wasserstein on the logits, vertical only | **72.74** | -0.77 | 69.40 | 1.415 | +0.054 |

## What the numbers can carry

Accuracy was logged to three decimals of error, so every value is a multiple of 0.10 and each one carries up to 0.05 of rounding. A difference between two of them carries up to **0.10**, and a difference of means over sixteen widths roughly 0.01 to 0.03.

A and A' are the same branch under different seeds. Their means differ by **0.01**, which is at that floor: what the pair establishes is that seed noise on the mean sits under it, not that it equals 0.01. Per width the same pair differs by as much as **0.40**, so the worst-width column is noise and should not be ranked.

So the gaps worth reading are the ones far outside that floor: A over C by 0.77, over B by 0.44, over D by 0.25. F against A, at +0.03, is not one of them.

NLL is printed to three decimals on a scale near 1.3, so it resolves to 0.001 and none of this applies to it.

## Accuracy by width

| width | MACs (M) | F | A' | A | D | B | C |
|---|---|---|---|---|---|---|---|
| 0.25 | 35.1 | 70.30 | 69.80 | 70.10 | 70.30 | **70.40** | 69.40 |
| 0.30 | 60.5 | **71.30** | 70.70 | 70.80 | **71.30** | 71.10 | 70.40 |
| 0.35 | 72.7 | **72.00** | 71.50 | 71.50 | 71.60 | 71.40 | 70.90 |
| 0.40 | 84.8 | 72.40 | **72.60** | 72.20 | 72.20 | 71.80 | 71.30 |
| 0.45 | 118.0 | **72.80** | 72.70 | **72.80** | 72.60 | 72.20 | 72.20 |
| 0.50 | 139.3 | **73.50** | 73.10 | 73.20 | 72.80 | 72.60 | 72.30 |
| 0.55 | 163.2 | **73.60** | 73.40 | 73.40 | 73.30 | 72.90 | 72.60 |
| 0.60 | 207.6 | 73.60 | **73.80** | **73.80** | 73.70 | 73.30 | 73.00 |
| 0.65 | 227.7 | 74.00 | **74.20** | 73.90 | 74.00 | 73.60 | 73.40 |
| 0.70 | 280.2 | 74.20 | **74.50** | 74.30 | 74.10 | 73.60 | 73.30 |
| 0.75 | 312.8 | **74.60** | **74.60** | **74.60** | 74.10 | 74.10 | 73.60 |
| 0.80 | 347.9 | 74.60 | 74.70 | **74.80** | 74.40 | 74.20 | 73.90 |
| 0.85 | 411.6 | 74.80 | 74.70 | **75.10** | 74.30 | 74.20 | 74.10 |
| 0.90 | 439.8 | 74.90 | **75.20** | 75.10 | 74.60 | 74.50 | 74.20 |
| 0.95 | 511.6 | 74.90 | **75.40** | **75.40** | 74.50 | 74.70 | 74.60 |
| 1.00 | 555.5 | 75.20 | **75.60** | 75.30 | 74.50 | 74.70 | 74.70 |

## Against A, by width

Every branch that has moved at all has moved the same way: ahead at the narrow widths, behind at the wide ones. Five times here, and three more in the literature.

| width | F | A' | D | B | C |
|---|---|---|---|---|---|
| 0.25 | +0.20 | -0.30 | +0.20 | +0.30 | -0.70 |
| 0.30 | +0.50 | -0.10 | +0.50 | +0.30 | -0.40 |
| 0.35 | +0.50 | · | +0.10 | -0.10 | -0.60 |
| 0.40 | +0.20 | +0.40 | · | -0.40 | -0.90 |
| 0.45 | · | -0.10 | -0.20 | -0.60 | -0.60 |
| 0.50 | +0.30 | -0.10 | -0.40 | -0.60 | -0.90 |
| 0.55 | +0.20 | · | -0.10 | -0.50 | -0.80 |
| 0.60 | -0.20 | · | -0.10 | -0.50 | -0.80 |
| 0.65 | +0.10 | +0.30 | +0.10 | -0.30 | -0.50 |
| 0.70 | -0.10 | +0.20 | -0.20 | -0.70 | -1.00 |
| 0.75 | · | · | -0.50 | -0.50 | -1.00 |
| 0.80 | -0.20 | -0.10 | -0.40 | -0.60 | -0.90 |
| 0.85 | -0.30 | -0.40 | -0.80 | -0.90 | -1.00 |
| 0.90 | -0.20 | +0.10 | -0.50 | -0.60 | -0.90 |
| 0.95 | -0.50 | · | -0.90 | -0.70 | -0.80 |
| 1.00 | -0.10 | +0.30 | -0.80 | -0.60 | -0.60 |

A dot is a difference at or inside the rounding floor.

## NLL by width

The one place a branch has separated from A by more than the measurement can be blamed for.

| width | F | A | D | B | C |
|---|---|---|---|---|---|
| 0.25 | 1.296 | **1.272** | 1.400 | 1.465 | 1.453 |
| 0.30 | 1.291 | **1.287** | 1.395 | 1.482 | 1.429 |
| 0.35 | 1.301 | **1.286** | 1.381 | 1.476 | 1.433 |
| 0.40 | **1.291** | 1.295 | 1.384 | 1.518 | 1.443 |
| 0.45 | **1.286** | 1.318 | 1.374 | 1.539 | 1.427 |
| 0.50 | **1.275** | 1.337 | 1.364 | 1.562 | 1.413 |
| 0.55 | **1.274** | 1.337 | 1.362 | 1.563 | 1.403 |
| 0.60 | **1.265** | 1.354 | 1.340 | 1.587 | 1.392 |
| 0.65 | **1.259** | 1.389 | 1.347 | 1.611 | 1.412 |
| 0.70 | **1.242** | 1.395 | 1.347 | 1.622 | 1.406 |
| 0.75 | **1.222** | 1.391 | 1.348 | 1.627 | 1.409 |
| 0.80 | **1.205** | 1.400 | 1.343 | 1.629 | 1.409 |
| 0.85 | **1.192** | 1.413 | 1.347 | 1.636 | 1.409 |
| 0.90 | **1.184** | 1.427 | 1.342 | 1.628 | 1.409 |
| 0.95 | **1.199** | 1.436 | 1.354 | 1.619 | 1.399 |
| 1.00 | **1.304** | 1.437 | 1.351 | 1.596 | 1.398 |

A's NLL climbs from 1.272 at width 0.25 to 1.437 at 1.00: it is least calibrated where it is most accurate, which is what a training error of 0.000 at the widest width predicts. F does not do that.

## Not settled

- Whether F is ahead of A at all. The accuracy gap is at the rounding floor; only the NLL gap is outside it.
- What F is doing, exactly. It has symmetry and it has a coupling between the two middle widths. E, the same term as plain KL, is what separates those.
- Sigma, properly. Two seeds put it under the floor; three would make it a number worth quoting.
- Everything at the feature tier. Those four branches died in the profiler on their first run and have not been rerun.
