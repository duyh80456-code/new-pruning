# Seed-3 fixed-dense-SW matched-control gate

Run in this order:

1. `kaggle_rq2_fixed_dense_sw_tau_probe_cpu.ipynb` (CPU). Attach one completed
   seed-3 `e2e_pairwise_pilot_v2` output. It reads the existing common E10
   checkpoint and `pure_sw/sw_policies/epoch_010.npz`, then exports three
   *offline* soft-SW policies for κ = 0.1, 1, 10 with τ = κ·Std(SW² scores).
   Inspect entropy, effective support, L1 from Uniform and max probability.
   This notebook does **not** train or choose κ.
2. Attach the original completed output, the small CPU probe output, and
   CIFAR-100 to `kaggle_rq2_fixed_dense_sw_gate_seed3_t4x2.ipynb`. Set
   `SELECTED_KAPPA` to one of the probed values using policy shape only.
   Its freeze artifact binds the choice to hashes of the common E10 checkpoint,
   E10 SW file and probe. It checks that the existing Uniform E100 anchor came
   from the same E10 state.
3. The GPU notebook trains `fixed_dense_sw` and `fixed_dense_shuffled` from the
   same E10 model, optimizer, scheduler, data order and pair RNG on T4×2.
   Uniform is **reused**, not retrained. Both policies stay fixed for epochs
   11–100. For compute parity, both new branches perform and discard the same
   training-derived geometry sweeps at epochs 10,20,…,90 as the old branches.

The shuffled control is a deterministic permutation of the 14 width labels
of the SW policy's 91-vector. Therefore it retains exactly the same sorted
probabilities, entropy, support, maximum probability and uniform 1/7 width
marginals. Its SW alignment is minimized among 256 fixed-seed permutations,
using no accuracy. Both policies have zero post-E10 churn.

The compact ZIP `rq2-fixed-dense-sw-seed3-analysis.zip` contains each new
branch's `dense_metrics.csv`, `pair_stats.csv`, `train_log.csv`,
`policy_summary.json`, width marginals and provenance, plus the Uniform
comparison and frozen policy. It excludes large checkpoints; the full Kaggle
output tree retains them for resume/verification. All dense accuracy is on the
fixed 5k **validation** split; test is never used here.

This is one **development** seed and a matched-control gate, not multi-seed
confirmation. A seed-3 win for SW over shuffled would support incremental
pair-structure value, but any generalization claim needs fresh seeds.
