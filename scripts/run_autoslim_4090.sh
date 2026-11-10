#!/usr/bin/env bash
# AutoSlim on CIFAR-100 ResNet-50, everything on one card: steps 1 and 2
# (supernet and greedy search, scripts/autoslim_search.py, seed 1995) once,
# then step 3 at the three seeds, KS 1995, KT 2026, KU 2006, with the
# authors' four budgets (MACs of widths 0.25, 0.5, 0.75, 1.0). The search's
# JSON stays on this machine and the three training configs read it from
# results/autoslim_search_r50.json, so nothing has to be copied in between.
# Run from the repo root inside tmux. The search is skipped when its JSON
# exists (the supernet checkpoint is kept as well, so a crash during the
# search does not retrain the supernet). Each training run resumes from its
# last checkpoint, so after a crash run this again. A failed run is reported
# and the queue moves on to the next one.
set -o pipefail
SEARCH=results/autoslim_search_r50.json
FAILED=()

if [ ! -f "$SEARCH" ]; then
    echo "=== search $(date)"
    python scripts/autoslim_search.py --supernet_ckpt logs/autoslim_supernet_r50.pt \
        --out "$SEARCH" 2>&1 | tee -a run_autoslim_search.log
    if [ ! -f "$SEARCH" ]; then
        echo "=== search FAILED, no configurations to train"
        exit 1
    fi
fi

for b in ks_autoslim_r50_seed1995 kt_autoslim_r50_seed2026 \
         ku_autoslim_r50_seed2006; do
    echo "=== $b $(date)"
    python train.py app:apps/cifar100_$b.yml 2>&1 | tee -a run_${b}_b4.log \
        || { echo "=== $b FAILED"; FAILED+=("$b"); }
done

# AutoSlim's step 3 with all sixteen configurations in one network, as a
# note that it does not carry over: resumes the first KS run (epoch 33)
echo "=== ks16 $(date)"
python train.py app:apps/cifar100_ks16_autoslim_r50_seed1995.yml 2>&1     | tee -a run_ks_autoslim_r50_seed1995.log || { echo "=== ks16 FAILED"; FAILED+=(ks16); }

echo "=== done $(date), failed: ${FAILED[*]:-none}"
