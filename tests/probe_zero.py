"""AV measures confidence as entropy.max() - entropy: a batch order
statistic, not a fixed reference. The banner says the teacher reaches a
training error of 0.000, so ask what that expression does once entropy
underflows to exactly 0 in float32, and compare against the fixed
reference log(C) that the quantity actually has.
"""
import math
import torch

C = 100
LOG_C = math.log(C)


def run(entropy, mode):
    if mode == 'batch max':
        w = entropy.max() - entropy
    elif mode == 'log C':
        w = LOG_C - entropy
    else:
        w = entropy
    return w / w.mean().clamp_min(1e-12)


print('a memorised teacher: most entropies underflow to exactly 0,')
print('a few samples are still uncertain\n')
print('{:>6} {:>10} {:>12} {:>12} {:>12}'.format(
    'hard', 'H.max()', 'AV max w', 'AV w on hard', 'logC max w'))
for hard in (8, 4, 2, 1, 0):
    entropy = torch.zeros(96)
    if hard:
        entropy[:hard] = torch.linspace(0.05, 0.4, hard)
    av = run(entropy, 'batch max')
    fixed = run(entropy, 'log C')
    print('{:6d} {:10.4f} {:12.3f} {:12.3f} {:12.3f}'.format(
        hard, entropy.max(), av.max(),
        av[:hard].max() if hard else float('nan'), fixed.max()))

print()
print('the last row is the absorbing state: every entropy exactly 0.')
entropy = torch.zeros(96)
av = run(entropy, 'batch max')
au = run(entropy, 'entropy')
fixed = run(entropy, 'log C')
print('  AV weights  -> all {:.1f}   (KD term is identically zero)'
      .format(av.max()))
print('  AU weights  -> all {:.1f}   (same hole, same arithmetic)'
      .format(au.max()))
print('  logC weights-> all {:.1f}   (survives it)'.format(fixed.max()))
