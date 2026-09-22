"""What the two kd_weighting modes actually do to the per-sample weights.

AV collapsed: the narrow widths sat at exactly ln(100) for 95 epochs.
Before blaming the idea, look at the arithmetic that implements it.
Batch of 96, 100 classes, teacher sharpened the way training sharpens it.
"""
import torch

torch.manual_seed(1995)
N, C = 96, 100


def weights(soft, mode):
    entropy = -(soft * soft.clamp_min(1e-12).log()).sum(1)
    w = entropy if mode == 'entropy' else entropy.max() - entropy
    return w / w.mean().clamp_min(1e-12), entropy


print('{:>7} {:>9} {:>9} {:>10} {:>9} {:>10}'.format(
    'sharp', 'H mean', 'H spread', 'AU max w', 'AU zeros', 'AV max w'))
base = torch.randn(N, C)
for sharp in (1.0, 3.0, 6.0, 10.0, 20.0, 40.0):
    soft = torch.softmax(base * sharp, dim=1)
    au, H = weights(soft, 'entropy')
    av, _ = weights(soft, 'confidence')
    print('{:7.0f} {:9.2e} {:9.2e} {:10.1f} {:9d} {:10.1f}'.format(
        sharp, H.mean(), H.std(), au.max(), int((au < 0.01).sum()),
        av.max()))

print()
print('the case the log shows: teacher memorised, every sample one-hot')
for spike in (1e-3, 1e-5, 1e-7, 1e-9):
    soft = torch.full((N, C), spike / (C - 1))
    soft[:, 0] = 1.0 - spike
    soft = soft + torch.rand(N, C) * spike * 1e-3   # tiny per-sample jitter
    soft = soft / soft.sum(1, keepdim=True)
    au, H = weights(soft, 'entropy')
    av, _ = weights(soft, 'confidence')
    print('  spike {:.0e}   H {:.3e}   AU max w {:8.2f}   AV max w {:12.2f}'
          .format(spike, H.mean(), au.max(), av.max()))
