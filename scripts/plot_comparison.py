"""K against published methods on ResNet-50, CIFAR-100, from results/runs.json

    python scripts/plot_comparison.py

Writes results/figures/compare_r50.png and .pdf: top-1 against width at
the sixteen widths 0.25-1.00, for K, K with four heads (DO) and K with
far free-width pairs (DM) against the published methods. Every run here is seed 1995 under the same
protocol, read after BN recalibration. WKD-L in US-Net (74.72) and DYNAS
(73.92) sit far below and are left out so the close methods stay
readable.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results', 'figures')

# letter, label, colour, emphasised. DO and DM are one seed each, so
# their lead over DD is not yet a result.
METHODS = [
    ('DM', 'K + far pairs (ours)', '#8c1d18', True),
    ('DO', 'K + 4 heads (ours)', '#ff7f0e', True),
    ('DD', 'K (ours)', '#d62728', True),
    ('CH', 'SOLAR (WACV 2026)', '#1f77b4', False),
    ('BX', 'US-Net (ICCV 2019)', '#333333', False),
    ('CF', 'Scala (NeurIPS 2024)', '#2ca02c', False),
    ('CV', 'AlphaNet (ICML 2021)', '#9467bd', False),
]


def main():
    with open(os.path.join(ROOT, 'results', 'runs.json'),
              encoding='utf-8') as handle:
        store = json.load(handle)
    widths = store['widths']
    top1 = {r['letter']: r['top1'] for r in store['runs']}

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for letter, label, colour, ours in METHODS:
        mean = sum(top1[letter]) / len(top1[letter])
        ax.plot(widths, top1[letter], marker='o' if ours else '.',
                markersize=5 if ours else 6, linewidth=2.4 if ours else 1.3,
                color=colour, label='{} - {:.2f}'.format(label, mean),
                zorder=3 if ours else 2)
    ax.set_xlabel('Width multiplier')
    ax.set_ylabel('Top-1 accuracy (%)')
    ax.set_xticks([0.25, 0.4, 0.55, 0.7, 0.85, 1.0])
    ax.set_title('ResNet-50, CIFAR-100: accuracy against width '
                 '(legend: mean over 16 widths)', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, 'compare_r50.' + ext), dpi=200,
                    bbox_inches='tight')
    print('wrote', os.path.join(OUT, 'compare_r50.png'))


if __name__ == '__main__':
    main()
