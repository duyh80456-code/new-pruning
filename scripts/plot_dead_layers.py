"""figure: where the dead channels are, from scripts/dead_layers.py output

    python scripts/plot_dead_layers.py <out.png> <label>=<dead_layers.json> ...

e.g. python scripts/plot_dead_layers.py results/figures/dead_layers.png \
         "collapsed (CR)=results/dead_layers_cr.json"

One row of panels per run. Left: the fraction of dead channels at the
output of every residual block (the ReLU after the sum, the stream the
next block reads), per test width; the y labels carry each width's top-1.
Right: at width 1.0, the same fraction split by where a channel sits,
inside the prefix width 0.25 also uses, or past the prefix of width 0.60,
which only the wider widths have.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

out_path = sys.argv[1]
runs = [arg.split('=', 1) for arg in sys.argv[2:]]

INK = '#0b0b0b'
INK_2 = '#52514e'
GRID = '#e4e3df'
SURFACE = '#ffffff'
# sequential blue, near-surface at zero dead
RAMP = LinearSegmentedColormap.from_list(
    'dead', ['#f4f8fd', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
PREFIX = '#2a78d6'   # categorical slot 1
REST = '#eb6834'     # categorical slot 2


def residual_outputs(order):
    """the stem ReLU and every block's post-sum ReLU, in forward order"""
    return [n for n in order if n.endswith('post_relu') or n.count('.') == 2
            and n.startswith('features.0.')]


def stage_bounds(names, counts=(3, 4, 6, 3)):
    """column index after which each stage ends (stem is column 0)"""
    bounds, at = [], 0
    for c in counts[:-1]:
        at += c
        bounds.append(at + 0.5)
    return bounds


plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK_2,
                     'axes.labelcolor': INK, 'xtick.color': INK_2,
                     'ytick.color': INK_2, 'text.color': INK})
fig, axes = plt.subplots(len(runs), 2, figsize=(10, 3.6 * len(runs)),
                         gridspec_kw={'width_ratios': [1.6, 1]},
                         squeeze=False, facecolor=SURFACE)

for row, (label, path) in enumerate(runs):
    data = json.load(open(path))
    names = residual_outputs(data['relu_order'])
    widths = sorted(data['widths'], key=float)
    grid = []
    for w in widths:
        by = {l['name']: l['dead'] for l in data['widths'][w]['layers']}
        grid.append([by[n] for n in names])
    columns = ['stem'] + [str(i) for i in range(1, len(names))]

    ax = axes[row][0]
    im = ax.imshow(grid, cmap=RAMP, vmin=0, vmax=1, aspect='auto',
                   origin='lower', interpolation='nearest')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(columns)
    ax.set_yticks(range(len(widths)))
    ax.set_yticklabels(['{:.2f}  ({:.1f}%)'.format(
        float(w), data['widths'][w]['acc']) for w in widths], fontsize=7.5)
    for b in stage_bounds(names):
        ax.axvline(b, color=SURFACE, linewidth=2)
    for i, name in enumerate(['stage 1', 'stage 2', 'stage 3', 'stage 4']):
        centre = [2, 5.5, 10.5, 15][i]
        ax.text(centre, len(widths) - 0.3, name, ha='center', va='bottom',
                fontsize=8, color=INK_2)
    ax.set_xlabel('residual block output (ReLU after the sum)')
    ax.set_ylabel('width  (top-1)')
    ax.set_title('{}: dead channels per block and width'.format(label),
                 loc='left', fontsize=10, color=INK, pad=14)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label('fraction of channels dead', color=INK_2)
    bar.outline.set_visible(False)

    ax = axes[row][1]
    split = {e['name']: e for e in data['full_width_split']}
    xs = list(range(len(names)))
    prefix = [split[n]['prefix_0.25']['dead'] for n in names]
    rest = [split[n]['rest']['dead'] or 0.0 for n in names]
    ax.plot(xs, prefix, color=PREFIX, linewidth=2, marker='o', markersize=4,
            label='channels width 0.25 also uses')
    ax.plot(xs, rest, color=REST, linewidth=2, marker='o', markersize=4,
            label='channels only widths > 0.60 have')
    ax.set_xticks(xs)
    ax.set_xticklabels(['s'] + columns[1:], fontsize=7.5)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel('residual block output (s = stem)')
    ax.set_ylabel('fraction dead at width 1.0')
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.set_title('width 1.0: where its dead channels sit', loc='left',
                 fontsize=10, color=INK, pad=14)

fig.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
fig.savefig(out_path, dpi=200, facecolor=SURFACE)
print('wrote', out_path)
