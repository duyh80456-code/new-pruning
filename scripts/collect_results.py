"""keep results/runs.json and RESULTS.md in step with what has been run

    python scripts/collect_results.py ingest kaggle-kd-fs.ipynb
    python scripts/collect_results.py render

ingest reads the post-calibration validation lines out of an executed
notebook, or a plain log, and adds or replaces those runs. render writes
RESULTS.md from the store. ingest renders too.

The parser looks for the lines calibration prints, which carry epoch -1:

    [f_jeffreys_pair] 22.3s  val  0.65  -1/100: loss: 1.259, top1_error: ...

Only those. The per-epoch validation lines use training-time batch norm
statistics at two widths and are not comparable.
"""
import json
import os
import re
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = os.path.join(ROOT, 'results', 'runs.json')
PAGE = os.path.join(ROOT, 'RESULTS.md')

CAL_LINE = re.compile(
    r'\[(?P<run>[\w.]+)\][^\n]*?\bval\s+(?P<width>[0-9.]+)\s+-1/\d+:\s+'
    r'loss:\s+(?P<loss>[0-9.eE+-]+),\s+top1_error:\s+(?P<top1>[0-9.]+)')


def load():
    with open(STORE, encoding='utf-8') as handle:
        return json.load(handle)


def save(store):
    with open(STORE, 'w', encoding='utf-8') as handle:
        json.dump(store, handle, indent=2, ensure_ascii=False)
        handle.write('\n')


def text_of(path):
    if not path.endswith('.ipynb'):
        with open(path, encoding='utf-8', errors='replace') as handle:
            return handle.read()
    with open(path, encoding='utf-8') as handle:
        notebook = json.load(handle)
    chunks = []
    for cell in notebook['cells']:
        for out in cell.get('outputs', []):
            if 'text' in out:
                chunks.append(''.join(out['text']))
            elif 'data' in out and 'text/plain' in out['data']:
                chunks.append(''.join(out['data']['text/plain']))
    return ''.join(chunks)


def ingest(path):
    store = load()
    widths = store['widths']
    found = {}
    for match in CAL_LINE.finditer(text_of(path)):
        run = match.group('run')
        found.setdefault(run, {})[float(match.group('width'))] = (
            100.0 * (1.0 - float(match.group('top1'))),
            float(match.group('loss')))

    if not found:
        raise SystemExit(
            'no post-calibration validation lines in {}.\n'
            'A run that crashed, or one whose calibration never ran, has '
            'nothing to record.'.format(os.path.basename(path)))

    for run, table in sorted(found.items()):
        missing = [w for w in widths if w not in table]
        if missing:
            print('skipped {}: no value at {}'.format(run, missing))
            continue
        entry = {
            'name': run,
            'letter': '?',
            'what': 'ingested from {}'.format(os.path.basename(path)),
            'top1': [round(table[w][0], 2) for w in widths],
            'nll': [table[w][1] for w in widths],
        }
        existing = [r for r in store['runs'] if r['name'] == run]
        if existing:
            existing[0].update(
                {k: v for k, v in entry.items() if k not in ('letter',
                                                             'what')})
            print('updated {}'.format(run))
        else:
            store['runs'].append(entry)
            print('added   {}  (set its letter and description by hand)'
                  .format(run))
    save(store)



# What each config key does, for the 'How each branch is built' section.
# Keyed by the key alone where the key is the whole story, and by
# key=value where the value picks between mechanisms.
LEGEND = [
    ('kd_loss=wasserstein', 'the vertical KD term becomes entropic '
     'transport over a class cost matrix, instead of soft cross entropy'),
    ('kd_loss=alpha', 'the vertical KD term becomes an adaptive '
     'alpha-divergence, AlphaNet as published'),
    ('kd_temperature', 'the logit teacher is softened by this factor '
     'before the student is matched to it, with a T^2 rescale so the '
     'gradient stays comparable'),
    ('kd_weighting=entropy', 'each sample\'s KD loss is scaled by the '
     'teacher\'s entropy on it, normalised to mean one'),
    ('kd_weighting=confidence', 'the same scaling, reversed: strongest '
     'where the teacher is most certain'),
    ('cost_source', 'where the class cost matrix comes from. `fc` is the '
     'distance between classifier rows, `confusion` is what the teacher '
     'mistakes for what, `identity` is every pair equally far apart'),
    ('feature_kd', 'adds a vertical term between the student\'s pooled '
     'features and the teacher\'s, on top of the logit KD'),
    ('horizontal_kd', 'adds a term between two co-sampled widths that '
     'stand in no teacher relation to each other'),
    ('horizontal_where', 'which tier the horizontal term sits at: '
     '`logit`, `feature`, or `both`'),
    ('horizontal_loss', 'what compares the two widths at the logit tier: '
     '`wasserstein`, `jeffreys`, or `kl`'),
    ('horizontal_pairs=all', 'every pair among the co-sampled students is '
     'coupled, not just the two middle ones'),
    ('feature_loss=sliced', 'transport along random one dimensional '
     'projections instead of a full plan'),
    ('feature_loss=channel', 'exact transport along each shared channel, '
     'one dimension at a time, which the nesting makes meaningful'),
    ('feature_loss=bures', 'the closed form between two Gaussians fitted '
     'to the clouds: mean gap plus a covariance term'),
    ('feature_loss=unbalanced', 'transport that may leave mass unmatched, '
     'the marginals penalised rather than enforced'),
    ('feature_loss=mse', 'sample i against sample i, no rematching'),
    ('feature_loss=mmd', 'the clouds matched with a kernel, no plan'),
    ('feature_align', 'how two widths are put in a comparable space. '
     '`prefix` truncates the wide features to the narrow width, which '
     'the nesting allows; `gram` compares which samples each width '
     'considers similar'),
    ('feature_layers', 'the depths the feature term is applied at, '
     'instead of the final pooled tap alone'),
    ('feature_classwise', 'transport runs between class means rather '
     'than between samples'),
    ('feature_ground=cosine', 'the ground cost is angle rather than '
     'distance'),
    ('feature_weight', 'the weight on the feature terms, set by matching '
     'gradient norms rather than loss values'),
    ('horizontal_weight', 'the weight on the horizontal term'),
    ('weight_schedule=narrow', 'the extra term is faded out toward the '
     'wide widths, where it was measured to hurt'),
    ('sinkhorn_eps', 'the entropic blur. Small collapses the plan onto a '
     'permutation, large spreads mass over many partners'),
    ('sinkhorn_iters', 'fixed point iterations in the Sinkhorn solve'),
    ('sinkhorn_debiased', 'subtracts the self-transport terms, so two '
     'identical clouds read exactly zero'),
    ('wasserstein_p=1.0', 'an absolute ground cost instead of a squared '
     'one, so the gradient does not decay as the widths converge'),
    ('sliced_projections', 'how many random directions stand in for the '
     'full plan'),
    ('sliced_reduce=max', 'keeps the worst projection rather than the '
     'average of them'),
    ('unbalanced_tau', 'the price of leaving mass unmatched. Large '
     'recovers the balanced plan'),
    ('bures_diagonal', 'keeps only the per channel variances and drops '
     'every cross channel term'),
    ('spread_weight', 'a repulsion on the classifier rows, the one term '
     'here with the opposite sign to the rest'),
    ('spread_neighbours', 'how many nearest rows the repulsion charges '
     'for, which makes it a minimum margin rather than a mean spread'),
    ('width_sampling=log', 'the sandwich rule draws its free widths flat '
     'in log width, which puts half of them below 0.50 against a third '
     'for uniform'),
    ('width_sampling=macs', 'the same draw made flat in compute. MACs '
     'measured at width^1.965 here, so this leans toward the wide end'),
    ('teacher_chain', 'each width learns from the next larger one in the '
     'batch instead of every width learning from the widest'),
    ('num_sample_training', 'how many widths are run per step, the '
     'sandwich rule\'s two ends included'),
    ('random_seed', 'the seed, for a repeat of a branch already run'),
    ('confusion_momentum', 'how fast the confusion embedding updates'),
    ('confusion_warmup', 'steps before the confusion cost is trusted'),
    ('cost_normalize', 'divides the class cost by its own mean'),
    ('alpha_min', 'lower end of the adaptive alpha range'),
    ('alpha_max', 'upper end of the adaptive alpha range'),
    ('alpha_iw_clip', 'importance weight clip in the alpha-divergence'),
]


def config_keys(branch):
    """every top level key a branch's config sets"""
    path = os.path.join(ROOT, 'apps', 'cifar100_{}.yml'.format(branch))
    if not os.path.exists(path):
        return None
    found = {}
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.split('#')[0].rstrip()
            if not line or line.startswith(' ') or ':' not in line:
                continue
            key, value = line.split(':', 1)
            found[key.strip()] = value.strip()
    return found


def explain(key, value):
    """the legend entry for one setting, most specific first"""
    for name, text in LEGEND:
        if name == '{}={}'.format(key, value):
            return text
    for name, text in LEGEND:
        if name == key:
            return text
    return None


def mean(values):
    return sum(values) / len(values)


def render():
    store = load()
    widths = store['widths']
    macs = store['macs_millions']
    runs = store['runs']
    by_name = {r['name']: r for r in runs}
    reference = by_name[store['reference']]
    ranked = sorted(runs, key=lambda r: -mean(r['top1']))

    def nll_mean(run):
        return mean(run['nll']) if run.get('nll') else run.get('nll_mean')

    out = []
    w = out.append

    w('# Results')
    w('')
    w('Universally slimmable ResNet-18 on CIFAR-100, sixteen widths from '
      '0.25 to 1.00, accuracy after BN post-statistics. One seed each '
      'unless the row says otherwise. Generated from `results/runs.json` '
      'by `scripts/collect_results.py`.')
    w('')
    w('## Where it stands')
    w('')
    w('| | branch | mean top-1 | vs A | worst width | mean NLL | vs A |')
    w('|---|---|---|---|---|---|---|')
    for run in ranked:
        accuracy = mean(run['top1'])
        nll = nll_mean(run)
        w('| **{}** | {} | **{:.2f}** | {:+.2f} | {:.2f} | {} | {} |'.format(
            run['letter'], run['what'], accuracy,
            accuracy - mean(reference['top1']), min(run['top1']),
            '{:.3f}'.format(nll) if nll else '-',
            '{:+.3f}'.format(nll - nll_mean(reference)) if nll else '-'))
    w('')

    floor = 0.10
    w('## What the numbers can carry')
    w('')
    w('Accuracy was logged to three decimals of error, so every value is '
      'a multiple of 0.10 and each one carries up to 0.05 of rounding. A '
      'difference between two of them carries up to **{:.2f}**, and a '
      'difference of means over sixteen widths roughly 0.01 to 0.03.'
      .format(floor))
    w('')
    a_prime = by_name.get('a_seed2')
    if a_prime:
        gap = abs(mean(a_prime['top1']) - mean(reference['top1']))
        w('A and A\' are the same branch under different seeds. Their means '
          'differ by **{:.2f}**, which is at that floor: what the pair '
          'establishes is that seed noise on the mean sits under it, not '
          'that it equals 0.01. Per width the same pair differs by as much '
          'as **{:.2f}**, so the worst-width column is noise and should '
          'not be ranked.'.format(
              gap, max(abs(x - y) for x, y in
                       zip(a_prime['top1'], reference['top1']))))
        w('')
    w('So the gaps worth reading are the ones far outside that floor: A '
      'over C by {:.2f}, over B by {:.2f}, over D by {:.2f}. F against A, '
      'at {:+.2f}, is not one of them.'.format(
          mean(reference['top1']) - mean(by_name['c_wasserstein']['top1']),
          mean(reference['top1']) - mean(by_name['b_alpha']['top1']),
          mean(reference['top1']) - mean(by_name['d_wasserstein_pair']['top1']),
          mean(by_name['f_jeffreys_pair']['top1'])
          - mean(reference['top1'])))
    w('')
    w('NLL is printed to three decimals on a scale near 1.3, so it '
      'resolves to 0.001 and none of this applies to it.')
    w('')

    w('## Accuracy by width')
    w('')
    header = '| width | MACs (M) |' + ''.join(
        ' {} |'.format(r['letter']) for r in ranked)
    w(header)
    w('|---|---|' + '---|' * len(ranked))
    for index, width in enumerate(widths):
        row = '| {:.2f} | {:.1f} |'.format(width, macs[index])
        best = max(r['top1'][index] for r in ranked)
        for run in ranked:
            value = run['top1'][index]
            row += ' {} |'.format(
                '**{:.2f}**'.format(value) if value == best
                else '{:.2f}'.format(value))
        w(row)
    best = max(mean(r['top1']) for r in ranked)
    row = '| **mean** | {:.1f} |'.format(mean(macs))
    for run in ranked:
        value = mean(run['top1'])
        row += ' {} |'.format(
            '**{:.2f}**'.format(value) if value == best
            else '{:.2f}'.format(value))
    w(row)
    w('')
    w('The mean row is the column each branch is ranked by, which the '
      'table above it could not be read off before.')
    w('')

    w('## Against A, by width')
    w('')
    w('Every branch that has moved at all has moved the same way: ahead at '
      'the narrow widths, behind at the wide ones. Five times here, and '
      'three more in the literature.')
    w('')
    others = [r for r in ranked if r['name'] != reference['name']]
    w('| width |' + ''.join(' {} |'.format(r['letter']) for r in others))
    w('|---|' + '---|' * len(others))
    for index, width in enumerate(widths):
        row = '| {:.2f} |'.format(width)
        for run in others:
            delta = run['top1'][index] - reference['top1'][index]
            row += ' {} |'.format(
                '{:+.2f}'.format(delta) if abs(delta) > floor / 2
                else '·')
        w(row)
    w('')
    w('A dot is a difference at or inside the rounding floor.')
    w('')

    w('## NLL by width')
    w('')
    w('The one place a branch has separated from A by more than the '
      'measurement can be blamed for.')
    w('')
    have_nll = [r for r in ranked if r.get('nll')]
    w('| width |' + ''.join(' {} |'.format(r['letter']) for r in have_nll))
    w('|---|' + '---|' * len(have_nll))
    for index, width in enumerate(widths):
        row = '| {:.2f} |'.format(width)
        best = min(r['nll'][index] for r in have_nll)
        for run in have_nll:
            value = run['nll'][index]
            row += ' {} |'.format(
                '**{:.3f}**'.format(value) if value == best
                else '{:.3f}'.format(value))
        w(row)
    lowest = min(mean(r['nll']) for r in have_nll)
    row = '| **mean** |'
    for run in have_nll:
        value = mean(run['nll'])
        row += ' {} |'.format(
            '**{:.3f}**'.format(value) if value == lowest
            else '{:.3f}'.format(value))
    w(row)
    w('')
    w("A's NLL climbs from {:.3f} at width 0.25 to {:.3f} at 1.00: it is "
      'least calibrated where it is most accurate, which is what a '
      'training error of 0.000 at the widest width predicts. F does not '
      'do that.'.format(reference['nll'][0], reference['nll'][-1]))
    w('')

    w('## How each branch is built')
    w('')
    w('Each row names the closest branch above it and lists only what '
      'differs, so the line that makes a branch itself is the line you '
      'read. Diffed against A, most of this table would be K repeated '
      'ten times with the distinguishing setting arriving last.')
    w('')
    w('Read out of `apps/cifar100_<name>.yml` when this page was '
      'written, so a branch cannot be described here as something its '
      'config has stopped being.')
    # Brackets are only worth pointing at once both ends have run.
    BRACKETS = [('w_eps_002', 'y_eps_050', 'blur'),
                ('as_log_widths', 'at_macs_widths',
                 'where the free widths are drawn'),
                ('au_entropy_kd', 'av_confidence_kd',
                 'which samples KD attends to'),
                ('ac_weight_half', 'ad_weight_double',
                 'the weight on the feature terms')]
    # A bracket only reads as a bracket if both ends actually trained.
    # An end that collapsed says nothing about the axis, so naming it
    # beside the working pairs would claim an answer that is not there.
    def alive(name):
        return mean(by_name[name]['top1']) > mean(reference['top1']) - 5.0

    have = [(a, b, what) for a, b, what in BRACKETS
            if a in by_name and b in by_name]
    pairs = ['{} against {} on {}'.format(
        by_name[a]['letter'], by_name[b]['letter'], what)
        for a, b, what in have if alive(a) and alive(b)]
    broken = ['{} against {} on {}, where {} collapsed'.format(
        by_name[a]['letter'], by_name[b]['letter'], what,
        by_name[a]['letter'] if not alive(a) else by_name[b]['letter'])
        for a, b, what in have if not (alive(a) and alive(b))]
    if pairs:
        w('')
        w('Some branches are only readable in pairs, one leaning each '
          'way from K: {}. A bracket where both ends win says the axis '
          'does not matter, which is an answer the winning end alone '
          'cannot give.'.format('; '.join(pairs)))
    if broken:
        w('')
        w('One bracket does not close: {}. A collapsed end is not a '
          'losing end, so it cannot stand as the control its pair '
          'needed, and the axis stays open.'.format('; '.join(broken)))
    w('')
    # Diffed against the nearest of a few ancestors rather than always
    # against A. Most of the table is K with one line changed, and
    # against A that line arrives tenth.
    ancestors = [(name, config_keys(name)) for name in
                 ('k_feature_pair', 'i_feature_vertical', 'c_wasserstein',
                  reference['name'])]
    ancestors = [(n, k) for n, k in ancestors if k]
    letter_of = {r['name']: r['letter'] for r in runs}

    def diff(settings, base):
        return [(k, v) for k, v in sorted(settings.items())
                if base.get(k) != v and k != 'log_dir']

    for run in ranked:
        settings = config_keys(run['name'])
        if settings is None:
            continue
        best_name, best_changed = None, None
        for name, base in ancestors:
            if name == run['name']:
                continue
            changed = diff(settings, base)
            if best_changed is None or len(changed) < len(best_changed):
                best_name, best_changed = name, changed
        if not best_changed:
            w('**{}** - {}. Identical to {} in config; they differ by '
              'seed.'.format(run['letter'], run['what'],
                             letter_of.get(best_name, best_name)))
            w('')
            continue
        if run['name'] == reference['name']:
            w('**{}** - {}. The reference every row below is measured '
              'against.'.format(run['letter'], run['what']))
            w('')
            continue
        w('**{}** - {}'.format(run['letter'], run['what']))
        w('')
        w('{}, with:'.format(letter_of.get(best_name, best_name)))
        w('')
        for key, value in best_changed:
            text = explain(key, value)
            w('* `{}: {}`{}'.format(
                key, value, ' - ' + text if text else ''))
        w('')

    w('## Which channels the prefix holds')
    w('')
    w('US-Net slices channels as `weight[:k]`, so which channels a '
      'narrow width gets is decided by initialisation and never '
      'revisited. Sorting them by importance, the way Once-for-All '
      'does for elastic width, is the obvious thing to try, and it '
      'would have cost a session.')
    w('')
    w('`scripts/channel_order.py` reads a checkpoint and compares the '
      'importance mass the first k channels hold against the mass the '
      'best k would hold. The control is the same checkpoint with each '
      "layer's output channels shuffled, so the shapes and the values "
      'are identical and only the order is destroyed. Read on K, at '
      'width 0.25:')
    w('')
    w('| criterion | shuffled | as trained | best possible | recovered |')
    w('|---|---|---|---|---|')
    w('| L1 of the conv filter | 0.247 | 0.468 | 0.472 | 99% |')
    w('| L2 of the conv filter | 0.247 | 0.520 | 0.522 | 99% |')
    w('| absolute batch-norm gain | 0.248 | 0.303 | 0.357 | 51% |')
    w('| read by the next layer | 0.247 | 0.474 | 0.475 | 100% |')
    w('| distance from the other filters | 0.249 | 0.388 | 0.389 | 99% |')
    w('| Fisher at width 1.00 | 0.250 | 0.798 | 0.835 | 94% |')
    w('| Taylor at width 1.00 | 0.259 | 0.603 | 0.626 | 94% |')
    w('| Taylor at width 0.50 | 0.248 | 0.753 | 0.770 | 97% |')
    w('')
    w('Seven of the eight say the sorting is done. The one that does '
      'not is the batch-norm gain, and it is also the one that agrees '
      'least with the others: ranked against the Taylor score, every '
      'other criterion correlates between 0.89 and 0.95 and the gain '
      'manages 0.603, falling to 0.127 in one layer. Being both the '
      'outlier in the answer and the outlier in the ordering is what a '
      'bad measure looks like, not a revealing one.')
    w('')
    w('The L1 of a conv filter is not durable in principle: scale a '
      'filter by c, divide the gain of that channel by c, and the '
      'function is unchanged, because the normalisation removes the '
      'scale. For one commit this page treated that as a reason to '
      'disbelieve the L1 row and believe the gain. It is not. Training '
      'does not wander through that freedom, because `train.py:267` '
      'decays the conv weights and `train.py:271` leaves every '
      'one-dimensional parameter alone, so the conv scale is pinned '
      'and the gain is the one left to drift. Which is why L1 tracks '
      'Taylor at 0.892 here and the gain does not, and why Once-for-All '
      'can sort by L1 and get a usable order out of it.')
    w('')
    w('There is a second reason, and it is in this repository rather '
      'than in the measurement. Ranking channels by the batch-norm '
      'gain is Network Slimming, and that method trains with an L1 '
      'penalty on the gains, which is what drives them apart and '
      'makes the small ones mean something. `train.py:271` gives '
      'every one-dimensional parameter a weight decay of zero, so the '
      'gains here are trained under no penalty at all. The criterion '
      'is being read outside the regime it was built for.')
    w('')
    w('The row above it says the rest. The gain records how far a channel '
      'is turned up and nothing about whether anything downstream '
      'reads it; how much the next layer reads each channel is a '
      'separate quantity, and that one is sorted to 100%. A channel '
      'can be quiet and still matter. The Taylor score multiplies the '
      'gain by the gradient reaching it and so carries both halves, '
      'which is why it lands with the majority.')
    w('')
    w('One of the eight is not asking about importance at all. '
      'Distance from the other filters is the FPGM criterion, which '
      'ranks a channel by how much of a duplicate it is, and the '
      'sandwich rule presses directly for a prefix that is good and '
      'only indirectly for one that is varied. It was the likeliest '
      'place to find headroom and it reads 99%.')
    w('')
    w('The widths agree on the order too, which was the objection '
      'that would have closed the idea for every criterion at once. '
      'Rank correlation between the Taylor orderings is 0.905 between '
      'widths 1.00 and 0.50, 0.882 between 1.00 and 0.25, and 0.981 '
      'between 0.50 and 0.25, with no layer below 0.74. That is three '
      'of the sixteen widths, not all of them, though the three '
      'include both ends of the range and the 1.00 against 0.25 pair '
      'is the one with the most room to disagree.')
    w('')
    w('Every row above is read on a finished checkpoint, which cannot '
      'say when the prefix got that way. BH and BI print the same '
      'overlap at the top of every epoch, and they hold the narrow end '
      'back for the first 10 and 25 epochs, so between them they '
      'measure it. Under L1, with only width 1.00 training:')
    w('')
    w('| epoch | BH, warm-up of 10 | BI, warm-up of 25 |')
    w('|---|---|---|')
    w('| 1 | 0.254 | 0.254 |')
    w('| 5 | 0.226 | 0.232 |')
    w('| 10 | 0.231, permuted after | 0.244 |')
    w('| 25 | - | 0.231, permuted after |')
    w('| the epoch after the permutation | 0.844 | 0.830 |')
    w('| 99 | 0.991 | 0.994 |')
    w('')
    w('Twenty five epochs at full width leave the ordering where '
      'initialisation put it. It does not drift up and it does not '
      'drift down: every reading in both warm-ups sits between 0.216 '
      'and 0.254, against a floor of 0.25 at this fraction. So the 99% '
      'a finished K reads is not what training does to a network, it '
      'is what the narrow widths do to it, and the window these '
      'branches were built to open stays open for at least 25 epochs.')
    w('')
    w('The jump is the other half. Both permutations rank by Taylor '
      'and the column reports L1, so the reading has no obligation to '
      'move at all, and it goes to 0.84 at once. That is the two '
      'criteria agreeing on a mid-run model about as well as the 0.892 '
      'rank correlation on a finished one.')
    w('')
    w('None of that says the sorting is worth doing. BH and BI land at '
      '73.71 and 73.95, below K at 74.26, and their narrow widths '
      'trained for 90 and 75 epochs rather than 99. BF and BG take the '
      'same two warm-ups and do not permute, and until they run there '
      'is no way to tell the warm-up apart from the sort.')
    w('')
    w('Nothing here is sorted, and saying the sandwich rule sorts the '
      'channels gets the mechanism backwards. The index mapping is '
      'fixed before the first step and never moves: width 0.25 is '
      'channels 0 to 127 at the start and at the end. What changes is '
      'what those positions learn. A channel in the prefix runs at '
      'every sampled width, and at 0.25 the prefix has to classify '
      'with nothing else, while a channel at index 400 is only ever '
      'alive above width 0.78 and is never asked to stand alone. The '
      'prefix does not collect the important channels. It grows them.')
    w('')
    w('Which is why the idea inverts here and not at Once-for-All. '
      'There, width is the last axis to be made elastic, so until '
      'that stage every channel lives at every step and no position is '
      'asked for anything the others are not. The ordering really is '
      'arbitrary, the shuffled floor of 25 per cent is what it looks '
      'like, and a permutation has to supply what training did not. '
      'Here there is nothing to move, and moving anyway would take a '
      'channel that grew up in the tail, trained only at wide widths, '
      'and ask it to work inside a subnet it has never run in.')
    w('')
    w('Six per cent of the available ordering is left at width 1.00 '
      'and three at 0.50, and no criterion tried here finds more. '
      'This section said closed, then open, then closed again, and '
      'the middle reading rested on the single measure that the other '
      'seven contradict.')
    w('')

    w('## Not settled')
    w('')
    w('- Whether F is ahead of A at all. The accuracy gap is at the '
      'rounding floor; only the NLL gap is outside it.')
    w('- What F is doing, exactly. It has symmetry and it has a coupling '
      'between the two middle widths. E, the same term as plain KL, is '
      'what separates those.')
    w('- Sigma, properly. Two seeds put it under the floor; three would '
      'make it a number worth quoting. It has stopped being a '
      'precaution: the four branches below decide their own reading on '
      'it.')
    w('- Why AV collapsed. Everything at width 0.70 and above '
      'trained; everything at 0.65 and below sits at exactly chance, '
      '1.00 accuracy and NLL ln(100). A cliff, not a slope, which is '
      'the shape AJ and AN already showed. What has been measured is '
      'that the weight itself does not explode: across teacher '
      'sharpness from uniform to memorised the per-sample weight stays '
      'between 0 and about 4 with mean 1, so a blown-up KD term is '
      'ruled out rather than merely unlikely. What has not been found '
      'is the path from that weighting to dead leading channels. It is '
      'undiagnosed, not explained.')
    w('- One defect the probe did find, which is not yet shown to be '
      'the cause. AV measures confidence as `entropy.max() - entropy`, '
      'taking its zero point from a batch order statistic that itself '
      'drifts to zero as the teacher memorises. When every entropy in '
      'a batch underflows to exactly zero both AU and AV divide zero '
      'by zero and hand KD a weight of exactly zero for every sample. '
      'The fixed reference the quantity actually has, log(C) minus '
      'entropy, survives that case and does not move with the batch. '
      'A rerun of this axis should use it.')
    w('- Whether AW is ahead of K, and the table should not be read '
      'as saying it is. AW is first at 74.32 against 74.26, a gap of '
      '0.06 where a difference of means over sixteen widths carries '
      '0.01 to 0.03 of rounding alone. The number that settles it is '
      'not the mean: AW is ahead of K at eight widths out of sixteen, '
      'behind at eight, scattered from -0.26 to +0.71. A real gain '
      'does not look like that. K against A is ahead at all sixteen, '
      'which is what one does look like. On this evidence AW ties K '
      'and the ordering between them is a coin.')
    w('- What AW is still worth. It reaches K from somewhere else - it '
      'changes which width teaches which and never touches the '
      'transport term - and it is the first branch off that axis to '
      'get there. Its NLL is 1.091 against 1.127, about twice the '
      'gap two seeds of A showed, which is weak but points the same '
      'way. Two mechanisms arriving at the same place is worth a '
      'second seed on both, which is the run to do next.')
    w('- Whether K sits on a peak or on a high draw. AS, AT, AC and AD '
      'are K with one knob moved in four different directions - the '
      'free widths drawn narrower, the free widths drawn wider, the '
      'feature weight halved, the feature weight doubled - and all four '
      'land between 73.82 and 73.96, which is 0.30 to 0.44 below K. '
      'Four perturbations that share nothing mechanically do not '
      'usually agree by accident, so either 1.0 and a uniform draw are '
      'both genuinely best, or the 74.26 for K is a high seed and the '
      'branch '
      'is really worth about 73.9. Both readings fit this table. Only '
      'sigma separates them, and it is the same measurement the bullet '
      'above asks for.')
    w('')
    w('- And BH and BI make it six. They are K with a warm-up and a '
      'Taylor permutation, which shares nothing mechanically with a '
      'free-width draw or a feature weight, and they land at 73.71 and '
      '73.95. Six perturbations of K, none of them related to any '
      'other, now sit between 73.71 and 73.96 while K sits at 74.26 '
      'alone. That is the strongest evidence in this table for the '
      'second reading, that K drew high and this family is worth about '
      '73.9. It is not proof: BH and BI also trained their narrow '
      'widths for 90 and 75 epochs rather than 99, so they had a '
      'reason to land low that the other four did not. The measurement '
      'that would settle it is still a second seed of K.')
    w('')

    with open(PAGE, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(out))
    print('wrote {} ({} runs)'.format(
        os.path.relpath(PAGE, ROOT), len(runs)))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('ingest', 'render'):
        print(__doc__)
        return 1
    if sys.argv[1] == 'ingest':
        if len(sys.argv) < 3:
            print('ingest needs a notebook or log to read')
            return 1
        for path in sys.argv[2:]:
            ingest(path)
    render()
    return 0


if __name__ == '__main__':
    sys.exit(main())
