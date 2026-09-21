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
    w('')
    w("A's NLL climbs from {:.3f} at width 0.25 to {:.3f} at 1.00: it is "
      'least calibrated where it is most accurate, which is what a '
      'training error of 0.000 at the widest width predicts. F does not '
      'do that.'.format(reference['nll'][0], reference['nll'][-1]))
    w('')

    w('## Not settled')
    w('')
    w('- Whether F is ahead of A at all. The accuracy gap is at the '
      'rounding floor; only the NLL gap is outside it.')
    w('- What F is doing, exactly. It has symmetry and it has a coupling '
      'between the two middle widths. E, the same term as plain KL, is '
      'what separates those.')
    w('- Sigma, properly. Two seeds put it under the floor; three would '
      'make it a number worth quoting.')
    w('- Everything at the feature tier. Those four branches died in the '
      'profiler on their first run and have not been rerun.')
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
