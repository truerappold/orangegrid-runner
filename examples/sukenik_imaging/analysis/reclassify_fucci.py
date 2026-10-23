"""
Re-call cell-cycle phases from an existing FUCCI CSV, without re-segmenting.

Cellpose and the tracking are the expensive part and neither depends on the
phase thresholds, so there is no reason to re-run them while tuning.  Every
input the classifier uses is already a column in the CSV, so this reads it back,
applies new thresholds, and reports what changed.

    python reclassify_fucci.py A1-1_fucci.csv --venus-high 450 --cherry-low 80
    python reclassify_fucci.py A1-1_fucci.csv --suggest
    python reclassify_fucci.py A1-1_fucci.csv --plot traces.png --tracks 3 7 12
"""

import argparse

import numpy as np
import pandas as pd

import fucci_pipeline as fp


def suggest(df):
    """Where the low and high modes of each marker separate.

    Uses Otsu on the pooled per-cell values -- the same logic as thresholding an
    image, applied to a distribution of cell intensities.
    """
    from skimage.filters import threshold_otsu

    out = {}
    for col, lo_key, hi_key in (('mvenus_p90', 'venus_low', 'venus_high'),
                                ('mcherry_p90', 'cherry_low', 'cherry_high')):
        v = df[col].replace([np.inf, -np.inf], np.nan).dropna().values
        if len(v) < 20:
            continue
        t = float(threshold_otsu(v))
        out[lo_key] = round(t * 0.8, 1)
        out[hi_key] = round(t * 1.2, 1)
        print(f'{col:12s} n={len(v):6d}  Otsu split at {t:8.1f}'
              f'   -> {lo_key}={out[lo_key]}, {hi_key}={out[hi_key]}')
        for p in (5, 25, 50, 75, 95):
            print(f'{"":12s}   {p:>2}% = {np.percentile(v, p):8.1f}')
    print('\nA single Otsu split assumes two modes. Check the plot before '
          'trusting it -- the dark window means the low mode mixes G1-negative '
          'cells with genuinely marker-free early-S cells.')
    return out


def main():
    ap = argparse.ArgumentParser(description='Re-call phases from a FUCCI CSV.')
    ap.add_argument('csv')
    ap.add_argument('--out', default=None, help='write the re-called CSV here')
    ap.add_argument('--suggest', action='store_true',
                    help='propose thresholds from the data and exit')
    for k, v in fp.PHASE_THRESHOLDS.items():
        ap.add_argument('--' + k.replace('_', '-'), type=float, default=None,
                        help=f'default {v}')
    ap.add_argument('--plot', default=None, help='write marker traces here')
    ap.add_argument('--tracks', nargs='*', type=int, default=None,
                    help='track ids to plot (default: the longest few)')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f'{len(df)} rows, {df["track_id"].nunique()} tracks, '
          f'{df["timepoint"].nunique()} frames\n')

    if args.suggest:
        suggest(df)
        return

    thresholds = {k: getattr(args, k) for k in fp.PHASE_THRESHOLDS
                  if getattr(args, k, None) is not None}
    if thresholds:
        print('overriding: ' + ', '.join(f'{k}={v}' for k, v in thresholds.items()))

    before = df['phase'].value_counts() if 'phase' in df else None
    out = fp.classify_table(df, thresholds=thresholds or None)
    after = out['phase'].value_counts()

    print(f'\n{"phase":12s} {"before":>8s} {"after":>8s}')
    for ph in fp.PHASES:
        b = int(before.get(ph, 0)) if before is not None else 0
        print(f'{ph:12s} {b:8d} {int(after.get(ph, 0)):8d}')
    if before is not None:
        changed = int((df['phase'].values != out['phase'].values).sum())
        print(f'\n{changed} of {len(df)} rows changed call '
              f'({100*changed/max(len(df),1):.1f}%)')

    args.plot = fp.output_path(args.plot)
    args.out = fp.output_path(args.out)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        lengths = out.groupby('track_id').size().sort_values(ascending=False)
        ids = args.tracks or list(lengths.index[:4])
        fig, axes = plt.subplots(len(ids), 1, figsize=(9, 2.6 * len(ids)),
                                 squeeze=False, sharex=True)
        colours = {'G1': '#2ca02c', 'earlyS': '#7f7f7f', 'S': '#1f77b4',
                   'G2': '#ff7f0e', 'M': '#d62728', 'ambiguous': '#dddddd'}
        for ax, tid in zip(axes[:, 0], ids):
            t = out[out['track_id'] == tid].sort_values('timepoint')
            ax.plot(t['time_min'] / 60, t['mvenus_p90'], '-o', ms=3,
                    color='#b8a600', label='mVenus p90')
            ax.plot(t['time_min'] / 60, t['mcherry_p90'], '-o', ms=3,
                    color='#c0392b', label='mCherry p90')
            for ph, g in t.groupby('phase'):
                ax.scatter(g['time_min'] / 60,
                           np.full(len(g), -0.06 * t[['mvenus_p90',
                                                      'mcherry_p90']].max().max()),
                           s=28, marker='s', color=colours.get(ph, '#000'),
                           label=ph)
            ax.set_ylabel(f'track {tid}')
            ax.grid(alpha=.15)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        seen, h2, l2 = set(), [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l); h2.append(h); l2.append(l)
        axes[0, 0].legend(h2, l2, frameon=False, fontsize=7, ncol=4)
        axes[-1, 0].set_xlabel('time (h)')
        fig.suptitle('marker traces with phase calls (squares along the bottom)')
        fig.tight_layout()
        fig.savefig(args.plot, dpi=160)
        print(f'wrote {args.plot}')

    if args.out:
        out.to_csv(args.out, index=False)
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
