"""
Work out why cells stop tracking, from a timelapse_match_distances.csv.

Two things this exists to prevent you from being misled by:

1. `cell_label` is NOT a cell identity.  Every timepoint is segmented
   independently and relabelled 1..N, so label 26 at t2 and label 26 at t15 are
   unrelated objects.  Only `track_id` persists.  (An earlier version of this
   script keyed continuity on cell_label and produced nonsense.)

2. A percentile of the distance column is not an event detector.  Unmatched
   cells accumulate, so once the unmatched fraction crosses 10% the p90 leaps
   from the true-match cluster (~1 px) into the spurious tail (~150 px) with
   nothing whatever having happened in the images at that frame.  Read the
   survival curve, not the p90.

    python diagnose_drops.py <timelapse_match_distances.csv> [--well B1 --tile 1]
"""

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning)


def main():
    ap = argparse.ArgumentParser(description='Diagnose why tracks are lost.')
    ap.add_argument('distances_csv')
    ap.add_argument('--well', default=None)
    ap.add_argument('--tile', type=int, default=None)
    ap.add_argument('--big-dist', type=float, default=25.0,
                    help='distance above which a match is considered spurious '
                         'rather than real, for the nuclear-vs-cell comparison')
    args = ap.parse_args()

    df = pd.read_csv(args.distances_csv).replace([np.inf, -np.inf], np.nan)
    if args.well:
        df = df[df['well'] == args.well]
    if args.tile is not None:
        df = df[df['tile'] == args.tile]
    if df.empty:
        raise SystemExit('no records after filtering')

    print(f'{len(df)} records, timepoints {df["timepoint"].min()}'
          f'..{df["timepoint"].max()}\n')

    # ---- ledger, keyed on track_id (the only persistent identity) -----------
    rows = []
    prev_live = None
    for tp, g in df.groupby('timepoint', sort=True):
        live = set(g.loc[g['track_id'].notna(), 'track_id'].astype(int))
        n_seg = len(g)
        unmatched = n_seg - len(live)
        rec = {'timepoint': tp, 'n_segmented': n_seg, 'n_tracked': len(live),
               'unmatched': unmatched,
               'pct_unmatched': 100 * unmatched / max(n_seg, 1),
               'tracks_lost_here': (len(prev_live - live)
                                    if prev_live is not None else 0),
               'median_dist': g['match_dist'].median(),
               'p90_dist': g['match_dist'].quantile(.90)}
        rows.append(rec)
        prev_live = live
    led = pd.DataFrame(rows).set_index('timepoint')
    print('per-timepoint ledger  (tracks_lost_here = tracks alive at t-1 but not at t)')
    print(led.to_string(float_format=lambda v: f'{v:.1f}'))

    # ---- survival, the number that actually matters -------------------------
    first, last = led['n_tracked'].iloc[0], led['n_tracked'].iloc[-1]
    steps = len(led) - 1
    print(f'\ntrack survival: {first} -> {last} over {steps} steps = '
          f'{100 * last / max(first, 1):.0f}% retained')
    if steps and first:
        print(f'implied per-frame retention: '
              f'{(last / first) ** (1 / steps) * 100:.1f}%')
    worst = led['tracks_lost_here'].idxmax()
    print(f'worst single frame for track loss: t{worst} '
          f'({int(led.loc[worst, "tracks_lost_here"])} tracks)')

    # ---- explain any p90 step change ---------------------------------------
    cross = led[led['pct_unmatched'] > 10]
    if len(cross) and led['pct_unmatched'].iloc[0] <= 10:
        tc = cross.index[0]
        print(f'\nNOTE: the unmatched fraction crosses 10% at t{tc}, so the p90 '
              f'column jumps there by construction — it stops reporting real '
              f'matches and starts reporting the distance to an unrelated cell. '
              f'That is a property of the statistic, not of the images. '
              f'tracks_lost_here is the column to read.')

    # ---- nuclear vs whole-cell movement ------------------------------------
    if 'match_dist_cell' in df.columns and df['match_dist_cell'].notna().any():
        sub = df[df['match_dist'].notna() & df['match_dist_cell'].notna()]
        big = sub[sub['match_dist'] > args.big_dist]
        print(f'\n{len(big)} records with nuclear distance > {args.big_dist:g} px:')
        if len(big):
            mn, mc = big['match_dist'].median(), big['match_dist_cell'].median()
            print(f'  nuclear centroid moved  median {mn:.1f} px')
            print(f'  cell centroid moved     median {mc:.1f} px')
            if mc > 0 and mn / mc > 3:
                print(f'  -> nuclear jumps ~{mn / mc:.0f}x further than the cell: the '
                      f'cells are not moving, the NUCLEUS is being reassigned '
                      f'(merged Cellpose mask holding two nuclei, or a boundary '
                      f'shift handing a nucleus to a neighbour). Fix refine_nuclei.')
            else:
                print('  -> both centroids moved together, so these are genuinely '
                      'different objects: new cells with no predecessor, not '
                      'mis-assigned ones.')
    else:
        print(f'\nmatch_dist_cell is empty — your CSV predates the fix that '
              f'computes it for unmatched cells too. Re-run maskPrint_timelapse.py '
              f'to get the nuclear-vs-cell comparison.')

    print(f'\nBy design only timepoint 1 seeds tracks, so a cell that first '
          f'appears later can never be tracked and will be reported unmatched at '
          f'every subsequent frame. With {led["n_segmented"].min()}-'
          f'{led["n_segmented"].max()} cells segmented per frame, some of the '
          f'unmatched count is that, not failure.')


if __name__ == '__main__':
    main()
