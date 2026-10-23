"""
Undo spurious divisions in an existing FUCCI CSV, without re-segmenting.

Cellpose sometimes splits one nucleus into two fragments for a frame or two.
On area that is indistinguishable from mitosis -- two objects, each about half
the parent, summing to roughly one -- so every geometric test in the tracker
passes and a division is recorded.  What gives it away is what happens next: a
real pair of daughters goes on existing side by side, while a fragment heals
and from then on there is only one object, handed alternately to the two
daughter ids.

This reads a *_fucci.csv, reverts the divisions whose daughters never coexist
for long, merges those tracks back together, and writes a corrected CSV.
Nothing is re-segmented, so it takes seconds.

    python prune_divisions.py A9-2_fucci.csv                  # report only
    python prune_divisions.py A9-2_fucci.csv --write          # fix in place
    python prune_divisions.py A9-2_fucci.csv -o fixed.csv
    python prune_divisions.py A9-2_fucci.csv --min-overlap 4

Start without --write: the report lists every division it would undo and how
many frames the daughters actually coexisted, so you can see the separation
before changing anything.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import fucci_pipeline as fp
import timelapse_tracking as tt

LINEAGE_COLS = ('parent_track_id', 'root_track_id', 'generation',
                'track_end_reason', 'frames_since_division',
                'frames_until_division')


def rebuild(df):
    """(tracks, lineage, areas) from a written CSV.

    Everything the tracker needs is already in the file -- track_id,
    timepoint, cell_label and the lineage columns -- so the structures can be
    reconstructed exactly and fed back through the same pruning code the
    pipeline uses.  No second implementation to drift.
    """
    tracks, areas = {}, {}
    for tp, g in df.groupby('timepoint'):
        tp = int(tp)
        tracks[tp] = {int(r.cell_label): int(r.track_id) for r in g.itertuples()}
        areas[tp] = {int(r.cell_label): float(r.area_px) for r in g.itertuples()}

    lineage = {}
    for tid, g in df.groupby('track_id'):
        first = g.iloc[0]
        par = first.get('parent_track_id')
        lineage[int(tid)] = {
            'parent_track_id': None if pd.isna(par) else int(par),
            'root_track_id': int(first.get('root_track_id', tid))
            if not pd.isna(first.get('root_track_id', np.nan)) else int(tid),
            'generation': int(first.get('generation', 0) or 0),
            'start_tp': int(g.timepoint.min()),
            'end_tp': int(g.timepoint.max()),
            'end_reason': str(first.get('track_end_reason', 'series_end')),
        }
    return tracks, lineage, areas


def coexistence_table(df):
    """One row per two-daughter division: how long the daughters overlap."""
    presence = {int(t): set(g.timepoint) for t, g in df.groupby('track_id')}
    kids = {}
    for tid, g in df.groupby('track_id'):
        par = g.iloc[0].get('parent_track_id')
        if not pd.isna(par):
            kids.setdefault(int(par), []).append(int(tid))
    rows = []
    for par, ks in kids.items():
        if len(ks) != 2:
            continue
        a, b = ks
        t0 = min(min(presence[a]), min(presence[b]))
        rows.append({'parent': par, 'daughter_a': a, 'daughter_b': b,
                     'timepoint': t0,
                     'both_frames': len(presence[a] & presence[b]),
                     'n_a': len(presence[a]), 'n_b': len(presence[b])})
    return pd.DataFrame(rows).sort_values('both_frames') if rows else pd.DataFrame()


def apply_pruning(df, min_overlap=3, log=print, heal_window=0,
                  heal_frac=0.75):
    """Return (corrected_df, reverted) with spurious divisions merged away."""
    tracks, lineage, areas = rebuild(df)
    tracks, lineage, reverted = tt.prune_spurious_divisions(
        tracks, lineage, areas_per_tp=areas, min_overlap=min_overlap, log=log,
        heal_window=heal_window, heal_frac=heal_frac)
    if not reverted:
        return df.copy(), reverted

    timepoints = sorted(tracks)
    timing = fp.division_timing(tracks, lineage, timepoints)

    # Rows whose (timepoint, cell_label) no longer has a track were the
    # fragment half of a frame where both daughters held a label; the larger
    # object was kept and this one is dropped.
    keep, new_tid = [], []
    for r in df.itertuples():
        tid = tracks.get(int(r.timepoint), {}).get(int(r.cell_label))
        keep.append(tid is not None)
        new_tid.append(tid)
    out = df[pd.Series(keep, index=df.index)].copy()
    out['track_id'] = [t for t, k in zip(new_tid, keep) if k]

    info = out.track_id.map(lambda t: lineage.get(int(t), {}))
    out['parent_track_id'] = [i.get('parent_track_id') for i in info]
    out['root_track_id'] = [i.get('root_track_id') for i in info]
    out['generation'] = [i.get('generation', 0) for i in info]
    out['track_end_reason'] = [i.get('end_reason', '') for i in info]
    out['frames_since_division'] = [
        timing.get((int(t), int(tp)), (np.nan, np.nan))[0]
        for t, tp in zip(out.track_id, out.timepoint)]
    out['frames_until_division'] = [
        timing.get((int(t), int(tp)), (np.nan, np.nan))[1]
        for t, tp in zip(out.track_id, out.timepoint)]
    if 'median_area_track' in out.columns:
        med = out.groupby('track_id').area_px.transform('median')
        out['median_area_track'] = med
        if 'area_frac' in out.columns:
            out['area_frac'] = out.area_px / med.replace(0, np.nan)
    return out.sort_values(['timepoint', 'track_id']).reset_index(drop=True), reverted


def main():
    ap = argparse.ArgumentParser(
        description='Undo spurious divisions in a FUCCI CSV, no re-segmentation.')
    ap.add_argument('csv')
    ap.add_argument('-o', '--output', default=None)
    ap.add_argument('--write', action='store_true',
                    help='overwrite the input file')
    ap.add_argument('--min-overlap', type=int, default=3,
                    help='frames the two daughters must both be present for')
    ap.add_argument('--heal-window', type=int, default=8,
                    help='also revert when one daughter is lost within this '
                         'many frames and the other is back to a whole nucleus '
                         '(default 8; 0 disables)')
    ap.add_argument('--heal-frac', type=float, default=0.75)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f'{len(df)} rows, {df.track_id.nunique()} tracks, '
          f'{df.timepoint.nunique()} frames')

    tab = coexistence_table(df)
    if len(tab):
        print(f'\n{len(tab)} two-daughter divisions. Frames on which both '
              f'daughters are present:')
        counts = tab.both_frames.value_counts().sort_index()
        for v, n in counts.items():
            mark = '  <- would be reverted' if v < args.min_overlap else ''
            print(f'   {v:3d} frame(s): {n:3d} division(s){mark}')
        print('\n   A clear gap here means the threshold is doing real work; '
              'no gap\n   means --min-overlap is a judgement call, so check '
              'the overlay.')

    out, reverted = apply_pruning(df, min_overlap=args.min_overlap,
                                  heal_window=args.heal_window,
                                  heal_frac=args.heal_frac)
    if not reverted:
        print('\nnothing to revert.')
        return

    print(f'\n{df.track_id.nunique()} tracks -> {out.track_id.nunique()} tracks, '
          f'{len(df)} rows -> {len(out)} rows')
    before = df.groupby('track_id').size()
    after = out.groupby('track_id').size()
    print(f'median track length {int(before.median())} -> {int(after.median())} '
          f'frames;  longest {before.max()} -> {after.max()}')
    nd_before = (df.groupby('track_id').track_end_reason.first() == 'divided').sum()
    nd_after = (out.groupby('track_id').track_end_reason.first() == 'divided').sum()
    print(f'divisions {nd_before} -> {nd_after}')

    dest = fp.output_path(args.output) or (args.csv if args.write else None)
    if dest:
        out.to_csv(dest, index=False)
        print(f'\nwrote {dest}')
    else:
        print('\n(report only -- pass --write or -o PATH to save)')


if __name__ == '__main__':
    main()
