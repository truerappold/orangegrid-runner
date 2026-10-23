"""
NeoPIP-FUCCI timelapse analysis -- ONE well/tile at a time.

Segments and tracks cells on brightfield, measures nuclear mCherry and mVenus in
every frame, follows divisions, and writes a long-format CSV with one row per
cell per frame carrying the cell-cycle classifiers.

    # see what it does on one tile, a few frames at a time
    python analyzeDirFUCCI.py <experiment_dir> --well A1 --tile 1 --max-frames 10 -v

    # the whole tile
    python analyzeDirFUCCI.py <experiment_dir> --well A1 --tile 1

    # list what is available, and which wells have a condition
    python analyzeDirFUCCI.py <experiment_dir> --list

<experiment_dir> follows the same convention as the other analysis scripts:

    26-10-03_DoubleThymidine_1/
        Pics/          the .vsi files
        seqs.csv       one row per well, first column the well id

If there is no Pics/ subfolder the .vsi files are read from <experiment_dir>
itself, so a flat folder still works without renaming anything.  seqs.csv is
optional: without it the run proceeds and the condition columns are left blank.
EVERY column of seqs.csv is copied onto every row of the output, so it does not
matter whether you call the column "construct", "condition" or anything else.

Tune in this order:
  1  --max-frames 10 and check the per-frame cell counts look sane
  2  set BACKGROUND in fucci_pipeline.py from an empty region of your images
  3  run a full tile, plot mvenus_p90 / mcherry_p90 against time for long
     tracks, and set PHASE_THRESHOLDS where the populations separate
  4  --max-dist from the printed centroid-movement summary

Output columns, grouped:
  identity      well, tile, track_id, timepoint, time_min, cell_label
  lineage       parent_track_id, root_track_id, generation, track_end_reason,
                frames_since_division, frames_until_division
  morphology    area_px, area_um2, perimeter_px, circularity, solidity,
                eccentricity, axis_major_px, axis_minor_px, area_frac,
                median_area_track, centroid_row, centroid_col
  intensity     mcherry_* and mvenus_* as mean/median/p75/p90/max/total,
                cherry_venus_ratio, bf_mean
  call          phase

Everything the phase call uses is also a column, so the call can be recomputed
downstream without re-running the pipeline -- see reclassify_fucci.py.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

import fucci_pipeline as fp
import fucci_imaging as fi
import timelapse_tracking as tt


def resolve_pics_dir(experiment_dir, pics_subdir):
    """<exp>/Pics if it exists, else <exp> itself.

    Keeps the convention used by the other scripts without forcing a reorganise:
    a flat folder of .vsi files still works unchanged.
    """
    if pics_subdir:
        candidate = os.path.join(experiment_dir, pics_subdir)
        if os.path.isdir(candidate):
            return candidate
    return experiment_dir


def load_seqs(experiment_dir, filename='seqs.csv'):
    """{well: {column: value}} from seqs.csv, or {} if there is none.

    The well id is taken from the first column whatever it is called.  All other
    columns are carried through verbatim, so the condition column can be named
    anything.
    """
    path = os.path.join(experiment_dir, filename)
    if not os.path.exists(path):
        return {}, None
    df = pd.read_csv(path)
    if df.empty or df.shape[1] < 1:
        return {}, path
    well_col = df.columns[0]
    df[well_col] = df[well_col].astype(str).str.strip()
    out = {}
    for _, row in df.iterrows():
        well = row[well_col]
        out[well] = {c: row[c] for c in df.columns if c != well_col}
    return out, path


def summarise(df, res, log=print, args_max_gap=0):
    """The printouts that make a test run worth doing."""
    log('')
    log('=' * 68)
    n_tracks = df['track_id'].nunique()
    log(f'tracks: {n_tracks}   rows: {len(df)}   frames: {res["n_frames"]}')

    lengths = df.groupby('track_id').size()
    log(f'track length (frames): min {lengths.min()}, median '
        f'{int(lengths.median())}, max {lengths.max()}')
    full = int((lengths == res['n_frames']).sum())
    log(f'tracks spanning every frame: {full} / {n_tracks}')

    ends = {}
    for info in res['lineage'].values():
        ends[info['end_reason']] = ends.get(info['end_reason'], 0) + 1
    log('track endings: ' + ', '.join(f'{k}={v}' for k, v in sorted(ends.items())))

    divisions = sum(1 for i in res['lineage'].values()
                    if i['end_reason'] == 'divided')
    daughters = sum(1 for i in res['lineage'].values()
                    if i.get('parent_track_id') is not None)
    maxgen = max((i.get('generation', 0) for i in res['lineage'].values()),
                 default=0)
    log(f'divisions observed: {divisions}  (producing {daughters} daughter '
        f'tracks; deepest generation {maxgen})')

    # A division that misses a threshold is otherwise indistinguishable from a
    # tracking collision: both just drop the cells.  If mitoses are being
    # missed, the candidates are almost always here, with the test that killed
    # them -- so this is the thing to read before touching Cellpose.
    rev = res.get('division_reverted') or []
    if rev:
        log(f'divisions REVERTED as one nucleus split in two: {len(rev)}  '
            f'(daughters never coexisted long enough)')
        for r in rev[:6]:
            log('   ' + tt.describe_reverted(r))
        if len(rev) > 6:
            log(f'   ... and {len(rev) - 6} more')

    rejects = res.get('division_rejects') or []
    if rejects:
        by_reason = {}
        for rj in rejects:
            by_reason.setdefault(rj['reason'], []).append(rj)
        log(f'division candidates REJECTED: {len(rejects)}  (' + ', '.join(
            f'{k}={len(v)}' for k, v in sorted(by_reason.items())) + ')')
        for reason, group in sorted(by_reason.items()):
            log(f'   {reason}:')
            for rj in group[:6]:
                detail = ', '.join(f'{k}={v}' for k, v in rj.items()
                                   if k not in ('reason', 'parent', 'daughters',
                                                'timepoint'))
                log(f'     t{rj["timepoint"]} parent {rj["parent"]} -> '
                    f'{rj["daughters"]}  {detail}')
            if len(group) > 6:
                log(f'     ... and {len(group) - 6} more')
        hints = {
            'daughter_frac': 'widen --daughter-frac (e.g. 0.20 0.95)',
            'sum_frac': 'widen --daughter-sum-frac (e.g. 0.50 2.00)',
            'symmetry': 'lower --daughter-symmetry (e.g. 0.25)',
            'distance': 'raise --division-max-dist',
            'midpoint': 'raise --division-midpoint (e.g. 0.8) if these are '
                        'real divisions',
            'no_area': 'areas missing -- this is a bug, tell me',
        }
        log('   -> ' + '; '.join(hints[r] for r in sorted(by_reason)
                                 if r in hints))
    elif divisions == 0:
        log('   no division candidates were even proposed: a mitosis has to '
            'show up as TWO cells claiming the same parent, which needs '
            '--division-max-dist small enough that a daughter is nearest its '
            'own parent and not some neighbour')

    gaps = [int(r['gap']) for r in res['records']
            if r.get('event') == 'matched' and r.get('gap')]
    if gaps:
        hist = {}
        for g in gaps:
            hist[g] = hist.get(g, 0) + 1
        log(f'tracks reconnected across a gap: {len(gaps)}  ('
            + ', '.join(f'{k} frame{"s" if k > 1 else ""}={v}'
                        for k, v in sorted(hist.items())) + ')')
        if args_max_gap and max(hist) >= args_max_gap:
            log(f'   NOTE: reconnections reach the --max-gap limit of '
                f'{args_max_gap}; raise it to see whether more cells come back')
    elif args_max_gap:
        log(f'no track was reconnected across a gap (--max-gap '
            f'{args_max_gap}) -- if cells are still being lost, they are '
            f'being lost to distance or collision, not to going dark')

    rec = pd.DataFrame(res['records'])
    if len(rec) and 'match_dist' in rec:
        d = rec.loc[rec['timepoint'] > 1, 'match_dist'].replace(
            [np.inf, -np.inf], np.nan).dropna()
        if len(d):
            log('centroid movement between frames (px): ' + '  '.join(
                f'{p}%={d.quantile(p/100):.1f}' for p in (50, 75, 90, 95, 99)))
            log('  -> set --max-dist in the gap between the real matches and '
                'the spurious tail')
        dropped = rec[rec['dropped']]
        if len(dropped):
            log('cells dropped by reason: ' + ', '.join(
                f'{k}={v}' for k, v in
                dropped['drop_reason'].value_counts().items()))
            if (dropped['drop_reason'] == 'refractory').any():
                log('   refractory = an extra object next to a cell that '
                    'divided less than --min-division-interval frames ago; '
                    'treated as a Cellpose split, not a second mitosis')

    if res.get('fuse_totals'):
        ft = res['fuse_totals']
        log('dual segmentation, summed over all frames: '
            + ', '.join(f'{k}={v}' for k, v in ft.items()))
        log('   merged = brightfield pieces joined because one nucleus spanned '
            'them; split = one brightfield object holding two nuclei; '
            'fluor_only = nucleus brightfield missed; dropped_small = '
            'nucleus-less debris')
    if 'nuc_source' in df.columns:
        vc = df['nuc_source'].value_counts()
        log('nucleus each row was measured on: ' + ', '.join(
            f'{k}={v} ({100 * v / len(df):.0f}%)' for k, v in vc.items()))
        log('   carried = this track\'s nucleus shape from the nearest frame '
            'it was segmented, moved with the cell; disc = no segmented '
            'nucleus anywhere in the track')
    if 'division_kind' in df.columns:
        kinds = {}
        for info in res['lineage'].values():
            k = info.get('division_kind')
            if k:
                kinds[k] = kinds.get(k, 0) + 1
        if kinds:
            log('daughter tracks by how they were found: ' + ', '.join(
                f'{k}={v}' for k, v in sorted(kinds.items())))
            log('   pair = both daughters seen; single = a mitotic-flagged cell '
                'flattened out at daughter size with no sibling found; '
                'late_sibling = that sibling turned up afterwards')
    if 'mitotic_flag' in df.columns:
        n_flag = df.loc[df.mitotic_flag.astype(bool), 'track_id'].nunique()
        log(f'tracks flagged mitotic (round and shrunk) at least once: {n_flag}')

    if 'mitosis_no_division' in df.columns:
        sus = df[df.mitosis_no_division].groupby('track_id').mitosis_seen_tp.first()
        if len(sus):
            log(f'tracks that looked mitotic but never divided '
                f'(mitosis_no_division; probably a missed daughter): {len(sus)}: '
                + ', '.join(f'{int(t)} (round t{int(v)})' for t, v in sus.items()))

    if 'max_step_px' in df.columns:
        per = df.groupby('track_id')[['max_msd_px2', 'max_step_px']].first()
        log('motion per track (filter on these columns): largest MSD px^2 '
            + '  '.join(f'p{q}={per.max_msd_px2.quantile(q / 100):.0f}'
                        for q in (50, 90, 99))
            + ' | largest jump px '
            + '  '.join(f'p{q}={per.max_step_px.quantile(q / 100):.0f}'
                        for q in (50, 90, 99)))
        op = df['off_path_px'].dropna()
        if len(op):
            log('frames off their own path (off_path_px; slips / misplaced '
                'nuclei): ' + '  '.join(f'p{q}={op.quantile(q / 100):.0f}'
                                        for q in (50, 99))
                + f'; {int((op > 40).sum())} frame(s) > 40 px')
        log('   histogram + cutoff: python diagnose_tracks.py <this csv> '
            '--off-path-cutoff N --step-cutoff N')

    if 'mask_source' in df.columns:
        n_imp = int((df.mask_source == 'interpolated').sum())
        if n_imp:
            log(f'rows measured on an interpolated position: {n_imp} '
                f'({100 * n_imp / len(df):.1f}% of all rows) -- these have '
                f'intensities but no shape')

    log('')
    log('phase calls:')
    counts = df['phase'].value_counts()
    for ph in fp.PHASES:
        n = int(counts.get(ph, 0))
        log(f'   {ph:10s} {n:6d}  {100*n/max(len(df),1):5.1f}%')
    amb = 100 * int(counts.get('ambiguous', 0)) / max(len(df), 1)
    if amb > 25:
        log(f'   NOTE: {amb:.0f}% ambiguous -- PHASE_THRESHOLDS and BACKGROUND '
            f'in fucci_pipeline.py almost certainly need setting for your data')

    log('')
    log('intensity distributions (background-subtracted whole-cell p90):')
    for col in ('mvenus_p90', 'mcherry_p90'):
        v = df[col].dropna()
        if len(v):
            log(f'   {col:12s} ' + '  '.join(
                f'{p}%={v.quantile(p/100):7.1f}' for p in (5, 25, 50, 75, 95)))
    log('  -> the thresholds should sit between the low and high modes of these')
    log('=' * 68)


def main():
    ap = argparse.ArgumentParser(
        description='NeoPIP-FUCCI brightfield tracking and cell-cycle features, '
                    'one well/tile at a time.')
    ap.add_argument('experiment_dir',
                    help='folder containing Pics/ and seqs.csv, or a flat folder '
                         'of .vsi files')
    ap.add_argument('--pics-subdir', default='Pics',
                    help="subfolder holding the .vsi files; ignored if it does "
                         "not exist, so a flat folder works too (default Pics)")
    ap.add_argument('--seqs', default='seqs.csv',
                    help='per-well condition table in the experiment folder '
                         '(default seqs.csv); optional')
    ap.add_argument('--well', help='e.g. A1')
    ap.add_argument('--tile', type=int, help='e.g. 1')
    ap.add_argument('--list', action='store_true',
                    help='list the well/tiles found and exit')
    ap.add_argument('--output-dir', default=None,
                    help='where to write the CSV (default: '
                         'neoPIP_FUCCI_outputs/ beside this script)')
    ap.add_argument('--max-frames', type=int, default=0,
                    help='process only the first N frames (0 = all). Use a small '
                         'number for the first look')
    ap.add_argument('--max-dist', type=float, default=40.0,
                    help='max cell-centroid movement in px between consecutive '
                         'frames; further than this and the track is cut '
                         '(default 40)')
    ap.add_argument('--first-step-max-dist', type=float, default=None)
    ap.add_argument('--match-method', choices=['staged', 'nearest', 'hungarian'],
                    default='staged',
                    help="'staged' (default): one-to-one matching first, then "
                         'divisions are looked for only among the cells left '
                         "over. 'nearest' is the old rule, where two cells "
                         'claiming one parent are a division or both dropped')
    ap.add_argument('--min-division-interval', type=int, default=24,
                    help='frames a daughter must exist before it may divide '
                         'again (default 24 = 6 h at 15 min/frame). Extra '
                         'objects next to a younger daughter are dropped as '
                         "'refractory' instead of starting a new division. "
                         '0 disables')
    ap.add_argument('--division-midpoint', type=float, default=0.6,
                    help="the parent's position must lie near the midpoint of "
                         'the two daughters: max distance from that midpoint, '
                         'as a fraction of half the daughter separation '
                         '(default 0.6). Stops a cell that carries on in place '
                         'plus a neighbour being called a division. Negative '
                         'disables')
    ap.add_argument('--heal-window', type=int, default=8,
                    help='revert a division when one daughter is lost within '
                         'this many frames and the other grows back to a whole '
                         'nucleus -- a Cellpose split that healed (default 8; '
                         '0 disables)')
    ap.add_argument('--heal-frac', type=float, default=0.75,
                    help='area, as a fraction of the parent median, that counts '
                         "as 'whole again' for --heal-window (default 0.75)")
    ap.add_argument('--no-divisions', action='store_true',
                    help='do not follow divisions (they are followed by default '
                         'here, since frames_since_division needs them)')
    ap.add_argument('--division-max-dist', type=float, default=None,
                    help='how far a daughter may sit from the parent centroid. '
                         'Keep this WELL BELOW --max-dist: with a large '
                         '--max-dist a daughter is often nearest some other '
                         "neighbour instead of its parent, and the two-claimants "
                         'signature that identifies a division never forms '
                         '(default: same as --max-dist)')
    ap.add_argument('--fill-gaps', action='store_true',
                    help='measure the frames where a cell went too dim to '
                         'segment. Its position is interpolated from either '
                         'side of the gap and a disc of the track median area '
                         'is measured there, so the dark window gets real '
                         'intensity numbers instead of missing rows. Costs a '
                         'second read pass (no Cellpose). Shape columns are '
                         'left blank on those rows -- they would describe the '
                         'disc, not the cell')
    ap.add_argument('--min-division-overlap', type=int, default=3,
                    help='frames the two daughters must both be present for '
                         'before a division is believed. Cellpose splitting '
                         'one nucleus in half passes every area test, so '
                         'persistence is what separates it from mitosis; '
                         '0 disables the check')
    ap.add_argument('--gap-dist-scale', type=float, default=1.0,
                    help='how much the distance cutoff grows per missed frame. '
                         '1.0 means a cell gone for 3 frames may be 3x '
                         '--max-dist away, which is what it has had time to '
                         'do; 0 keeps the single-frame cutoff for every gap')
    ap.add_argument('--no-motion-prediction', action='store_true',
                    help='match a gapped track against where it was last seen '
                         'rather than where its last velocity says it should '
                         'be now')
    ap.add_argument('--max-gap', type=int, default=0,
                    help='let a track survive this many frames in which its cell '
                         'was not segmented, then reconnect. Essential with '
                         '--segment-on sum, where cells go dark in early S; with '
                         '0 a single missed frame ends the track for good')
    ap.add_argument('--daughter-frac', type=float, nargs=2,
                    metavar=('MIN', 'MAX'), default=None,
                    help='each daughter area as a fraction of the parent, for '
                         'accepting a division (default 0.25 0.85)')
    ap.add_argument('--daughter-sum-frac', type=float, nargs=2,
                    metavar=('MIN', 'MAX'), default=None,
                    help='the two daughters together vs the parent '
                         '(default 0.60 1.60). Raise MAX if real divisions are '
                         'being rejected because the parent mask was '
                         'under-segmented')
    ap.add_argument('--daughter-symmetry', type=float, default=None,
                    help='smaller daughter / larger daughter, minimum '
                         '(default 0.40)')
    ap.add_argument('--min-area-px', type=int, default=400,
                    help='discard segmented objects smaller than this (default 400)')
    ap.add_argument('--pixel-size', type=float, default=None,
                    help='um per pixel, to add an area_um2 column')
    ap.add_argument('--flow-threshold', type=float, default=0.4)
    ap.add_argument('--cellprob-threshold', type=float, default=0.0)
    ap.add_argument('--tile-norm-blocksize', type=int, default=0)
    ap.add_argument('--diameter', type=float, default=None,
                    help='Cellpose diameter in px. In v4 this is a rescaling '
                         'hint, not a size prior: diameter=60 downsamples by 2x. '
                         'Default None leaves the image at native scale')
    ap.add_argument('--segment-on', choices=fi.SEGMENT_CHANNELS, default='bf',
                    help="channel Cellpose runs on. 'sum' = mCherry + mVenus "
                         'background-subtracted, which is much higher contrast '
                         'than brightfield and cuts far fewer cells in half -- '
                         'but both markers are near zero in early S, so those '
                         "cells drop out of the mask. 'bf' is the only channel "
                         'present in every phase. '
                         "'dual' runs Cellpose on BOTH: brightfield gives the "
                         'cells, the marker sum gives the nuclei, each corrects '
                         'the other, and a cell that goes dark keeps its '
                         'brightfield mask and gets its nucleus carried from '
                         'the nearest frame where it was visible (default bf)')
    dual = ap.add_argument_group(
        'dual mode', 'only used with --segment-on dual; see fucci_dual.py')
    dual.add_argument('--mitotic-circularity', type=float, default=0.80,
                      help='a cell at least this round (brightfield outline), '
                           'shrunk as below and with the brightfield halo is '
                           'flagged as about to divide, and its division is '
                           'actively searched for (default 0.80; 0.85 missed 6 '
                           'of 48 divisions on A9-2, 0.80 added them with no '
                           'false ones; 0 disables the flag)')
    dual.add_argument('--mitotic-area-frac', type=float, default=0.8,
                      help='...and at most this fraction of its own pre-mitotic '
                           'median area (default 0.8). The halo test is what '
                           'rejects false flags; 0.6 missed A9-2 cell 11, a '
                           'clear mitosis whose interphase outline was already '
                           'small, while 0.8 added it and changed nothing else')
    dual.add_argument('--mitosis-window', type=int, default=8,
                      help='frames after the last round frame during which the '
                           'flag stays on (default 8 = 2 h)')
    dual.add_argument('--single-daughter-frac', type=float, default=0.75,
                      help='a flagged cell that flattens out at no more than '
                           'this fraction of its pre-mitotic area, with no '
                           'second daughter found, is recorded as a division '
                           'with one daughter seen: it gets a new id instead '
                           'of carrying on as the parent (default 0.75)')
    dual.add_argument('--mitotic-rim', type=float, default=0.85,
                      help='...and its brightfield rim at most this dark '
                           'relative to the background outside it (default '
                           '0.85; mitotic cells measured 0.59-0.75)')
    dual.add_argument('--mitotic-interior', type=float, default=1.03,
                      help='...and its interior at least this bright (default '
                           '1.03; mitotic 1.055-1.117, a flat cell whose '
                           'outline Cellpose shrank to a round blob ~1.00)')
    dual.add_argument('--allow-unflagged-divisions', action='store_true',
                      help='let a cell divide without first being flagged '
                           'mitotic. Off by default in dual mode: every real '
                           'division checked was flagged first, and the '
                           'unflagged ones were flat cells cut in half')
    dual.add_argument('--link-overlap', type=float, default=0.3,
                      help='masks overlapping by this fraction of the smaller '
                           'one may link however far their centroids moved -- '
                           'what keeps a cell linked while it rounds up '
                           '(default 0.3)')
    dual.add_argument('--min-cell-px', type=int, default=1500,
                      help='brightfield objects with no nucleus smaller than '
                           'this are dropped as debris, e.g. the bridge between '
                           'two daughters (default 1500)')
    dual.add_argument('--orphan-grow-px', type=int, default=15,
                      help='a nucleus brightfield missed becomes a cell grown '
                           'this far around it (default 15)')
    dual.add_argument('--track-on', choices=['cell', 'nucleus'],
                      default='cell',
                      help='position the tracker links on: the brightfield '
                           'cell centroid (default) or the segmented nucleus '
                           'centroid, falling back to the cell centroid in '
                           'frames without one. Tried on A9-2: no measurable '
                           'difference, since mask overlap decides most links')
    dual.add_argument('--reuse-masks', action='store_true',
                      help='load the Cellpose masks saved by an earlier dual '
                           'run of this tile instead of segmenting again, so '
                           'only fusion and tracking are redone')
    ap.add_argument('--cellpose-model', default=None,
                    help='Cellpose model to load, e.g. cpsam. Default None uses '
                         "whatever the installed cellpose picks, which is cpsam "
                         'on 4.0.4 — pin it explicitly for the grid run so an '
                         'environment with a different default cannot silently '
                         'change your segmentation')
    ap.add_argument('--measure-background', type=int, nargs='?', const=3,
                    default=0, metavar='N',
                    help='segment N frames (default 3) and report the per-channel '
                         'background from pixels outside every cell, then exit. '
                         'Run this FIRST and put the numbers into BACKGROUND in '
                         'fucci_pipeline.py -- a wrong background makes every '
                         'cell read double-positive')
    ap.add_argument('--save-overlay', action='store_true',
                    help='also write an ImageJ composite hyperstack with BF, '
                         'mCherry, mVenus, cell outlines and track-id labels, so '
                         'the masks and tracking can be checked by eye. Holds the '
                         'frames in memory: ~30 MB per frame')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='print a line per frame')
    args = ap.parse_args()

    experiment_dir = os.path.abspath(args.experiment_dir)
    if not os.path.isdir(experiment_dir):
        ap.error(f'not a directory: {experiment_dir}')
    image_dir = resolve_pics_dir(experiment_dir, args.pics_subdir)
    seqs, seqs_path = load_seqs(experiment_dir, args.seqs)
    if seqs_path:
        print(f'conditions: {len(seqs)} well(s) from {seqs_path}')
    else:
        print(f'no {args.seqs} in {experiment_dir} — condition columns will be '
              f'blank')
    if image_dir != experiment_dir:
        print(f'images:     {image_dir}')

    filelist = glob.glob(os.path.join(image_dir, '*.vsi'))
    tiles, problems = fp.discover_tiles(filelist)
    for p in problems[:20]:
        print(f'  {p}')
    if not tiles:
        raise SystemExit(f'No complete well/tiles found in {image_dir}')

    if args.list:
        print(f'{len(tiles)} well/tile(s) with all three channels:')
        for w, t in tiles:
            tag = ''
            if seqs:
                tag = ('  ' + ', '.join(f'{k}={v}' for k, v in seqs[w].items())
                       if w in seqs else '  (no row in ' + args.seqs + ')')
            print(f'   {w}-{t}{tag}')
        if seqs:
            missing = sorted({w for w, _t in tiles} - set(seqs))
            if missing:
                print(f'\n{len(missing)} well(s) imaged but absent from '
                      f'{args.seqs}: {", ".join(missing)}')
        return

    if not args.well or args.tile is None:
        ap.error('--well and --tile are required (or use --list)')
    if (args.well, args.tile) not in tiles:
        raise SystemExit(f'{args.well}-{args.tile} not found or missing a '
                         f'channel; use --list to see what is available')

    well_meta = seqs.get(args.well, {})
    if seqs and args.well not in seqs:
        print(f'WARNING: {args.well} has no row in {args.seqs}; the condition '
              f'columns will be blank for this tile')

    out_dir = args.output_dir or fp.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # Heavy imports here, not at module scope, so --list and --help work without
    # a GPU or a JVM.
    import imagej
    from cellpose import models, io as cpio
    from pystackreg import StackReg

    cpio.logger_setup()
    try:
        if args.cellpose_model:
            model = models.CellposeModel(gpu=True,
                                         pretrained_model=args.cellpose_model)
        else:
            model = models.CellposeModel(gpu=True)
    except Exception as exc:
        raise SystemExit(f'Could not initialise Cellpose: {exc}')
    weights = getattr(model, 'pretrained_model', None)
    print(f'cellpose model: {weights if weights else "unknown"}')
    java_mem = os.environ.get('JAVA_MEM')       # e.g. 8g; set on OrangeGrid
    if java_mem:
        import scyjava
        scyjava.config.add_option(f'-Xmx{java_mem}')
    try:
        ij = imagej.init('sc.fiji:fiji:2.14.0')
    except Exception as exc:
        raise SystemExit(f'Could not initialise ImageJ: {exc}')
    print(f'ImageJ {ij.getVersion()}')
    sr = StackReg(StackReg.RIGID_BODY)

    div_thresholds = {}
    if args.daughter_frac:
        div_thresholds['min_daughter_frac'] = args.daughter_frac[0]
        div_thresholds['max_daughter_frac'] = args.daughter_frac[1]
    if args.daughter_sum_frac:
        div_thresholds['min_sum_frac'] = args.daughter_sum_frac[0]
        div_thresholds['max_sum_frac'] = args.daughter_sum_frac[1]
    if args.daughter_symmetry is not None:
        div_thresholds['min_symmetry'] = args.daughter_symmetry
    if args.division_midpoint >= 0 and not args.no_divisions:
        div_thresholds['max_midpoint_ratio'] = args.division_midpoint
    div_thresholds = div_thresholds or None

    cp_kwargs = dict(flow_threshold=args.flow_threshold,
                     cellprob_threshold=args.cellprob_threshold,
                     tile_norm_blocksize=args.tile_norm_blocksize,
                     diameter=args.diameter)
    log = print if args.verbose else (lambda *a, **k: None)

    if args.measure_background:
        print(f'\n=== background for {args.well}-{args.tile} ===')
        bg = fi.measure_background(ij, sr, model, image_dir, args.well, args.tile,
                                   n_frames=args.measure_background,
                                   cellpose_kwargs=cp_kwargs, log=print)
        print(f"\nfield covered by cells: {bg.get('field_covered_pct', float('nan')):.0f}%")
        names = {fp.CH_MCHERRY: 'mCherry', fp.CH_MVENUS: 'mVenus'}
        print(f"\n{'channel':10s} {'median':>9s} {'mean':>9s} {'p25':>9s} "
              f"{'p75':>9s} {'p95':>9s}")
        for ch, name in names.items():
            if ch not in bg:
                continue
            s_ = bg[ch]
            print(f"{name:10s} {s_['median']:9.1f} {s_['mean']:9.1f} "
                  f"{s_['p25']:9.1f} {s_['p75']:9.1f} {s_['p95']:9.1f}")
        print('\nPut the medians into BACKGROUND in fucci_pipeline.py:')
        for ch, name in names.items():
            if ch in bg:
                print(f'    CH_{name.upper()}: {bg[ch]["median"]:.1f},')
        print('\nIf p95 is far above the median the "background" still contains '
              'cells — raise --diameter or loosen Cellpose so fewer are missed.')
        return

    print(f'\n=== {args.well}-{args.tile} ===')
    print(f'segmenting on: {args.segment_on}')
    print(f'background (edit in fucci_pipeline.py): '
          f'mCherry={fp.BACKGROUND[fp.CH_MCHERRY]}, '
          f'mVenus={fp.BACKGROUND[fp.CH_MVENUS]}')

    res = fi.process_tile(
        ij, sr, model, image_dir, args.well, args.tile,
        max_dist=args.max_dist,
        cellpose_kwargs=cp_kwargs,
        match_method=args.match_method,
        first_step_max_dist=args.first_step_max_dist,
        allow_division=not args.no_divisions,
        division_max_dist=args.division_max_dist,
        division_thresholds=div_thresholds, max_gap=args.max_gap,
        gap_dist_scale=args.gap_dist_scale,
        min_division_overlap=args.min_division_overlap,
        min_division_interval=args.min_division_interval,
        heal_window=args.heal_window, heal_frac=args.heal_frac,
        dual_kwargs=dict(
            mitotic_circularity=args.mitotic_circularity or None,
            mitotic_area_frac=args.mitotic_area_frac,
            mitosis_window=args.mitosis_window,
            single_daughter_frac=args.single_daughter_frac,
            link_overlap=args.link_overlap, min_cell_px=args.min_cell_px,
            mitotic_halo=dict(max_rim=args.mitotic_rim,
                              min_interior=args.mitotic_interior),
            division_requires_mitosis=(not args.allow_unflagged_divisions
                                       and bool(args.mitotic_circularity)),
            orphan_grow_px=args.orphan_grow_px, min_nuc_px=args.min_area_px,
            track_on=args.track_on),
        masks_dir=os.path.join(out_dir, f'{args.well}-{args.tile}_masks'),
        reuse_masks=args.reuse_masks,
        fill_gaps=args.fill_gaps,
        predict_motion=not args.no_motion_prediction,
        min_area_px=args.min_area_px,
        max_frames=args.max_frames,
        pixel_size_um=args.pixel_size, segment_on=args.segment_on,
        keep_masks=args.save_overlay, keep_planes=args.save_overlay, log=log)

    if not res['rows']:
        raise SystemExit('No cells survived tracking -- nothing written. Try a '
                         'larger --max-dist or looser Cellpose settings.')

    df = fp.classify_table(pd.DataFrame(res['rows']))
    # one value per track, repeated on its rows, so a track can be filtered
    # with a single comparison (diagnose_tracks.py histograms them)
    df = fp.add_motion_columns(df)
    df = fp.add_division_check(df)
    for col, val in well_meta.items():
        df[col] = val

    lead = (['well', 'tile'] + list(well_meta) +
            ['track_id', 'timepoint', 'time_min', 'phase',
             'cell_label', 'parent_track_id', 'root_track_id', 'generation',
             'track_end_reason', 'division_kind', 'mitotic_flag',
             'frames_since_division', 'frames_until_division',
             'nuc_source', 'cell_source', 'mitosis_no_division',
             'mitosis_seen_tp', 'off_path_px', 'max_off_path_px',
             'max_msd_px2', 'max_step_px', 'max_step_sq_px2'])
    cols = [c for c in lead if c in df.columns]
    df = df[cols + sorted(c for c in df.columns if c not in cols)]

    # match distances first and the main CSV last, each written whole and
    # renamed: the main CSV existing is what marks a tile as finished, so an
    # interrupted job must never leave a partial one
    rec_csv = os.path.join(out_dir, f'{args.well}-{args.tile}_match_distances.csv')
    pd.DataFrame(res['records']).to_csv(rec_csv + '.part', index=False)
    os.replace(rec_csv + '.part', rec_csv)

    out_csv = os.path.join(out_dir, f'{args.well}-{args.tile}_fucci.csv')
    df.to_csv(out_csv + '.part', index=False)
    os.replace(out_csv + '.part', out_csv)

    summarise(df, res, args_max_gap=args.max_gap)
    print(f'\nwrote {out_csv}')
    print(f'wrote {rec_csv}')

    if args.save_overlay:
        ov = os.path.join(out_dir, f'{args.well}-{args.tile}_overlay.tif')
        shape = fi.write_overlay(ov, res['planes'], res['masks'], res['tracks'],
                                 res['centroids'], res['timepoints'],
                                 pixel_size_um=args.pixel_size,
                                 nuc_masks=res.get('nuc_masks') or None)
        print(f'wrote {ov}  ({shape[0]} frames x {shape[1]} channels)')
        print('  open it with File > Open in Fiji; channel 4 is the cell '
              'outline, channel 5 the track-id labels'
              + (', channel 6 the nucleus each cell was measured on'
                 if res.get('nuc_masks') else ''))


if __name__ == '__main__':
    main()
