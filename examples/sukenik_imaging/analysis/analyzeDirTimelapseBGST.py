"""
Timelapse version of analyzeDirPerturbBGST.py.

Accepts a series of any length.  Each filename prefix is a block (no prefix = 0,
'1' = 1, ...), and a block's .vsi may itself contain many frames — a basal
snapshot plus a 20-frame timelapse under prefix 1 is 21 timepoints.  Blocks are
concatenated in numerical order.  Every timepoint gets its own
Cellpose mask on the direct mNG channel; masks are linked frame to frame by
nearest nuclear centroid, so a cell keeps one track_id across the whole series.

Output is wide: one row per track, measurements suffixed _t1 .. _tN, plus
n_timepoints / complete_track so partial tracks can be filtered downstream.

    python analyzeDirTimelapseBGST.py <dir> --max-dist 25
    python analyzeDirTimelapseBGST.py <dir> --max-dist 25 --match-method hungarian

Pick --max-dist with maskPrint_timelapse.py and tune_y.py before running this on
a whole plate.  The default of 25 px (~8 um at 0.325 um/px) is a starting guess,
not a calibrated value.
"""

import argparse
import glob
import os

import imagej
import numpy as np
import pandas as pd
import tqdm
from cellpose import models, io
from pystackreg import StackReg

import timelapse_tracking as tt
import timelapse_imaging as ti


def main(dir, args, ij, model):
    sr = StackReg(StackReg.RIGID_BODY)

    try:
        seqs = pd.read_csv(os.path.join(dir, 'seqs.csv'), index_col=0)
    except FileNotFoundError:
        print(f'Error: seqs.csv not found in {dir}. Please provide a valid directory.')
        return

    pics_dir = os.path.join(dir, args.pics_subdir) if args.pics_subdir else dir
    filelist = glob.glob(os.path.join(pics_dir, '*.vsi'))
    if not filelist:
        print(f'No VSI files found in {pics_dir}.')
        return

    series, problems = tt.discover_blocks(filelist)
    if problems:
        problems_file = os.path.join(dir, 'timelapse_problems.txt')
        with open(problems_file, 'w') as f:
            f.write('\n'.join(problems) + '\n')
        print(f'{len(problems)} series problem(s) written to {problems_file}')
        for p in problems[:10]:
            print(f'  {p}')

    wells_in_images = sorted({w for (w, _t) in series})
    missing = [w for w in wells_in_images if w not in seqs.index]
    if missing:
        with open(os.path.join(dir, 'missing_wells.txt'), 'w') as f:
            f.write('Wells found in images but not in seqs.csv:\n')
            f.write('\n'.join(missing) + '\n')
        print(f'{len(missing)} well(s) not in seqs.csv — see missing_wells.txt')

    keys = sorted(k for k in series if k[0] in seqs.index)
    if not keys:
        print(f'No matching wells between image files and seqs.csv in {dir}.')
        return

    nblocks = {k: len(series[k]) for k in keys}
    print(f'{len(keys)} well/tile series; blocks per series: '
          f'min {min(nblocks.values())}, max {max(nblocks.values())} '
          f'(frames inside each block are counted when the block is opened)')

    cellpose_kwargs = dict(flow_threshold=args.flow_threshold,
                           cellprob_threshold=args.cellprob_threshold,
                           tile_norm_blocksize=args.tile_norm_blocksize)

    all_rows = []
    all_records = []
    pbar = tqdm.tqdm(total=len(keys), desc='Processing well/tile series', ncols=150)
    for well, tile in keys:
        construct = seqs.loc[well, 'construct']
        blocks = series[(well, tile)]
        print(f'on well {well} construct {construct} tile {tile} '
              f'(blocks {blocks})')

        try:
            res = ti.process_tile(
                ij, sr, model, pics_dir, well, tile, blocks,
                cellpose_kwargs, max_dist=args.max_dist,
                match_method=args.match_method,
                first_step_max_dist=args.first_step_max_dist,
                allow_division=args.track_divisions,
                division_max_dist=args.division_max_dist,
                hole_area=args.hole_area, dapi_mode=args.dapi_threshold,
                dapi_sigma=args.dapi_sigma, dapi_contrast=args.dapi_contrast,
                do_background=not args.no_background, keep_masks=False,
                log=print)
        except Exception as exc:                       # keep the plate going
            print(f'  ERROR on {well} tile {tile}: {type(exc).__name__}: {exc}')
            pbar.update(1)
            continue

        all_records.extend(res['records'])
        if res['measurements']:
            wide = tt.assemble_wide(
                res['measurements'],
                meta={'construct': construct, 'well': well, 'tile': tile},
                timepoints=res['timepoints'], lineage=res['lineage'])
            if args.min_track_length > 1:
                wide = wide[wide['n_timepoints'] >= args.min_track_length]
            all_rows.append(wide)
        pbar.update(1)
    pbar.close()

    if not all_rows:
        print('No cells survived tracking — nothing written.')
        return

    cells_df = pd.concat(all_rows, ignore_index=True)
    cells_df = tt.order_columns(cells_df)
    out_csv = os.path.join(dir, args.output)
    cells_df.to_csv(out_csv, index=False)
    print(f'Saved {len(cells_df)} tracks to {out_csv}')

    rec_df = pd.DataFrame(all_records)
    rec_csv = os.path.join(dir, 'timelapse_match_distances.csv')
    rec_df.to_csv(rec_csv, index=False)
    print(f'Saved {len(rec_df)} match records to {rec_csv} '
          f'(feed this to tune_y.py to re-test cutoffs without re-segmenting)')

    n_complete = int(cells_df['complete_track'].sum())
    print(f'  complete tracks: {n_complete} / {len(cells_df)} '
          f'({100 * n_complete / len(cells_df):.0f}%)')
    if 'end_reason' in cells_df.columns:
        print('  track endings: ' + ', '.join(
            f'{k}={v}' for k, v in cells_df['end_reason'].value_counts().items()))
    if 'generation' in cells_df.columns and cells_df['generation'].max() > 0:
        n_div = int((cells_df['generation'] > 0).sum())
        print(f'  {n_div} daughter track(s) from divisions; max generation '
              f'{int(cells_df["generation"].max())}')
    if len(rec_df):
        dropped = rec_df[rec_df['dropped']]
        if len(dropped):
            print('  drops by reason: '
                  + ', '.join(f'{k}={v}' for k, v in
                              dropped['drop_reason'].value_counts().items()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Timelapse cell tracking and FRET analysis; wide CSV output.')
    parser.add_argument('directory', help='Directory containing Pics/ and seqs.csv')
    parser.add_argument('--max-dist', type=float, default=25.0,
                        help='y: max centroid movement in px between consecutive '
                             'timepoints; further than this and the cell is dropped '
                             '(default 25)')
    parser.add_argument('--match-method', choices=['nearest', 'hungarian'],
                        default='nearest',
                        help="'nearest' = your rule (nearest centroid, drop all "
                             "cells that collide on one target). 'hungarian' = "
                             "globally optimal one-to-one matching, which cannot "
                             "collide and so keeps more cells (default nearest)")
    parser.add_argument('--min-track-length', type=int, default=1,
                        help='discard tracks shorter than this many timepoints '
                             '(default 1 = keep everything, filter later on '
                             'n_timepoints / complete_track)')
    parser.add_argument('--flow-threshold', type=float, default=0.4,
                        help='Cellpose flow threshold (default 0.4, as in '
                             'analyzeDirPerturbBGST.py; maskPrint_pipeline used 1.0)')
    parser.add_argument('--cellprob-threshold', type=float, default=0.0)
    parser.add_argument('--tile-norm-blocksize', type=int, default=0)
    parser.add_argument('--hole-area', type=int, default=200,
                        help='fill nuclear holes smaller than this (default 200)')
    parser.add_argument('--first-step-max-dist', type=float, default=None,
                        help='override y for the t1->t2 step only. The basal .vsi '
                             'and the first timelapse frame are usually separated '
                             'by a longer interval than frames within the stack '
                             '(~110 s vs ~35 s in the example data), so cells can '
                             'legitimately move further across that one join. '
                             'tune_y.py flags this if it sees it')
    parser.add_argument('--track-divisions', action='store_true',
                        help='follow dividing cells: a two-way collision that '
                             'passes the area tests is treated as mitosis, the '
                             'parent track ends and two daughter tracks begin '
                             'with parent_track_id / root_track_id / generation '
                             'set. Without this, both daughters are dropped as a '
                             'collision, which is the old behaviour')
    parser.add_argument('--division-max-dist', type=float, default=None,
                        help='how far a daughter may sit from the parent '
                             'centroid (default: same as --max-dist)')
    parser.add_argument('--dapi-threshold', choices=['per-cell', 'global'],
                        default='per-cell',
                        help='per-cell thresholds DAPI inside each Cellpose mask '
                             'separately. A single global Otsu cannot cope with '
                             "the field's dynamic range and shaves dim nuclei "
                             'down to their bright cores (default per-cell)')
    parser.add_argument('--dapi-sigma', type=float, default=2.0,
                        help='Gaussian smoothing on DAPI before thresholding; 0 '
                             'disables. Without it the threshold cuts through '
                             'pixel noise and the outlines come out ragged '
                             '(default 2.0)')
    parser.add_argument('--dapi-contrast', type=float, default=1.5,
                        help='reject a cell as having no nucleus unless its '
                             'bright DAPI median exceeds its dim median by this '
                             'factor (default 1.5)')
    parser.add_argument('--pics-subdir', default='Pics',
                        help="subfolder holding the .vsi files; pass '' if they "
                             "sit directly in <directory> (default Pics)")
    parser.add_argument('--no-background', action='store_true',
                        help='skip flat background subtraction')
    parser.add_argument('--output', default='cells_df_BGSTtimelapse.csv')
    args = parser.parse_args()

    io.logger_setup()
    model = None
    try:
        model = models.CellposeModel(gpu=True)
    except Exception as exc:
        print(f'Error initializing Cellpose model: {exc}')
    if not model:
        raise SystemExit('Cellpose model could not be initialized. Exiting.')
    try:
        ij = imagej.init('sc.fiji:fiji:2.14.0')
    except Exception as exc:
        raise SystemExit(f'Error initializing ImageJ: {exc}')
    print(f'ImageJ version: {ij.getVersion()}')
    main(args.directory, args, ij, model)
