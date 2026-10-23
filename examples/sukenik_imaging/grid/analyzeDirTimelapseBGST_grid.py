"""
analyzeDirTimelapseBGST.py for OrangeGrid: one well/tile series per job.

Same pipeline, same numbers.  It calls the original's own per-tile function
(timelapse_imaging.process_tile) with the same options, after swapping in the
crop-based per-cell steps from fast_timelapse.py (identical results, ~20x
faster on those steps).  What changes is the packaging:

* one tile per run (--well / --tile), written to its own files
      <well>-<tile>_BGSTtimelapse.csv        the wide per-track table
      <well>-<tile>_match_distances.csv      the per-cell match records
  and combine_timelapse.py joins them into cells_df_BGSTtimelapse.csv and
  timelapse_match_distances.csv exactly as the original writes them;
* a tile whose CSV exists is skipped (so an evicted or resubmitted job only
  does what is missing) and files are written whole and then renamed.

Usage
    python analyzeDirTimelapseBGST_grid.py EXP_DIR --well A2 --tile 1 [original options]
    python analyzeDirTimelapseBGST_grid.py EXP_DIR --list

Every option of analyzeDirTimelapseBGST.py is accepted with the same meaning
and default (--max-dist, --match-method, --track-divisions, --dapi-* ...).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.environ.get('CODE_DIR', os.path.join(HERE, '..', 'analysis'))
sys.path.insert(0, os.path.abspath(CODE_DIR))
sys.path.insert(0, HERE)

import timelapse_tracking as tt  # noqa: E402


def add_original_options(ap):
    """The options of analyzeDirTimelapseBGST.py, same names and defaults."""
    ap.add_argument('--max-dist', type=float, default=25.0)
    ap.add_argument('--match-method', choices=['nearest', 'hungarian'],
                    default='nearest')
    ap.add_argument('--min-track-length', type=int, default=1)
    ap.add_argument('--flow-threshold', type=float, default=0.4)
    ap.add_argument('--cellprob-threshold', type=float, default=0.0)
    ap.add_argument('--tile-norm-blocksize', type=int, default=0)
    ap.add_argument('--hole-area', type=int, default=200)
    ap.add_argument('--first-step-max-dist', type=float, default=None)
    ap.add_argument('--track-divisions', action='store_true')
    ap.add_argument('--division-max-dist', type=float, default=None)
    ap.add_argument('--dapi-threshold', choices=['per-cell', 'global'],
                    default='per-cell')
    ap.add_argument('--dapi-sigma', type=float, default=2.0)
    ap.add_argument('--dapi-contrast', type=float, default=1.5)
    ap.add_argument('--pics-subdir', default='Pics')
    ap.add_argument('--no-background', action='store_true')


def series_for(exp_dir, pics_subdir):
    pics = os.path.join(exp_dir, pics_subdir) if pics_subdir else exp_dir
    series, problems = tt.discover_blocks(glob.glob(os.path.join(pics, '*.vsi')))
    return pics, series, problems


def tile_paths(out_dir, well, tile):
    return (os.path.join(out_dir, f'{well}-{tile}_BGSTtimelapse.csv'),
            os.path.join(out_dir, f'{well}-{tile}_match_distances.csv'))


def write_whole(df, path):
    df.to_csv(path + '.part', index=False)
    os.replace(path + '.part', path)


def run_tile(ij, sr, model, args, pics, series, seqs, well, tile, out_dir,
             log=print):
    """The body of the original main() loop for one (well, tile)."""
    import timelapse_imaging as ti

    construct = seqs.loc[well, 'construct']
    blocks = series[(well, tile)]
    log(f'on well {well} construct {construct} tile {tile} (blocks {blocks})')
    cellpose_kwargs = dict(flow_threshold=args.flow_threshold,
                           cellprob_threshold=args.cellprob_threshold,
                           tile_norm_blocksize=args.tile_norm_blocksize)
    res = ti.process_tile(
        ij, sr, model, pics, well, tile, blocks, cellpose_kwargs,
        max_dist=args.max_dist, match_method=args.match_method,
        first_step_max_dist=args.first_step_max_dist,
        allow_division=args.track_divisions,
        division_max_dist=args.division_max_dist,
        hole_area=args.hole_area, dapi_mode=args.dapi_threshold,
        dapi_sigma=args.dapi_sigma, dapi_contrast=args.dapi_contrast,
        do_background=not args.no_background, keep_masks=False, log=log)
    wide = pd.DataFrame()
    if res['measurements']:
        wide = tt.assemble_wide(
            res['measurements'],
            meta={'construct': construct, 'well': well, 'tile': tile},
            timepoints=res['timepoints'], lineage=res['lineage'])
        if args.min_track_length > 1:
            wide = wide[wide['n_timepoints'] >= args.min_track_length]
    csv, rec = tile_paths(out_dir, well, tile)
    write_whole(pd.DataFrame(res['records']), rec)
    write_whole(wide, csv)                 # last: its existence marks "done"
    return len(wide), len(res['records'])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('directory', help='experiment folder with Pics/ and seqs.csv')
    ap.add_argument('--well')
    ap.add_argument('--tile', type=int, nargs='*', default=None,
                    help='tile(s) of --well (default: all of that well)')
    ap.add_argument('--output-dir', default=None,
                    help='default: <directory>/BGSTtimelapse_tiles')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--original-speed', action='store_true',
                    help='do not swap in the fast per-cell steps (for checking)')
    add_original_options(ap)
    args = ap.parse_args()

    exp = os.path.abspath(args.directory)
    seqs = pd.read_csv(os.path.join(exp, 'seqs.csv'), index_col=0)
    pics, series, problems = series_for(exp, args.pics_subdir)
    for p in problems:
        print(f'  {p}')
    keys = sorted(k for k in series if k[0] in seqs.index)
    if args.list:
        for w, t in keys:
            print(f'{w}-{t}: blocks {series[(w, t)]}')
        return
    if args.well:
        keys = [k for k in keys if k[0] == args.well
                and (not args.tile or k[1] in args.tile)]
    if not keys:
        raise SystemExit('nothing to do: no matching well/tile in images and seqs.csv')
    out_dir = args.output_dir or os.path.join(exp, 'BGSTtimelapse_tiles')
    os.makedirs(out_dir, exist_ok=True)
    todo = [k for k in keys
            if args.force or not os.path.exists(tile_paths(out_dir, *k)[0])]
    if not todo:
        print('every requested tile already has a CSV (use --force to redo)')
        return

    if not args.original_speed:
        import fast_timelapse
        fast_timelapse.install()
    from cellpose import io, models
    from pystackreg import StackReg
    io.logger_setup()
    model = models.CellposeModel(gpu=True)
    print(f'cellpose on GPU: {model.gpu}')
    import imagej
    import scyjava
    if os.environ.get('JAVA_MEM'):
        scyjava.config.add_option(f"-Xmx{os.environ['JAVA_MEM']}")
    ij = imagej.init('sc.fiji:fiji:2.14.0')
    print(f'ImageJ version: {ij.getVersion()}')
    sr = StackReg(StackReg.RIGID_BODY)

    for well, tile in todo:
        t0 = time.time()
        n_tracks, n_rec = run_tile(ij, sr, model, args, pics, series, seqs,
                                   well, tile, out_dir)
        print(f'{well}-{tile}: {n_tracks} tracks, {n_rec} match records, '
              f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    sys.exit(main())
