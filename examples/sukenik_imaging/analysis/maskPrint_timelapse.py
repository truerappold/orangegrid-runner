"""
Diagnostic version of the timelapse pipeline: prints the labelled mask of every
timepoint so the tracking cutoff y (--max-dist) can be judged by eye.

The key difference from maskPrint_pipeline.py is that cells are coloured by
**track id**, not by their per-frame Cellpose label.  A correctly tracked cell
is therefore the same colour in every panel, and a mis-assignment shows up as a
colour change rather than as a number you have to read.  Dropped cells are drawn
as grey outlines annotated with why they were dropped, and a white line joins
each matched centroid to its position in the previous timepoint.

    # one tile, look at it, adjust, repeat
    python maskPrint_timelapse.py <dir> <out_dir> --wells A1 --tiles 9 --max-dist 25

    # then sweep the cutoff offline against the saved distances, no re-segmenting
    python tune_y.py <out_dir>/timelapse_match_distances.csv

Judging y by eye is slow; the distance histogram this script writes is the
faster route. If tracking is working there is a tight peak of true matches near
zero and a long tail of spurious ones, and y belongs in the gap. Use the images
to confirm the choice on a couple of tiles rather than to discover it.
"""

import argparse
import glob
import os

import imagej
import matplotlib
import numpy as np
import pandas as pd
import tqdm
from cellpose import models, io
from pystackreg import StackReg

matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib import colormaps                                   # noqa: E402
import skimage as ski                                              # noqa: E402

import timelapse_tracking as tt                                    # noqa: E402
import timelapse_imaging as ti                                     # noqa: E402


def track_colors(track_ids, seed=0):
    """Stable colour per track id, so a track keeps its colour across panels."""
    rng = np.random.default_rng(seed)
    cmap = colormaps['hsv']
    ids = sorted(track_ids)
    shuffled = rng.permutation(len(ids)) / max(len(ids), 1)
    return {tid: cmap(float(v))[:3] for tid, v in zip(ids, shuffled)}


def draw_tile(res, timepoints, well, tile, max_dist, out_dir, dpi=200):
    """One figure per tile: a panel per timepoint, cells coloured by track."""
    masks = res['masks']
    tracks = res['tracks']
    ncen = res['nuc_centroids']
    all_tracks = {tid for tp in tracks for tid in tracks[tp].values()}
    colors = track_colors(all_tracks)

    n = len(timepoints)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows),
                             facecolor='black', squeeze=False)
    axes = axes.flatten()

    drop_by_tp = {}
    for rec in res['records']:
        if rec['dropped']:
            drop_by_tp.setdefault(rec['timepoint'], {})[rec['cell_label']] = rec

    for i, tp in enumerate(timepoints):
        ax = axes[i]
        ax.set_facecolor('black')
        mask_cell, mask_nuc = masks[tp][1], masks[tp][2]

        rgb = np.zeros((*mask_cell.shape, 3), dtype=np.float32)
        for label, tid in tracks.get(tp, {}).items():
            rgb[mask_cell == label] = colors[tid]
        # dropped cells in dark grey so they are visible but clearly not tracked
        for label in drop_by_tp.get(tp, {}):
            rgb[mask_cell == label] = (0.28, 0.28, 0.28)
        ax.imshow(rgb, interpolation='nearest')

        for label, tid in tracks.get(tp, {}).items():
            nucleus = mask_nuc == label
            if np.any(nucleus):
                for c in ski.measure.find_contours(nucleus.astype(float), 0.5):
                    ax.plot(c[:, 1], c[:, 0], 'w-', linewidth=1.0, alpha=0.9)
            if label in ncen.get(tp, {}):
                r, c = ncen[tp][label]
                ax.text(c, r, str(tid), color='white', fontsize=7,
                        ha='center', va='center', weight='bold')

        # link each matched centroid back to where it was in the previous frame
        if i > 0:
            prev_tp = timepoints[i - 1]
            prev_by_track = {tid: ncen[prev_tp][lab]
                             for lab, tid in tracks.get(prev_tp, {}).items()
                             if lab in ncen.get(prev_tp, {})}
            for label, tid in tracks.get(tp, {}).items():
                if label in ncen.get(tp, {}) and tid in prev_by_track:
                    r0, c0 = prev_by_track[tid]
                    r1, c1 = ncen[tp][label]
                    ax.plot([c0, c1], [r0, r1], '-', color='white',
                            linewidth=0.8, alpha=0.55)

        for label, rec in drop_by_tp.get(tp, {}).items():
            if label in ncen.get(tp, {}):
                r, c = ncen[tp][label]
                d = rec.get('match_dist', np.nan)
                txt = rec['drop_reason']
                if np.isfinite(d):
                    txt += f'\n{d:.0f}px'
                ax.text(c, r, txt, color='#ff6b6b', fontsize=5.5,
                        ha='center', va='center')

        n_kept = len(tracks.get(tp, {}))
        n_drop = len(drop_by_tp.get(tp, {}))
        ax.set_title(f't{tp}   kept {n_kept}   dropped {n_drop}',
                     color='white', fontsize=11)
        ax.axis('off')

    for ax in axes[n:]:
        ax.axis('off')

    fig.suptitle(f'{well} tile {tile} — coloured by track id, y = {max_dist:g} px',
                 color='white', fontsize=14)
    fig.tight_layout()
    out = os.path.join(out_dir, f'{well}_tile{tile}_tracks.png')
    fig.savefig(out, dpi=dpi, facecolor='black', bbox_inches='tight')
    plt.close(fig)
    return out


def draw_distance_histogram(records, max_dist, out_dir):
    """Pooled nearest-neighbour distances — the fast way to choose y."""
    df = pd.DataFrame(records)
    if df.empty or 'match_dist' not in df:
        return None
    d = df['match_dist'].replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (vals, label) in zip(axes, [
            (d, 'nuclear centroid'),
            (df['match_dist_cell'].replace([np.inf, -np.inf], np.nan).dropna()
             if 'match_dist_cell' in df else pd.Series(dtype=float),
             'whole-cell centroid')]):
        if vals.empty:
            ax.set_visible(False)
            continue
        ax.hist(vals, bins=60, color='#2471a3', alpha=.8)
        ax.axvline(max_dist, color='#c0392b', lw=2,
                   label=f'y = {max_dist:g} px')
        ax.set_xlabel(f'distance to nearest previous {label} (px)')
        ax.set_ylabel('cells')
        ax.set_title(f'{label}: median {vals.median():.1f}, '
                     f'90th pct {vals.quantile(.9):.1f} px', fontsize=10)
        ax.legend(frameon=False)
    fig.suptitle('Choose y in the gap between true matches and the spurious tail',
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(out_dir, 'match_distance_histogram.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out


def main(dir, out_dir, args, ij, model):
    sr = StackReg(StackReg.RIGID_BODY)
    os.makedirs(out_dir, exist_ok=True)

    try:
        seqs = pd.read_csv(os.path.join(dir, 'seqs.csv'), index_col=0)
    except FileNotFoundError:
        print(f'Error: seqs.csv not found in {dir}.')
        return

    pics_dir = os.path.join(dir, args.pics_subdir) if args.pics_subdir else dir
    filelist = glob.glob(os.path.join(pics_dir, '*.vsi'))
    if not filelist:
        print(f'No VSI files found in {pics_dir}.')
        return

    series, problems = tt.discover_blocks(filelist)
    for p in problems[:20]:
        print(f'  {p}')

    keys = sorted(k for k in series if k[0] in seqs.index)
    if args.wells:
        keys = [k for k in keys if k[0] in args.wells]
    if args.tiles:
        keys = [k for k in keys if k[1] in args.tiles]
    if not keys:
        print('No well/tile series selected.')
        return
    print(f'{len(keys)} well/tile series selected')

    cellpose_kwargs = dict(flow_threshold=args.flow_threshold,
                           cellprob_threshold=args.cellprob_threshold,
                           tile_norm_blocksize=args.tile_norm_blocksize)

    all_records = []
    pbar = tqdm.tqdm(total=len(keys), desc='Drawing well/tile series', ncols=150)
    for well, tile in keys:
        blocks = series[(well, tile)]
        print(f'on well {well} tile {tile} (blocks {blocks})')
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
                do_background=not args.no_background, keep_masks=True,
                log=print)
        except Exception as exc:
            print(f'  ERROR on {well} tile {tile}: {type(exc).__name__}: {exc}')
            pbar.update(1)
            continue
        all_records.extend(res['records'])
        out = draw_tile(res, res['timepoints'], well, tile, args.max_dist, out_dir)
        print(f'  saved {out}')
        pbar.update(1)
    pbar.close()

    rec_csv = os.path.join(out_dir, 'timelapse_match_distances.csv')
    pd.DataFrame(all_records).to_csv(rec_csv, index=False)
    print(f'Saved match records to {rec_csv}')
    hist = draw_distance_histogram(all_records, args.max_dist, out_dir)
    if hist:
        print(f'Saved distance histogram to {hist}')
    print(f'\nNext: python tune_y.py {rec_csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Print per-timepoint labelled masks coloured by track id.')
    parser.add_argument('directory', help='Directory containing Pics/ and seqs.csv')
    parser.add_argument('output_dir', help='Where to write the figures')
    parser.add_argument('--wells', nargs='*', default=None,
                        help='limit to these wells, e.g. --wells A1 B2')
    parser.add_argument('--tiles', nargs='*', type=int, default=None,
                        help='limit to these tiles, e.g. --tiles 9')
    parser.add_argument('--max-dist', type=float, default=25.0)
    parser.add_argument('--match-method', choices=['nearest', 'hungarian'],
                        default='nearest')
    parser.add_argument('--flow-threshold', type=float, default=0.4)
    parser.add_argument('--cellprob-threshold', type=float, default=0.0)
    parser.add_argument('--tile-norm-blocksize', type=int, default=0)
    parser.add_argument('--hole-area', type=int, default=200)
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
    parser.add_argument('--no-background', action='store_true')
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
    main(args.directory, args.output_dir, args, ij, model)
