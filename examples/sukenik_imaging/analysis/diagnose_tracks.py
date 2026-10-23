"""
Diagnostic plots for one tile's tracking, from the files a run already wrote.
No ImageJ, no GPU: it reads the CSV and the saved masks.

    python diagnose_tracks.py neoPIP_FUCCI_outputs/A9-2_fucci.csv --track 5

Three outputs, all written to neoPIP_FUCCI_outputs/ (or --output-dir):

1. <tile>_lineage<ID>.png + <tile>_lineage<ID>_centroids.csv
   The path of one cell's centroid through every frame, drawn on the
   max-projection of the fluorescence (which shows where nuclei went).  Each
   track in the lineage has its own colour, so the colour changes at a
   division and both daughters are followed, and their daughters too.  The
   centroid of every frame is also printed and saved as a table.

2. <tile>_outlines_age.png (or _age_pertrack / _time)
   The outlines of a lineage in every frame on one image: left the
   brightfield cell mask, right the nucleus mask the intensities were
   measured on (carried nuclei included).  Light blue -> purple by time since
   each track began, restarting at every division so each daughter starts
   light again (--outline-scale per_track: every track spans the whole map;
   --outline-color time: absolute time in the movie).

3. <tile>_msd.png + <tile>_msd.csv
   Hairplots, one line per track: mean squared displacement against lag
   time, and the root of it (RMSD, a distance), with the population mean in
   bold.

4. <tile>_motion_hist.png + <tile>_track_motion.csv
   Histograms (number of tracks on y) of each track's largest MSD value, its
   largest single frame-to-frame jump, and its largest off-path frame (a
   frame where the track left its own path for 1-2 frames and came back: a
   slip onto another object, or a nucleus mask placed beside the nucleus).
   --msd-cutoff / --step-cutoff / --off-path-cutoff draw a cutoff and count
   what it removes; --off-path-cutoff also lists the frames.  The same
   numbers are columns in the analyzeDirFUCCI CSV (max_msd_px2, max_step_px,
   max_off_path_px per track; off_path_px per frame).

--track takes any track id; its whole lineage (from the founder cell at t1
down through every division) is drawn.  Without --track the lineage with the
most divisions and frames is picked.  --outline-tracks picks the cells for
output 2 (each brings its lineage); by default just the chosen lineage.
Positions are nucleus centroids by default (--centroid cell for the
brightfield centroid the tracker links on).

Needs a run made after diagnostic masks were added (they are saved to
<tile>_masks/fused_tNNN.npz and reference.npz).  Output 3 only needs the CSV.
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd

import fucci_pipeline as fp


# --------------------------------------------------------------------------- #
# lineage bookkeeping
# --------------------------------------------------------------------------- #
def track_table(df):
    """One row per track: parent, root, first/last frame, frames present."""
    g = df.groupby('track_id')
    out = pd.DataFrame({
        'parent': g['parent_track_id'].first(),
        'root': g['root_track_id'].first(),
        'start': g['timepoint'].min(),
        'end': g['timepoint'].max(),
        'n': g.size(),
        'end_reason': g['track_end_reason'].first(),
    })
    out['root'] = out['root'].fillna(pd.Series(out.index, index=out.index))
    return out


def lineage_of(tracks, tid):
    """All track ids sharing tid's founder, parents before daughters."""
    root = tracks.loc[tid, 'root']
    members = tracks[tracks['root'] == root]
    return list(members.sort_values(['start', 'n'],
                                    ascending=[True, False]).index)


def pick_lineage(tracks):
    """The founder whose lineage has the most divisions, then the most frames."""
    agg = tracks.groupby('root').agg(n_tracks=('n', 'size'), frames=('n', 'sum'))
    best = agg.sort_values(['n_tracks', 'frames'], ascending=False).index[0]
    return int(best)


def branch_colors(ids):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('tab10')
    return {t: cmap(i % 10) for i, t in enumerate(ids)}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def tile_stem(csv_path):
    base = os.path.basename(csv_path)
    return re.sub(r'_fucci\.csv$', '', base) if base.endswith('_fucci.csv') \
        else os.path.splitext(base)[0]


def load_reference(masks_dir):
    p = os.path.join(masks_dir, 'reference.npz')
    if not os.path.exists(p):
        return None
    r = np.load(p)
    return {'bf': r['bf'], 'fluor_max': r['fluor_max']}


def fused(masks_dir, tp):
    p = os.path.join(masks_dir, f'fused_t{tp:03d}.npz')
    if not os.path.exists(p):
        return None, None
    m = np.load(p)
    return m['cell'], m['nuc']


add_position = fp.add_position


def stretch(img, lo=1, hi=99.5):
    a, b = np.percentile(img, (lo, hi))
    return np.clip((img.astype(float) - a) / max(b - a, 1e-9), 0, 1)


# --------------------------------------------------------------------------- #
# 1. one lineage's path
# --------------------------------------------------------------------------- #
def plot_lineage(df, tracks, tid, ref, out_png, out_csv, margin=120, log=print):
    import matplotlib.pyplot as plt

    ids = lineage_of(tracks, tid)
    col = branch_colors(ids)
    sub = df[df['track_id'].isin(ids)].sort_values(['track_id', 'timepoint'])

    rows = []
    for t in ids:
        g = sub[sub['track_id'] == t]
        par = tracks.loc[t, 'parent']
        for _, r in g.iterrows():
            rows.append({'track_id': t, 'parent_track_id': par,
                         'timepoint': int(r.timepoint),
                         'time_h': (r.timepoint - 1) * fp.FRAME_INTERVAL_MIN / 60,
                         'centroid_row': round(r.pos_row, 1),
                         'centroid_col': round(r.pos_col, 1),
                         'nuc_source': r.get('nuc_source', ''),
                         'phase': r.get('phase', '')})
    table = pd.DataFrame(rows)
    table.to_csv(out_csv, index=False)

    log(f'lineage of track {tid} (founder {int(tracks.loc[tid, "root"])}): '
        f'{len(ids)} track(s)')
    for t in ids:
        info = tracks.loc[t]
        par = '' if pd.isna(info.parent) else f', daughter of {int(info.parent)}'
        log(f'  track {t}: t{info.start}-t{info.end}, {info.n} frames, '
            f'ends {info.end_reason}{par}')
        g = table[table['track_id'] == t]
        log('    ' + '  '.join(f't{r.timepoint}:({r.centroid_row:.0f},'
                               f'{r.centroid_col:.0f})'
                               for r in g.itertuples()))

    r0, r1 = sub.pos_row.min() - margin, sub.pos_row.max() + margin
    c0, c1 = sub.pos_col.min() - margin, sub.pos_col.max() + margin
    fig, ax = plt.subplots(figsize=(8, 8 * (r1 - r0) / max(c1 - c0, 1) + 0.6))
    if ref is not None:
        H, W = ref['fluor_max'].shape
        rr0, rr1 = int(max(r0, 0)), int(min(r1, H))
        cc0, cc1 = int(max(c0, 0)), int(min(c1, W))
        bg = stretch(np.sqrt(ref['fluor_max'][rr0:rr1, cc0:cc1]))
        ax.imshow(bg, cmap='gray', extent=(cc0, cc1, rr1, rr0), alpha=0.8)
    for t in ids:
        g = sub[sub['track_id'] == t]
        xs, ys, ts = list(g.pos_col), list(g.pos_row), list(g.timepoint)
        par = tracks.loc[t, 'parent']
        if not pd.isna(par):
            pg = sub[sub['track_id'] == par]
            if len(pg):
                xs = [pg.pos_col.iloc[-1]] + xs
                ys = [pg.pos_row.iloc[-1]] + ys
                ts = [pg.timepoint.iloc[-1]] + ts
        for i in range(len(xs) - 1):
            gap = ts[i + 1] - ts[i] > 1
            ax.plot(xs[i:i + 2], ys[i:i + 2], color=col[t], lw=2,
                    ls='--' if gap else '-')
        hollow = (g['nuc_source'] != 'segmented') if 'nuc_source' in g \
            else np.zeros(len(g), bool)
        ax.scatter(g.pos_col[hollow], g.pos_row[hollow], s=14,
                   facecolor='none', edgecolor=col[t], zorder=3)
        ax.scatter(g.pos_col[~hollow], g.pos_row[~hollow], s=10, color=col[t],
                   zorder=3, label=f'track {t}: t{tracks.loc[t, "start"]}-'
                                   f't{tracks.loc[t, "end"]}')
        ax.annotate(str(t), (g.pos_col.iloc[0], g.pos_row.iloc[0]),
                    color='white', fontsize=9, fontweight='bold',
                    xytext=(4, 4), textcoords='offset points')
        if tracks.loc[t, 'end_reason'] == 'divided':
            ax.scatter(g.pos_col.iloc[-1], g.pos_row.iloc[-1],
                       marker='*', s=220, color='yellow', edgecolor='black',
                       zorder=4)
    first = sub[sub['track_id'] == ids[0]].iloc[0]
    ax.scatter(first.pos_col, first.pos_row, marker='o', s=90,
               facecolor='none', edgecolor='white', lw=2, zorder=4)
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    ax.set_xlabel('column (px)')
    ax.set_ylabel('row (px)')
    ax.set_title(f'track {tid} lineage: centroid in every frame\n'
                 f'circle = start, star = division, dashed = missed frames, '
                 f'hollow dot = nucleus not segmented that frame', fontsize=9)
    ax.legend(fontsize=7, loc='best', framealpha=0.7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return ids


# --------------------------------------------------------------------------- #
# 2. outlines through time
# --------------------------------------------------------------------------- #
def blue_cmap():
    """Light blue -> dark blue -> purple.  Dark end last, so late frames
    stand out against the earlier ones they are drawn over."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        'lightblue_purple', ['#bfe6ff', '#5aaef0', '#1f5fbf', '#1a237e',
                             '#4a0e6b'])


def plot_outlines(df, tracks, tids, masks_dir, ref, out_png, every=1,
                  margin=40, log=print, color_by='age', scale='shared'):
    """color_by 'age': colour by time since the track began, so each
    daughter starts again at the colour its parent had in its first frame
    and the separate cells read as separate runs of the colour map.
    'time': colour by absolute time in the movie.
    scale (age only) 'shared': one colour bar in hours since birth, the same
    for every track; 'per_track': each track runs the whole colour map over
    its own life, so a short track still shows which way it went."""
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from skimage.measure import find_contours

    ids = []
    for t in tids:
        ids += [x for x in lineage_of(tracks, t) if x not in ids]
    sub = df[df['track_id'].isin(ids)]
    tps = sorted(sub.timepoint.unique())[::every]
    dt_h = fp.FRAME_INTERVAL_MIN / 60
    start = {t: int(tracks.loc[t, 'start']) for t in ids}
    life = {t: max(int(tracks.loc[t, 'end']) - start[t], 1) for t in ids}
    if color_by == 'time':
        norm = colors.Normalize(0, (df.timepoint.max() - 1) * dt_h)
        cb_label = 'time (h)'
    elif scale == 'per_track':
        norm = colors.Normalize(0, 1)
        cb_label = "fraction of the track's life (0 = first frame, 1 = last)"
    else:
        norm = colors.Normalize(0, max(life.values()) * dt_h)
        cb_label = 'time since the track began (h)'

    def colour_value(tid, tp):
        if color_by == 'time':
            return (tp - 1) * dt_h
        if scale == 'per_track':
            return (tp - start[tid]) / life[tid]
        return (tp - start[tid]) * dt_h

    cmap = blue_cmap()
    contours = {'cell': [], 'nuc': []}          # (colour value, [N,2] row/col)
    rmin = cmin = np.inf
    rmax = cmax = -np.inf
    for tp in tps:
        cell, nuc = fused(masks_dir, tp)
        if cell is None:
            continue
        here = sub[sub.timepoint == tp].dropna(subset=['cell_label'])
        for lab, tid in zip(here.cell_label.astype(int), here.track_id):
            th = colour_value(int(tid), tp)
            for kind, m in (('cell', cell), ('nuc', nuc)):
                rr, cc = np.nonzero(m == lab)
                if rr.size == 0:
                    continue
                r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
                crop = np.pad(m[r0:r1, c0:c1] == lab, 1)
                for cnt in find_contours(crop.astype(float), 0.5):
                    cnt = cnt + [r0 - 1, c0 - 1]
                    contours[kind].append((th, cnt))
                rmin, rmax = min(rmin, r0), max(rmax, r1)
                cmin, cmax = min(cmin, c0), max(cmax, c1)
    if not contours['cell']:
        log('  no saved masks found for these cells -- re-run the tile with '
            'this version of the pipeline (masks go to <tile>_masks/)')
        return
    r0, r1 = int(rmin - margin), int(rmax + margin)
    c0, c1 = int(cmin - margin), int(cmax + margin)
    if ref is not None:
        H, W = ref['bf'].shape
        r0, r1, c0, c1 = max(r0, 0), min(r1, H), max(c0, 0), min(c1, W)

    aspect = (r1 - r0) / max(c1 - c0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2 * aspect + 1.2))
    titles = {'cell': 'brightfield cell mask', 'nuc': 'nucleus mask (intensity)'}
    for ax, kind in zip(axes, ('cell', 'nuc')):
        if ref is not None:
            # kept dark so light-blue early outlines still show
            bg = (stretch(ref['bf'][r0:r1, c0:c1]) * 0.45 if kind == 'cell'
                  else stretch(np.sqrt(ref['fluor_max'][r0:r1, c0:c1])) * 0.6)
            ax.imshow(bg, cmap='gray', vmin=0, vmax=1,
                      extent=(c0, c1, r1, r0))
        for th, cnt in contours[kind]:
            ax.plot(cnt[:, 1], cnt[:, 0], color=cmap(norm(th)), lw=0.8,
                    alpha=0.9)
        for t in ids:
            g = sub[sub.track_id == t].sort_values('timepoint')
            ax.annotate(str(t), (g.centroid_col.iloc[0], g.centroid_row.iloc[0]),
                        color='white', fontsize=8, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.15', fc='black',
                                  alpha=0.5, lw=0))
        ax.set_xlim(c0, c1)
        ax.set_ylim(r1, r0)
        ax.set_title(titles[kind], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                      fraction=0.025, pad=0.01)
    cb.set_label(cb_label)
    how = ('time' if color_by == 'time' else
           'time since each track began (restarts at every division)')
    fig.suptitle(f'outlines of tracks {", ".join(map(str, ids))} in every '
                 f'{"" if every == 1 else f"{every}th "}frame, coloured by {how}',
                 fontsize=10)
    fig.savefig(out_png, dpi=170, bbox_inches='tight')
    plt.close(fig)
    log(f'  outlined {len(ids)} track(s) over {len(tps)} frames: {ids}')


# --------------------------------------------------------------------------- #
# 3. mean squared displacement
# --------------------------------------------------------------------------- #
track_msd = fp.track_msd


def plot_msd(df, out_png, out_csv, pixel_size=None, min_frames=8,
             max_lag_frac=0.5, min_pairs=4, centroid='nucleus', loglog=False,
             log=print):
    import matplotlib.pyplot as plt

    d = df.copy()
    if 'mask_source' in d:
        d = d[d['mask_source'] == 'segmented']
    d = add_position(d, centroid)
    rcol, ccol = 'pos_row', 'pos_col'
    scale = pixel_size or 1.0
    unit = 'µm' if pixel_size else 'px'
    dt_h = fp.FRAME_INTERVAL_MIN / 60

    rows, curves = [], {}
    for tid, g in d.groupby('track_id'):
        if len(g) < min_frames:
            continue
        g = g.sort_values('timepoint')
        span = g.timepoint.max() - g.timepoint.min()
        max_lag = max(1, int(span * max_lag_frac))
        rc = g[[rcol, ccol]].to_numpy(float) * scale
        m = track_msd(g.timepoint.to_numpy(), rc, max_lag, min_pairs)
        if not m:
            continue
        curves[tid] = m
        for lag, v in m.items():
            rows.append({'track_id': tid, 'lag_frames': lag,
                         'lag_h': lag * dt_h, f'msd_{unit}2': v,
                         f'rmsd_{unit}': np.sqrt(v),
                         'generation': g.generation.iloc[0]
                         if 'generation' in g else np.nan})
    table = pd.DataFrame(rows)
    table.to_csv(out_csv, index=False)
    if table.empty:
        log('  no track is long enough for an MSD (--min-frames)')
        return
    mean = table.groupby('lag_frames')[f'msd_{unit}2'].agg(['mean', 'count'])
    mean = mean[mean['count'] >= max(3, len(curves) // 10)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, root in zip(axes, (False, True)):
        for tid, m in curves.items():
            lags = np.array(sorted(m)) * dt_h
            v = np.array([m[k] for k in sorted(m)])
            ax.plot(lags, np.sqrt(v) if root else v, color='tab:blue',
                    lw=0.6, alpha=0.35)
        mv = mean['mean'].to_numpy()
        ax.plot(mean.index.to_numpy() * dt_h, np.sqrt(mv) if root else mv,
                color='black', lw=2.5, label=f'mean of {len(curves)} tracks')
        ax.set_xlabel('lag time (h)')
        ax.set_ylabel(f'RMSD ({unit})' if root else f'MSD ({unit}²)')
        ax.set_title('root mean squared displacement' if root
                     else 'mean squared displacement', fontsize=10)
        if loglog:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.legend(fontsize=8)
    fig.suptitle(f'one line per track ({centroid} centroid, tracks with '
                 f'>= {min_frames} frames, lags up to {max_lag_frac:.0%} of '
                 f'each track)', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # slope of the mean on log-log: 1 = random walk, 2 = straight line
    lg = mean.index.to_numpy() * dt_h
    ok = (lg > 0) & (mean['mean'].to_numpy() > 0)
    if ok.sum() >= 3:
        alpha = np.polyfit(np.log(lg[ok]), np.log(mean['mean'].to_numpy()[ok]), 1)[0]
        log(f'  MSD of {len(curves)} tracks; mean MSD at 1 h = '
            f'{np.interp(1.0, lg, mean["mean"]):.0f} {unit}², log-log slope '
            f'{alpha:.2f} (1 = random walk, 2 = straight-line motion)')


# --------------------------------------------------------------------------- #
# 4. histogram of each track's largest MSD / largest jump, for a cutoff
# --------------------------------------------------------------------------- #
def plot_motion_hist(motion, out_png, msd_cutoff=None, step_cutoff=None,
                     off_path_cutoff=None, bins=40, log=print):
    """Two histograms, number of tracks on y: the peak of each track's MSD
    curve (log-spaced bins -- it spans orders of magnitude), and its largest
    single frame-to-frame jump.  Optional cutoffs are drawn and counted."""
    import matplotlib.pyplot as plt

    m = pd.DataFrame.from_dict(motion, orient='index')
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    panels = (('max_msd_px2', 'largest MSD value per track (px²)', msd_cutoff,
               True),
              ('max_step_px', 'largest frame-to-frame jump per track (px)',
               step_cutoff, False),
              ('max_off_path_px', 'largest off-path frame per track (px)\n'
               '(one frame away from its own path: slip or misplaced nucleus)',
               off_path_cutoff, False))
    for ax, (col, label, cut, logx) in zip(axes, panels):
        v = m[col].dropna()
        v = v[v > 0] if logx else v
        if v.empty:
            ax.set_title(f'{label}: no data')
            continue
        edges = (np.logspace(np.log10(v.min()), np.log10(v.max()), bins + 1)
                 if logx else np.linspace(0, v.max(), bins + 1))
        ax.hist(v, bins=edges, color='tab:blue', edgecolor='white')
        if logx:
            ax.set_xscale('log')
        ax.set_xlabel(label)
        ax.set_ylabel('number of tracks')
        q = v.quantile([0.5, 0.9, 0.95, 0.99])
        txt = (f'n={len(v)}  median {q[0.5]:.0f}  p90 {q[0.9]:.0f}  '
               f'p95 {q[0.95]:.0f}  p99 {q[0.99]:.0f}')
        if cut is not None:
            above = int((v > cut).sum())
            ax.axvline(cut, color='red', ls='--', lw=1.5)
            txt += f'\ncutoff {cut:g}: {above} track(s) above ({100 * above / len(v):.0f}%)'
        ax.set_title(txt, fontsize=8)
        log(f'  {col}: ' + txt.replace('\n', ' | '))
    fig.suptitle('pick a cutoff: tracks above it are the ones to inspect or '
                 'filter out', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description='Lineage path, outlines through time and MSD for one tile.')
    ap.add_argument('csv', help='a <well>-<tile>_fucci.csv from analyzeDirFUCCI')
    ap.add_argument('--track', type=int, default=None,
                    help='any track id; its whole lineage is drawn (default: '
                         'the lineage with the most divisions)')
    ap.add_argument('--outline-tracks', type=int, nargs='*', default=None,
                    help='tracks for the outline image; each brings its whole '
                         'lineage (default: just --track\'s lineage)')
    ap.add_argument('--outline-color', choices=['age', 'time'], default='age',
                    help="outline colour: 'age' = time since the track began, "
                         'restarting at each division so every daughter starts '
                         "light again (default); 'time' = time in the movie")
    ap.add_argument('--outline-scale', choices=['shared', 'per_track'],
                    default='shared',
                    help="with --outline-color age: 'shared' = one colour bar in "
                         "hours for all tracks (default); 'per_track' = each "
                         'track spans the whole colour map, so short tracks '
                         'show their direction too')
    ap.add_argument('--every', type=int, default=1,
                    help='draw outlines of every Nth frame (default 1)')
    ap.add_argument('--masks-dir', default=None,
                    help='default: <tile>_masks beside the CSV')
    ap.add_argument('--output-dir', default=None,
                    help='default: neoPIP_FUCCI_outputs/')
    ap.add_argument('--pixel-size', type=float, default=None,
                    help='um per pixel, so the MSD is in um^2 instead of px^2')
    ap.add_argument('--min-frames', type=int, default=8,
                    help='shortest track included in the MSD (default 8)')
    ap.add_argument('--centroid', choices=['nucleus', 'cell'], default='nucleus',
                    help='point plotted and used for the MSD: the nucleus '
                         'centroid (default; segmented or carried, falls back '
                         'to the cell centroid where there is none) or the '
                         'brightfield cell centroid the tracker links on')
    ap.add_argument('--loglog', action='store_true', help='log-log MSD axes')
    ap.add_argument('--msd-cutoff', type=float, default=None,
                    help='draw this max-MSD cutoff (px^2) on the histogram and '
                         'count the tracks above it')
    ap.add_argument('--step-cutoff', type=float, default=None,
                    help='same for the largest frame-to-frame jump (px)')
    ap.add_argument('--off-path-cutoff', type=float, default=None,
                    help='same for the largest off-path frame (px); also lists '
                         'every frame above it')
    ap.add_argument('--add-columns', action='store_true',
                    help='also write max_msd_px2 / max_step_px / '
                         'max_step_sq_px2 / n_steps into the CSV itself (for '
                         'CSVs made before analyzeDirFUCCI added them)')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    stem = tile_stem(args.csv)
    out_dir = args.output_dir or fp.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    masks_dir = args.masks_dir or os.path.join(os.path.dirname(
        os.path.abspath(args.csv)), f'{stem}_masks')
    df = add_position(df, args.centroid)
    tracks = track_table(df)
    ref = load_reference(masks_dir) if os.path.isdir(masks_dir) else None
    if ref is None:
        print(f'note: no {masks_dir}/reference.npz -- plots are drawn without '
              f'a background image, and the outline image needs the saved '
              f'masks from a run of this version of the pipeline')

    tid = args.track if args.track is not None else pick_lineage(tracks)
    if tid not in tracks.index:
        raise SystemExit(f'track {tid} is not in {args.csv}')

    print('\n1. lineage path')
    p1 = os.path.join(out_dir, f'{stem}_lineage{tid}.png')
    c1 = os.path.join(out_dir, f'{stem}_lineage{tid}_centroids.csv')
    plot_lineage(df, tracks, tid, ref, p1, c1)
    print(f'  wrote {p1}\n  wrote {c1}')

    print('\n2. outlines through time')
    otids = args.outline_tracks or [tid]
    suffix = ('time' if args.outline_color == 'time' else
              'age' if args.outline_scale == 'shared' else 'age_pertrack')
    p2 = os.path.join(out_dir, f'{stem}_outlines_{suffix}.png')
    if os.path.isdir(masks_dir):
        plot_outlines(df, tracks, otids, masks_dir, ref, p2, every=args.every,
                      color_by=args.outline_color, scale=args.outline_scale)
        if os.path.exists(p2):
            print(f'  wrote {p2}')
    else:
        print(f'  skipped: no masks folder {masks_dir}')

    print('\n3. MSD')
    p3 = os.path.join(out_dir, f'{stem}_msd.png')
    c3 = os.path.join(out_dir, f'{stem}_msd.csv')
    plot_msd(df, p3, c3, pixel_size=args.pixel_size, min_frames=args.min_frames,
             centroid=args.centroid, loglog=args.loglog)
    print(f'  wrote {p3}\n  wrote {c3}')

    print('\n4. largest MSD / jump per track')
    raw = pd.read_csv(args.csv)
    motion = fp.track_motion(raw, args.centroid, min_frames=args.min_frames)
    p4 = os.path.join(out_dir, f'{stem}_motion_hist.png')
    c4 = os.path.join(out_dir, f'{stem}_track_motion.csv')
    plot_motion_hist(motion, p4, msd_cutoff=args.msd_cutoff,
                     step_cutoff=args.step_cutoff,
                     off_path_cutoff=args.off_path_cutoff)
    if args.off_path_cutoff is not None:
        rows = fp.add_motion_columns(raw, args.centroid,
                                     min_frames=args.min_frames)
        bad = rows[rows['off_path_px'] > args.off_path_cutoff]
        cols = [c for c in ('track_id', 'timepoint', 'off_path_px',
                            'nuc_source', 'cell_source', 'mitotic_flag')
                if c in bad]
        print(f'  {len(bad)} frame(s) in {bad.track_id.nunique()} track(s) '
              f'off their path by more than {args.off_path_cutoff:g} px '
              f'(mitotic_flag True = probably real rounding, not an error):')
        print(bad.sort_values('off_path_px', ascending=False)[cols]
              .round(1).to_string(index=False))
    pd.DataFrame.from_dict(motion, orient='index').rename_axis(
        'track_id').to_csv(c4)
    print(f'  wrote {p4}\n  wrote {c4}')
    chk = fp.add_division_check(raw)
    sus = chk[chk.mitosis_no_division].groupby('track_id').mitosis_seen_tp.first()
    print(f'\n5. tracks that looked mitotic but never divided: {len(sus)}'
          + (': ' + ', '.join(f'{int(t)} (round t{int(v)})'
                              for t, v in sus.items()) if len(sus) else ''))
    if args.add_columns:
        out = fp.add_division_check(fp.add_motion_columns(
            raw, args.centroid, min_frames=args.min_frames))
        out.to_csv(args.csv, index=False)
        print(f'  added the motion and division-check columns to {args.csv}')


if __name__ == '__main__':
    main()
