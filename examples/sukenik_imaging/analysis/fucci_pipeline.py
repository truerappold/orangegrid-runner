"""
NeoPIP-FUCCI timelapse: brightfield tracking plus per-cell cell-cycle features.

Built on timelapse_tracking.py (nearest-nuclear-centroid matching, division
following, lineage bookkeeping) but for a different experiment layout and a
different measurement problem.

Layout (BF_timelapse_20261003_DoubleThymidine_1)
------------------------------------------------
Flat folder, no prefixes, three .vsi per well/tile, each holding the whole
series:  <WELL>-<TILE>_<CHANNEL>.vsi  with CHANNEL in
    "BF Cam 1", "mCherryNeoPIPFUCCI", "mVenusNeoPIPFUCCI"
Each is 2048 x 2048, C=1, Z=1, T=96, 16-bit, 15 min per frame.

Why brightfield
---------------
PIP-FUCCI has a window in early S where mVenus-PIP has just been destroyed and
mCherry-geminin has not yet accumulated, so BOTH markers are near zero.  A cell
segmented on fluorescence would simply vanish for those frames and its track
would break exactly at the transition that matters most.  Brightfield is present
in every frame regardless of cycle phase, so cells are segmented and tracked on
brightfield and the fluorescence is only ever measured, never used to find cells.

All three channels come off one camera, so only frame-to-frame stage drift needs
correcting -- there is no fixed inter-camera offset.

Measurement
-----------
Cellpose segments whole cells on brightfield.  Both markers are nuclear, so a
whole-cell mean is diluted by cytoplasm; the dilution factor is roughly constant
per cell, so for *classification* the thresholds absorb it.  Upper percentiles
within the cell (p75, p90) are reported alongside the mean because they are much
less diluted while still being defined in every frame, including the dark window.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

import timelapse_tracking as tt

# --------------------------------------------------------------------------- #
# channel layout for this experiment
# --------------------------------------------------------------------------- #
CH_BF = 0
CH_MCHERRY = 1
CH_MVENUS = 2
N_CHANNELS = 3

CHANNEL_FILES = {
    CH_BF: 'BF Cam 1',
    CH_MCHERRY: 'mCherryNeoPIPFUCCI',
    CH_MVENUS: 'mVenusNeoPIPFUCCI',
}

FRAME_INTERVAL_MIN = 15.0

# Every FUCCI script writes here by default, so results from different
# experiments land in one place that can be inspected without hunting.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'neoPIP_FUCCI_outputs')


def output_path(name):
    """A bare filename goes into OUTPUT_DIR; anything with a directory is kept."""
    if name is None or os.path.dirname(name):
        return name
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, name)

# --------------------------------------------------------------------------- #
# EDIT ME: empirically determined background, in raw camera counts
# --------------------------------------------------------------------------- #
# Subtracted as a flat constant per channel, then clipped at zero.  Read these
# off an empty region of your own images -- the defaults are placeholders, not
# measurements.  They set the zero point for every intensity threshold below, so
# getting them wrong shifts every phase call.
# Measured on A1-1 with `--measure-background` (pixels outside every cell mask,
# dilated 15 px first).  These are the 25th percentile rather than the median,
# because at ~49% confluence the "background" region is narrow gaps between
# cells and the median there is pulled up by halo; p25 is closer to true empty
# field.  Re-measure on a cell-free well when you have one.
BACKGROUND = {
    CH_MCHERRY: 1890.0,   # median was 2175
    CH_MVENUS: 1628.0,    # median was 1825
    CH_BF: 0.0,           # brightfield is not background-subtracted
}

# --------------------------------------------------------------------------- #
# EDIT ME: phase-calling thresholds
# --------------------------------------------------------------------------- #
# PIP-FUCCI dynamics these encode:
#   mVenus-PIP   HIGH in G1 -> destroyed sharply at the G1/S transition
#                -> re-accumulates through G2
#   mCherry-Gem  LOW in G1 -> accumulates through S and G2/M
#                -> destroyed at anaphase
#   both near zero = early S, just after the G1/S transition
#
# Intensities below are background-subtracted whole-cell p90 values, in camera
# counts.  Start by plotting mcherry_p90 and mvenus_p90 against timepoint for a
# few long tracks, then set the cutoffs where the populations separate.
PHASE_THRESHOLDS = {
    'venus_high': 300.0,    # mVenus clearly present (G1, or re-accumulated G2)
    'venus_low': 120.0,     # mVenus clearly destroyed
    'cherry_high': 300.0,   # geminin well accumulated (G2/M)
    'cherry_low': 120.0,    # geminin not yet accumulated
    # morphology gate for mitosis: cells round up and shrink in projection
    'mitotic_circularity': 0.85,
    'mitotic_area_frac': 0.75,   # area relative to that track's own median
}

PHASES = ('M', 'G1', 'earlyS', 'S', 'G2', 'ambiguous')

VSI_RE = re.compile(r'^(?P<well>[A-H]\d{1,2})-(?P<tile>\d+)_(?P<channel>.+)\.vsi$')


# --------------------------------------------------------------------------- #
# 1. discovery
# --------------------------------------------------------------------------- #
def parse_vsi_name(basename):
    """'A1-2_mCherryNeoPIPFUCCI.vsi' -> {'well': 'A1', 'tile': 2, 'channel': ...}"""
    m = VSI_RE.match(basename)
    if m is None:
        return None
    d = m.groupdict()
    d['tile'] = int(d['tile'])
    return d


def build_filename(image_dir, well, tile, channel):
    return os.path.join(image_dir, f'{well}-{tile}_{channel}.vsi')


def discover_tiles(filelist, channels=None):
    """Sorted [(well, tile)] that have all three channels, plus any problems."""
    channels = list(CHANNEL_FILES.values()) if channels is None else channels
    seen = defaultdict(set)
    for path in filelist:
        info = parse_vsi_name(os.path.basename(path))
        if info:
            seen[(info['well'], info['tile'])].add(info['channel'])
    tiles, problems = [], []
    for key, chans in sorted(seen.items(), key=lambda kv: (kv[0][0][0],
                                                           int(kv[0][0][1:]),
                                                           kv[0][1])):
        missing = [c for c in channels if c not in chans]
        if missing:
            problems.append(f'{key[0]} tile {key[1]}: missing '
                            f'{", ".join(missing)} — skipped')
        else:
            tiles.append(key)
    return tiles, problems


# --------------------------------------------------------------------------- #
# 2. per-cell features
# --------------------------------------------------------------------------- #
def subtract_background(plane):
    """Flat per-channel background subtraction, clipped at zero.

    Returns float32 so the clipping does not wrap around as it would in uint16.
    """
    out = plane.astype(np.float32)
    for ch, bg in BACKGROUND.items():
        if ch < out.shape[2] and bg:
            out[:, :, ch] = np.clip(out[:, :, ch] - bg, 0, None)
    return out


def measure_frame(plane_corrected, mask_cell, pixel_size_um=None):
    """All per-cell features for one frame.

    Returns {label: {feature: value}}.  regionprops is called once for the whole
    frame with a multichannel intensity image, which is much faster than
    indexing each label separately.
    """
    import skimage as ski

    if mask_cell.max() == 0:
        return {}

    props = ski.measure.regionprops(mask_cell, intensity_image=plane_corrected)
    out = {}
    for p in props:
        area = float(p.area)
        perim = float(p.perimeter)
        circ = (4 * np.pi * area / perim ** 2) if perim > 0 else np.nan
        rec = {
            'area_px': area,
            'perimeter_px': perim,
            'circularity': circ,
            'solidity': float(p.solidity),
            'eccentricity': float(p.eccentricity),
            'axis_major_px': float(p.axis_major_length),
            'axis_minor_px': float(p.axis_minor_length),
            'centroid_row': float(p.centroid[0]),
            'centroid_col': float(p.centroid[1]),
        }
        if pixel_size_um:
            rec['area_um2'] = area * pixel_size_um ** 2

        rec.update(intensity_features(p.image_intensity[p.image]))
        out[int(p.label)] = rec
    return out


INTENSITY_KEYS = tuple(
    [f'{n}_{s}' for n in ('mcherry', 'mvenus')
     for s in ('mean', 'median', 'p75', 'p90', 'max', 'total')]
    + ['bf_mean', 'cherry_venus_ratio'])


def intensity_features(pix):
    """Marker statistics from an [n_pixels, n_channels] array.

    Shared by every way a region gets measured (segmented cell, segmented
    nucleus, carried nucleus) so the columns mean the same thing whichever
    mask produced them.
    """
    rec = {}
    if pix.shape[0] == 0:
        return {k: np.nan for k in INTENSITY_KEYS}
    for ch, name in ((CH_MCHERRY, 'mcherry'), (CH_MVENUS, 'mvenus')):
        v = pix[:, ch]
        rec[f'{name}_mean'] = float(v.mean())
        rec[f'{name}_median'] = float(np.median(v))
        rec[f'{name}_p75'] = float(np.percentile(v, 75))
        rec[f'{name}_p90'] = float(np.percentile(v, 90))
        rec[f'{name}_max'] = float(v.max())
        rec[f'{name}_total'] = float(v.sum())
    rec['bf_mean'] = float(pix[:, CH_BF].mean())
    den = rec['mvenus_p90']
    rec['cherry_venus_ratio'] = (rec['mcherry_p90'] / den
                                 if den > 0 else np.nan)
    return rec


# --------------------------------------------------------------------------- #
# 3. division timing
# --------------------------------------------------------------------------- #
def division_timing(tracks, lineage, timepoints):
    """{(track_id, timepoint): (frames_since_division, frames_until_division)}

    frames_since_division is NaN for a founder track -- a cell present at the
    start of the series was never observed dividing, so its age is unknown
    rather than zero.  frames_until_division is NaN unless the track actually
    ends in an observed division.
    """
    out = {}
    for tid, info in lineage.items():
        born = info['start_tp'] if info.get('parent_track_id') is not None else None
        divides = info['end_tp'] if info.get('end_reason') == 'divided' else None
        for tp in timepoints:
            since = (tp - born) if born is not None and tp >= born else np.nan
            until = (divides - tp) if divides is not None and tp <= divides else np.nan
            out[(tid, tp)] = (since, until)
    return out


# --------------------------------------------------------------------------- #
# 4. phase calling
# --------------------------------------------------------------------------- #
def classify_phase(feat, median_area=None, thresholds=None):
    """Provisional phase for one cell in one frame.

    Deliberately a short, readable decision tree rather than anything fitted:
    every input it uses is also written to the CSV, so the call can be
    recomputed or overridden downstream without re-running the pipeline.

    Order matters.  Mitosis is tested first on morphology alone, because a cell
    in mitosis can carry almost any marker combination on its way through.
    """
    t = dict(PHASE_THRESHOLDS)
    if thresholds:
        t.update(thresholds)

    circ = feat.get('circularity', np.nan)
    area = feat.get('area_px', np.nan)
    V = feat.get('mvenus_p90', np.nan)
    C = feat.get('mcherry_p90', np.nan)

    if not np.isfinite(V) or not np.isfinite(C):
        return 'ambiguous'

    area_frac = (area / median_area
                 if median_area and np.isfinite(area) and median_area > 0
                 else np.nan)
    if (np.isfinite(circ) and circ >= t['mitotic_circularity']
            and np.isfinite(area_frac) and area_frac <= t['mitotic_area_frac']):
        return 'M'

    v_hi, v_lo = V >= t['venus_high'], V < t['venus_low']
    c_hi, c_lo = C >= t['cherry_high'], C < t['cherry_low']

    if v_hi and c_lo:
        return 'G1'          # PIP present, geminin not yet
    if v_lo and c_lo:
        return 'earlyS'      # the dark window just after the G1/S transition
    if v_lo and not c_lo:
        return 'S'           # PIP destroyed, geminin accumulating
    if not v_lo and c_hi:
        return 'G2'          # PIP re-accumulating with geminin high
    return 'ambiguous'


def classify_table(df, thresholds=None):
    """Add median_area_track, area_frac and phase columns to a long-format table."""
    df = df.copy()
    med = df.groupby('track_id')['area_px'].median().rename('median_area_track')
    df = df.join(med, on='track_id')
    df['area_frac'] = df['area_px'] / df['median_area_track']
    df['phase'] = [
        classify_phase(row, median_area=row.get('median_area_track'),
                       thresholds=thresholds)
        for row in df.to_dict('records')
    ]
    return df


# --------------------------------------------------------------------------- #
# 5. motion per track (MSD), for filtering
# --------------------------------------------------------------------------- #
MSD_DEFAULTS = dict(min_frames=8, max_lag_frac=0.5, min_pairs=4)


def add_position(df, centroid='nucleus'):
    """Columns pos_row / pos_col: the point motion is measured on.

    'nucleus' (default) uses the nucleus centroid -- in dual mode every row has
    one, segmented or carried -- and falls back to the cell centroid only for
    rows without it (a non-dual CSV, or an interpolated gap row).  The
    brightfield cell centroid moves whenever the outline changes shape, which
    on a faint flat cell is often by tens of px between frames.
    """
    df = df.copy()
    df['pos_row'], df['pos_col'] = df['centroid_row'], df['centroid_col']
    if centroid == 'nucleus' and 'nuc_centroid_row' in df:
        ok = df['nuc_centroid_row'].notna()
        df.loc[ok, 'pos_row'] = df.loc[ok, 'nuc_centroid_row']
        df.loc[ok, 'pos_col'] = df.loc[ok, 'nuc_centroid_col']
    return df


def track_msd(t_frames, rc, max_lag, min_pairs):
    """Time-averaged MSD of one track at lags 1..max_lag frames.

    Pairs are taken by actual timepoint, so a missed frame removes the pairs
    that need it instead of shifting every later lag by one.
    """
    pos = {int(t): np.asarray(p, float) for t, p in zip(t_frames, rc)}
    out = {}
    for lag in range(1, max_lag + 1):
        d2 = [np.sum((pos[t + lag] - pos[t]) ** 2) for t in pos if t + lag in pos]
        if len(d2) >= min_pairs:
            out[lag] = float(np.mean(d2))
    return out


def off_path(t_frames, rc, half=3):
    """Per frame: distance (px) from the median position of the frames
    around it (up to `half` either side, itself excluded).

    A cell moving fast stays near the median of its neighbours along its own
    path; a frame where the track briefly sat on another object -- or where
    the nucleus mask landed beside the nucleus -- stands far off it, and a
    1-2 frame excursion still does.  Needs at least one frame on each side,
    so the first and last frame of a track (where every neighbour lies ahead
    or behind and ordinary motion would look like an excursion) are NaN.
    """
    pos = {int(t): np.asarray(p, float) for t, p in zip(t_frames, rc)}
    out = []
    for t in (int(x) for x in t_frames):
        before = [pos[x] for x in range(t - half, t) if x in pos]
        after = [pos[x] for x in range(t + 1, t + half + 1) if x in pos]
        if before and after:
            # as many frames from each side, nearest first: a lopsided set
            # drags the median along the path and fast motion looks off-path
            k = min(len(before), len(after))
            med = np.median(before[-k:] + after[:k], axis=0)
            out.append(float(np.hypot(*(pos[t] - med))))
        else:
            out.append(np.nan)
    return np.array(out)


def track_motion(df, centroid='nucleus', min_frames=None, max_lag_frac=None,
                 min_pairs=None):
    """Per-track motion summary: {track_id: {column: value}}.

        max_msd_px2      peak of the track's MSD curve (the same curve
                         diagnose_tracks.py plots); NaN if the track is
                         shorter than min_frames
        max_step_px      largest displacement between two consecutive frames
                         -- one big value here is the signature of a track
                         hopping to another object
        max_step_sq_px2  the same, squared (one-frame squared displacement)
        n_steps          consecutive-frame pairs the step was taken over
        max_off_path_px  largest off_path_px of the track (see off_path)

    Only segmented rows count (interpolated gap rows are a guess).
    """
    p = dict(MSD_DEFAULTS)
    p.update({k: v for k, v in dict(min_frames=min_frames,
                                    max_lag_frac=max_lag_frac,
                                    min_pairs=min_pairs).items() if v is not None})
    d = df
    if 'mask_source' in d:
        d = d[d['mask_source'] == 'segmented']
    d = add_position(d, centroid)
    out = {}
    for tid, g in d.groupby('track_id'):
        g = g.sort_values('timepoint')
        t = g['timepoint'].to_numpy()
        rc = g[['pos_row', 'pos_col']].to_numpy(float)
        consec = np.diff(t) == 1
        steps = np.hypot(*np.diff(rc, axis=0).T)[consec] if len(t) > 1 else []
        rec = {'max_step_px': float(np.max(steps)) if len(steps) else np.nan,
               'n_steps': int(len(steps))}
        rec['max_step_sq_px2'] = rec['max_step_px'] ** 2
        dev = off_path(t, rc)
        rec['max_off_path_px'] = (float(np.nanmax(dev))
                                  if np.isfinite(dev).any() else np.nan)
        rec['max_msd_px2'] = np.nan
        if len(g) >= p['min_frames']:
            span = t.max() - t.min()
            m = track_msd(t, rc, max(1, int(span * p['max_lag_frac'])),
                          p['min_pairs'])
            if m:
                rec['max_msd_px2'] = max(m.values())
        out[int(tid)] = rec
    return out


def add_motion_columns(df, centroid='nucleus', **kw):
    """Motion columns for filtering.

    Per track (repeated on its rows): see track_motion.
    Per frame: off_path_px -- this frame's distance from the track's local
    path (see off_path).  Filter rows on it to drop just the bad frame, or
    tracks on max_off_path_px to drop the whole track.  Mitotic rounding also
    moves the centroid, so check mitotic_flag before calling a high value an
    error.
    """
    m = pd.DataFrame.from_dict(track_motion(df, centroid, **kw), orient='index')
    df = df.drop(columns=[c for c in list(m.columns) + ['off_path_px']
                          if c in df.columns])
    df = df.join(m, on='track_id')
    df['off_path_px'] = np.nan
    seg = df if 'mask_source' not in df else df[df['mask_source'] == 'segmented']
    pos = add_position(seg, centroid)
    for _tid, g in pos.groupby('track_id'):
        g = g.sort_values('timepoint')
        df.loc[g.index, 'off_path_px'] = off_path(
            g['timepoint'].to_numpy(), g[['pos_row', 'pos_col']].to_numpy(float))
    return df


# --------------------------------------------------------------------------- #
# 6. mitosis seen but no division recorded, for review
# --------------------------------------------------------------------------- #
def add_division_check(df, circularity=0.80, max_rim=0.85, min_interior=1.03,
                       min_after=2, newborn_frames=3):
    """Flag tracks that looked mitotic but never divided.

    A frame "looks mitotic" if the brightfield cell is round and has the halo
    of a rounded-up cell (bf_rim / bf_interior, dual mode only) -- the same
    tests the tracker's mitosis flag uses, minus the shrink test, so this
    also catches cells the tracker did not flag.  A track with such a frame
    that does not end 'divided', and continues at least `min_after` frames
    afterwards (so a cell still in mitosis at the end of the movie is not
    flagged), is where a daughter was probably missed and the other one
    carried on under the parent's id.

    A daughter is still round and haloed for a frame or two after
    cytokinesis, so the first `newborn_frames` frames of a track born from a
    division do not count.

    Adds per track: mitosis_no_division (bool) and mitosis_seen_tp (last
    mitotic-looking frame, NaN if none).  No-op without the halo columns.
    """
    df = df.copy()
    df['mitosis_no_division'] = False
    df['mitosis_seen_tp'] = np.nan
    if 'bf_rim' not in df or 'bf_interior' not in df:
        return df
    looks = ((df['circularity'] >= circularity) & (df['bf_rim'] <= max_rim)
             & (df['bf_interior'] >= min_interior))
    if 'mask_source' in df:
        looks &= df['mask_source'] == 'segmented'
    if 'parent_track_id' in df:
        born = df.groupby('track_id')['timepoint'].transform('min')
        looks &= ~(df['parent_track_id'].notna()
                   & (df['timepoint'] < born + newborn_frames))
    for tid, g in df[looks].groupby('track_id'):
        rows = df['track_id'] == tid
        last = int(g['timepoint'].max())
        df.loc[rows, 'mitosis_seen_tp'] = last
        ended = df.loc[rows, 'track_end_reason'].iloc[0]
        after = int((df.loc[rows, 'timepoint'] > last).sum())
        if ended != 'divided' and after >= min_after:
            df.loc[rows, 'mitosis_no_division'] = True
    return df
