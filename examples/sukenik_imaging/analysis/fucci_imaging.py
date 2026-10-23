"""
Imaging layer for the NeoPIP-FUCCI pipeline: loading, drift correction,
brightfield segmentation.

Separated from fucci_pipeline.py because this half needs a JVM (pyimagej), a GPU
(cellpose) and pystackreg.  The heavy imports live inside the functions that
need them, so the feature and phase-calling code in fucci_pipeline stays
importable and testable on its own.
"""

from __future__ import annotations

import numpy as np

import fucci_pipeline as fp
import timelapse_tracking as tt


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def _axis_index(ds):
    """{'X':0,'Y':1,'Time':2,...} -- ImageJ2 drops singleton axes, so read labels."""
    out = {}
    try:
        for i in range(int(ds.numDimensions())):
            out[str(ds.axis(i).type().getLabel())] = i
    except Exception:
        ndim = int(ds.numDimensions())
        names = ['X', 'Y'] + (['Time'] if ndim >= 3 else [])
        out = {n: i for i, n in enumerate(names[:ndim])}
    return out


def _n_frames(ds):
    a = _axis_index(ds)
    return int(ds.dimension(a['Time'])) if 'Time' in a else 1


def _plane_slice(ds, frame):
    a = _axis_index(ds)
    idx = [slice(None)] * int(ds.numDimensions())
    if 'Time' in a:
        idx[a['Time']] = frame
    elif frame not in (0, None):
        raise ValueError(f'frame {frame} requested but the Dataset has no Time axis')
    return tuple(idx)


def _release(ds):
    """net.imagej.Dataset has no close(); dispose the backing img if it has one."""
    try:
        img = ds.getImgPlus().getImg()
        if hasattr(img, 'dispose'):
            img.dispose()
    except Exception:
        pass


class TileReader:
    """Holds the three channel Datasets of one well/tile open while frames are read.

    Each .vsi here is ~800 MB on disk, so they are opened once per tile and
    planes pulled lazily rather than re-opened per frame.
    """

    def __init__(self, ij, image_dir, well, tile):
        self.ij = ij
        self.datasets = {}
        for ch, name in fp.CHANNEL_FILES.items():
            path = fp.build_filename(image_dir, well, tile, name)
            self.datasets[ch] = ij.io().open(path)
        counts = {ch: _n_frames(d) for ch, d in self.datasets.items()}
        if len(set(counts.values())) != 1:
            self.close()
            raise ValueError(f'{well} tile {tile}: channels disagree on frame '
                             f'count {counts} — cannot pair them into timepoints')
        self.n_frames = next(iter(counts.values()))

    def frame(self, frame_idx):
        """One timepoint as [H, W, 3] uint16: BF, mCherry, mVenus."""
        planes = []
        for ch in (fp.CH_BF, fp.CH_MCHERRY, fp.CH_MVENUS):
            ds = self.datasets[ch]
            f = None if self.n_frames == 1 else frame_idx
            planes.append(np.asarray(
                self.ij.py.from_java(ds[_plane_slice(ds, f)])))
        return np.dstack(planes).astype(np.uint16)

    def close(self):
        for ds in self.datasets.values():
            _release(ds)
        self.datasets.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# --------------------------------------------------------------------------- #
# drift correction
# --------------------------------------------------------------------------- #
def register_to_reference(plane, bf_reference, sr):
    """Warp a frame onto the first frame, in place.

    All three channels come off one camera, so there is no inter-camera offset
    to correct -- only stage drift.  The transform is measured once on
    brightfield and applied to all three channels, which keeps them mutually
    aligned by construction.

    Registering every frame to frame 1 rather than to its predecessor matters:
    chained registration accumulates drift over 96 frames, which would force the
    tracking cutoff to grow with time.
    """
    import skimage as ski

    if bf_reference is None:
        return plane

    def binar(img):
        if img.max() == img.min():
            return np.zeros_like(img, dtype=bool)
        return img >= ski.filters.threshold_otsu(img)

    ref, mov = binar(bf_reference), binar(plane[:, :, fp.CH_BF])
    if not (ref.any() and mov.any()):
        return plane
    matrix = sr.register(ref, mov)
    for ch in range(plane.shape[2]):
        plane[:, :, ch] = ski.transform.warp(
            plane[:, :, ch], matrix, preserve_range=True).astype('uint16')
    return plane


# --------------------------------------------------------------------------- #
# segmentation
# --------------------------------------------------------------------------- #
SEGMENT_CHANNELS = ('bf', 'sum', 'mcherry', 'mvenus', 'dual')


def segmentation_image(plane_raw, plane_corrected, segment_on):
    """The 2-D image Cellpose is run on.

    'sum' adds the two background-subtracted nuclear markers.  Fluorescence is
    far higher contrast than brightfield, so Cellpose cuts far fewer cells in
    half -- but note what summing does NOT fix: PIP-FUCCI has a window in early
    S where mVenus has just been destroyed and mCherry has not yet accumulated,
    so the sum is near zero there too and those cells vanish from the mask
    entirely.  Brightfield is the only channel present in every phase.  Use
    'sum' when segmentation quality is the limit and 'bf' when continuity is,
    and compare the per-frame counts.
    """
    if segment_on == 'bf':
        return plane_raw[:, :, fp.CH_BF]
    if segment_on == 'mcherry':
        return plane_corrected[:, :, fp.CH_MCHERRY]
    if segment_on == 'mvenus':
        return plane_corrected[:, :, fp.CH_MVENUS]
    if segment_on == 'sum':
        return (plane_corrected[:, :, fp.CH_MCHERRY]
                + plane_corrected[:, :, fp.CH_MVENUS])
    raise ValueError(f'segment_on must be one of {SEGMENT_CHANNELS}, '
                     f'got {segment_on!r}')


def segment_cells_bf(bf, model, flow_threshold=0.4, cellprob_threshold=0.0,
                     tile_norm_blocksize=0, batch_size=8, diameter=None):
    """Cellpose whole-cell segmentation on brightfield.

    Brightfield is lower contrast than a fluorescence channel, so expect to
    retune flow_threshold and cellprob_threshold on real tiles -- the defaults
    are carried over from the fluorescence pipeline and are a starting point
    only.  Cells are segmented here in every frame regardless of cycle phase,
    which is the whole reason for using brightfield.
    """
    kw = dict(batch_size=batch_size, flow_threshold=flow_threshold,
              cellprob_threshold=cellprob_threshold, progress=False,
              normalize={'tile_norm_blocksize': tile_norm_blocksize})
    if diameter:
        kw['diameter'] = diameter
    masks, _flows, _styles = model.eval(bf, **kw)
    return masks


def drop_edge_labels(mask):
    """Remove labels touching any border, then relabel 1..N.

    Edge-touching cells are partial, so both their area and their mean intensity
    are biased -- and area feeds the mitosis morphology gate.
    """
    import skimage as ski
    from skimage.segmentation import relabel_sequential

    edge = np.unique(np.concatenate(
        [mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]]))
    mask = mask.copy()
    mask[np.isin(mask, edge)] = 0
    mask, _, _ = relabel_sequential(mask)
    return mask.astype(np.int32)


# --------------------------------------------------------------------------- #
# one tile, end to end
# --------------------------------------------------------------------------- #
def process_tile(ij, sr, model, image_dir, well, tile, max_dist,
                 cellpose_kwargs=None, match_method='nearest',
                 first_step_max_dist=None, allow_division=True,
                 division_max_dist=None, division_thresholds=None, max_gap=0,
                 gap_dist_scale=1.0, predict_motion=True,
                 min_division_overlap=3, fill_gaps=False,
                 min_division_interval=0, heal_window=0, heal_frac=0.75,
                 dual_kwargs=None, masks_dir=None, reuse_masks=False,
                 min_area_px=0, max_frames=0,
                 pixel_size_um=None, segment_on='bf', keep_masks=False,
                 keep_planes=False, log=print):
    """Segment, track and measure a whole tile.

    Peak memory is one frame (2048 x 2048 x 3 uint16, ~25 MB) plus whatever the
    JVM holds for the three open files; the per-frame features are small and
    tracking only needs the centroids.  keep_masks retains the label images for
    diagnostics, which costs ~8 MB per frame.

    Tracking uses the *cell* centroid here, not a nuclear one: there is no
    nuclear mask to use, because the markers disappear entirely in early S.
    """
    if segment_on == 'dual':
        return _process_tile_dual(
            ij, sr, model, image_dir, well, tile, max_dist,
            cellpose_kwargs=cellpose_kwargs,
            first_step_max_dist=first_step_max_dist,
            allow_division=allow_division,
            division_max_dist=division_max_dist,
            division_thresholds=division_thresholds, max_gap=max_gap,
            gap_dist_scale=gap_dist_scale, predict_motion=predict_motion,
            min_division_overlap=min_division_overlap, fill_gaps=fill_gaps,
            min_division_interval=min_division_interval,
            heal_window=heal_window, heal_frac=heal_frac,
            dual_kwargs=dual_kwargs, masks_dir=masks_dir,
            reuse_masks=reuse_masks, max_frames=max_frames,
            pixel_size_um=pixel_size_um, keep_masks=keep_masks,
            keep_planes=keep_planes, log=log)

    reader = TileReader(ij, image_dir, well, tile)
    n = reader.n_frames
    if max_frames and max_frames < n:
        log(f'  limiting to the first {max_frames} of {n} frames')
        n = max_frames

    cellpose_kwargs = cellpose_kwargs or {}
    bf_reference = None
    features = {}          # {timepoint: {label: feature dict}}
    centroids = {}         # {timepoint: {label: (row, col)}}
    areas = {}             # {timepoint: {label: area}} for the division test
    masks = {}
    kept_planes = {}
    ref = {}

    try:
        for i in range(n):
            tp = i + 1
            plane = reader.frame(i)
            plane = register_to_reference(plane, bf_reference, sr)
            if tp == 1:
                bf_reference = plane[:, :, fp.CH_BF].copy()

            corrected = fp.subtract_background(plane)
            seg_img = segmentation_image(plane, corrected, segment_on)
            mask = drop_edge_labels(
                segment_cells_bf(seg_img, model, **cellpose_kwargs))
            if min_area_px:
                import skimage as ski
                from skimage.segmentation import relabel_sequential
                keep = np.zeros(mask.shape, mask.dtype)
                for p in ski.measure.regionprops(mask):
                    if p.area >= min_area_px:
                        keep[mask == p.label] = p.label
                mask, _, _ = relabel_sequential(keep)
                mask = mask.astype(np.int32)

            feats = fp.measure_frame(corrected, mask, pixel_size_um)
            if masks_dir:
                import fucci_dual as fd
                fd.save_fused(masks_dir, tp, mask, mask)
                fd.update_reference(ref, plane, corrected)

            features[tp] = feats
            centroids[tp] = {lab: (v['centroid_row'], v['centroid_col'])
                             for lab, v in feats.items()}
            areas[tp] = {lab: v['area_px'] for lab, v in feats.items()}
            if keep_masks:
                masks[tp] = mask
            if keep_planes:
                kept_planes[tp] = plane.copy()
            log(f'  frame {tp:3d}/{n}: {len(feats)} cells segmented '
                f'(on {segment_on})')
            del plane, corrected, seg_img
    finally:
        reader.close()
    if masks_dir:
        import fucci_dual as fd
        fd.save_reference(masks_dir, ref)

    timepoints = sorted(features)
    div_rejects = []
    tracks, records, lineage = tt.track_series(
        centroids, max_dist, method=match_method,
        first_step_max_dist=first_step_max_dist, areas_per_tp=areas,
        allow_division=allow_division, division_max_dist=division_max_dist,
        division_thresholds=division_thresholds, max_gap=max_gap,
        division_rejects=div_rejects, gap_dist_scale=gap_dist_scale,
        predict_motion=predict_motion,
        min_division_interval=min_division_interval)

    div_reverted = []
    if allow_division and min_division_overlap > 0:
        tracks, lineage, div_reverted = tt.prune_spurious_divisions(
            tracks, lineage, areas_per_tp=areas,
            min_overlap=min_division_overlap, log=log,
            heal_window=heal_window, heal_frac=heal_frac)

    timing = fp.division_timing(tracks, lineage, timepoints)

    rows = []
    for tp in timepoints:
        for lab, feat in features[tp].items():
            tid = tracks.get(tp, {}).get(lab)
            if tid is None:
                continue
            since, until = timing.get((tid, tp), (np.nan, np.nan))
            info = lineage.get(tid, {})
            row = {'well': well, 'tile': tile, 'track_id': int(tid),
                   'timepoint': tp,
                   'time_min': (tp - 1) * fp.FRAME_INTERVAL_MIN,
                   'cell_label': lab,
                   'frames_since_division': since,
                   'frames_until_division': until,
                   'parent_track_id': info.get('parent_track_id'),
                   'root_track_id': info.get('root_track_id'),
                   'generation': info.get('generation'),
                   'track_end_reason': info.get('end_reason')}
            row.update(feat)
            row['mask_source'] = 'segmented'
            rows.append(row)

    if fill_gaps:
        targets = gap_targets(tracks, centroids, areas)
        filled = measure_gap_frames(ij, sr, image_dir, well, tile, targets,
                                    bf_reference, pixel_size_um, log=log)
        for tp, by_track in filled.items():
            for tid, feat in by_track.items():
                since, until = timing.get((tid, tp), (np.nan, np.nan))
                info = lineage.get(tid, {})
                row = {'well': well, 'tile': tile, 'track_id': int(tid),
                       'timepoint': tp,
                       'time_min': (tp - 1) * fp.FRAME_INTERVAL_MIN,
                       'cell_label': np.nan,
                       'frames_since_division': since,
                       'frames_until_division': until,
                       'parent_track_id': info.get('parent_track_id'),
                       'root_track_id': info.get('root_track_id'),
                       'generation': info.get('generation'),
                       'track_end_reason': info.get('end_reason')}
                row.update(feat)
                for k in IMPUTED_BLANK:
                    if k in row:
                        row[k] = np.nan
                row['mask_source'] = 'interpolated'
                rows.append(row)
        rows.sort(key=lambda r: (r['timepoint'], r['track_id']))

    return {'rows': rows, 'tracks': tracks, 'records': records,
            'lineage': lineage, 'timepoints': timepoints, 'masks': masks,
            'planes': kept_planes, 'centroids': centroids, 'n_frames': n,
            'division_rejects': div_rejects,
            'division_reverted': div_reverted}


def _process_tile_dual(ij, sr, model, image_dir, well, tile, max_dist,
                       cellpose_kwargs=None, first_step_max_dist=None,
                       allow_division=True, division_max_dist=None,
                       division_thresholds=None, max_gap=0, gap_dist_scale=1.0,
                       predict_motion=True, min_division_overlap=3,
                       fill_gaps=False, min_division_interval=0, heal_window=0,
                       heal_frac=0.75, dual_kwargs=None, masks_dir=None,
                       reuse_masks=False, max_frames=0, pixel_size_um=None,
                       keep_masks=False, keep_planes=False, log=print):
    """--segment-on dual: Cellpose on brightfield AND on the marker sum.

    The fusion, tracking and nucleus carrying live in fucci_dual.run_dual; this
    only supplies the frames and the two Cellpose masks.  With masks_dir every
    frame's raw masks are saved there, and with reuse_masks they are loaded
    instead of re-running Cellpose -- so fusion and tracking can be retuned in
    minutes rather than the ~30 s per frame the two Cellpose passes cost.
    """
    import os
    import fucci_dual as fd

    dual_kwargs = dict(dual_kwargs or {})
    cellpose_kwargs = cellpose_kwargs or {}
    reader = TileReader(ij, image_dir, well, tile)
    n = reader.n_frames
    if max_frames and max_frames < n:
        log(f'  limiting to the first {max_frames} of {n} frames')
        n = max_frames
    if masks_dir:
        os.makedirs(masks_dir, exist_ok=True)
    state = {'ref': None}

    def read_frame(i):
        plane = register_to_reference(reader.frame(i), state['ref'], sr)
        if i == 0:
            state['ref'] = plane[:, :, fp.CH_BF].copy()
        return plane

    def get_masks(tp, plane, corrected):
        path = os.path.join(masks_dir, f't{tp:03d}.npz') if masks_dir else None
        if path and reuse_masks and os.path.exists(path):
            try:
                m = np.load(path)
                return m['bf'], m['fl']
            except Exception as exc:          # e.g. a job killed mid-write
                log(f'  frame {tp}: cached masks unreadable ({exc}); '
                    f'segmenting again')
        bf = segment_cells_bf(plane[:, :, fp.CH_BF], model, **cellpose_kwargs)
        fl = segment_cells_bf(segmentation_image(plane, corrected, 'sum'),
                              model, **cellpose_kwargs)
        if path:
            # written whole, then renamed: an interrupted job (OrangeGrid
            # evicts jobs from scavenged desktops) never leaves a truncated
            # file that a --reuse-masks restart would trust
            tmp = path + '.part.npz'
            np.savez_compressed(tmp, bf=bf.astype(np.int32),
                                fl=fl.astype(np.int32))
            os.replace(tmp, path)
        return bf, fl

    track_kwargs = dict(
        first_step_max_dist=first_step_max_dist, allow_division=allow_division,
        division_max_dist=division_max_dist,
        division_thresholds=division_thresholds, max_gap=max_gap,
        gap_dist_scale=gap_dist_scale, predict_motion=predict_motion,
        min_division_interval=min_division_interval)
    track_on = dual_kwargs.pop('track_on', 'cell')
    for k in ('mitotic_circularity', 'mitotic_area_frac', 'mitosis_window',
              'single_daughter_frac', 'link_overlap', 'mitotic_halo',
              'division_requires_mitosis'):
        if k in dual_kwargs:
            track_kwargs[k] = dual_kwargs.pop(k)
    try:
        res = fd.run_dual(read_frame, n, get_masks, max_dist, well=well,
                          tile=tile, fuse_params=dual_kwargs,
                          track_kwargs=track_kwargs,
                          min_division_overlap=min_division_overlap,
                          heal_window=heal_window, heal_frac=heal_frac,
                          pixel_size_um=pixel_size_um, keep_masks=keep_masks,
                          keep_planes=keep_planes, log=log, save_dir=masks_dir,
                          track_on=track_on)
    finally:
        reader.close()

    if fill_gaps:
        # frames where even brightfield lost the cell; rare in this mode.  The
        # disc is sized like the track's nucleus, not its whole cell, so it
        # measures the same thing as every other row.
        targets = gap_targets(res['tracks'], res['centroids'], res['nuc_areas'])
        filled = measure_gap_frames(ij, sr, image_dir, well, tile, targets,
                                    state['ref'], pixel_size_um, log=log)
        lineage, timing = res['lineage'], res['timing']
        for tp, by_track in filled.items():
            for tid, feat in by_track.items():
                since, until = timing.get((tid, tp), (np.nan, np.nan))
                info = lineage.get(tid, {})
                row = {'well': well, 'tile': tile, 'track_id': int(tid),
                       'timepoint': tp,
                       'time_min': (tp - 1) * fp.FRAME_INTERVAL_MIN,
                       'cell_label': np.nan,
                       'frames_since_division': since,
                       'frames_until_division': until,
                       'parent_track_id': info.get('parent_track_id'),
                       'root_track_id': info.get('root_track_id'),
                       'generation': info.get('generation'),
                       'track_end_reason': info.get('end_reason'),
                       'division_kind': info.get('division_kind', '')}
                row.update(feat)
                for k in IMPUTED_BLANK:
                    if k in row:
                        row[k] = np.nan
                row['mask_source'] = 'interpolated'
                row['nuc_source'] = 'interpolated'
                res['rows'].append(row)
        res['rows'].sort(key=lambda r: (r['timepoint'], r['track_id']))
    return res


# --------------------------------------------------------------------------- #
# measuring the frames where a cell went dark
# --------------------------------------------------------------------------- #
# Morphology measured on a synthetic disc is not morphology -- it is a
# description of the disc.  Circularity in particular would come back as 1.0
# for every filled frame, and the mitosis rule fires on high circularity, so
# leaving it in would relabel the whole dark window as M.  These are blanked.
IMPUTED_BLANK = ('perimeter_px', 'circularity', 'solidity', 'eccentricity',
                 'axis_major_px', 'axis_minor_px', 'area_px', 'area_um2')


def gap_targets(tracks, centroids, areas, min_radius=6.0):
    """{timepoint: [(track_id, (row, col), radius_px)]} for unmeasured frames.

    A cell whose fluorescence falls below detection has no mask, so no row, so
    no intensity for exactly the frames where "the signal is gone" is the
    observation you want.  The track itself survives (gap closing bridges it),
    which means the cell's position is known on both sides and can be
    interpolated.  That gives somewhere defensible to measure.
    """
    from collections import defaultdict

    seen = defaultdict(dict)
    for tp, mapping in tracks.items():
        for lab, tid in mapping.items():
            if lab in centroids.get(tp, {}):
                seen[tid][tp] = centroids[tp][lab]

    track_area = {}
    for tid, by_tp in seen.items():
        vals = [areas.get(tp, {}).get(lab)
                for tp in by_tp
                for lab, t2 in tracks.get(tp, {}).items() if t2 == tid]
        vals = [v for v in vals if v]
        track_area[tid] = float(np.median(vals)) if vals else None

    targets = defaultdict(list)
    for tid, by_tp in seen.items():
        area = track_area.get(tid)
        if not area:
            continue
        radius = max(float(np.sqrt(area / np.pi)), min_radius)
        ts = sorted(by_tp)
        for lo, hi in zip(ts, ts[1:]):
            if hi - lo <= 1:
                continue
            for tp in range(lo + 1, hi):
                f = (tp - lo) / (hi - lo)
                r = by_tp[lo][0] + f * (by_tp[hi][0] - by_tp[lo][0])
                c = by_tp[lo][1] + f * (by_tp[hi][1] - by_tp[lo][1])
                targets[tp].append((tid, (r, c), radius))
    return dict(targets)


def measure_gap_frames(ij, sr, image_dir, well, tile, targets, bf_reference,
                       pixel_size_um=None, log=print):
    """Re-read only the frames that need filling and measure a disc on each.

    A second pass over the file rather than holding every plane in memory: at
    2048x2048x3 channels a 96-frame tile is ~2.4 GB, and which frames are
    needed is not known until tracking has finished.  Cellpose, which is the
    expensive part, is not run again.
    """
    import skimage as ski

    if not targets:
        return {}
    wanted = sorted(targets)
    log(f'  filling {sum(len(v) for v in targets.values())} unmeasured '
        f'cell-frames across {len(wanted)} frames (second read pass)')
    out = {}
    reader = TileReader(ij, image_dir, well, tile)
    try:
        for tp in wanted:
            plane = reader.frame(tp - 1)
            plane = register_to_reference(plane, bf_reference, sr)
            corrected = fp.subtract_background(plane)
            lab_img = np.zeros(corrected.shape[:2], np.int32)
            order = []
            for i, (tid, (r, c), radius) in enumerate(targets[tp], start=1):
                rr, cc = ski.draw.disk((r, c), radius, shape=lab_img.shape)
                lab_img[rr, cc] = i
                order.append(tid)
            feats = fp.measure_frame(corrected, lab_img, pixel_size_um)
            out[tp] = {order[i - 1]: f for i, f in feats.items()
                       if 1 <= i <= len(order)}
    finally:
        reader.close()
    return out


# --------------------------------------------------------------------------- #
# background measurement
# --------------------------------------------------------------------------- #
def measure_background(ij, sr, model, image_dir, well, tile, n_frames=3,
                       cellpose_kwargs=None, dilate_px=15, log=print):
    """Report per-channel background from pixels OUTSIDE every cell mask.

    Guessing the background is the single easiest way to wreck the phase calls:
    it sets the zero point for every intensity threshold, so if it is 10x too
    low then every cell reads double-positive and the whole plate comes back as
    one phase.  This measures it instead.

    The union of cell masks is dilated before being excluded, so halos and
    out-of-focus skirts around cells do not get counted as background.
    """
    import skimage as ski

    cellpose_kwargs = cellpose_kwargs or {}
    reader = TileReader(ij, image_dir, well, tile)
    picks = np.unique(np.linspace(0, reader.n_frames - 1,
                                  min(n_frames, reader.n_frames)).astype(int))
    stats = {ch: [] for ch in (fp.CH_MCHERRY, fp.CH_MVENUS)}
    coverage = []
    try:
        for i in picks:
            plane = reader.frame(int(i))
            corrected = fp.subtract_background(plane)
            seg_img = segmentation_image(plane, corrected, segment_on)
            mask = drop_edge_labels(
                segment_cells_bf(seg_img, model, **cellpose_kwargs))
            cells = mask > 0
            if dilate_px:
                cells = ski.morphology.dilation(
                    cells, footprint=np.ones((dilate_px, dilate_px), bool))
            bg = ~cells
            coverage.append(100 * cells.mean())
            if not bg.any():
                log(f'  frame {i+1}: cells cover the whole field, skipped')
                continue
            for ch in stats:
                stats[ch].append(plane[:, :, ch][bg])
            log(f'  frame {i+1}: {mask.max()} cells, '
                f'{100*cells.mean():.0f}% of the field covered')
            del plane, mask
    finally:
        reader.close()

    out = {}
    for ch, chunks in stats.items():
        if not chunks:
            continue
        v = np.concatenate(chunks).astype(np.float64)
        out[ch] = {'median': float(np.median(v)),
                   'mean': float(v.mean()),
                   'p25': float(np.percentile(v, 25)),
                   'p75': float(np.percentile(v, 75)),
                   'p95': float(np.percentile(v, 95)),
                   'n_px': int(v.size)}
    out['field_covered_pct'] = float(np.mean(coverage)) if coverage else np.nan
    return out


# --------------------------------------------------------------------------- #
# overlay for eyeballing masks and tracking
# --------------------------------------------------------------------------- #
_LUTS = [(1, 1, 1), (0.9, 0.2, 0.2), (0.2, 0.9, 0.2), (0.2, 0.8, 1.0), (1, 1, 1)]
_LUT_NAMES = ['BF', 'mCherry', 'mVenus', 'cell_outline', 'track_id']


def _font(size):
    from PIL import ImageFont
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont('DejaVu Sans'), size)
    except Exception:
        return ImageFont.load_default()


def _outline(mask, width=3):
    import skimage as ski
    from skimage.segmentation import find_boundaries
    b = find_boundaries(mask, mode='inner')
    if width > 1:
        fp_ = np.ones((width, width), bool)
        try:
            b = ski.morphology.dilation(b, footprint=fp_)
        except (AttributeError, TypeError):
            b = ski.morphology.binary_dilation(b, footprint=fp_)
    return b


def write_overlay(path, planes, masks, tracks, centroids, timepoints,
                  font_size=44, line_width=3, pixel_size_um=None,
                  nuc_masks=None):
    """One ImageJ composite hyperstack: BF, mCherry, mVenus, outlines, track ids.

    Written as a single multi-frame file rather than one per frame, so Fiji opens
    it with the channel and frame sliders already correct -- no Image Sequence
    import and no Stack to Hyperstack guessing.

    Labels are track ids, so a correctly tracked cell keeps its number across
    every frame and a tracking error reads as a number that jumps.

    With nuc_masks (dual mode) a sixth channel outlines the nucleus each
    tracked cell was measured on, carried nuclei included.
    """
    import tifffile
    from PIL import Image, ImageDraw

    font = _font(font_size)
    frames = []
    for tp in timepoints:
        plane, mask = planes[tp], masks[tp]
        txt = Image.new('I;16', (mask.shape[1], mask.shape[0]), 0)
        draw = ImageDraw.Draw(txt)
        for lab, tid in tracks.get(tp, {}).items():
            if lab in centroids.get(tp, {}):
                r, c = centroids[tp][lab]
                draw.text((c, r), str(tid), fill=65535, font=font, anchor='mm')
        tracked = (np.isin(mask, list(tracks.get(tp, {})))
                   if tracks.get(tp) else np.zeros(mask.shape, bool))
        chans = [
            plane[:, :, fp.CH_BF].astype(np.uint16),
            plane[:, :, fp.CH_MCHERRY].astype(np.uint16),
            plane[:, :, fp.CH_MVENUS].astype(np.uint16),
            np.where(_outline(np.where(tracked, mask, 0), line_width),
                     np.uint16(65535), np.uint16(0)),
            np.asarray(txt, dtype=np.uint16),
        ]
        if nuc_masks is not None:
            nm = nuc_masks[tp]
            keep = np.isin(nm, list(tracks.get(tp, {}))) if tracks.get(tp) \
                else np.zeros(nm.shape, bool)
            chans.append(np.where(_outline(np.where(keep, nm, 0), 2),
                                  np.uint16(65535), np.uint16(0)))
        frames.append(np.stack(chans))

    data = np.stack(frames)
    ramp = np.arange(256, dtype=np.float64)
    lut_cols = list(_LUTS) + ([(1.0, 0.0, 1.0)] if nuc_masks is not None else [])
    names = _LUT_NAMES + (['nucleus_outline'] if nuc_masks is not None else [])
    luts = [np.stack([(ramp * r).astype(np.uint8), (ramp * g).astype(np.uint8),
                      (ramp * b).astype(np.uint8)]) for r, g, b in lut_cols]
    kw = {}
    if pixel_size_um:
        kw['resolution'] = (1 / pixel_size_um, 1 / pixel_size_um)
    tifffile.imwrite(path, data, imagej=True, photometric='minisblack',
                     metadata={'axes': 'TCYX', 'mode': 'composite',
                               'unit': 'um', 'LUTs': luts,
                               'Labels': names}, **kw)
    return data.shape
