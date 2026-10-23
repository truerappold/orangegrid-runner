"""
Imaging layer for the timelapse pipeline: loading, registration, segmentation.

Split from timelapse_tracking.py because this half needs a JVM (pyimagej), a GPU
(cellpose) and pystackreg, so it cannot be unit tested.  Both driver scripts
import `process_tile` from here rather than each keeping their own copy — the
existing analyzeDirPerturbBGST / _imageprint / maskPrint_pipeline trio drifted
apart exactly that way (different Cellpose channels, different bleedthrough
coefficients), and that is worth not repeating.

Registration
------------
Two cameras: the FRET donor (mTQ2) and DAPI sit on camera 1, the FRET acceptor
and direct mNG on camera 2.  So each timepoint needs

    time matrix   donor_t  -> donor_1     applied to donor_t and DAPI_t
    camera matrix acceptor -> warped donor_t   applied to acceptor_t and mNG_t

Every timepoint therefore lands in the timepoint-1 frame.  Registering each
timepoint to timepoint 1 rather than to its predecessor matters: chained
registration accumulates drift, which would force the tracking cutoff to grow
with t.  For T=2 this reproduces the original script's behaviour exactly.
"""

from __future__ import annotations

import numpy as np
import skimage as ski
from skimage.segmentation import relabel_sequential

import timelapse_tracking as tt


# --------------------------------------------------------------------------- #
# segmentation
# --------------------------------------------------------------------------- #
def segment_cells(img, model, flow_threshold=0.4, cellprob_threshold=0.0,
                  tile_norm_blocksize=0, batch_size=32):
    """Cellpose on the direct mNG channel."""
    masks, _flows, _styles = model.eval(
        img, batch_size=batch_size, flow_threshold=flow_threshold,
        progress=False, cellprob_threshold=cellprob_threshold,
        normalize={'tile_norm_blocksize': tile_norm_blocksize})
    return masks


def segment_dapi(img, sigma=2.0):
    """Global Otsu on a smoothed DAPI image.

    Smoothing first is not cosmetic.  On a raw DAPI plane the threshold cuts
    through pixel noise, which gives ragged perimeters and a swarm of specks:
    measured on the example tile, raw Otsu produced 176 connected components of
    which 144 were under 50 px, and the real nuclei had a median circularity of
    0.77.  At sigma=2 that becomes 15 specks and 0.84.

    Kept mainly as a fallback; segment_dapi_per_cell is the better default
    because one global threshold cannot cope with the dynamic range of a whole
    field (see its docstring).
    """
    if sigma and sigma > 0:
        img = ski.filters.gaussian(img, sigma=sigma, preserve_range=True)
    if img.max() == img.min():
        return np.zeros_like(img, dtype=bool)
    return img > ski.filters.threshold_otsu(img)


def segment_dapi_per_cell(img, mask_cell, sigma=2.0, min_contrast=1.5,
                          min_pixels=200):
    """Threshold DAPI inside each cell separately.

    A single global Otsu fails here because the DAPI field spans a huge range.
    On the example tile the image median is 257 and global Otsu lands at 21075 —
    the 99.25th percentile — so only the brightest nuclei are captured at their
    true extent and dim ones are shaved down to their cores.  That is what makes
    nuclei look misshapen rather than merely rough.

    Thresholding within each cell instead recovered roughly twice the nuclear
    area on that tile (77011 px vs 41194) and found 36 nuclei instead of 28,
    with no loss of shape quality.

    min_contrast rejects cells with no real nucleus: Otsu always returns a split,
    even on flat noise, so a cell is only accepted when the bright population's
    median exceeds the dim population's by this factor.
    """
    if sigma and sigma > 0:
        img = ski.filters.gaussian(img, sigma=sigma, preserve_range=True)
    out = np.zeros(img.shape, dtype=bool)
    for cell_id in np.unique(mask_cell):
        if cell_id == 0:
            continue
        sel = mask_cell == cell_id
        vals = img[sel]
        if vals.size < min_pixels or vals.max() == vals.min():
            continue
        thresh = ski.filters.threshold_otsu(vals)
        hi, lo = vals[vals > thresh], vals[vals <= thresh]
        if hi.size == 0 or lo.size == 0:
            continue
        if np.median(hi) < min_contrast * np.median(lo):
            continue                      # flat cell: no nucleus to find
        out[sel & (img > thresh)] = True
    return out


def drop_edge_labels(mask):
    """Remove labels touching any border, then relabel 1..N."""
    edge = np.unique(np.concatenate(
        [mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]]))
    mask = mask.copy()
    mask[np.isin(mask, edge)] = 0
    mask, _, _ = relabel_sequential(mask)
    return mask.astype(np.int32)


# --------------------------------------------------------------------------- #
# loading + registration
# --------------------------------------------------------------------------- #
def _otsu_binary(img):
    if img.max() == img.min():
        return np.zeros_like(img, dtype=bool)
    return img >= ski.filters.threshold_otsu(img)


def axis_index(ds):
    """{'X': 0, 'Y': 1, 'Channel': 2, 'Time': 3} for whatever axes this Dataset has.

    ImageJ2 drops singleton axes, so a C=1,T=1 file is 2-D while a C=2,T=20 file
    is 4-D.  Reading the axis labels rather than assuming a fixed order means the
    same code handles a basal snapshot and a 20-frame stack.
    """
    out = {}
    try:
        for i in range(int(ds.numDimensions())):
            out[str(ds.axis(i).type().getLabel())] = i
    except Exception:
        # Fall back to the SCIFIO XYCZT order with singletons dropped.
        ndim = int(ds.numDimensions())
        names = ['X', 'Y'] + (['Channel'] if ndim >= 3 else []) + \
                (['Time'] if ndim >= 4 else [])
        out = {n: i for i, n in enumerate(names[:ndim])}
    return out


def dataset_n_frames(ds):
    """Number of timepoints inside one .vsi (1 when there is no Time axis)."""
    amap = axis_index(ds)
    if 'Time' not in amap:
        return 1
    return int(ds.dimension(amap['Time']))


def _plane_slice(ds, channel=None, frame=None):
    """Index tuple selecting one 2-D plane out of a Dataset of any arity."""
    amap = axis_index(ds)
    ndim = int(ds.numDimensions())
    idx = [slice(None)] * ndim
    if channel is not None:
        if 'Channel' in amap:
            idx[amap['Channel']] = channel
        elif channel != 0:
            raise ValueError(f'channel {channel} requested but the Dataset has '
                             f'no Channel axis')
    if frame is not None:
        if 'Time' in amap:
            idx[amap['Time']] = frame
        elif frame != 0:
            raise ValueError(f'frame {frame} requested but the Dataset has no '
                             f'Time axis')
    return tuple(idx)


def read_plane(ij, ds, channel=None, frame=None):
    """One 2-D numpy plane, pulled lazily so a 20-frame stack is not fully loaded."""
    return np.asarray(ij.py.from_java(ds[_plane_slice(ds, channel, frame)]))


def release_dataset(ds):
    """net.imagej.Dataset has no close(); dispose the backing img if it has one.

    Matters more here than in the original script: a 20-frame FRET block is a
    ~425 MB Java object, and a plate's worth of tiles opens a lot of them.
    """
    try:
        img = ds.getImgPlus().getImg()
        if hasattr(img, 'dispose'):
            img.dispose()
    except Exception:
        pass


class BlockReader:
    """Holds the three channel Datasets of one block open while its frames are read.

    Opening the files once per block instead of once per frame is the difference
    between reading a 425 MB .vsi twenty times and reading it once.
    """

    def __init__(self, ij, pics_dir, well, tile, block):
        self.ij = ij
        self.datasets = {}
        for channel in tt.CHANNEL_FILES:
            path = tt.build_filename(pics_dir, well, tile, block, channel)
            self.datasets[channel] = ij.io().open(path)
        counts = {c: dataset_n_frames(d) for c, d in self.datasets.items()}
        if len(set(counts.values())) != 1:
            self.close()
            raise ValueError(
                f'{well} tile {tile} block {block}: channels disagree on frame '
                f'count {counts} — cannot pair them into timepoints')
        self.n_frames = next(iter(counts.values()))

    def frame(self, frame_idx, shape=(2304, 2304)):
        """One timepoint as an [H, W, 4] plane."""
        plane = np.zeros((*shape, tt.N_CHANNELS), dtype='uint16')
        f = None if self.n_frames == 1 else frame_idx
        # 'FRET, FRET' is two-channel: index 1 is the donor (mTurquoise2),
        # index 0 the acceptor (mNeonGreen), as in the original script.
        plane[:, :, tt.CH_DONOR] = read_plane(
            self.ij, self.datasets['FRET, FRET'], channel=1, frame=f)
        plane[:, :, tt.CH_ACCEPTOR] = read_plane(
            self.ij, self.datasets['FRET, FRET'], channel=0, frame=f)
        plane[:, :, tt.CH_MNG] = read_plane(
            self.ij, self.datasets['mNG'], frame=f)
        plane[:, :, tt.CH_DAPI] = read_plane(
            self.ij, self.datasets['DAPI'], frame=f)
        return plane

    def close(self):
        for ds in self.datasets.values():
            release_dataset(ds)
        self.datasets.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def register_to_reference(plane, donor_reference, sr):
    """Warp one timepoint into the reference (timepoint 1) frame, in place.

    donor_reference is the timepoint-1 donor channel, or None for timepoint 1
    itself (which only needs the camera-to-camera correction).
    """
    if donor_reference is not None:
        ref = _otsu_binary(donor_reference)
        mov = _otsu_binary(plane[:, :, tt.CH_DONOR])
        if ref.any() and mov.any():
            t_time = sr.register(ref, mov)
            for ch in (tt.CH_DONOR, tt.CH_DAPI):
                plane[:, :, ch] = ski.transform.warp(
                    plane[:, :, ch], t_time, preserve_range=True).astype('uint16')

    ref = _otsu_binary(plane[:, :, tt.CH_DONOR])
    mov = _otsu_binary(plane[:, :, tt.CH_ACCEPTOR])
    if ref.any() and mov.any():
        t_cam = sr.register(ref, mov)
        for ch in (tt.CH_ACCEPTOR, tt.CH_MNG):
            plane[:, :, ch] = ski.transform.warp(
                plane[:, :, ch], t_cam, preserve_range=True).astype('uint16')
    return plane


# --------------------------------------------------------------------------- #
# one timepoint, end to end
# --------------------------------------------------------------------------- #
def process_timepoint(plane, model, cellpose_kwargs, hole_area=200,
                      do_background=True, dapi_mode='per-cell', dapi_sigma=2.0,
                      dapi_contrast=1.5):
    """Segment and measure one registered timepoint.

    Returns (measurements, nuc_centroids, cell_centroids, masks) where
    `measurements` is {cell_label: {param: value}} and `masks` is
    (mask_cell_orig, mask_cell, mask_nuc) for the diagnostic script.

    Order follows the original script: the Cellpose mask is copied before
    background subtraction (that copy supplies area_cell_original and
    circularity), background subtraction then erodes the working mask, and the
    nuclei are derived from the eroded mask.
    """
    mask_orig = drop_edge_labels(segment_cells(plane[:, :, tt.CH_MNG], model,
                                               **cellpose_kwargs))
    if mask_orig.max() == 0:
        return {}, {}, {}, (mask_orig, mask_orig, mask_orig)

    mask_cell = mask_orig.copy()
    if do_background:
        for ch, (bg, thresh) in tt.BG_SUBTRACT.items():
            plane[:, :, ch], mask_cell = tt.background_subtract(
                plane[:, :, ch], mask_cell, bg, thresh)

    # DAPI is thresholded over the *un-eroded* cell footprint, so the threshold
    # sees the whole nucleus even where background subtraction later trims the
    # cell mask; refine_nuclei then intersects with the eroded mask as before.
    if dapi_mode == 'per-cell':
        dapi_bin = segment_dapi_per_cell(plane[:, :, tt.CH_DAPI], mask_orig,
                                         sigma=dapi_sigma,
                                         min_contrast=dapi_contrast)
    else:
        dapi_bin = segment_dapi(plane[:, :, tt.CH_DAPI], sigma=dapi_sigma)
    mask_nuc = tt.refine_nuclei(mask_cell, dapi_bin, area_threshold=hole_area)

    nuc_centroids = tt.centroids_by_label(mask_nuc)
    cell_centroids = tt.centroids_by_label(mask_cell)

    measurements = {}
    for label in sorted(nuc_centroids):
        vals = tt.measure_cell(plane, mask_orig, mask_cell, mask_nuc, label)
        if vals is None:
            continue
        vals['centroid_nuc_row'], vals['centroid_nuc_col'] = nuc_centroids[label]
        if label in cell_centroids:
            vals['centroid_cell_row'], vals['centroid_cell_col'] = cell_centroids[label]
        measurements[label] = vals

    # a cell whose nucleus vanished under background subtraction is not a valid
    # tracking target either, so keep the centroid sets consistent
    nuc_centroids = {k: v for k, v in nuc_centroids.items() if k in measurements}
    return measurements, nuc_centroids, cell_centroids, (mask_orig, mask_cell, mask_nuc)


def process_tile(ij, sr, model, pics_dir, well, tile, blocks,
                 cellpose_kwargs, max_dist, match_method='nearest',
                 first_step_max_dist=None, allow_division=False,
                 division_max_dist=None, division_thresholds=None,
                 hole_area=200, do_background=True, dapi_mode='per-cell',
                 dapi_sigma=2.0, dapi_contrast=1.5,
                 keep_masks=False, shape=(2304, 2304), log=print):
    """Load, register, segment, measure and track a whole tile.

    `blocks` is the list of filename-prefix blocks (0 = unprefixed).  Each block
    is opened once and its frames read lazily, giving global timepoints 1..N
    across all blocks.

    Planes are discarded as soon as a timepoint has been measured, so peak
    memory is one timepoint (~42 MB at 2304x2304x4 uint16) plus whatever the JVM
    holds for the open block — measurements are small, and tracking only needs
    the centroids.  `keep_masks=True` retains the masks for the diagnostic script.

    Returns dict with 'measurements' (flat list, track_id attached), 'records'
    (per-cell match distances and drop reasons), 'tracks', 'timepoints',
    'timepoint_source' ({global t: (block, frame)}) and 'masks'.
    """
    donor_reference = None
    per_tp_measurements = {}
    nuc_centroids = {}
    cell_centroids = {}
    masks = {}
    timepoint_source = {}

    tp = 0
    for block in blocks:
        with BlockReader(ij, pics_dir, well, tile, block) as reader:
            log(f'    block {block}: {reader.n_frames} frame(s)')
            for frame in range(reader.n_frames):
                tp += 1
                timepoint_source[tp] = (block, frame)
                plane = reader.frame(frame, shape=shape)
                plane = register_to_reference(
                    plane, None if tp == 1 else donor_reference, sr)
                if tp == 1:
                    donor_reference = plane[:, :, tt.CH_DONOR].copy()

                meas, ncen, ccen, tp_masks = process_timepoint(
                    plane, model, cellpose_kwargs, hole_area=hole_area,
                    do_background=do_background, dapi_mode=dapi_mode,
                    dapi_sigma=dapi_sigma, dapi_contrast=dapi_contrast)
                if not meas:
                    log(f'      timepoint {tp} (block {block} frame {frame}): '
                        f'no usable cells')
                per_tp_measurements[tp] = meas
                nuc_centroids[tp] = ncen
                cell_centroids[tp] = ccen
                if keep_masks:
                    masks[tp] = tp_masks
                del plane

    timepoints = sorted(per_tp_measurements)
    areas_per_tp = {tp: {lab: v.get('area_cell')
                         for lab, v in per_tp_measurements[tp].items()}
                    for tp in timepoints}
    tracks, records, lineage = tt.track_series(
        nuc_centroids, max_dist, method=match_method,
        first_step_max_dist=first_step_max_dist, areas_per_tp=areas_per_tp,
        allow_division=allow_division, division_max_dist=division_max_dist,
        division_thresholds=division_thresholds)

    # Attach a whole-cell-centroid distance computed *independently* of the
    # nuclear match, for every cell whether it matched or not.  The pair is
    # diagnostic: if the cell centroid barely moved but the nuclear centroid
    # jumped ~a cell width, the nucleus was reassigned by refine_nuclei (two
    # nuclei inside one merged Cellpose mask, or a boundary shift moving a
    # nucleus between neighbours) rather than the cell having actually moved.
    def _nearest(point, candidates):
        if point is None or not candidates:
            return np.nan, np.nan
        best_d, best_lab = np.inf, np.nan
        for lab, c in candidates.items():
            d = float(np.hypot(point[0] - c[0], point[1] - c[1]))
            if d < best_d:
                best_d, best_lab = d, lab
        return best_d, best_lab

    for rec in records:
        tp, lab = rec['timepoint'], rec['cell_label']
        rec['well'], rec['tile'] = well, tile
        rec['block'], rec['frame'] = timepoint_source.get(tp, (np.nan, np.nan))
        rec['n_cells_this_tp'] = len(nuc_centroids.get(tp, {}))
        i = timepoints.index(tp) if tp in timepoints else 0
        if i == 0:
            rec['match_dist_cell'] = np.nan
            rec['nearest_cell_label'] = np.nan
            continue
        prev_tp = timepoints[i - 1]
        live_prev_cells = {l: cell_centroids[prev_tp][l]
                           for l in tracks.get(prev_tp, {})
                           if l in cell_centroids.get(prev_tp, {})}
        d, near_lab = _nearest(cell_centroids.get(tp, {}).get(lab), live_prev_cells)
        rec['match_dist_cell'] = d
        rec['nearest_cell_label'] = near_lab

    measurements = []
    for tp in timepoints:
        for label, vals in per_tp_measurements[tp].items():
            track_id = tracks.get(tp, {}).get(label)
            if track_id is None:
                continue                      # dropped by the distance or collision rule
            row = dict(vals)
            row['track_id'] = int(track_id)
            row['timepoint'] = tp
            row['cell_label'] = label
            measurements.append(row)

    return {'measurements': measurements, 'records': records, 'tracks': tracks,
            'lineage': lineage,
            'nuc_centroids': nuc_centroids, 'cell_centroids': cell_centroids,
            'timepoints': timepoints, 'timepoint_source': timepoint_source,
            'masks': masks}
