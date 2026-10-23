"""
Crop-based versions of the three per-cell steps of the timelapse FRET
pipeline that dominate its run time.  Each has the same signature and gives
the same result, bit for bit, as the function it replaces:

    fast_segment_dapi_per_cell  -> timelapse_imaging.segment_dapi_per_cell
    fast_refine_nuclei          -> timelapse_tracking.refine_nuclei
    fast_measure_cell           -> timelapse_tracking.measure_cell

The originals build a full 2304 x 2304 boolean image for every cell (and,
in measure_cell, index full frames ~70 times per cell), at every timepoint.
These work on each cell's bounding box, padded by a pixel so that hole
filling and perimeter -- which look at a pixel's neighbours -- see exactly
the same neighbourhood as on the full frame.  Pixels are visited in the same
row-major order, so even floating-point sums come out identical.

install() swaps them into the two modules, so timelapse_imaging.process_tile
itself -- registration, Cellpose, background subtraction, tracking -- runs
unchanged.  test_timelapse_grid.py checks each against its original.
"""

from __future__ import annotations

import numpy as np


def _boxes(labels):
    from scipy import ndimage as ndi
    return ndi.find_objects(labels)


def _cut(sl, shape):
    """(top, bottom, left, right): True where the crop edge is inside the image."""
    return (sl[0].start > 0, sl[0].stop < shape[0],
            sl[1].start > 0, sl[1].stop < shape[1])


def _pad(sl, pad, shape):
    return (slice(max(sl[0].start - pad, 0), min(sl[0].stop + pad, shape[0])),
            slice(max(sl[1].start - pad, 0), min(sl[1].stop + pad, shape[1])))


def fill_holes_like_full_frame(binary, area_threshold, fill_fn, cut=None):
    """Hole filling on a crop that matches hole filling on the whole image.

    remove_small_holes fills every background region below the size limit.
    On the full frame the background around an object is one huge region and
    is never filled; on a tight crop it can be small, and would be.  Every
    background region that touches the crop's edge is part of that outside
    region on the full frame (the crop is padded, so its edge ring is
    background), so those are put back after filling.  Enclosed holes lie
    wholly inside the crop and are filled exactly as on the full frame.

    cut = (top, bottom, left, right): which crop edges were cut through the
    image.  An edge that is the image's own border does not lead anywhere, so
    a pocket of background against it is a real (fillable) region on the full
    frame too.  None = all four were cut.
    """
    from scipy import ndimage as ndi
    filled = fill_fn(binary, area_threshold)
    bg_lab, _n = ndi.label(~binary)              # 4-connected, as remove_small_holes
    top, bottom, left, right = cut if cut is not None else (True,) * 4
    rims = [bg_lab[0, :]] * top + [bg_lab[-1, :]] * bottom + \
        [bg_lab[:, 0]] * left + [bg_lab[:, -1]] * right
    if not rims:
        return filled
    edge = np.unique(np.concatenate(rims))
    outside = np.isin(bg_lab, edge[edge > 0])
    filled[outside] = False
    return filled


def fast_segment_dapi_per_cell(img, mask_cell, sigma=2.0, min_contrast=1.5,
                               min_pixels=200):
    import skimage as ski
    if sigma and sigma > 0:
        img = ski.filters.gaussian(img, sigma=sigma, preserve_range=True)
    out = np.zeros(img.shape, dtype=bool)
    boxes = _boxes(mask_cell)
    for cell_id in np.unique(mask_cell):
        if cell_id == 0 or boxes[cell_id - 1] is None:
            continue
        sl = boxes[cell_id - 1]
        sel = mask_cell[sl] == cell_id
        crop = img[sl]
        vals = crop[sel]
        if vals.size < min_pixels or vals.max() == vals.min():
            continue
        thresh = ski.filters.threshold_otsu(vals)
        hi, lo = vals[vals > thresh], vals[vals <= thresh]
        if hi.size == 0 or lo.size == 0:
            continue
        if np.median(hi) < min_contrast * np.median(lo):
            continue
        out[sl][sel & (crop > thresh)] = True
    return out


def fast_refine_nuclei(mask_cell, dapi_binary, area_threshold=200):
    import skimage as ski
    import timelapse_tracking as tt

    mask_nuc = (dapi_binary > 0).astype(mask_cell.dtype) * mask_cell
    boxes = _boxes(mask_cell)
    for cell_id in np.unique(mask_cell):
        if cell_id == 0 or boxes[cell_id - 1] is None:
            continue
        # every pixel the original can touch for this cell lies in its box:
        # its nucleus pixels start inside the cell and the filled largest
        # component stays inside that component's own box
        sl = _pad(boxes[cell_id - 1], 1, mask_cell.shape)
        sub = mask_nuc[sl]
        cell_dapi = sub == cell_id
        if not np.any(cell_dapi):
            continue
        labeled = ski.measure.label(cell_dapi)
        regions = ski.measure.regionprops(labeled)
        if not regions:
            continue
        largest = max(regions, key=lambda r: r.area)
        keep = fill_holes_like_full_frame(
            labeled == largest.label, area_threshold, tt._remove_small_holes,
            cut=_cut(sl, mask_cell.shape))
        sub[mask_cell[sl] == cell_id] = 0
        sub[keep] = cell_id
    return mask_nuc


def fast_measure_cell(image_tp, mask_cell_orig, mask_cell, mask_nuc, label):
    # boxes are computed once per timepoint's mask, not once per cell; the
    # cache holds the mask itself and checks identity, so a later frame can
    # never be given an earlier frame's boxes
    if _cache.get('mask') is not mask_cell_orig:
        boxes = _boxes(mask_cell_orig)
        _cache['mask'] = mask_cell_orig
        _cache['boxes'] = {i + 1: _pad(b, 1, mask_cell_orig.shape)
                           for i, b in enumerate(boxes) if b is not None}
    sl = _cache['boxes'].get(label)
    if sl is None:
        return None
    # mask_cell and mask_nuc of this label lie inside the original mask's box
    # (background subtraction only removes pixels; the nucleus is cut from
    # the cell and its holes are inside it)
    return _ORIGINAL['measure_cell'](image_tp[sl], mask_cell_orig[sl],
                                     mask_cell[sl], mask_nuc[sl], label)


_cache = {}


def _originals():
    import timelapse_imaging as ti
    import timelapse_tracking as tt
    return dict(segment_dapi_per_cell=ti.segment_dapi_per_cell,
                refine_nuclei=tt.refine_nuclei,
                measure_cell=tt.measure_cell)


# captured at import, before anything is swapped
_ORIGINAL = _originals()


def install():
    """Swap the fast versions into timelapse_imaging / timelapse_tracking."""
    import timelapse_imaging as ti
    import timelapse_tracking as tt
    ti.segment_dapi_per_cell = fast_segment_dapi_per_cell
    tt.refine_nuclei = fast_refine_nuclei
    tt.measure_cell = fast_measure_cell


def uninstall():
    import timelapse_imaging as ti
    import timelapse_tracking as tt
    ti.segment_dapi_per_cell = _ORIGINAL['segment_dapi_per_cell']
    tt.refine_nuclei = _ORIGINAL['refine_nuclei']
    tt.measure_cell = _ORIGINAL['measure_cell']
