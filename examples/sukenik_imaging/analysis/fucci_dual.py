"""
Dual segmentation for NeoPIP-FUCCI: brightfield finds the cells, the summed
fluorescence finds the nuclei, and each is used to check the other.

Why both
--------
Fluorescence alone (--segment-on sum) gives clean nuclei, but PIP-FUCCI has
windows where a cell is too dim to segment -- early S, and a freshly divided
daughter -- and a cell with no mask cannot be tracked or measured.  Brightfield
alone finds every cell in every phase, but Cellpose on brightfield sometimes
splits one cell in two or joins two touching cells.  Run both and the cell
outline always exists (brightfield), while the nuclei say whether that outline
is right.

Fusion rules, applied in this order
-----------------------------------
1. Nucleus fragments that touch and sit in the same brightfield cell are one
   nucleus that Cellpose cut in half; they are rejoined.
2. A nucleus spread over two or more brightfield objects means brightfield cut
   the cell up; those objects are merged.  Vetoed when one of them already has
   its own nucleus, or when two of them are round (that is two daughters
   sharing a mitotic blob, not one cell cut in half).
3. A brightfield object holding two or more nuclei is two cells joined; it is
   split between the nuclei (watershed from each nucleus).
4. A nucleus lying mostly outside every brightfield cell means brightfield
   missed the cell; the nucleus, grown outwards by a few px, becomes the cell.
   (Optional, off by default: extend_cells instead gives it to a nucleus-less
   cell already holding >= 25% of it -- see FUSE_DEFAULTS for why it is off.)
   (One spread over several cells is not missed -- each takes its part.)
5. A brightfield object with no nucleus is kept (it is a dim cell, which is the
   point) unless it is smaller than min_cell_px, which removes the cytoplasmic
   bridge between two daughters and other debris.

What gets measured where
------------------------
Shape (area_px, circularity, ...) is the brightfield cell, so the mitosis rule
in classify_phase sees whole-cell rounding.  Marker intensities (mcherry_*,
mvenus_*) are the nucleus, as before with --segment-on sum, so thresholds stay
comparable.  Whole-cell intensities are kept as cell_mcherry_* etc.

When a cell has no segmented nucleus in a frame, its nucleus is CARRIED: the
same track's nucleus shape from the nearest frame where it was segmented is
placed at the same offset from the cell centroid and clipped to the cell.  With
no segmented nucleus anywhere in the track, a disc of the tile's median nucleus
area is used.  `nuc_source` says which: segmented | carried | disc.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

import fucci_pipeline as fp

FUSE_DEFAULTS = dict(
    nuc_in_cell=0.5,          # share of a nucleus inside a cell for it to belong
    straddle_share=0.25,      # each of >=2 cells holding this much -> merge them
    touch_px=2,               # fragments closer than ~2x this are one nucleus
    orphan_grow_px=15,        # fluorescence-only cell = nucleus grown this far
    min_cell_px=1500,         # nucleus-less objects smaller than this dropped
    min_nuc_px=400,           # fluorescence objects smaller than this ignored
    daughter_circularity=0.80,  # round objects are not merged (daughters)
    # Rule 4a, off by default.  On A9-2 it cut area flips 185 -> 174 but on
    # track 57 (cell 4's daughter) gave the daughter's nucleus to a neighbour
    # whose own nucleus was unsegmented that frame and whose outline had
    # swallowed the daughter -- the daughter then vanished for t14-t27.  A
    # separate fluor_only cell is clumsier but keeps the cell alive.
    extend_cells=False,
)


# --------------------------------------------------------------------------- #
# label bookkeeping
# --------------------------------------------------------------------------- #
def overlap_table(a, b):
    """{(label_a, label_b): pixels in both} for every non-zero pair."""
    sel = (a > 0) & (b > 0)
    if not sel.any():
        return {}
    nb = int(b.max()) + 1
    key = a[sel].astype(np.int64) * nb + b[sel].astype(np.int64)
    u, cnt = np.unique(key, return_counts=True)
    return {(int(k // nb), int(k % nb)): int(c) for k, c in zip(u, cnt)}


def overlap_fractions(prev, curr, min_frac=0.05):
    """{(prev_label, curr_label): overlap / smaller area}.

    The smaller area, not the union, because the interesting cases are
    lopsided: a flat cell rounding up for mitosis shrinks to a fraction of its
    footprint but sits almost entirely inside it.
    """
    pa = np.bincount(prev.ravel())
    ca = np.bincount(curr.ravel())
    out = {}
    for (p, c), n in overlap_table(prev, curr).items():
        f = n / max(min(pa[p], ca[c]), 1)
        if f >= min_frac:
            out[(p, c)] = float(f)
    return out


class _Union:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _apply_union(mask, uf):
    if not uf.parent:
        return mask
    lut = np.arange(int(mask.max()) + 1, dtype=mask.dtype)
    for x in list(uf.parent):
        if x < len(lut):
            lut[x] = uf.find(x)
    return lut[mask]


def _shares(fl, cell):
    """{nucleus: {cell: fraction of the nucleus inside that cell}}."""
    fa = np.bincount(fl.ravel())
    out = defaultdict(dict)
    for (n, c), px in overlap_table(fl, cell).items():
        out[n][c] = px / fa[n]
    return out


def _owners(shares, thresh):
    """{cell: {nuclei mostly inside it}}."""
    own = defaultdict(set)
    for n, s in shares.items():
        c = max(s, key=s.get)
        if s[c] >= thresh:
            own[c].add(n)
    return own


def _touching_pairs(fl, touch_px):
    import skimage as ski
    grown = ski.segmentation.expand_labels(fl, distance=touch_px)
    pairs = set()
    for a, b in ((grown[:, :-1], grown[:, 1:]), (grown[:-1, :], grown[1:, :])):
        sel = (a > 0) & (b > 0) & (a != b)
        if sel.any():
            for x, y in zip(a[sel].tolist(), b[sel].tolist()):
                pairs.add((min(x, y), max(x, y)))
    return pairs


def _circularity(mask):
    import skimage as ski
    out = {}
    for p in ski.measure.regionprops(mask):
        per = p.perimeter
        out[p.label] = 4 * np.pi * p.area / per ** 2 if per > 0 else 0.0
    return out


# --------------------------------------------------------------------------- #
# fusion
# --------------------------------------------------------------------------- #
def fuse_masks(bf, fl, params=None, exclude=None):
    """Combine a brightfield cell mask and a fluorescence nucleus mask.

    exclude: bool image of brightfield cells removed for touching the border.
    Their nuclei are removed too; otherwise rule 4 would resurrect every
    partial edge cell as a fluorescence-only cell.

    Returns (cell, nuc, info, log):
        cell  int32 labels 1..N, the final cells
        nuc   int32, same label as the cell it belongs to, 0 elsewhere
        info  {label: {'cell_source': 'bf' | 'bf_merged' | 'bf_split' |
                       'bf_extended' | 'fluor_only', 'has_nucleus': bool}}
        log   {'rejoined_nuclei', 'merged', 'split', 'fluor_only',
               'dropped_small'} counts for the per-frame printout
    """
    import skimage as ski
    from scipy import ndimage as ndi
    from skimage.segmentation import relabel_sequential, watershed

    p = dict(FUSE_DEFAULTS)
    p.update(params or {})
    bf = np.array(bf, np.int32)          # copies: the steps below edit in place
    fl = np.array(fl, np.int32)
    counts = dict(rejoined_nuclei=0, merged=0, split=0, extended=0,
                  fluor_only=0, dropped_small=0)
    source = {}

    if fl.max():
        fa = np.bincount(fl.ravel())
        drop = [n for n in range(1, len(fa)) if 0 < fa[n] < p['min_nuc_px']]
        if exclude is not None and exclude.any():
            inside = np.bincount(fl[exclude], minlength=len(fa))
            drop += [n for n in range(1, len(fa))
                     if fa[n] and inside[n] / fa[n] >= p['nuc_in_cell']]
        if drop:
            fl = np.where(np.isin(fl, drop), 0, fl)

    # 1. rejoin touching nucleus fragments inside one cell
    if fl.max():
        sh = _shares(fl, bf)
        best = {n: max(s, key=s.get) for n, s in sh.items()}
        uf = _Union()
        for a, b in _touching_pairs(fl, p['touch_px']):
            if best.get(a) is not None and best.get(a) == best.get(b):
                uf.union(a, b)
                counts['rejoined_nuclei'] += 1
        fl = _apply_union(fl, uf)

    # 2. merge brightfield pieces that share one nucleus
    cell = bf
    if fl.max() and bf.max():
        sh = _shares(fl, bf)
        own = _owners(sh, p['nuc_in_cell'])
        circ = _circularity(bf)
        uf = _Union()
        for n, s in sh.items():
            big = [c for c, v in s.items() if v >= p['straddle_share']]
            if len(big) < 2:
                continue
            if any(own.get(c, set()) - {n} for c in big):
                continue
            if sum(circ.get(c, 0) >= p['daughter_circularity'] for c in big) >= 2:
                continue
            for c in big[1:]:
                uf.union(big[0], c)
            counts['merged'] += len(big) - 1
        cell = _apply_union(bf, uf)
        for x in uf.parent:
            source[uf.find(x)] = 'bf_merged'

    # 3. split cells holding two or more nuclei
    if fl.max() and cell.max():
        own = _owners(_shares(fl, cell), p['nuc_in_cell'])
        next_lab = int(cell.max()) + 1
        objs = ndi.find_objects(cell)
        for c, ns in own.items():
            if len(ns) < 2 or c > len(objs) or objs[c - 1] is None:
                continue
            sl = objs[c - 1]
            region = cell[sl] == c
            markers = np.zeros(region.shape, np.int32)
            for i, n in enumerate(sorted(ns), start=1):
                markers[(fl[sl] == n) & region] = i
            if markers.max() < 2:
                continue
            dist = ndi.distance_transform_edt(markers == 0)
            ws = watershed(dist, markers, mask=region)
            sub = cell[sl]
            for i in range(2, int(ws.max()) + 1):
                sub[ws == i] = next_lab
                source[next_lab] = 'bf_split'
                next_lab += 1
            source[c] = 'bf_split'
            counts['split'] += int(ws.max()) - 1

    # 4a. a nucleus hanging off a nucleus-less cell belongs to it: a faint flat
    #     cell's brightfield outline often stops short of its own nucleus, and
    #     making the nucleus a separate cell gives the tracker two objects for
    #     one cell to hop between
    if p['extend_cells'] and fl.max() and cell.max():
        sh = _shares(fl, cell)
        own = _owners(sh, p['nuc_in_cell'])
        claimed = set()
        for n, s in sh.items():
            if sum(s.values()) >= p['nuc_in_cell']:
                continue
            # part of it must already be inside: merely touching a dim
            # neighbour attached nuclei to the wrong cell on A9-2
            cand = {c: v for c, v in s.items()
                    if v >= p['straddle_share'] and not own.get(c)
                    and c not in claimed}
            if not cand:
                continue
            c = max(cand, key=cand.get)
            cell[(fl == n) & (cell == 0)] = c
            claimed.add(c)
            source[c] = 'bf_extended'
            counts['extended'] += 1

    # 4b. nuclei with no brightfield cell become cells
    if fl.max():
        sh = _shares(fl, cell)
        next_lab = int(cell.max()) + 1
        disk = ski.morphology.disk(p['orphan_grow_px'])
        objs = ndi.find_objects(fl)
        pad = p['orphan_grow_px'] + 1
        for n in range(1, len(objs) + 1):
            if objs[n - 1] is None:
                continue
            s = sh.get(n, {})
            # mostly inside cells, even if split between them, is not missed
            if s and sum(s.values()) >= p['nuc_in_cell']:
                continue
            r0, r1 = max(objs[n - 1][0].start - pad, 0), objs[n - 1][0].stop + pad
            c0, c1 = max(objs[n - 1][1].start - pad, 0), objs[n - 1][1].stop + pad
            nucl = fl[r0:r1, c0:c1] == n
            grown = ndi.binary_dilation(nucl, structure=disk)
            sub = cell[r0:r1, c0:c1]
            sub[(grown & (sub == 0)) | nucl] = next_lab
            source[next_lab] = 'fluor_only'
            next_lab += 1
            counts['fluor_only'] += 1

    # 5. nucleus per cell, then drop small nucleus-less debris
    nuc = np.zeros_like(cell)
    if fl.max() and cell.max():
        sh = _shares(fl, cell)
        fa = np.bincount(fl.ravel())
        taken = {}
        shared = []
        for n, s in sh.items():
            c = max(s, key=s.get)
            if s[c] < p['nuc_in_cell']:
                # one blob over several cells (a mitotic cell's signal over
                # both daughters): each cell takes its own part of it
                shared += [(c2, n) for c2, v in s.items()
                           if v >= p['straddle_share']]
                continue
            if c in taken and fa[taken[c]] >= fa[n]:
                continue
            taken[c] = n
        for c, n in shared:
            taken.setdefault(c, n)
        for c, n in taken.items():
            nuc[(fl == n) & (cell == c)] = c
    has_nuc = set(np.unique(nuc[nuc > 0]).tolist())
    ca = np.bincount(cell.ravel())
    small = [c for c in range(1, len(ca)) if ca[c] and c not in has_nuc
             and ca[c] < p['min_cell_px']]
    if small:
        cell[np.isin(cell, small)] = 0
        counts['dropped_small'] = len(small)

    cell, fw, _inv = relabel_sequential(cell)
    cell = cell.astype(np.int32)
    nuc = fw[nuc].astype(np.int32)
    info = {}
    for old in range(1, len(fw)):
        new = int(fw[old])
        if new:
            info[new] = {'cell_source': source.get(old, 'bf'),
                         'has_nucleus': old in has_nuc}
    return cell, nuc, info, counts


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
# A cell rounded up for mitosis is refractile: in brightfield it has a dark rim
# and a bright interior.  Round alone is not enough -- Cellpose sometimes
# shrinks a flat cell's outline to a small round shape around its nucleus,
# which is exactly as round and as small.  Measured on A9-2 (15 frames, every
# round-flagged cell checked by eye): real mitoses had rim 0.59-0.75 and
# interior 1.055-1.117 relative to the background just outside the cell; the
# false flags and interphase cells 0.92-0.97 and 1.000-1.013.
MITOTIC_HALO = dict(max_rim=0.85, min_interior=1.03)


def bf_halo(bf, mask):
    """(rim, interior): brightfield at the cell edge and inside, relative to
    the median of a ring 2-12 px outside.  rim is the 5th percentile of a band
    +-4 px around the outline, interior the mean 6 px in from it."""
    from scipy import ndimage as ndi
    ring = ndi.binary_dilation(mask, iterations=12) & \
        ~ndi.binary_dilation(mask, iterations=2)
    if not ring.any():
        return np.nan, np.nan
    bg = float(np.median(bf[ring]))
    if bg <= 0:
        return np.nan, np.nan
    band = ndi.binary_dilation(mask, iterations=4) & \
        ~ndi.binary_erosion(mask, iterations=4)
    inner = ndi.binary_erosion(mask, iterations=6)
    rim = float(np.percentile(bf[band], 5)) / bg if band.any() else np.nan
    interior = float(bf[inner].mean()) / bg if inner.any() else np.nan
    return rim, interior

SHAPE_KEYS = ('area_px', 'perimeter_px', 'circularity', 'solidity',
              'eccentricity', 'axis_major_px', 'axis_minor_px', 'area_um2')


def measure_dual(corrected, cell, nuc, info, pixel_size_um=None):
    """Per-cell features for one fused frame.

    Returns (feats, nuc_shapes, pending):
        feats       {label: row dict}; cells without a nucleus have no marker
                    columns yet -- carry_nuclei fills them after tracking
        nuc_shapes  {label: (bool crop, (d_row, d_col) from the cell centroid)}
                    for every segmented nucleus, the source of carried shapes
        pending     {label: (bbox, cell crop, image crop)} for cells whose
                    nucleus has to be carried in; the crops are small, so the
                    frame does not have to be read a second time
    """
    from scipy import ndimage as ndi

    cf = fp.measure_frame(corrected, cell, pixel_size_um)
    nf = fp.measure_frame(corrected, nuc) if nuc.max() else {}
    feats, nuc_shapes, pending = {}, {}, {}
    objs = ndi.find_objects(cell)
    bf = corrected[:, :, fp.CH_BF]
    pad = 14
    nobjs = ndi.find_objects(nuc) if nuc.max() else []
    for lab, rec in cf.items():
        out = {k: v for k, v in rec.items() if k not in fp.INTENSITY_KEYS}
        for k in fp.INTENSITY_KEYS:
            out[f'cell_{k}'] = rec[k]
        out['cell_source'] = info.get(lab, {}).get('cell_source', 'bf')
        sl = objs[lab - 1]
        big = (slice(max(sl[0].start - pad, 0), sl[0].stop + pad),
               slice(max(sl[1].start - pad, 0), sl[1].stop + pad))
        out['bf_rim'], out['bf_interior'] = bf_halo(bf[big], cell[big] == lab)
        if lab in nf:
            n = nf[lab]
            for k in fp.INTENSITY_KEYS:
                out[k] = n[k]
            out['nuc_area_px'] = n['area_px']
            out['nuc_circularity'] = n['circularity']
            out['nuc_centroid_row'] = n['centroid_row']
            out['nuc_centroid_col'] = n['centroid_col']
            out['nuc_source'] = 'segmented'
            sl = nobjs[lab - 1]
            nuc_shapes[lab] = (nuc[sl] == lab,
                               (n['centroid_row'] - rec['centroid_row'],
                                n['centroid_col'] - rec['centroid_col']),
                               (sl[0].start, sl[1].start))
        else:
            sl = objs[lab - 1]
            pending[lab] = ((sl[0].start, sl[1].start),
                            cell[sl] == lab,
                            corrected[sl].astype(np.float32))
        feats[lab] = out
    return feats, nuc_shapes, pending


def _place(shape, shape_origin, shape_centroid, target_centroid, bbox_origin,
           bbox_shape):
    """Paste a bool crop so its centroid lands on target; return bbox crop."""
    out = np.zeros(bbox_shape, bool)
    dr = int(round(target_centroid[0] - shape_centroid[0]))
    dc = int(round(target_centroid[1] - shape_centroid[1]))
    r0 = shape_origin[0] + dr - bbox_origin[0]
    c0 = shape_origin[1] + dc - bbox_origin[1]
    h, w = shape.shape
    rr0, cc0 = max(r0, 0), max(c0, 0)
    rr1, cc1 = min(r0 + h, bbox_shape[0]), min(c0 + w, bbox_shape[1])
    if rr1 > rr0 and cc1 > cc0:
        out[rr0:rr1, cc0:cc1] = shape[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0]
    return out


def _signal_peak(crop, cmask, radius, peak_mads):
    """(row, col) in the crop of the brightest smoothed marker signal, or None.

    Only the cell's own pixels are smoothed and only its interior (at least
    0.6 radius from the outline) is searched: a dark cell's brightest pixels
    are otherwise the glow of a neighbour's nucleus across the shared edge.
    """
    from scipy import ndimage as ndi
    sig = (crop[:, :, fp.CH_MCHERRY] + crop[:, :, fp.CH_MVENUS]) * cmask
    sigma = max(radius / 2, 1.0)
    norm = ndi.gaussian_filter(cmask.astype(np.float32), sigma)
    sm = ndi.gaussian_filter(sig, sigma) / np.maximum(norm, 1e-3)
    interior = cmask & (ndi.distance_transform_edt(cmask) >= 0.6 * radius)
    vals = sm[interior]
    if vals.size == 0:
        return None
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) or 1.0
    masked = np.where(interior, sm, -np.inf)
    idx = np.unravel_index(int(np.argmax(masked)), sm.shape)
    if sm[idx] < med + peak_mads * mad:
        return None
    return idx


def carry_nuclei(tracks, features, nuc_shapes, pending, min_keep=0.3,
                 peak_mads=4.0):
    """Give every tracked cell without a segmented nucleus a nucleus mask.

    The donor is the same track's segmented nucleus nearest in time, placed at
    the offset from the cell centroid it had then and clipped to the current
    cell.  If less than `min_keep` of it survives the clipping, or the track
    never had a segmented nucleus, a disc of the tile's median nucleus area
    sits on the brightest (smoothed) marker signal inside the cell instead --
    a dim nucleus Cellpose could not segment is usually still the brightest
    thing in its cell -- or at the centroid if there is no clear peak
    (peak below the cell median + `peak_mads` MADs: a truly dark cell, whose
    intensity is ~0 wherever it is measured).

    Fills the marker columns of `features` in place and returns
    {(tp, label): (bbox origin, bool crop)} -- the masks used, for the overlay.
    """
    lab_of = defaultdict(dict)          # track -> {tp: label}
    for tp, m in tracks.items():
        for lab, tid in m.items():
            lab_of[tid][tp] = lab
    areas = [s[0].sum() for by in nuc_shapes.values() for s in by.values()]
    med = float(np.median(areas)) if areas else 2500.0
    radius = np.sqrt(med / np.pi)
    used = {}
    for tid, by_tp in lab_of.items():
        donors = [t for t, lab in by_tp.items() if lab in nuc_shapes.get(t, {})]
        for tp, lab in by_tp.items():
            if lab not in pending.get(tp, {}):
                continue
            origin, cmask, crop = pending[tp][lab]
            feat = features[tp][lab]
            cen = (feat['centroid_row'], feat['centroid_col'])
            placed, src, from_tp = None, 'disc', np.nan
            if donors:
                t2 = min(donors, key=lambda t: (abs(t - tp), t))
                shape, off, sorigin = nuc_shapes[t2][by_tp[t2]]
                d = features[t2][by_tp[t2]]
                scen = (d['nuc_centroid_row'], d['nuc_centroid_col'])
                target = (cen[0] + off[0], cen[1] + off[1])
                cand = _place(shape, sorigin, scen, target, origin,
                              cmask.shape) & cmask
                if cand.sum() >= min_keep * shape.sum():
                    placed, src, from_tp = cand, 'carried', t2
            if placed is None:
                centre = _signal_peak(crop, cmask, radius, peak_mads)
                if centre is None:
                    centre = (cen[0] - origin[0], cen[1] - origin[1])
                rr, cc = np.ogrid[:cmask.shape[0], :cmask.shape[1]]
                placed = (((rr - centre[0]) ** 2 + (cc - centre[1]) ** 2)
                          <= radius ** 2) & cmask
                if not placed.any():
                    placed = cmask
            feat.update(fp.intensity_features(crop[placed]))
            feat['nuc_area_px'] = float(placed.sum())
            feat['nuc_circularity'] = np.nan
            pr, pc = np.nonzero(placed)
            feat['nuc_centroid_row'] = float(pr.mean() + origin[0])
            feat['nuc_centroid_col'] = float(pc.mean() + origin[1])
            feat['nuc_source'] = src
            feat['nuc_carried_from_tp'] = from_tp
            used[(tp, lab)] = (origin, placed)
    return used


# --------------------------------------------------------------------------- #
# saved masks, for diagnostics
# --------------------------------------------------------------------------- #
# <masks_dir>/fused_tNNN.npz  cell, nuc: the final labels of that frame, the
#                             labels the CSV's cell_label refers to; nuc
#                             includes carried / disc nuclei
# <masks_dir>/reference.npz   bf: frame-1 brightfield; fluor_max: max over all
#                             frames of mCherry + mVenus (background-subtracted)
def fused_path(save_dir, tp):
    import os
    return os.path.join(save_dir, f'fused_t{tp:03d}.npz')


def save_fused(save_dir, tp, cell, nuc):
    import os
    os.makedirs(save_dir, exist_ok=True)
    path = fused_path(save_dir, tp)
    tmp = path + '.part.npz'
    np.savez_compressed(tmp, cell=cell.astype(np.uint16),
                        nuc=nuc.astype(np.uint16))
    os.replace(tmp, path)


def load_fused(save_dir, tp):
    m = np.load(fused_path(save_dir, tp))
    return m['cell'], m['nuc']


def add_carried(save_dir, used):
    """Paste the carried / disc nuclei from carry_nuclei into the saved frames."""
    by_tp = defaultdict(list)
    for (tp, lab), (origin, crop) in used.items():
        by_tp[tp].append((lab, origin, crop))
    for tp, items in by_tp.items():
        cell, nuc = load_fused(save_dir, tp)
        for lab, origin, crop in items:
            sub = nuc[origin[0]:origin[0] + crop.shape[0],
                      origin[1]:origin[1] + crop.shape[1]]
            sub[crop] = lab
        save_fused(save_dir, tp, cell, nuc)


def update_reference(ref, plane, corrected):
    fl = corrected[:, :, fp.CH_MCHERRY] + corrected[:, :, fp.CH_MVENUS]
    if 'bf' not in ref:
        ref['bf'] = plane[:, :, fp.CH_BF].copy()
        ref['fluor_max'] = fl.astype(np.float32)
    else:
        np.maximum(ref['fluor_max'], fl, out=ref['fluor_max'])


def save_reference(save_dir, ref):
    import os
    if ref:
        np.savez_compressed(os.path.join(save_dir, 'reference.npz'),
                            bf=ref['bf'], fluor_max=ref['fluor_max'])


# --------------------------------------------------------------------------- #
# one tile, end to end
# --------------------------------------------------------------------------- #
def track_position(feat, track_on='nucleus'):
    if track_on == 'nucleus' and feat.get('nuc_source') == 'segmented':
        return (feat['nuc_centroid_row'], feat['nuc_centroid_col'])
    return (feat['centroid_row'], feat['centroid_col'])


def split_edge(raw):
    """(mask without border-touching labels, bool image of what was removed)."""
    import fucci_imaging as fi
    kept = fi.drop_edge_labels(raw)
    return kept, (raw > 0) & (kept == 0)


def run_dual(read_frame, n, get_masks, max_dist, well='', tile=0,
             fuse_params=None, track_kwargs=None, min_division_overlap=3,
             heal_window=0, heal_frac=0.75, pixel_size_um=None,
             keep_masks=False, keep_planes=False, log=print, save_dir=None,
             track_on='cell'):
    """Segment (twice), fuse, track and measure one tile.

    read_frame(i)                  -> registered [H, W, 3] uint16 plane, i 0-based
    get_masks(tp, plane, corrected) -> (brightfield labels, fluorescence labels),
                                      both raw from Cellpose (edges not removed)

    Taking those two as functions keeps this independent of ImageJ and of
    Cellpose: fucci_imaging passes its readers, a test passes cached arrays.

    save_dir: write every frame's final masks there (see save_fused), which is
    what diagnose_tracks.py draws from.

    track_on: the position the tracker links on.  'cell' (default) is the
    brightfield cell centroid.  'nucleus' uses the segmented nucleus centroid,
    falling back to the cell centroid in frames without one (a carried nucleus
    is placed after tracking, so it cannot be used here).  Measured on A9-2,
    96 frames: no difference in nucleus jumps (13 vs 13) or area flips, and
    one division fewer -- linking is decided mostly by mask overlap, so the
    point used barely matters.
    Returns the same dict as fucci_imaging.process_tile.
    """
    track_kwargs = dict(track_kwargs or {})
    max_gap = track_kwargs.get('max_gap', 0)
    features, centroids, areas, circs, halo = {}, {}, {}, {}, {}
    halo_thr = dict(MITOTIC_HALO)
    halo_thr.update(track_kwargs.pop('mitotic_halo', None) or {})
    nuc_shapes, pending, overlaps = {}, {}, {}
    masks, nuc_masks, kept_planes = {}, {}, {}
    fuse_totals = defaultdict(int)
    buffer = []                                   # [(tp, cell mask)]
    ref = {}
    for i in range(n):
        tp = i + 1
        plane = read_frame(i)
        corrected = fp.subtract_background(plane)
        bf_raw, fl_raw = get_masks(tp, plane, corrected)
        bf, edge = split_edge(np.asarray(bf_raw, np.int32))
        fl, _ = split_edge(np.asarray(fl_raw, np.int32))
        cell, nuc, info, cnt = fuse_masks(bf, fl, fuse_params, exclude=edge)
        for k, v in cnt.items():
            fuse_totals[k] += v
        feats, shapes, pend = measure_dual(corrected, cell, nuc, info,
                                           pixel_size_um)
        features[tp], nuc_shapes[tp], pending[tp] = feats, shapes, pend
        centroids[tp] = {lab: track_position(v, track_on)
                         for lab, v in feats.items()}
        areas[tp] = {lab: v['area_px'] for lab, v in feats.items()}
        circs[tp] = {lab: v['circularity'] for lab, v in feats.items()}
        halo[tp] = {lab: bool(v['bf_rim'] <= halo_thr['max_rim']
                              and v['bf_interior'] >= halo_thr['min_interior'])
                    for lab, v in feats.items()}
        for tprev, mprev in buffer:
            overlaps[(tprev, tp)] = overlap_fractions(mprev, cell)
        buffer = (buffer + [(tp, cell)])[-(max_gap + 1):]
        if keep_masks:
            masks[tp] = cell.astype(np.uint16)
            nuc_masks[tp] = nuc.astype(np.uint16)
        if save_dir:
            save_fused(save_dir, tp, cell, nuc)
            update_reference(ref, plane, corrected)
        if keep_planes:
            kept_planes[tp] = plane.copy()
        log(f'  frame {tp:3d}/{n}: {len(feats)} cells '
            f'({len(shapes)} with a fluorescent nucleus); fused: '
            + ', '.join(f'{k}={v}' for k, v in cnt.items() if v))
    timepoints = sorted(features)

    div_rejects = []
    import timelapse_tracking as tt
    tracks, records, lineage = tt.track_series(
        centroids, max_dist, method='staged', areas_per_tp=areas,
        overlaps=overlaps, circularity_per_tp=circs,
        mitotic_evidence_per_tp=halo,
        division_rejects=div_rejects, **track_kwargs)
    div_reverted = []
    if track_kwargs.get('allow_division') and min_division_overlap > 0:
        tracks, lineage, div_reverted = tt.prune_spurious_divisions(
            tracks, lineage, areas_per_tp=areas,
            min_overlap=min_division_overlap, log=log,
            heal_window=heal_window, heal_frac=heal_frac)

    used = carry_nuclei(tracks, features, nuc_shapes, pending)
    if save_dir:
        add_carried(save_dir, used)
        save_reference(save_dir, ref)
    if keep_masks:
        for (tp, lab), (origin, crop) in used.items():
            sub = nuc_masks[tp][origin[0]:origin[0] + crop.shape[0],
                                origin[1]:origin[1] + crop.shape[1]]
            sub[crop] = lab

    mitotic = {(r['timepoint'], r['cell_label']) for r in records
               if r.get('mitotic')}
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
                   'track_end_reason': info.get('end_reason'),
                   'division_kind': info.get('division_kind', ''),
                   'mitotic_flag': (tp, lab) in mitotic}
            row.update(feat)
            row['mask_source'] = 'segmented'
            rows.append(row)

    nuc_area = {tp: {lab: f.get('nuc_area_px') for lab, f in fs.items()}
                for tp, fs in features.items()}
    return {'rows': rows, 'tracks': tracks, 'records': records,
            'lineage': lineage, 'timepoints': timepoints, 'masks': masks,
            'nuc_masks': nuc_masks, 'planes': kept_planes,
            'centroids': centroids, 'n_frames': n,
            'division_rejects': div_rejects, 'division_reverted': div_reverted,
            'fuse_totals': dict(fuse_totals), 'nuc_areas': nuc_area,
            'timing': timing}
