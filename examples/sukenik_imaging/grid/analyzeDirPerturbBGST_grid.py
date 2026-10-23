"""
analyzeDirPerturbBGST.py, rewritten to run one well (or one tile) per
OrangeGrid job.  Same images in, same numbers out.

What is the same
----------------
Every image operation, in the same order, with the same parameters:
registration (pystackreg rigid body on Otsu masks), Cellpose on the
before-donor image, edge-mask removal, the flat background subtraction with
its per-cell noise trimming, the DAPI largest-component + hole filling, and
every per-cell quantity and formula.  The CSV has the same columns in the same
order, and the same leading unnamed index column (all 0) the original wrote.
test_fret_grid.py checks this against the original on the example tile.

What is different
-----------------
* Each per-cell step works on a crop around that cell instead of the whole
  2304 x 2304 frame.  The original dilated a full-frame image once per cell
  per channel in the background step and indexed full frames ~60 times per
  cell in the measurement; the crops give identical results (the crops are
  padded so nothing at their edge behaves differently) at a fraction of the
  cost.
* Rows are collected and turned into one table at the end; the original
  concatenated the table once per cell, which is quadratic in cell count.
* The Cellpose model is created once (the original created it twice).
* Tiles are found from the file names instead of assumed to be 1-9, and a
  missing tile is reported rather than crashing the run.
* One CSV per tile (<well>-<tile>_BGSTperturbq.csv), so a job that is
  interrupted keeps what it finished and a re-run skips finished tiles.
  `ogrun combine OUT --pattern '*_BGSTperturbq.csv' --no-job-column` joins
  them into the original's cells_df_BGSTperturbq.csv layout.

Usage
-----
    python analyzeDirPerturbBGST_grid.py EXPERIMENT_DIR --well A2 [--tiles 1 2]
    python analyzeDirPerturbBGST_grid.py EXPERIMENT_DIR --list   # wells/tiles found

EXPERIMENT_DIR holds Pics/ (the .vsi files) and seqs.csv, as before.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd

CHANNELS = ['FRET, FRET', 'mNG', 'DAPI']
SHAPE = (2304, 2304)

# background (flat value, noise threshold) per channel, as in the original
BG = {0: (102, 24), 1: (105, 24), 2: (101, 20),
      4: (102, 24), 5: (105, 24), 6: (101, 20)}
BLEED_DIRECT, BLEED_DONOR = 0.070, 0.4163

NAME_RE = re.compile(r'^(?P<after>1?)(?P<well>[A-Za-z]+\d+)-(?P<tile>\d+)_(?P<ch>.+)\.vsi$')


# --------------------------------------------------------------------------- #
# file discovery
# --------------------------------------------------------------------------- #
def pics_dir(exp_dir):
    return os.path.join(exp_dir, 'Pics')


def discover(exp_dir):
    """{well: [tiles]} for tiles that have all 3 channels before AND after,
    plus a list of problems (incomplete tiles)."""
    have = {}
    for p in glob.glob(os.path.join(pics_dir(exp_dir), '*.vsi')):
        m = NAME_RE.match(os.path.basename(p))
        if m and m.group('ch') in CHANNELS:
            key = (m.group('well'), int(m.group('tile')))
            have.setdefault(key, set()).add((m.group('after') == '1',
                                             m.group('ch')))
    need = {(a, c) for a in (False, True) for c in CHANNELS}
    tiles, problems = {}, []
    for (well, tile), got in sorted(have.items()):
        if got >= need:
            tiles.setdefault(well, []).append(tile)
        else:
            missing = sorted(('1' if a else '') + f'{well}-{tile}_{c}.vsi'
                             for a, c in need - got)
            problems.append(f'{well}-{tile}: missing {", ".join(missing)}')
    return tiles, problems


def load_seqs(exp_dir):
    return pd.read_csv(os.path.join(exp_dir, 'seqs.csv'), index_col=0)


# --------------------------------------------------------------------------- #
# image steps (behaviour identical to the original)
# --------------------------------------------------------------------------- #
def _bbox(sl, pad, shape):
    return (slice(max(sl[0].start - pad, 0), min(sl[0].stop + pad, shape[0])),
            slice(max(sl[1].start - pad, 0), min(sl[1].stop + pad, shape[1])))


def background_subtract(img, mask, background, threshold):
    """Same as backgroundSubtract() in the original.

    The original, for each cell id in ascending order: takes the pixels of
    that cell (in the mask as already trimmed by earlier cells) that are at or
    below the noise threshold, dilates them with a 5x5 square, and zeroes the
    mask there -- which can also trim a neighbour.  Done here on the cell's
    bounding box plus 2 px (the dilation's reach), in the same order, so every
    pixel decision is the same.
    """
    import skimage as ski
    from scipy import ndimage as ndi

    tmp = img.astype(np.int32) - background
    tmp[tmp < 1] = 1
    out = mask.copy()
    noisy = tmp <= threshold
    fp5 = ski.morphology.footprint_rectangle((5, 5))
    objs = ndi.find_objects(mask)
    for mask_id in np.unique(mask):
        if mask_id == 0 or mask_id > len(objs) or objs[mask_id - 1] is None:
            continue
        sl = _bbox(objs[mask_id - 1], 2, mask.shape)
        noise = noisy[sl] & (out[sl] == mask_id)
        if not noise.any():
            continue
        dil = ski.morphology.binary_dilation(noise, footprint=fp5)
        out[sl][dil] = 0
    return tmp.astype(np.uint16), out


def _remove_small_holes(binary, area_threshold):
    import skimage as ski
    try:
        return ski.morphology.remove_small_holes(binary, max_size=area_threshold)
    except TypeError:
        return ski.morphology.remove_small_holes(binary,
                                                 area_threshold=area_threshold)


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


def clean_dapi(maskDAPI, maskDAPIafter, maskFRET, bbox_src):
    """Per cell: keep the largest DAPI component (before and after), fill
    holes up to 200 px.  Same rule and order as the original, on crops."""
    import skimage as ski
    from scipy import ndimage as ndi

    objs = ndi.find_objects(bbox_src)
    for cell_id in np.unique(maskFRET):
        if cell_id == 0:
            continue
        sl = _bbox(objs[cell_id - 1], 1, maskFRET.shape)
        cell_dapi = maskDAPI[sl] == cell_id
        cell_dapi_after = maskDAPIafter[sl] == cell_id
        if not np.any(cell_dapi) or not np.any(cell_dapi_after):
            continue
        lab = ski.measure.label(cell_dapi)
        lab_after = ski.measure.label(cell_dapi_after)
        regions = ski.measure.regionprops(lab)
        regions_after = ski.measure.regionprops(lab_after)
        if len(regions_after) == 0:
            continue
        big = max(regions, key=lambda r: r.area)
        big_after = max(regions_after, key=lambda r: r.area)
        cut = (sl[0].start > 0, sl[0].stop < maskFRET.shape[0],
               sl[1].start > 0, sl[1].stop < maskFRET.shape[1])
        filled = fill_holes_like_full_frame(lab == big.label, 200,
                                            _remove_small_holes, cut=cut)
        filled_after = fill_holes_like_full_frame(lab_after == big_after.label,
                                                  200, _remove_small_holes,
                                                  cut=cut)
        in_cell = maskFRET[sl] == cell_id
        sub, sub_after = maskDAPI[sl], maskDAPIafter[sl]
        sub[in_cell] = 0
        sub[filled] = cell_id
        sub_after[in_cell] = 0
        sub_after[filled_after] = cell_id
    return maskDAPI, maskDAPIafter


def quantile_means(values, suffix):
    """Mean intensity within each of 10 quantile bins (as the original)."""
    q = np.percentile(values, np.linspace(0, 100, 11))
    out = {}
    for i in range(10):
        if i == 9:
            sel = (values >= q[i]) & (values <= q[i + 1])
        else:
            sel = (values >= q[i]) & (values < q[i + 1])
        out[f'directA_Q{i + 1}_{suffix}'] = (np.mean(values[sel])
                                             if np.any(sel) else np.nan)
    return out


def measure_cells(image, maskFRETo, maskFRET, maskDAPI, construct, well, tile):
    """One row per cell, every column of the original, in its order."""
    import skimage as ski
    from scipy import ndimage as ndi

    rows = []
    objs = ndi.find_objects(maskFRETo)
    for cell in np.arange(1, np.max(maskFRET) + 1):
        if cell > len(objs) or objs[cell - 1] is None:
            continue
        sl = _bbox(objs[cell - 1], 1, maskFRET.shape)
        sel_cell_original = maskFRETo[sl] == cell
        sel_cell = maskFRET[sl] == cell
        sel_nuc = maskDAPI[sl] == cell
        sel_cyto = np.bitwise_xor(sel_cell, sel_nuc)
        if not np.any(sel_nuc) or not np.any(sel_cyto):
            continue
        im = image[sl]
        v = {}
        for name, ch_b, ch_a in (('D', 0, 4), ('A', 1, 5), ('directA', 2, 6)):
            for reg, sel in (('cell', sel_cell), ('cyto', sel_cyto),
                             ('nuc', sel_nuc)):
                b, a = im[:, :, ch_b][sel], im[:, :, ch_a][sel]
                v[f'{name}_{reg}'] = np.mean(b)
                v[f'{name}_{reg}_after'] = np.mean(a)
                v[f'{name}_{reg}_max'] = np.max(b)
                v[f'{name}_{reg}_max_after'] = np.max(a)
        for reg in ('cell', 'cyto', 'nuc'):
            for suf in ('', '_after'):
                v[f'A_{reg}_corr{suf}'] = (v[f'A_{reg}{suf}']
                                           - BLEED_DIRECT * v[f'directA_{reg}{suf}']
                                           - BLEED_DONOR * v[f'D_{reg}{suf}'])
                v[f'Ef_{reg}{suf}'] = v[f'A_{reg}_corr{suf}'] / (
                    v[f'A_{reg}_corr{suf}'] + v[f'D_{reg}{suf}'])
            v[f'Ef_{reg}_delta'] = v[f'Ef_{reg}_after'] - v[f'Ef_{reg}']
        area_o = np.sum(sel_cell_original)
        perim = ski.measure.perimeter(sel_cell_original)
        circ = 4 * np.pi * area_o / (perim ** 2) if perim > 0 else np.nan

        row = {'construct': construct, 'well': well, 'tile': tile,
               'cell_idx': cell, 'area_cell_original': area_o,
               'area_cell': np.sum(sel_cell), 'area_cyto': np.sum(sel_cyto),
               'area_nuc': np.sum(sel_nuc)}
        for k in ('D_cell', 'D_cell_after', 'D_cyto', 'D_cyto_after', 'D_nuc',
                  'D_nuc_after', 'A_cell', 'A_cell_after', 'A_cyto',
                  'A_cyto_after', 'A_nuc', 'A_nuc_after', 'A_cell_corr',
                  'A_cell_corr_after', 'A_cyto_corr', 'A_cyto_corr_after',
                  'A_nuc_corr', 'A_nuc_corr_after', 'Ef_cell', 'Ef_cell_after',
                  'Ef_cell_delta', 'Ef_cyto', 'Ef_cyto_after', 'Ef_cyto_delta',
                  'Ef_nuc', 'Ef_nuc_after', 'Ef_nuc_delta', 'directA_cell',
                  'directA_cell_after', 'directA_cyto', 'directA_cyto_after',
                  'directA_nuc', 'directA_nuc_after', 'D_cell_max',
                  'D_cell_max_after', 'D_cyto_max', 'D_cyto_max_after',
                  'D_nuc_max', 'D_nuc_max_after', 'A_cell_max',
                  'A_cell_max_after', 'A_cyto_max', 'A_cyto_max_after',
                  'A_nuc_max', 'A_nuc_max_after', 'directA_cell_max',
                  'directA_cell_max_after'):
            row[k] = v[k]
        row['circ_cell_original'] = circ
        row.update(quantile_means(im[:, :, 2][sel_cyto], 'cyto'))
        row.update(quantile_means(im[:, :, 2][sel_nuc], 'nuc'))
        row.update(quantile_means(im[:, :, 6][sel_cyto], 'cyto_after'))
        row.update(quantile_means(im[:, :, 6][sel_nuc], 'nuc_after'))
        rows.append(row)
    return rows


def process_tile(ij, model, sr, exp_dir, well, tile, construct, log=print):
    """The original's per-tile body.  Returns a list of row dicts."""
    import skimage as ski
    from skimage.segmentation import relabel_sequential

    pics = pics_dir(exp_dir)
    image = np.zeros([SHAPE[0], SHAPE[1], 8], dtype='uint16')
    tmatrix = tmatrixP = tmatrixW = None
    maskFRET = maskDAPI = maskDAPIafter = None

    def otsu(x):
        return x >= ski.filters.threshold_otsu(x)

    def warp(x, m):
        return ski.transform.warp(x, m, preserve_range=True).astype('uint16')

    for channel in CHANNELS:
        dataset = ij.io().open(os.path.join(pics, f'{well}-{tile}_{channel}.vsi'))
        after = ij.io().open(os.path.join(pics, f'1{well}-{tile}_{channel}.vsi'))
        if channel == 'FRET, FRET':
            image[:, :, 0] = ij.py.from_java(dataset[:, :, 1])
            image[:, :, 4] = ij.py.from_java(after[:, :, 1])
            tmatrixP = sr.register(otsu(image[:, :, 0]), otsu(image[:, :, 4]))
            tmatrixW = sr.register(otsu(image[:, :, 0]), otsu(image[:, :, 4]))
            image[:, :, 4] = warp(image[:, :, 4], tmatrixP)
            masks, _flows, _styles = model.eval(
                image[:, :, 0], batch_size=32, flow_threshold=0.4,
                progress=False, cellprob_threshold=0.0,
                normalize={'tile_norm_blocksize': 0})
            maskFRET = masks
            edge = np.unique(np.concatenate([maskFRET[0, :], maskFRET[-1, :],
                                             maskFRET[:, 0], maskFRET[:, -1]]))
            maskFRET[np.isin(maskFRET, edge)] = 0
            maskFRET, _, _ = relabel_sequential(maskFRET)

            image[:, :, 1] = ij.py.from_java(dataset[:, :, 0])
            image[:, :, 5] = ij.py.from_java(after[:, :, 0])
            tmatrix = sr.register(otsu(image[:, :, 0]), otsu(image[:, :, 1]))
            image[:, :, 1] = warp(image[:, :, 1], tmatrix)
            tmatrixP = sr.register(otsu(image[:, :, 4]), otsu(image[:, :, 5]))
            image[:, :, 5] = warp(image[:, :, 5], tmatrixP)
        elif channel == 'mNG':
            image[:, :, 2] = ij.py.from_java(dataset)
            image[:, :, 2] = warp(image[:, :, 2], tmatrix)
            image[:, :, 6] = ij.py.from_java(after)
            image[:, :, 6] = warp(image[:, :, 6], tmatrixP)
        elif channel == 'DAPI':
            image[:, :, 3] = ij.py.from_java(dataset)
            image[:, :, 7] = ij.py.from_java(after)
            image[:, :, 7] = warp(image[:, :, 7], tmatrixW)
            maskDAPI = image[:, :, 3] > ski.filters.threshold_otsu(image[:, :, 3])
            maskDAPIafter = image[:, :, 7] > ski.filters.threshold_otsu(
                image[:, :, 7])

    if np.max(maskFRET) == 0:
        log(f'Warning: No cells detected in well {well}, tile {tile}. Skipping...')
        return []

    maskFRETo = maskFRET.copy()
    for ch in (0, 1, 2, 4, 5, 6):
        image[:, :, ch], maskFRET = background_subtract(image[:, :, ch], maskFRET,
                                                        *BG[ch])
    maskDAPI = maskDAPI * maskFRET
    maskDAPIafter = maskDAPIafter * maskFRET
    maskDAPI, maskDAPIafter = clean_dapi(maskDAPI, maskDAPIafter, maskFRET,
                                         maskFRETo)
    return measure_cells(image, maskFRETo, maskFRET, maskDAPI, construct, well,
                         tile)


def tile_csv(out_dir, well, tile):
    return os.path.join(out_dir, f'{well}-{tile}_BGSTperturbq.csv')


def write_rows(rows, path):
    """Same layout as the original: a leading unnamed index column of 0s."""
    df = pd.DataFrame(rows)
    df.index = [0] * len(df)
    tmp = path + '.part'
    df.to_csv(tmp)
    os.replace(tmp, path)            # never leave a half-written CSV behind


def init_imagej():
    import imagej
    import scyjava
    mem = os.environ.get('JAVA_MEM')
    if mem:
        scyjava.config.add_option(f'-Xmx{mem}')
    return imagej.init('sc.fiji:fiji:2.14.0')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('directory', help='experiment folder with Pics/ and seqs.csv')
    ap.add_argument('--well', help='well to process (e.g. A2); default all')
    ap.add_argument('--tiles', type=int, nargs='*', default=None,
                    help='tiles to process (default: every complete tile)')
    ap.add_argument('--output-dir', default=None,
                    help='default: <directory>/BGSTperturbq_tiles')
    ap.add_argument('--force', action='store_true',
                    help='redo tiles whose CSV already exists')
    ap.add_argument('--list', action='store_true',
                    help='print the wells/tiles found and exit')
    args = ap.parse_args()

    exp = os.path.abspath(args.directory)
    seqs = load_seqs(exp)
    tiles, problems = discover(exp)
    for p in problems:
        print(f'  incomplete: {p}')
    missing = sorted(w for w in tiles if w not in seqs.index)
    if missing:
        print(f'  wells imaged but not in seqs.csv (skipped): {", ".join(missing)}')
    tiles = {w: t for w, t in tiles.items() if w in seqs.index}
    if args.list:
        for w, ts in tiles.items():
            print(f'{w} ({seqs.loc[w, "construct"]}): tiles {ts}')
        return
    wells = [args.well] if args.well else sorted(tiles)
    for w in wells:
        if w not in tiles:
            raise SystemExit(f'{w}: no complete tiles (or not in seqs.csv)')
    out_dir = args.output_dir or os.path.join(exp, 'BGSTperturbq_tiles')
    os.makedirs(out_dir, exist_ok=True)

    todo = [(w, t) for w in wells for t in (args.tiles or tiles[w])
            if args.force or not os.path.exists(tile_csv(out_dir, w, t))]
    if not todo:
        print('every requested tile already has a CSV (use --force to redo)')
        return
    print(f'{len(todo)} tile(s) to process: '
          + ', '.join(f'{w}-{t}' for w, t in todo))

    from cellpose import models
    from pystackreg import StackReg
    model = models.CellposeModel(gpu=True)
    print(f'cellpose on GPU: {model.gpu}')
    ij = init_imagej()
    print(f'ImageJ {ij.getVersion()}')
    sr = StackReg(StackReg.RIGID_BODY)

    for w, t in todo:
        t0 = time.time()
        rows = process_tile(ij, model, sr, exp, w, t, seqs.loc[w, 'construct'])
        write_rows(rows, tile_csv(out_dir, w, t))
        print(f'{w}-{t}: {len(rows)} cells, {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    sys.exit(main())
