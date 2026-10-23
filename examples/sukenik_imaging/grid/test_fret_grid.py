"""Checks that analyzeDirPerturbBGST_grid.py computes what the original does.

    python test_fret_grid.py

The original's backgroundSubtract() is loaded straight from
../analysis/analyzeDirPerturbBGST.py (only that function -- importing
the module would start Cellpose) and compared on dense synthetic fields, where
neighbouring cells trim each other.  No GPU, no ImageJ.

The full pipeline (Cellpose, registration, DAPI, every column) was compared
end to end on real images before release: byte-identical CSVs on the example
tile and on dense fields built from A9-2 frames (see ../README.md).
"""

import ast
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyzeDirPerturbBGST_grid as g  # noqa: E402

ORIGINAL = os.path.join(HERE, '..', 'analysis',
                        'analyzeDirPerturbBGST.py')
FAILS = []


def check(name, cond, detail=''):
    print(f'  {"PASS" if cond else "FAIL"}  {name}   {detail if not cond else ""}')
    if not cond:
        FAILS.append(name)


def load_original_function(path, name):
    src = open(path).read()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {}
    exec('import numpy as np\nimport skimage as ski\n' + ast.get_source_segment(src, fn), ns)
    return ns[name]


def dense_field(seed, n=40, size=400):
    """Touching blob cells and a noisy image near the noise floor."""
    rng = np.random.default_rng(seed)
    from skimage.segmentation import expand_labels
    mask = np.zeros((size, size), np.int32)
    for i, (r, c) in enumerate(rng.integers(20, size - 20, (n, 2)), start=1):
        mask[r, c] = i
    mask = expand_labels(mask, 30)
    img = (rng.normal(120, 15, (size, size))).clip(0, 65535).astype(np.uint16)
    return img, mask


if not os.path.exists(ORIGINAL):
    print(f'original not found at {ORIGINAL}')
    raise SystemExit(1)
orig_bg = load_original_function(ORIGINAL, 'backgroundSubtract')

print('\n--- background subtraction: original vs grid ---')
for seed in range(4):
    img, mask = dense_field(seed)
    for bg, thr in ((102, 24), (105, 24), (101, 20)):
        ri, rm = orig_bg(img, mask, bg, thr)
        ni, nm = g.background_subtract(img, mask, bg, thr)
        trimmed = int(((mask > 0) & (rm == 0)).sum())
        check(f'seed {seed} bg {bg}/{thr}: identical ({trimmed} px trimmed)',
              np.array_equal(ri, ni) and np.array_equal(rm, nm))

print('\n--- DAPI clean-up: grid crops vs the original full-frame loop ---')


def original_dapi_loop(maskDAPI, maskDAPIafter, maskFRET):
    """The per-cell DAPI block of the original main(), verbatim apart from
    being wrapped in a function (full-frame arrays, area_threshold=200)."""
    import skimage as ski
    for cell_id in np.unique(maskFRET):
        if cell_id == 0:
            continue
        cell_dapi = (maskDAPI == cell_id)
        cell_dapi_after = (maskDAPIafter == cell_id)
        if not np.any(cell_dapi) or not np.any(cell_dapi_after):
            continue
        labeled_dapi = ski.measure.label(cell_dapi)
        labeled_dapi_after = ski.measure.label(cell_dapi_after)
        regions = ski.measure.regionprops(labeled_dapi)
        regions_after = ski.measure.regionprops(labeled_dapi_after)
        if len(regions_after) == 0:
            continue
        largest_region = max(regions, key=lambda r: r.area)
        largest_region_after = max(regions_after, key=lambda r: r.area)
        largest_mask = (labeled_dapi == largest_region.label)
        largest_mask_after = (labeled_dapi_after == largest_region_after.label)
        filled_mask = g._remove_small_holes(largest_mask, 200)
        filled_mask_after = g._remove_small_holes(largest_mask_after, 200)
        maskDAPI[maskFRET == cell_id] = 0
        maskDAPI[filled_mask] = cell_id
        maskDAPIafter[maskFRET == cell_id] = 0
        maskDAPIafter[filled_mask_after] = cell_id
    return maskDAPI, maskDAPIafter


rng = np.random.default_rng(7)
for case, (n, grow) in enumerate(((40, 25), (120, 8), (60, 12))):
    from skimage.segmentation import expand_labels
    size = 300
    seeds = np.zeros((size, size), np.int32)
    for i, (r, c) in enumerate(rng.integers(10, size - 10, (n, 2)), start=1):
        seeds[r, c] = i
    fret_mask = expand_labels(seeds, grow)           # grow 8: tiny cells, tight crops
    dapi_b = rng.random((size, size)) > 0.35         # speckled: many components + holes
    dapi_a = rng.random((size, size)) > 0.35
    md0, ma0 = original_dapi_loop(dapi_b * fret_mask, dapi_a * fret_mask, fret_mask)
    md1, ma1 = g.clean_dapi(dapi_b * fret_mask, dapi_a * fret_mask, fret_mask,
                            fret_mask)
    check(f'case {case} ({n} cells, size {grow}): identical before and after',
          np.array_equal(md0, md1) and np.array_equal(ma0, ma1))

print('\n--- quantile means match the original formula ---')
v = np.arange(1, 101, dtype=float)
q = g.quantile_means(v, 'cyto')
check('10 bins of 10 values', np.allclose([q[f'directA_Q{i}_cyto'] for i in range(1, 11)],
                                          [5.5 + 10 * i for i in range(10)]), q)

print('\n--- file discovery ---')
import tempfile  # noqa: E402
with tempfile.TemporaryDirectory() as d:
    os.makedirs(os.path.join(d, 'Pics'))
    for pre in ('', '1'):
        for ch in g.CHANNELS:
            open(os.path.join(d, 'Pics', f'{pre}B3-2_{ch}.vsi'), 'w').close()
    open(os.path.join(d, 'Pics', 'B3-5_DAPI.vsi'), 'w').close()      # incomplete
    tiles, problems = g.discover(d)
    check('complete tile found', tiles == {'B3': [2]}, tiles)
    check('incomplete tile reported, not crashed on',
          len(problems) == 1 and 'B3-5' in problems[0], problems)

print('\n' + ('ALL TESTS PASSED' if not FAILS else f'{len(FAILS)} FAILURE(S): {FAILS}'))
raise SystemExit(1 if FAILS else 0)
