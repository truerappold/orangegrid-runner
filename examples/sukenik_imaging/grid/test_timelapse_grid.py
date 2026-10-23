"""Checks the fast per-cell steps against the originals, bit for bit.

    python test_timelapse_grid.py

Dense synthetic fields (touching cells, nuclei with holes, background
trimming), no GPU, no ImageJ.  Each fast function must give exactly what the
function it replaces gives.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.environ.get(
    'CODE_DIR', os.path.join(HERE, '..', 'analysis'))))
sys.path.insert(0, HERE)
import fast_timelapse as ft  # noqa: E402
import timelapse_tracking as tt  # noqa: E402

O = ft._ORIGINAL
FAILS = []


def check(name, cond):
    print(f'  {"PASS" if cond else "FAIL"}  {name}')
    if not cond:
        FAILS.append(name)


def field(seed, n=35, size=360):
    from skimage.segmentation import expand_labels
    rng = np.random.default_rng(seed)
    seeds = np.zeros((size, size), np.int32)
    pts = rng.integers(15, size - 15, (n, 2))
    for i, (r, c) in enumerate(pts, start=1):
        seeds[r, c] = i
    cell = expand_labels(seeds, 28)
    img = rng.normal(150, 20, (size, size, 4)).clip(0, None)
    rr, cc = np.ogrid[:size, :size]
    for (r, c) in pts:                                   # bright nuclei, some with holes
        d2 = (rr - r) ** 2 + (cc - c) ** 2
        img[:, :, tt.CH_DAPI] += 900 * (d2 < 100) * (d2 > (4 if seed % 2 else 0))
        img[:, :, tt.CH_MNG] += 400 * (d2 < 400)
    return img.astype(np.uint16), cell


def same_meas(a, b):
    if (a is None) != (b is None):
        return False
    if a is None:
        return True
    return a.keys() == b.keys() and all(
        (np.isnan(a[k]) and np.isnan(b[k])) or a[k] == b[k] for k in a)


print('\n--- fast per-cell steps vs originals ---')
for seed in range(6):
    img, cell = field(seed)  # seeds 4-5 put cells against the image border
    if seed >= 4:
        cell = np.pad(cell[20:, 20:], ((0, 20), (0, 20)))
    trimmed = cell.copy()
    for ch, (bg, th) in tt.BG_SUBTRACT.items():
        img[:, :, ch], trimmed = tt.background_subtract(img[:, :, ch], trimmed, bg, th)
    d0 = O['segment_dapi_per_cell'](img[:, :, tt.CH_DAPI], cell)
    d1 = ft.fast_segment_dapi_per_cell(img[:, :, tt.CH_DAPI], cell)
    check(f'seed {seed}: DAPI per-cell threshold identical', np.array_equal(d0, d1))
    n0 = O['refine_nuclei'](trimmed, d0)
    n1 = ft.fast_refine_nuclei(trimmed, d0)
    check(f'seed {seed}: nuclei identical ({int(n0.max())} labelled)',
          np.array_equal(n0, n1))
    labs = range(1, int(cell.max()) + 1)
    m0 = {lab: O['measure_cell'](img, cell, trimmed, n0, lab) for lab in labs}
    m1 = {lab: ft.fast_measure_cell(img, cell, trimmed, n0, lab) for lab in labs}
    check(f'seed {seed}: all {sum(v is not None for v in m0.values())} '
          f'measurements identical', all(same_meas(m0[k], m1[k]) for k in labs))

print('\n--- install / uninstall ---')
import timelapse_imaging as ti  # noqa: E402
ft.install()
check('installed', tt.measure_cell is ft.fast_measure_cell
      and ti.segment_dapi_per_cell is ft.fast_segment_dapi_per_cell)
ft.uninstall()
check('restored', tt.measure_cell is O['measure_cell'])

print('\n' + ('ALL TESTS PASSED' if not FAILS else f'{len(FAILS)} FAILURE(S): {FAILS}'))
raise SystemExit(1 if FAILS else 0)
