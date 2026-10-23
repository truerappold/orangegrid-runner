"""Tests for fucci_dual.py and the mitosis-aware tracking it drives -- run with:
python test_fucci_dual.py

Synthetic masks only; no Cellpose, no ImageJ.
"""

import numpy as np

import fucci_dual as fd
import timelapse_tracking as tt

FAILS = []


def check(name, cond, detail=''):
    if cond:
        print(f'  PASS  {name}')
    else:
        print(f'  FAIL  {name}   {detail}')
        FAILS.append(name)


H = W = 200


def disc(mask, r, c, rad, lab):
    rr, cc = np.ogrid[:mask.shape[0], :mask.shape[1]]
    mask[(rr - r) ** 2 + (cc - c) ** 2 <= rad ** 2] = lab
    return mask


P = dict(min_cell_px=300, min_nuc_px=20, orphan_grow_px=5)

# ------------------------------------------------------------------ fusion --
print('\n--- brightfield cut one cell in two; its nucleus spans both ---')
bf = np.zeros((H, W), np.int32)
bf[40:120, 40:80] = 1
bf[40:120, 80:120] = 2
fl = disc(np.zeros((H, W), np.int32), 80, 80, 15, 1)
cell, nuc, info, cnt = fd.fuse_masks(bf, fl, P)
check('merged into one cell', cell.max() == 1 and cnt['merged'] == 1, cnt)
check('marked bf_merged', info[1]['cell_source'] == 'bf_merged', info)
check('the cell has its nucleus', info[1]['has_nucleus'] and (nuc == 1).sum() > 0)

print('\n--- two round daughters sharing one mitotic blob are NOT merged ---')
bf = disc(np.zeros((H, W), np.int32), 80, 60, 18, 1)
disc(bf, 80, 100, 18, 2)
fl = np.zeros((H, W), np.int32)
fl[70:90, 50:110] = 1
cell, nuc, info, cnt = fd.fuse_masks(bf, fl, P)
check('two cells remain', cell.max() == 2 and cnt['merged'] == 0
      and cnt['fluor_only'] == 0, cnt)
check('each daughter takes its own part of the blob',
      info[1]['has_nucleus'] and info[2]['has_nucleus'], info)

print('\n--- one brightfield object holding two separate nuclei ---')
bf = np.zeros((H, W), np.int32)
bf[40:120, 30:170] = 1
fl = disc(np.zeros((H, W), np.int32), 80, 60, 12, 1)
disc(fl, 80, 140, 12, 2)
cell, nuc, info, cnt = fd.fuse_masks(bf, fl, P)
check('split into two cells', cell.max() == 2 and cnt['split'] == 1, cnt)
check('each half holds one nucleus',
      all(info[c]['has_nucleus'] for c in (1, 2)), info)
check('split along the middle',
      cell[80, 60] != cell[80, 140] and cell[80, 60] > 0 and cell[80, 140] > 0)

print('\n--- touching nucleus fragments inside one cell are one nucleus ---')
bf = np.zeros((H, W), np.int32)
bf[40:120, 40:120] = 1
fl = np.zeros((H, W), np.int32)
fl[70:90, 60:80] = 1
fl[70:90, 81:100] = 2           # 1 px gap: Cellpose cut the nucleus in two
cell, nuc, info, cnt = fd.fuse_masks(bf, fl, P)
check('fragments rejoined, cell not split',
      cell.max() == 1 and cnt['rejoined_nuclei'] == 1 and cnt['split'] == 0, cnt)

print('\n--- extend_cells: a cell whose outline stops short of its nucleus ---')
bf = np.zeros((H, W), np.int32)
bf[40:120, 40:100] = 1                       # nucleus-less cell
fl = np.zeros((H, W), np.int32)
fl[70:90, 92:122] = 1                        # 8 of 30 columns (27%) inside
cell, nuc, info, cnt = fd.fuse_masks(bf, fl, P)
check('off by default: a fluor_only cell as before',
      cnt['extended'] == 0 and cnt['fluor_only'] == 1, cnt)
PX = dict(P, extend_cells=True)
cell, nuc, info, cnt = fd.fuse_masks(bf, fl, PX)
check('extended, not a second fluor_only cell',
      cell.max() == 1 and cnt['extended'] == 1 and cnt['fluor_only'] == 0, cnt)
check('the cell now owns the nucleus',
      info[1]['cell_source'] == 'bf_extended' and info[1]['has_nucleus'], info)
fl2 = np.zeros((H, W), np.int32)
fl2[70:90, 101:130] = 1                      # only touching, nothing inside
cell, nuc, info, cnt = fd.fuse_masks(bf, fl2, PX)
check('a nucleus merely touching a cell is not given to it',
      cnt['extended'] == 0 and cnt['fluor_only'] == 1, cnt)

print('\n--- a nucleus brightfield missed becomes a cell ---')
bf = np.zeros((H, W), np.int32)
bf[10:60, 10:60] = 1
fl = disc(np.zeros((H, W), np.int32), 30, 30, 8, 1)
disc(fl, 140, 140, 10, 2)
cell, nuc, info, cnt = fd.fuse_masks(bf, fl, P)
check('two cells', cell.max() == 2 and cnt['fluor_only'] == 1, cnt)
orphan = cell[140, 140]
check('the new one is fluor_only and larger than its nucleus',
      info[orphan]['cell_source'] == 'fluor_only'
      and (cell == orphan).sum() > (nuc == orphan).sum(), info)

print('\n--- nuclei of cells removed at the border are not resurrected ---')
bf_raw = np.zeros((H, W), np.int32)
bf_raw[0:50, 60:120] = 1        # touches the top edge
bf, edge = fd.split_edge(bf_raw)
fl = disc(np.zeros((H, W), np.int32), 25, 90, 10, 1)
cell, nuc, info, cnt = fd.fuse_masks(bf, fl, P, exclude=edge)
check('no cell made from an edge cell nucleus',
      cell.max() == 0 and cnt['fluor_only'] == 0, cnt)

print('\n--- nucleus-less objects: dim cells kept, debris dropped ---')
bf = np.zeros((H, W), np.int32)
bf[20:80, 20:80] = 1            # 3600 px, a dim cell
bf[150:160, 150:160] = 2        # 100 px of debris
cell, nuc, info, cnt = fd.fuse_masks(bf, np.zeros((H, W), np.int32), P)
check('the dim cell stays, the debris goes',
      cell.max() == 1 and cnt['dropped_small'] == 1, cnt)
check('the dim cell has no nucleus yet', not info[1]['has_nucleus'])

# --------------------------------------------------------- carried nucleus --
print('\n--- a cell that goes dark keeps a nucleus mask ---')
planes, cells, nucs = [], [], []
for t in range(3):
    img = np.zeros((H, W, 3), np.float32)
    c = np.zeros((H, W), np.int32)
    c[50:130, 50 + 10 * t:130 + 10 * t] = 1
    n = np.zeros((H, W), np.int32)
    if t == 0:
        n[70:90, 60:80] = 1     # off-centre in the cell, 20 px left
        img[70:90, 60:80, 2] = 500.0
    else:
        img[70:90, 60 + 10 * t:80 + 10 * t, 2] = 40.0   # dim, unsegmented
    planes.append(img)
    cells.append(c)
    nucs.append(n)
feats, shapes, pend = {}, {}, {}
for t in range(3):
    info = {1: {'cell_source': 'bf', 'has_nucleus': t == 0}}
    feats[t + 1], shapes[t + 1], pend[t + 1] = fd.measure_dual(
        planes[t], cells[t], nucs[t], info)
tracks = {1: {1: 7}, 2: {1: 7}, 3: {1: 7}}
used = fd.carry_nuclei(tracks, feats, shapes, pend)
check('frames 2 and 3 get a carried nucleus',
      feats[2][1]['nuc_source'] == 'carried'
      and feats[3][1]['nuc_source'] == 'carried', feats[2][1].get('nuc_source'))
check('it moved with the cell and kept its offset (sits on the dim signal)',
      np.isclose(feats[3][1]['mvenus_mean'], 40.0), feats[3][1]['mvenus_mean'])
check('the source frame is recorded', feats[3][1]['nuc_carried_from_tp'] == 1)
check('segmented frame is untouched',
      feats[1][1]['nuc_source'] == 'segmented'
      and np.isclose(feats[1][1]['mvenus_mean'], 500.0))
tracks_none = {2: {1: 9}, 3: {1: 9}}
f2 = {t: {1: dict(feats[t][1])} for t in (2, 3)}
fd.carry_nuclei(tracks_none, f2, {2: {}, 3: {}}, {t: pend[t] for t in (2, 3)})
check('a track never segmented gets a disc', f2[2][1]['nuc_source'] == 'disc')

# --------------------------------------------------------------- tracking --
print('\n--- a cell rounding up is linked by overlap, not distance ---')
# flat cell centroid at col 100; it rounds up at col 50, 50 px away -- far
# beyond max_dist -- but inside its old footprint
cen = {1: {1: (100.0, 100.0)}, 2: {1: (100.0, 50.0)}}
ar = {1: {1: 20000}, 2: {1: 5000}}
ov = {(1, 2): {(1, 1): 0.95}}
tr, _r, _l = tt.track_series(cen, 30, method='staged', areas_per_tp=ar)
check('distance alone loses it', tr[2] == {}, tr[2])
tr, _r, _l = tt.track_series(cen, 30, method='staged', areas_per_tp=ar,
                             overlaps=ov)
check('overlap keeps it', tr[2] == {1: 1}, tr[2])

THR = dict(min_daughter_frac=0.30, max_daughter_frac=1.10, min_sum_frac=0.80,
           max_sum_frac=2.20, min_symmetry=0.35, max_midpoint_ratio=0.6)
KW = dict(method='staged', allow_division=True, division_max_dist=60,
          division_thresholds=THR, mitotic_circularity=0.85, max_gap=2)

print('\n--- a mitotic cell divides past the bridge between its daughters ---')
# t1-2 flat; t3 round and shrunk (flagged); t4 left daughter 76 px away,
# right daughter 35 px, and the cytoplasmic bridge 17 px -- the closest.
cen = {1: {1: (370.0, 868.0)}, 2: {1: (371.0, 845.0)}, 3: {1: (371.0, 837.0)},
       4: {1: (384.0, 870.0), 2: (378.0, 822.0), 3: (353.0, 763.0)},
       5: {1: (398.0, 865.0), 3: (373.0, 766.0)}}
ar = {1: {1: 35000}, 2: {1: 30000}, 3: {1: 6900},
      4: {1: 4000, 2: 2100, 3: 8000}, 5: {1: 10000, 3: 9000}}
ci = {1: {1: 0.6}, 2: {1: 0.6}, 3: {1: 0.9},
      4: {1: 0.84, 2: 0.79, 3: 0.8}, 5: {1: 0.55, 3: 0.6}}
tr, rec, lin = tt.track_series(cen, 30, areas_per_tp=ar,
                               circularity_per_tp=ci, **KW)
kids = sorted(k for k, v in lin.items() if v['parent_track_id'] == 1)
check('the cell divides at t4', lin[1]['end_reason'] == 'divided'
      and lin[1]['end_tp'] == 3, lin[1])
check('the daughters are the two cells, not the bridge',
      sorted(lab for lab, t in tr[4].items() if t in kids) == [1, 3], tr[4])
check('both daughters are followed into t5', len(tr[5]) == 2, tr[5])
tr_nf, _r, lin_nf = tt.track_series(cen, 30, areas_per_tp=ar,
                                    circularity_per_tp=ci,
                                    **dict(KW, mitotic_circularity=None))
check('without the flag the far daughter is missed',
      3 not in tr_nf[4], (tr_nf[4], lin_nf))

print('\n--- one daughter seen: the cell gets a new id, not the parent id ---')
cen = {t: {1: (100.0, 100.0 + t)} for t in range(1, 9)}
ar = {1: {1: 20000}, 2: {1: 20000}, 3: {1: 6000}, 4: {1: 6000},
      5: {1: 9000}, 6: {1: 9500}, 7: {1: 10000}, 8: {1: 10500}}
ci = {1: {1: 0.6}, 2: {1: 0.6}, 3: {1: 0.92}, 4: {1: 0.9},
      5: {1: 0.6}, 6: {1: 0.55}, 7: {1: 0.55}, 8: {1: 0.55}}
cen[7][2] = (100.0, 140.0)       # the sibling finally segmented at t7
ar[7][2] = 9000
ci[7][2] = 0.6
cen[8][2] = (100.0, 142.0)
ar[8][2] = 9200
ci[8][2] = 0.6
tr, rec, lin = tt.track_series(cen, 30, areas_per_tp=ar,
                               circularity_per_tp=ci, **KW)
d1 = tr[5][1]
check('the parent ends where it flattened out',
      lin[1]['end_reason'] == 'divided' and lin[1]['end_tp'] == 4, lin[1])
check('the cell continues as a daughter (division_kind single)',
      d1 != 1 and lin[d1]['division_kind'] == 'single', lin.get(d1))
sib = tr[7].get(2)
check('the sibling found at t7 is adopted as the second daughter',
      sib is not None and lin[sib]['parent_track_id'] == 1
      and lin[sib]['division_kind'] == 'late_sibling', lin.get(sib))

print('\n--- a cell that rounds and flattens back to full size did not divide ---')
cen = {t: {1: (100.0, 100.0 + t)} for t in range(1, 8)}
ar = {1: {1: 20000}, 2: {1: 20000}, 3: {1: 6000}, 4: {1: 19000},
      5: {1: 20000}, 6: {1: 20000}, 7: {1: 20000}}
ci = {1: {1: 0.6}, 2: {1: 0.6}, 3: {1: 0.92}, 4: {1: 0.6},
      5: {1: 0.6}, 6: {1: 0.6}, 7: {1: 0.6}}
tr, rec, lin = tt.track_series(cen, 30, areas_per_tp=ar,
                               circularity_per_tp=ci, **KW)
check('one track throughout', set(lin) == {1}
      and all(tr[t] == {1: 1} for t in tr), lin)

print('\n--- overlap fractions are relative to the smaller mask ---')
a = np.zeros((50, 50), np.int32)
a[0:40, 0:40] = 1
b = np.zeros((50, 50), np.int32)
b[10:20, 10:20] = 3
check('a small mask inside a big one overlaps 1.0',
      fd.overlap_fractions(a, b) == {(1, 3): 1.0}, fd.overlap_fractions(a, b))

print('\n--- a never-segmented dim nucleus: the disc goes on the signal ---')
img = np.zeros((H, W, 3), np.float32)
c = np.zeros((H, W), np.int32)
c[40:160, 40:160] = 1
img[50:70, 50:70, 2] = 60.0            # dim nucleus in a corner of the cell
fz, sz, pz = fd.measure_dual(img, c, np.zeros((H, W), np.int32),
                             {1: {'cell_source': 'bf', 'has_nucleus': False}})
fd.carry_nuclei({1: {1: 3}}, {1: fz}, {1: sz}, {1: pz})
check('disc sits on the dim nucleus, not the empty centre',
      fz[1]['nuc_source'] == 'disc' and fz[1]['mvenus_p90'] > 30,
      fz[1]['mvenus_p90'])

print('\n--- brightfield halo: a refractile round cell vs a flat one ---')
bfimg = np.full((120, 120), 1000.0)
m = disc(np.zeros((120, 120), np.int32), 60, 60, 25, 1) == 1
ring = disc(np.zeros((120, 120), np.int32), 60, 60, 27, 1) == 1
bfimg[ring & ~m] = 600.0               # dark rim just outside the edge
bfimg[disc(np.zeros((120, 120), np.int32), 60, 60, 22, 1) == 1] = 1100.0
rim, interior = fd.bf_halo(bfimg, m)
check('mitotic-looking cell: dark rim, bright interior',
      rim <= fd.MITOTIC_HALO['max_rim']
      and interior >= fd.MITOTIC_HALO['min_interior'], (rim, interior))
rim, interior = fd.bf_halo(np.full((120, 120), 1000.0), m)
check('flat featureless cell: neither', rim > 0.85 and interior < 1.03,
      (rim, interior))

print('\n--- round without the halo is not mitotic ---')
cen = {t: {1: (100.0, 100.0 + t)} for t in range(1, 8)}
ar = {1: {1: 20000}, 2: {1: 20000}, 3: {1: 6000}, 4: {1: 9000},
      5: {1: 9000}, 6: {1: 9000}, 7: {1: 9000}}
ci = {1: {1: 0.6}, 2: {1: 0.6}, 3: {1: 0.92}, 4: {1: 0.6},
      5: {1: 0.6}, 6: {1: 0.6}, 7: {1: 0.6}}
no_halo = {t: {1: False} for t in cen}
tr, rec, lin = tt.track_series(cen, 30, areas_per_tp=ar, circularity_per_tp=ci,
                               mitotic_evidence_per_tp=no_halo, **KW)
check('no single-daughter division is invented', set(lin) == {1}, lin)
halo = dict(no_halo)
halo[3] = {1: True}
tr, rec, lin = tt.track_series(cen, 30, areas_per_tp=ar, circularity_per_tp=ci,
                               mitotic_evidence_per_tp=halo, **KW)
check('with the halo it is', lin[1]['end_reason'] == 'divided', lin[1])

print('\n--- division_requires_mitosis: a flat cell cut in half is not a division ---')
cen = {1: {1: (100.0, 100.0)}, 2: {1: (100.0, 101.0)},
       3: {1: (100.0, 85.0), 2: (100.0, 117.0)},
       4: {1: (100.0, 102.0)}}
ar = {1: {1: 20000}, 2: {1: 20000}, 3: {1: 10000, 2: 10000}, 4: {1: 20000}}
ci = {t: {lab: 0.6 for lab in cen[t]} for t in cen}
tr, rec, lin = tt.track_series(cen, 30, areas_per_tp=ar, circularity_per_tp=ci,
                               **dict(KW, max_gap=0))
check('without the rule it is accepted', lin[1]['end_reason'] == 'divided')
tr, rec, lin = tt.track_series(cen, 30, areas_per_tp=ar, circularity_per_tp=ci,
                               division_requires_mitosis=True,
                               **dict(KW, max_gap=0))
check('with it the track carries on', set(lin) == {1}
      and all(1 in tr[t].values() for t in tr), lin)
check('the other half is dropped as not_mitotic',
      any(r['drop_reason'] == 'not_mitotic' for r in rec if r['dropped']),
      [r['drop_reason'] for r in rec if r['dropped']])

print('\n' + ('ALL TESTS PASSED' if not FAILS else
              f'{len(FAILS)} FAILURE(S): {FAILS}'))
raise SystemExit(1 if FAILS else 0)
