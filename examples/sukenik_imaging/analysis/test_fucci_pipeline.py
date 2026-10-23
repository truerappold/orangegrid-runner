"""Unit tests for fucci_pipeline.py -- run with the Anaconda python.

Covers the parts that need no GPU or JVM: filename parsing, discovery,
background subtraction, per-cell features, division timing, and the phase calls.
"""
import numpy as np
import pandas as pd
import skimage as ski

import fucci_pipeline as fp

FAILS = []
def check(name, cond, detail=''):
    print(('  PASS  ' if cond else '  FAIL  ') + name + ('' if cond else f'   {detail}'))
    if not cond: FAILS.append(name)

print('\n--- filenames / discovery ---')
check('parses a channel with no prefix',
      fp.parse_vsi_name('A1-2_mCherryNeoPIPFUCCI.vsi') ==
      {'well': 'A1', 'tile': 2, 'channel': 'mCherryNeoPIPFUCCI'})
check('channel name with a space survives',
      fp.parse_vsi_name('H12-1_BF Cam 1.vsi')['channel'] == 'BF Cam 1')
check('two-digit column parses', fp.parse_vsi_name('H12-1_BF Cam 1.vsi')['well'] == 'H12')
check('non-matching name is None', fp.parse_vsi_name('notes.txt') is None)
check('build_filename round-trips',
      fp.build_filename('/d', 'A1', 2, 'BF Cam 1').endswith('A1-2_BF Cam 1.vsi'))

files = [f'/d/{w}-{t}_{c}.vsi' for w in ('A1', 'B2') for t in (1, 2)
         for c in fp.CHANNEL_FILES.values()]
tiles, probs = fp.discover_tiles(files)
check('all complete tiles found', tiles == [('A1', 1), ('A1', 2), ('B2', 1), ('B2', 2)], tiles)
check('no problems for a clean set', probs == [], probs)
partial = [f for f in files if 'B2-2_mVenusNeoPIPFUCCI' not in f]
tiles2, probs2 = fp.discover_tiles(partial)
check('tile missing a channel is excluded', ('B2', 2) not in tiles2, tiles2)
check('and it is reported', any('missing' in p for p in probs2), probs2)

print('\n--- background subtraction ---')
# Read the constants rather than hardcoding them: BACKGROUND is meant to be
# edited per experiment, and a fixture that pins its value fails every time
# someone does exactly what the script tells them to do.
bg_c = fp.BACKGROUND[fp.CH_MCHERRY]
bg_v = fp.BACKGROUND[fp.CH_MVENUS]
plane = np.zeros((8, 8, 3), np.uint16)
plane[:, :, fp.CH_MCHERRY] = int(bg_c) + 50
plane[:, :, fp.CH_MVENUS] = max(int(bg_v) - 40, 0)   # below bg -> must clip to 0
plane[:, :, fp.CH_BF] = 500
corr = fp.subtract_background(plane)
check('mCherry background removed', np.allclose(corr[:, :, fp.CH_MCHERRY], 50),
      corr[0, 0, fp.CH_MCHERRY])
check('below-background clips to zero, does not wrap',
      np.allclose(corr[:, :, fp.CH_MVENUS], 0), corr[0, 0, fp.CH_MVENUS])
check('brightfield untouched', np.allclose(corr[:, :, fp.CH_BF], 500))
check('result is float, so clipping is safe', corr.dtype == np.float32, corr.dtype)

print('\n--- per-cell features ---')
H = W = 200
yy, xx = np.mgrid[0:H, 0:W]
mask = np.zeros((H, W), np.int32)
disc = (yy - 60) ** 2 + (xx - 60) ** 2 <= 25 ** 2        # round cell
mask[disc] = 1
mask[130:150, 20:180] = 2                                 # elongated cell
pl = np.zeros((H, W, 3), np.float32)
pl[:, :, fp.CH_MCHERRY][mask == 1] = 400
pl[:, :, fp.CH_MVENUS][mask == 1] = 800
pl[:, :, fp.CH_MCHERRY][mask == 2] = 100
pl[:, :, fp.CH_MVENUS][mask == 2] = 50
feats = fp.measure_frame(pl, mask, pixel_size_um=0.5)
check('both cells measured', set(feats) == {1, 2}, set(feats))
f1, f2 = feats[1], feats[2]
check('round cell has high circularity', f1['circularity'] > 0.9, f1['circularity'])
check('elongated cell has low circularity', f2['circularity'] < 0.5, f2['circularity'])
check('round cell is not eccentric', f1['eccentricity'] < 0.3, f1['eccentricity'])
check('elongated cell is eccentric', f2['eccentricity'] > 0.9, f2['eccentricity'])
check('area matches the disc', abs(f1['area_px'] - np.pi * 25 ** 2) < 60, f1['area_px'])
check('calibrated area applied', np.isclose(f1['area_um2'], f1['area_px'] * 0.25))
check('mCherry p90 correct', np.isclose(f1['mcherry_p90'], 400), f1['mcherry_p90'])
check('mVenus p90 correct', np.isclose(f1['mvenus_p90'], 800), f1['mvenus_p90'])
check('ratio is cherry/venus', np.isclose(f1['cherry_venus_ratio'], 0.5),
      f1['cherry_venus_ratio'])
check('empty mask gives no features', fp.measure_frame(pl, np.zeros((H, W), np.int32)) == {})

print('\n--- division timing ---')
# track 1 founder, divides at t4; tracks 2 and 3 are its daughters
lineage = {
    1: {'parent_track_id': None, 'root_track_id': 1, 'generation': 0,
        'start_tp': 1, 'end_tp': 4, 'end_reason': 'divided'},
    2: {'parent_track_id': 1, 'root_track_id': 1, 'generation': 1,
        'start_tp': 5, 'end_tp': 8, 'end_reason': 'series_end'},
    3: {'parent_track_id': 1, 'root_track_id': 1, 'generation': 1,
        'start_tp': 5, 'end_tp': 6, 'end_reason': 'lost'},
}
timing = fp.division_timing({}, lineage, [1, 2, 3, 4, 5, 6, 7, 8])
check('founder has NaN frames_since_division (age unknown, not zero)',
      np.isnan(timing[(1, 1)][0]), timing[(1, 1)])
check('founder counts down to its division',
      timing[(1, 1)][1] == 3 and timing[(1, 4)][1] == 0,
      (timing[(1, 1)], timing[(1, 4)]))
check('daughter starts at 0 frames since division',
      timing[(2, 5)][0] == 0, timing[(2, 5)])
check('daughter ages one frame at a time',
      timing[(2, 8)][0] == 3, timing[(2, 8)])
check('daughter that never divides has NaN frames_until_division',
      np.isnan(timing[(2, 8)][1]), timing[(2, 8)])
check('no negative ages before a daughter exists',
      np.isnan(timing[(2, 3)][0]), timing[(2, 3)])

print('\n--- phase calls ---')
T = fp.PHASE_THRESHOLDS
def cell(V, C, circ=0.6, area=1000.0):
    return {'mvenus_p90': V, 'mcherry_p90': C, 'circularity': circ, 'area_px': area}

check('G1  = venus high, cherry low',
      fp.classify_phase(cell(600, 50), median_area=1000) == 'G1')
check('earlyS = both dark (the PIP-FUCCI gap)',
      fp.classify_phase(cell(50, 50), median_area=1000) == 'earlyS')
check('S   = venus destroyed, cherry accumulating',
      fp.classify_phase(cell(50, 200), median_area=1000) == 'S')
check('G2  = venus back up, cherry high',
      fp.classify_phase(cell(400, 600), median_area=1000) == 'G2')
check('M   = round and shrunken beats any marker combination',
      fp.classify_phase(cell(600, 50, circ=0.95, area=600), median_area=1000) == 'M')
check('round but not shrunken is not called M',
      fp.classify_phase(cell(600, 50, circ=0.95, area=1000), median_area=1000) == 'G1')
check('shrunken but not round is not called M',
      fp.classify_phase(cell(600, 50, circ=0.5, area=600), median_area=1000) == 'G1')
check('no median area -> morphology gate cannot fire',
      fp.classify_phase(cell(600, 50, circ=0.99, area=10), median_area=None) == 'G1')
check('NaN intensity -> ambiguous',
      fp.classify_phase(cell(np.nan, 50), median_area=1000) == 'ambiguous')
check('mid/mid falls through to ambiguous',
      fp.classify_phase(cell(200, 200), median_area=1000) == 'ambiguous')
check('every call is a declared phase',
      all(fp.classify_phase(cell(v, c), median_area=1000) in fp.PHASES
          for v in (0, 150, 500) for c in (0, 150, 500)))

print('\n--- classify_table ---')
df = pd.DataFrame([
    {'track_id': 1, 'timepoint': 1, 'area_px': 1000, 'circularity': 0.6,
     'mvenus_p90': 600, 'mcherry_p90': 50},
    {'track_id': 1, 'timepoint': 2, 'area_px': 1000, 'circularity': 0.6,
     'mvenus_p90': 50,  'mcherry_p90': 50},
    {'track_id': 1, 'timepoint': 3, 'area_px': 600,  'circularity': 0.95,
     'mvenus_p90': 50,  'mcherry_p90': 500},
])
out = fp.classify_table(df)
check('median area is per track', np.isclose(out['median_area_track'].iloc[0], 1000),
      out['median_area_track'].tolist())
check('area_frac computed', np.isclose(out['area_frac'].iloc[2], 0.6))
check('phases assigned along the track',
      out['phase'].tolist() == ['G1', 'earlyS', 'M'], out['phase'].tolist())
check('thresholds can be overridden',
      fp.classify_table(df, thresholds={'venus_high': 1e9})['phase'].iloc[0]
      != 'G1')

print('\n--- measuring the frames where a cell went dark ---')
import fucci_imaging as fi

# one track seen at t1, t2 and t5: t3 and t4 have no mask at all
tracks_g = {1: {1: 7}, 2: {2: 7}, 3: {}, 4: {}, 5: {3: 7}}
cents_g = {1: {1: (100.0, 100.0)}, 2: {2: (100.0, 110.0)}, 3: {}, 4: {},
           5: {3: (100.0, 140.0)}}
areas_g = {1: {1: 3140}, 2: {2: 3140}, 3: {}, 4: {}, 5: {3: 3140}}
tg = fi.gap_targets(tracks_g, cents_g, areas_g)
check('both unmeasured frames are targeted', sorted(tg) == [3, 4], sorted(tg))
check('one target per frame', all(len(v) == 1 for v in tg.values()), tg)
pos = {tp: tg[tp][0][1] for tp in tg}
check('position is interpolated, not held at the last sighting',
      abs(pos[3][1] - 120.0) < 1e-6 and abs(pos[4][1] - 130.0) < 1e-6, pos)
check('radius comes from the track median area, not one frame',
      abs(tg[3][0][2] - np.sqrt(3140 / np.pi)) < 1e-6, tg[3][0][2])
check('the right track id is carried', tg[3][0][0] == 7, tg[3][0])

# a gap at the start or end of a track is NOT filled: there is nothing to
# interpolate between, so any position would be invented
tracks_e = {1: {}, 2: {1: 9}, 3: {2: 9}, 4: {}}
cents_e = {1: {}, 2: {1: (50.0, 50.0)}, 3: {2: (50.0, 55.0)}, 4: {}}
areas_e = {1: {}, 2: {1: 1000}, 3: {2: 1000}, 4: {}}
check('frames before a track starts or after it ends are not filled',
      fi.gap_targets(tracks_e, cents_e, areas_e) == {},
      fi.gap_targets(tracks_e, cents_e, areas_e))

# a track with no interior gap produces nothing
check('a continuous track needs no filling',
      fi.gap_targets({1: {1: 3}, 2: {2: 3}}, {1: {1: (0.0, 0.0)},
                     2: {2: (0.0, 1.0)}}, {1: {1: 500}, 2: {2: 500}}) == {})

# the blanking list must cover circularity: a disc is perfectly circular, and
# the mitosis rule fires on high circularity, so an unblanked fill would
# relabel every dark frame as M
check('circularity is blanked on interpolated rows',
      'circularity' in fi.IMPUTED_BLANK)
check('so are the other shape columns',
      {'area_px', 'solidity', 'eccentricity', 'perimeter_px',
       'axis_major_px', 'axis_minor_px'} <= set(fi.IMPUTED_BLANK))
check('a NaN circularity cannot be called mitotic',
      fp.classify_phase({'circularity': np.nan, 'area_frac': np.nan,
                         'mvenus_p90': 50.0, 'mcherry_p90': 50.0}) != 'M',
      fp.classify_phase({'circularity': np.nan, 'area_frac': np.nan,
                         'mvenus_p90': 50.0, 'mcherry_p90': 50.0}))

# skimage.draw.disk is what places the synthetic mask
import skimage as ski
_lab = np.zeros((64, 64), np.int32)
_rr, _cc = ski.draw.disk((32, 32), 10, shape=_lab.shape)
_lab[_rr, _cc] = 1
check('skimage.draw.disk is available and fills a plausible area',
      abs(int((_lab == 1).sum()) - int(np.pi * 100)) < 40,
      int((_lab == 1).sum()))

print('\n' + ('ALL TESTS PASSED' if not FAILS else f'{len(FAILS)} FAILURE(S): {FAILS}'))

raise SystemExit(1 if FAILS else 0)
