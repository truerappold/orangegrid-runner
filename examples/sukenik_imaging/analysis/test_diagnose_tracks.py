"""Tests for diagnose_tracks.py -- run with: python test_diagnose_tracks.py"""

import numpy as np
import pandas as pd

import diagnose_tracks as dt

FAILS = []


def check(name, cond, detail=''):
    if cond:
        print(f'  PASS  {name}')
    else:
        print(f'  FAIL  {name}   {detail}')
        FAILS.append(name)


print('\n--- MSD ---')
t = np.arange(1, 11)
rc = np.stack([np.zeros(10), 3.0 * t], 1)          # straight line, 3 px/frame
m = dt.track_msd(t, rc, max_lag=4, min_pairs=1)
check('straight line: MSD = (3 * lag)^2', all(np.isclose(m[k], (3 * k) ** 2)
                                             for k in m), m)
t_gap = np.array([1, 2, 3, 5, 6])                     # frame 4 missing
rc_gap = np.stack([np.zeros(5), 3.0 * t_gap], 1)
m = dt.track_msd(t_gap, rc_gap, max_lag=2, min_pairs=1)
check('a missed frame does not shift the lags', np.isclose(m[1], 9)
      and np.isclose(m[2], 36), m)
check('lags with too few pairs are left out',
      2 not in dt.track_msd(t_gap, rc_gap, max_lag=2, min_pairs=3))

print('\n--- lineage ---')
df = pd.DataFrame({
    'track_id': [1, 1, 2, 2, 3, 3, 9],
    'timepoint': [1, 2, 3, 4, 3, 4, 1],
    'parent_track_id': [np.nan, np.nan, 1, 1, 1, 1, np.nan],
    'root_track_id': [1, 1, 1, 1, 1, 1, 9],
    'track_end_reason': ['divided'] * 2 + ['series_end'] * 5,
    'centroid_row': [0.0] * 7, 'centroid_col': [0.0] * 7})
tr = dt.track_table(df)
check('any member brings the whole lineage', dt.lineage_of(tr, 3) == [1, 2, 3],
      dt.lineage_of(tr, 3))
check('the lineage with divisions is picked by default', dt.pick_lineage(tr) == 1)

print('\n--- position column ---')
d2 = pd.DataFrame({'centroid_row': [1.0, 2.0], 'centroid_col': [1.0, 2.0],
                   'nuc_centroid_row': [5.0, np.nan],
                   'nuc_centroid_col': [6.0, np.nan]})
p = dt.add_position(d2, 'nucleus')
check('nucleus centroid used where present, cell centroid otherwise',
      list(p.pos_row) == [5.0, 2.0] and list(p.pos_col) == [6.0, 2.0])
check('cell mode ignores the nucleus', list(dt.add_position(d2, 'cell').pos_row)
      == [1.0, 2.0])

print('\n--- per-track motion columns ---')
import fucci_pipeline as fp
rows = []
for t in range(1, 11):                       # track 1: 3 px/frame, one 40 px slip at t6
    rows.append({'track_id': 1, 'timepoint': t, 'centroid_row': 0.0,
                 'centroid_col': 3.0 * t + (40.0 if t == 6 else 0.0)})
for t in (1, 2, 5, 6):                        # track 2: short, with a gap
    rows.append({'track_id': 2, 'timepoint': t, 'centroid_row': 0.0,
                 'centroid_col': 10.0 * t})
dm = pd.DataFrame(rows)
mo = fp.track_motion(dm, 'cell')
check('largest jump finds the slip', np.isclose(mo[1]['max_step_px'], 43.0),
      mo[1])
check('squared jump is the square', np.isclose(mo[1]['max_step_sq_px2'], 43.0 ** 2))
check('a jump across a gap is not counted as one step',
      np.isclose(mo[2]['max_step_px'], 10.0) and mo[2]['n_steps'] == 2, mo[2])
check('too short for an MSD: NaN, not a misleading number',
      np.isnan(mo[2]['max_msd_px2']))
straight = pd.DataFrame({'track_id': 3, 'timepoint': range(1, 11),
                         'centroid_row': 0.0,
                         'centroid_col': [3.0 * t for t in range(1, 11)]})
m3 = fp.track_motion(straight, 'cell')[3]
check('max MSD of a straight track is the MSD at its longest lag',
      np.isclose(m3['max_msd_px2'], (3 * 4) ** 2), m3)
col = fp.add_motion_columns(dm, 'cell')
check('columns repeated on every row of the track',
      col.groupby('track_id').max_step_px.nunique().eq(1).all()
      and len(col) == len(dm))

print('\n--- off-path frames ---')
t = np.arange(1, 13)
rc = np.stack([np.zeros(12), 15.0 * t], 1)     # fast but steady: 15 px/frame
rc_slip = rc.copy()
rc_slip[5] += [80.0, 0.0]                       # t6 sits on another object
rc_slip[6] += [70.0, 0.0]                       # ...and t7: a 2-frame excursion
dev_fast = fp.off_path(t, rc)
dev_slip = fp.off_path(t, rc_slip)
check('fast steady motion stays near its own path',
      np.nanmax(dev_fast) < 20, np.nanmax(dev_fast))
check('a 2-frame excursion is caught on both frames',
      dev_slip[5] > 60 and dev_slip[6] > 50, dev_slip[4:8])
check('first and last frame are NaN (no neighbours on one side)',
      np.isnan(dev_fast[0]) and np.isnan(dev_fast[-1]))
dfs = pd.DataFrame({'track_id': 1, 'timepoint': t, 'centroid_row': rc_slip[:, 0],
                    'centroid_col': rc_slip[:, 1]})
cs = fp.add_motion_columns(dfs, 'cell')
check('per-frame and per-track columns are added',
      cs.off_path_px.notna().sum() == 10
      and np.isclose(cs.max_off_path_px.iloc[0], np.nanmax(dev_slip)))

print('\n--- mitosis seen but no division ---')
def tr(tid, end, round_at, n=10):
    return [{'track_id': tid, 'timepoint': t, 'track_end_reason': end,
             'circularity': 0.9 if t == round_at else 0.5,
             'bf_rim': 0.6 if t == round_at else 0.95,
             'bf_interior': 1.1 if t == round_at else 1.0} for t in range(1, n + 1)]
dc = fp.add_division_check(pd.DataFrame(tr(1, 'series_end', 4) + tr(2, 'divided', 9, 9)
                                        + tr(3, 'series_end', 10) + tr(4, 'series_end', None)))
per = dc.groupby('track_id').mitosis_no_division.first()
check('round + halo, never divided: flagged', per[1] and
      dc[dc.track_id == 1].mitosis_seen_tp.iloc[0] == 4)
check('round then divided: not flagged', not per[2])
check('round in the last frame (still in mitosis at the end): not flagged', not per[3])
check('never round: not flagged', not per[4])
check('no halo columns: no-op', not fp.add_division_check(
    pd.DataFrame({'track_id': [1], 'timepoint': [1], 'circularity': [0.9],
                  'track_end_reason': ['lost']})).mitosis_no_division.any())

print('\n' + ('ALL TESTS PASSED' if not FAILS else
              f'{len(FAILS)} FAILURE(S): {FAILS}'))
raise SystemExit(1 if FAILS else 0)
