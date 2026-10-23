"""Unit tests for timelapse_tracking.py — run with: python test_timelapse_tracking.py

Covers the parts that do not need a GPU or a JVM: filename parsing, series
discovery, frame-to-frame matching, track propagation, and wide-table assembly.
"""

import numpy as np
import pandas as pd

import timelapse_tracking as tt

FAILS = []


def check(name, cond, detail=''):
    if cond:
        print(f'  PASS  {name}')
    else:
        print(f'  FAIL  {name}   {detail}')
        FAILS.append(name)


# --------------------------------------------------------------- parsing ----
print('\n--- filename parsing ---')
check('unprefixed name is block 0',
      tt.parse_vsi_name('A1-7_DAPI.vsi') ==
      {'well': 'A1', 'tile': 7, 'block': 0, 'channel': 'DAPI', 'prefix': ''})
check('prefix 1 -> block 1',
      tt.parse_vsi_name('1A1-7_DAPI.vsi')['block'] == 1)
check('prefix 2 -> block 2',
      tt.parse_vsi_name('2A1-7_DAPI.vsi')['block'] == 2)
check('prefix 19 -> block 19',
      tt.parse_vsi_name('19H12-9_mNG.vsi')['block'] == 19)
check('well with two-digit column parses',
      tt.parse_vsi_name('3H12-9_mNG.vsi')['well'] == 'H12')
check('channel name containing a comma survives',
      tt.parse_vsi_name('5B2-3_FRET, FRET.vsi')['channel'] == 'FRET, FRET')
check('prefix 1 + well A1 is not confused with well 1A',
      tt.parse_vsi_name('1A1-1_DAPI.vsi')['well'] == 'A1')
check('non-matching name returns None',
      tt.parse_vsi_name('notanimage.txt') is None)
check('round trip through build_filename',
      tt.build_filename('/p', 'A1', 7, 2, 'DAPI').endswith('2A1-7_DAPI.vsi'))
check('block_prefix(0) is empty', tt.block_prefix(0) == '')

# ------------------------------------------------------------- discovery ----
print('\n--- block discovery ---')
files = []
for b in (0, 1, 2):
    for ch in tt.CHANNEL_FILES:
        files.append(f'/p/Pics/{tt.block_prefix(b)}A1-1_{ch}.vsi')
series, problems = tt.discover_blocks(files)
check('complete 3-block series found', series == {('A1', 1): [0, 1, 2]}, series)
check('no problems reported for a clean series', problems == [], problems)

files_missing = [f for f in files if not f.endswith('2A1-1_mNG.vsi')]
series, problems = tt.discover_blocks(files_missing)
check('block missing a channel is excluded',
      series == {('A1', 1): [0, 1]}, series)
check('missing channel is reported', any('missing channel' in p for p in problems))

files_gap = []
for b in (0, 1, 3):
    for ch in tt.CHANNEL_FILES:
        files_gap.append(f'/p/Pics/{tt.block_prefix(b)}B3-2_{ch}.vsi')
series, problems = tt.discover_blocks(files_gap)
check('gap truncates rather than bridges', series == {('B3', 2): [0, 1]}, series)
check('gap is reported', any('gap at block' in p for p in problems))

files_no_b0 = []
for b in (1, 2):
    for ch in tt.CHANNEL_FILES:
        files_no_b0.append(f'/p/Pics/{tt.block_prefix(b)}C4-1_{ch}.vsi')
series, problems = tt.discover_blocks(files_no_b0)
check('tile without an unprefixed block is skipped', ('C4', 1) not in series, series)

# ------------------------------------------------------------- centroids ----
print('\n--- centroids ---')
m = np.zeros((10, 10), int)
m[1:3, 1:3] = 1          # centroid (1.5, 1.5)
m[6:9, 6:9] = 4          # centroid (7.0, 7.0)
cen = tt.centroids_by_label(m)
check('centroid of label 1', np.allclose(cen[1], (1.5, 1.5)), cen)
check('centroid of label 4', np.allclose(cen[4], (7.0, 7.0)), cen)
check('background excluded', set(cen) == {1, 4}, set(cen))
check('empty mask gives empty dict', tt.centroids_by_label(np.zeros((5, 5), int)) == {})

# -------------------------------------------------------------- matching ----
print('\n--- matching ---')
prev = {1: (10.0, 10.0), 2: (50.0, 50.0), 3: (90.0, 90.0)}

curr = {7: (11.0, 10.0), 8: (52.0, 50.0), 9: (91.0, 91.0)}
asg, dist, drop, _div = tt.match_timepoint(prev, curr, max_dist=10)
check('clean one-to-one match', asg == {7: 1, 8: 2, 9: 3}, asg)
check('no drops when all are close', drop == {}, drop)
check('distance recorded', np.isclose(dist[7], 1.0), dist)

curr = {7: (11.0, 10.0), 8: (80.0, 80.0)}
asg, dist, drop, _div = tt.match_timepoint(prev, curr, max_dist=5)
check('cell beyond cutoff is dropped', drop.get(8) == 'distance', drop)
check('distance still recorded for the dropped cell', np.isfinite(dist[8]))
check('cell inside cutoff survives', asg == {7: 1}, asg)

curr = {7: (10.0, 11.0), 8: (10.0, 9.0)}
asg, dist, drop, _div = tt.match_timepoint(prev, curr, max_dist=10)
check('both claimants of one previous cell are dropped',
      drop == {7: 'collision', 8: 'collision'}, drop)
check('nothing assigned on collision', asg == {}, asg)

asg, dist, drop, _div = tt.match_timepoint({}, {7: (1.0, 1.0)}, max_dist=10)
check('no previous cells -> flagged', drop == {7: 'no_previous_cells'}, drop)
asg, dist, drop, _div = tt.match_timepoint(prev, {}, max_dist=10)
check('no current cells -> empty result', (asg, drop) == ({}, {}))

# hungarian recovers a pair that 'nearest' throws away
# both current cells are nearest to prev 1, so 'nearest' collides and drops
# both; hungarian pairs 10->1 (d=1) and 11->2 (d=7) for a lower total cost.
prev2 = {1: (0.0, 0.0), 2: (0.0, 10.0)}
curr2 = {10: (0.0, 1.0), 11: (0.0, 3.0)}
asg_n, _, drop_n, _ = tt.match_timepoint(prev2, curr2, max_dist=10, method='nearest')
asg_h, _, drop_h, _ = tt.match_timepoint(prev2, curr2, max_dist=10, method='hungarian')
check('nearest drops the colliding pair', asg_n == {} and len(drop_n) == 2, (asg_n, drop_n))
check('hungarian keeps both', asg_h == {10: 1, 11: 2}, asg_h)

# hungarian must still honour the cutoff
asg_h2, _, drop_h2, _ = tt.match_timepoint({1: (0.0, 0.0)}, {10: (0.0, 99.0)},
                                        max_dist=10, method='hungarian')
check('hungarian respects the distance cutoff',
      asg_h2 == {} and drop_h2 == {10: 'distance'}, (asg_h2, drop_h2))

# ---------------------------------------------------------------- tracks ----
print('\n--- track propagation ---')
cen_tp = {
    1: {1: (10.0, 10.0), 2: (50.0, 50.0), 3: (90.0, 90.0)},
    2: {5: (11.0, 10.0), 6: (51.0, 50.0), 7: (91.0, 90.0)},
    3: {8: (12.0, 10.0), 9: (52.0, 50.0)},                    # cell 3 disappears
}
tracks, records, lin = tt.track_series(cen_tp, max_dist=5)
check('timepoint 1 seeds the track ids', tracks[1] == {1: 1, 2: 2, 3: 3}, tracks[1])
check('ids propagate through timepoint 2', tracks[2] == {5: 1, 6: 2, 7: 3}, tracks[2])
check('ids propagate through timepoint 3', tracks[3] == {8: 1, 9: 2}, tracks[3])
check('one record per cell per timepoint', len(records) == 3 + 3 + 2, len(records))

# a cell dropped at t2 must not be a matching target at t3
cen_break = {
    1: {1: (10.0, 10.0)},
    2: {5: (200.0, 200.0)},          # too far -> dropped
    3: {8: (201.0, 200.0)},          # near the dropped cell, but it has no track
}
tracks, records, lin = tt.track_series(cen_break, max_dist=5)
check('cell dropped at t2 ends the track', tracks[2] == {}, tracks[2])
check('t3 cannot inherit from a dropped cell', tracks[3] == {}, tracks[3])
rec3 = [r for r in records if r['timepoint'] == 3][0]
check('t3 drop reason is no_previous_cells',
      rec3['drop_reason'] == 'no_previous_cells', rec3)

# a track that survives every timepoint stays complete
cen_ok = {t: {t * 10: (10.0 + t, 10.0)} for t in (1, 2, 3, 4)}
tracks, records, lin = tt.track_series(cen_ok, max_dist=5)
check('single cell tracked through 4 timepoints',
      [tracks[t] for t in (1, 2, 3, 4)] == [{10: 10}, {20: 10}, {30: 10}, {40: 10}],
      tracks)

# the first step can get its own, looser cutoff
cen_join = {
    1: {1: (10.0, 10.0)},
    2: {5: (18.0, 10.0)},      # 8 px: too far for y=5, fine for a looser first step
    3: {8: (20.0, 10.0)},      # 2 px: fine for y=5
}
tracks, _, _ = tt.track_series(cen_join, max_dist=5)
check('without an override the long first step breaks the track',
      tracks[2] == {} and tracks[3] == {}, (tracks[2], tracks[3]))
tracks, _, _ = tt.track_series(cen_join, max_dist=5, first_step_max_dist=12)
check('first_step_max_dist rescues the t1->t2 join',
      tracks[2] == {5: 1} and tracks[3] == {8: 1}, (tracks[2], tracks[3]))

# -------------------------------------------------------------- division ----
print('\n--- division ---')
# one parent at (100,100) area 400; two daughters either side, area ~200 each
par_c = {1: (100.0, 100.0)}
par_a = {1: 400}
dau_c = {7: (100.0, 92.0), 8: (100.0, 108.0)}
dau_a = {7: 200, 8: 200}

asg, dist, drop, div = tt.match_timepoint(par_c, dau_c, 25, prev_areas=par_a,
                                          curr_areas=dau_a, allow_division=False)
check('without allow_division a division is still dropped as a collision',
      div == {} and drop == {7: 'collision', 8: 'collision'}, (div, drop))

asg, dist, drop, div = tt.match_timepoint(par_c, dau_c, 25, prev_areas=par_a,
                                          curr_areas=dau_a, allow_division=True)
check('symmetric halves are accepted as a division', div == {1: [7, 8]}, div)
check('accepted division produces no drops', drop == {}, drop)
check('accepted division is not in the plain assignment', asg == {}, asg)

# a wildly asymmetric split is over-segmentation, not mitosis
asg, dist, drop, div = tt.match_timepoint(
    par_c, dau_c, 25, prev_areas=par_a, curr_areas={7: 380, 8: 20},
    allow_division=True)
check('asymmetric split rejected (a fragment, not a daughter)',
      div == {} and drop == {7: 'collision', 8: 'collision'}, (div, drop))

# two daughters that together far exceed the parent are not one cell dividing
asg, dist, drop, div = tt.match_timepoint(
    par_c, dau_c, 25, prev_areas=par_a, curr_areas={7: 400, 8: 400},
    allow_division=True)
check('daughters summing to 2x the parent rejected', div == {}, div)

# a three-way collision is noise whatever the areas
asg, dist, drop, div = tt.match_timepoint(
    par_c, {7: (100.0, 95.0), 8: (100.0, 105.0), 9: (95.0, 100.0)}, 25,
    prev_areas=par_a, curr_areas={7: 130, 8: 130, 9: 130}, allow_division=True)
check('three-way collision still dropped', div == {} and len(drop) == 3, (div, drop))

# a daughter beyond division_max_dist is rejected
asg, dist, drop, div = tt.match_timepoint(
    par_c, {7: (100.0, 98.0), 8: (100.0, 130.0)}, 40, prev_areas=par_a,
    curr_areas=dau_a, allow_division=True, division_max_dist=10)
check('daughter outside division_max_dist rejected', div == {}, div)

print('\n--- lineage through a division ---')
cen = {
    1: {1: (100.0, 100.0), 2: (300.0, 300.0)},
    2: {1: (100.0, 101.0), 2: (300.0, 301.0)},
    3: {5: (100.0, 93.0), 6: (100.0, 109.0), 7: (300.0, 302.0)},   # cell 1 divides
    4: {5: (100.0, 92.0), 6: (100.0, 110.0), 7: (300.0, 303.0)},
}
ar = {1: {1: 400, 2: 400}, 2: {1: 400, 2: 400},
      3: {5: 200, 6: 200, 7: 400}, 4: {5: 200, 6: 200, 7: 400}}
tracks, records, lin = tt.track_series(cen, max_dist=25, areas_per_tp=ar,
                                       allow_division=True)
check('parent tracked up to the division', tracks[2][1] == 1, tracks[2])
check('two new daughter ids at the division', len(tracks[3]) == 3, tracks[3])
dau = sorted(tid for lab, tid in tracks[3].items() if lab in (5, 6))
check('daughters get fresh ids, not the parent id', all(t > 2 for t in dau), dau)
check('daughters both point at the parent',
      all(lin[t]['parent_track_id'] == 1 for t in dau),
      {t: lin[t]['parent_track_id'] for t in dau})
check('daughters share the parent root',
      all(lin[t]['root_track_id'] == 1 for t in dau), dau)
check('daughters are generation 1',
      all(lin[t]['generation'] == 1 for t in dau), dau)
check("parent end_reason is 'divided'", lin[1]['end_reason'] == 'divided',
      lin[1])
check('parent end_tp is the frame before the division', lin[1]['end_tp'] == 2,
      lin[1])
check('daughters persist to the last timepoint',
      sorted(t for lab, t in tracks[4].items() if lab in (5, 6)) == dau, tracks[4])
check('undivided neighbour keeps its original id', tracks[4][7] == 2, tracks[4])
check("undivided neighbour end_reason is 'series_end'",
      lin[2]['end_reason'] == 'series_end', lin[2])
div_recs = [r for r in records if r['event'] == 'division']
check('division events recorded', len(div_recs) == 2, len(div_recs))

# a lost track must be labelled 'lost', not 'series_end'
cen_lost = {1: {1: (10.0, 10.0)}, 2: {}, 3: {}}
tracks, _, lin_lost = tt.track_series(cen_lost, max_dist=5)
check("a track that vanishes is end_reason 'lost'",
      lin_lost[1]['end_reason'] == 'lost', lin_lost[1])

# ------------------------------- division radius independent of max_dist --
print('\n--- a division needs a wider radius than a match ---')
# Nuclei ~62 px across: at mitosis each daughter centroid lands ~40 px from
# where the parent's was, while interphase cells move <15 px.  The division
# radius therefore has to be settable ABOVE max_dist, or no candidate forms.
cen_d = {1: {1: (100.0, 100.0)}, 2: {5: (100.0, 60.0), 6: (100.0, 140.0)}}
ar_d = {1: {1: 3000}, 2: {5: 1600, 6: 1600}}
t_narrow, _r, _l = tt.track_series(cen_d, max_dist=30, areas_per_tp=ar_d,
                                   allow_division=True, division_max_dist=30)
check('daughters outside the division radius are not a division',
      t_narrow[2] == {}, t_narrow[2])
t_wide, _r, lin_w = tt.track_series(cen_d, max_dist=30, areas_per_tp=ar_d,
                                    allow_division=True, division_max_dist=60)
check('a division radius wider than max_dist finds them',
      len(t_wide[2]) == 2, t_wide[2])
check('and the parent is marked divided',
      lin_w[1]['end_reason'] == 'divided', lin_w[1])
# the wide radius must NOT quietly loosen ordinary tracking
cen_one = {1: {1: (100.0, 100.0)}, 2: {5: (100.0, 145.0)}}
t_one, _r, _l = tt.track_series(cen_one, max_dist=30, allow_division=True,
                                division_max_dist=60)
check('a lone cell 45 px away is still dropped, not matched',
      t_one[2] == {}, t_one[2])

print('\n--- dropped rows stay analysable ---')
cen_f = {1: {1: (0.0, 0.0), 2: (500.0, 500.0)}, 2: {7: (200.0, 0.0)}}
_t, recs_f, _l = tt.track_series(cen_f, max_dist=30)
r = [x for x in recs_f if x['timepoint'] == 2][0]
check('a dropped cell records the parent it claimed',
      r['matched_prev_label'] == 1, r)
check('and its own centroid', (r['centroid_row'], r['centroid_col']) == (200.0, 0.0), r)
check('while still being marked dropped on distance',
      r['dropped'] and r['drop_reason'] == 'distance', r)

# ------------------------------------------------- division rejection log --
print('\n--- why a division was rejected ---')
# two daughters that each pass the per-daughter test (0.84 of the parent) but
# together come to 1.68x it.  Real mitosis does not grow the mask that much, so
# the default rejects this -- the point is that the rejection is now visible.
cen = {1: {1: (100.0, 100.0)}, 2: {5: (100.0, 92.0), 6: (100.0, 108.0)}}
ar_big = {1: {1: 400}, 2: {5: 336, 6: 336}}
rej = []
tracks, _r, _l = tt.track_series(cen, max_dist=25, areas_per_tp=ar_big,
                                 allow_division=True, division_rejects=rej)
check('the rejected division is reported', len(rej) == 1, rej)
check('with the test that rejected it named',
      rej and rej[0]['reason'] == 'sum_frac', rej)
check('and the number that failed it',
      rej and rej[0]['sum_frac'] == 1.68, rej)
check('and the timepoint it happened at',
      rej and rej[0]['timepoint'] == 2, rej)
check('raising max_sum_frac then accepts it',
      len(tt.track_series(cen, max_dist=25, areas_per_tp=ar_big,
                          allow_division=True,
                          division_thresholds={'max_sum_frac': 1.8})[0][2]) == 2)

rej = []
tt.track_series({1: {1: (100.0, 100.0)},
                 2: {5: (100.0, 92.0), 6: (100.0, 108.0)}},
                max_dist=25, areas_per_tp={1: {1: 400}, 2: {5: 400, 6: 400}},
                allow_division=True, division_rejects=rej)
check('two full-size daughters fail on daughter_frac, not sum_frac',
      len(rej) == 1 and rej[0]['reason'] == 'daughter_frac', rej)
check('and report both fractions so you can see they are whole cells',
      rej and rej[0]['daughter_fracs'] == [1.0, 1.0], rej)

# daughters too far from the parent
cen_far = {1: {1: (100.0, 100.0)}, 2: {5: (100.0, 80.0), 6: (100.0, 120.0)}}
ar_far = {1: {1: 400}, 2: {5: 180, 6: 180}}
rej = []
tt.track_series(cen_far, max_dist=25, areas_per_tp=ar_far,
                allow_division=True, division_max_dist=5,
                division_rejects=rej)
check('a too-far division is reported as a distance failure',
      len(rej) == 1 and rej[0]['reason'] == 'distance', rej)
check('with both daughter distances shown',
      rej and len(rej[0]['daughter_dists']) == 2, rej)

# an asymmetric split
cen_as = {1: {1: (100.0, 100.0)}, 2: {5: (100.0, 96.0), 6: (100.0, 104.0)}}
ar_as = {1: {1: 400}, 2: {5: 150, 6: 130}}
rej = []
tt.track_series(cen_as, max_dist=25, areas_per_tp=ar_as, allow_division=True,
                division_rejects=rej)
check('a symmetric-enough split is not rejected', rej == [], rej)

# an accepted division logs nothing
cen_ok = {1: {1: (100.0, 100.0)}, 2: {5: (100.0, 96.0), 6: (100.0, 104.0)}}
ar_ok = {1: {1: 400}, 2: {5: 180, 6: 180}}
rej = []
tt.track_series(cen_ok, max_dist=25, areas_per_tp=ar_ok, allow_division=True,
                division_rejects=rej)
check('an accepted division logs no rejection', rej == [], rej)

# ------------------------- the parent reference for the division area test --
print('\n--- a mitotic parent is the smallest the cell ever is ---')
# Real numbers from track 12 of A9-2: a nucleus sitting at ~2880 px condenses
# to 1719 px as it divides, and the two daughters come back at ~3000 px each.
# Measured against that one condensed frame they look 1.7x "the parent" and a
# genuine division is thrown out; measured against the parent's recent median
# they are 1.0x and it is accepted, with no threshold changed.
cen_mit = {t: {1: (980.0 + t, 1710.0)} for t in range(20, 25)}
for _t, _labs in ((25, (14, 15)), (26, (14, 15)), (27, (14, 15))):
    cen_mit[_t] = {_labs[0]: (960.0 - (_t - 25) * 2, 1700.0),
                   _labs[1]: (1000.0 + (_t - 25) * 2, 1700.0)}
ar_mit = {20: {1: 2879}, 21: {1: 2828}, 22: {1: 10550}, 23: {1: 3503},
          24: {1: 1719}, 25: {14: 3025, 15: 2905},
          26: {14: 3000, 15: 2900}, 27: {14: 3000, 15: 2900}}
thr_mit = dict(min_daughter_frac=0.30, max_daughter_frac=1.10,
               min_sum_frac=0.80, max_sum_frac=2.20, min_symmetry=0.35)
rej_m = []
tr_old, _r, _l = tt.track_series(
    cen_mit, max_dist=60, areas_per_tp=ar_mit, allow_division=True,
    division_max_dist=60, division_thresholds=thr_mit, area_history=1,
    division_rejects=rej_m)
check('against the previous frame alone the division is rejected',
      tr_old[25] == {} and rej_m and rej_m[0]['reason'] == 'daughter_frac',
      rej_m)
tr_new, _r, lin_m = tt.track_series(
    cen_mit, max_dist=60, areas_per_tp=ar_mit, allow_division=True,
    division_max_dist=60, division_thresholds=thr_mit)
check('against the parent median it is accepted', len(tr_new[25]) == 2,
      tr_new[25])
check('the parent is marked divided at the right frame',
      lin_m[1]['end_reason'] == 'divided' and lin_m[1]['end_tp'] == 24, lin_m[1])
check('both daughters continue', len(tr_new[27]) == 2, tr_new[27])
# the median must also ignore a frame where Cellpose merged a neighbour in
check('a single blown-up frame does not drag the reference up',
      abs(float(np.median([2879, 2828, 10550, 3503, 1719])) - 2879) < 1)

# ------------------------------------ one nucleus split in two, not mitosis --
print('\n--- a fragmented nucleus is not a division ---')
# Cellpose splits one nucleus at t3; the halves pass every area test, then the
# split heals and from t4 on there is only ever one object.
cen_frag = {1: {1: (100.0, 100.0)}, 2: {1: (100.0, 102.0)},
            3: {5: (100.0, 92.0), 6: (100.0, 112.0)},
            4: {7: (100.0, 104.0)}, 5: {8: (100.0, 106.0)},
            6: {9: (100.0, 108.0)}}
ar_frag = {1: {1: 3000}, 2: {1: 3000}, 3: {5: 1400, 6: 1600},
           4: {7: 3000}, 5: {8: 3000}, 6: {9: 3000}}
tr, _r, lin_f = tt.track_series(cen_frag, max_dist=30, areas_per_tp=ar_frag,
                                allow_division=True, division_max_dist=60)
check('the split is first recorded as a division (every area test passes)',
      len(tr[3]) == 2 and lin_f[1]['end_reason'] == 'divided', tr[3])
tr, lin_f, rev = tt.prune_spurious_divisions(tr, lin_f, areas_per_tp=ar_frag,
                                             min_overlap=3)
check('persistence reverts it', len(rev) == 1, rev)
check('and the parent becomes one continuous track',
      [tr[t] and list(tr[t].values())[0] for t in sorted(tr)] == [1] * 6,
      {t: tr[t] for t in sorted(tr)})
check('the daughter ids are gone from the lineage',
      set(lin_f) == {1}, set(lin_f))
check('the parent is no longer marked divided',
      lin_f[1]['end_reason'] != 'divided', lin_f[1])
check('the fragment frame keeps the larger half, not both',
      len(tr[3]) == 1 and list(tr[3]) == [6], tr[3])

print('\n--- a real division survives the same check ---')
cen_real = {1: {1: (100.0, 100.0)}, 2: {1: (100.0, 102.0)},
            3: {5: (100.0, 88.0), 6: (100.0, 118.0)},
            4: {7: (100.0, 86.0), 8: (100.0, 120.0)},
            5: {9: (100.0, 84.0), 10: (100.0, 122.0)},
            6: {11: (100.0, 82.0), 12: (100.0, 124.0)}}
ar_real = {t: {lab: 1600 for lab in cen_real[t]} for t in cen_real}
ar_real[1] = {1: 3000}
ar_real[2] = {1: 3000}
tr2, _r, lin_r = tt.track_series(cen_real, max_dist=30, areas_per_tp=ar_real,
                                 allow_division=True, division_max_dist=60)
tr2, lin_r, rev2 = tt.prune_spurious_divisions(tr2, lin_r,
                                               areas_per_tp=ar_real,
                                               min_overlap=3)
check('daughters that coexist are kept', rev2 == [], rev2)
check('the parent stays divided', lin_r[1]['end_reason'] == 'divided', lin_r[1])
check('two tracks run to the end', len(tr2[6]) == 2, tr2[6])

print('\n--- a division near the end of the series is not punished ---')
cen_late = {1: {1: (100.0, 100.0)}, 2: {1: (100.0, 102.0)},
            3: {5: (100.0, 90.0), 6: (100.0, 114.0)}}
ar_late = {1: {1: 3000}, 2: {1: 3000}, 3: {5: 1500, 6: 1500}}
tr3, _r, lin_l = tt.track_series(cen_late, max_dist=30, areas_per_tp=ar_late,
                                 allow_division=True, division_max_dist=60)
tr3, lin_l, rev3 = tt.prune_spurious_divisions(tr3, lin_l,
                                               areas_per_tp=ar_late,
                                               min_overlap=3)
check('only one frame remains, so one frame of overlap is enough',
      rev3 == [], rev3)

print('\n--- nested splits resolve from the inside out ---')
# parent divides; one daughter is itself a fragment that heals.  Undoing the
# inner split lengthens that daughter, which can make the OUTER division
# genuine -- so the order of resolution matters.
cen_n = {1: {1: (0.0, 0.0)}, 2: {1: (0.0, 2.0)},
         3: {5: (0.0, -20.0), 6: (0.0, 24.0)},
         4: {7: (0.0, -22.0), 8: (0.0, 20.0), 9: (0.0, 30.0)},
         5: {10: (0.0, -24.0), 11: (0.0, 26.0)},
         6: {12: (0.0, -26.0), 13: (0.0, 27.0)},
         7: {14: (0.0, -28.0), 15: (0.0, 28.0)}}
ar_n = {1: {1: 3000}, 2: {1: 3000}, 3: {5: 1500, 6: 1500},
        4: {7: 1500, 8: 700, 9: 800}, 5: {10: 1500, 11: 1500},
        6: {12: 1500, 13: 1500}, 7: {14: 1500, 15: 1500}}
tr4, _r, lin_n = tt.track_series(cen_n, max_dist=30, areas_per_tp=ar_n,
                                 allow_division=True, division_max_dist=60)
tr4, lin_n, rev4 = tt.prune_spurious_divisions(tr4, lin_n, areas_per_tp=ar_n,
                                               min_overlap=3)
check('the outer, genuine division is kept',
      any(i['end_reason'] == 'divided' for i in lin_n.values()), lin_n)
check('two tracks survive to the last frame', len(tr4[7]) == 2, tr4[7])
check('generation is recomputed after merging',
      max(i['generation'] for i in lin_n.values()) == 1,
      {k: v['generation'] for k, v in lin_n.items()})

print('\n--- min_overlap=0 disables the check ---')
tr5, _r, lin5 = tt.track_series(cen_frag, max_dist=30, areas_per_tp=ar_frag,
                                allow_division=True, division_max_dist=60)
_t, _l, rev5 = tt.prune_spurious_divisions(tr5, lin5, areas_per_tp=ar_frag,
                                           min_overlap=0)
check('nothing is reverted when the check is off', rev5 == [], rev5)

# -------------------------------------------- crossing a gap while moving --
print('\n--- a moving cell that goes dark for several frames ---')
# 12 px/frame, invisible at t4-t6, back at t7 sixty px from where it was last
# seen.  This is the post-division dark window: both markers are gone, so a
# cell segmented on fluorescence vanishes for a few frames and then has to be
# found again somewhere else entirely.
cen_mv = {1: {1: (100.0, 100.0)}, 2: {1: (100.0, 112.0)}, 3: {1: (100.0, 124.0)},
          4: {}, 5: {}, 6: {}, 7: {9: (100.0, 184.0)}}
t_old, _r, _l = tt.track_series(cen_mv, max_dist=30, max_gap=3,
                                gap_dist_scale=0.0, predict_motion=False)
check('stale position + flat cutoff loses it', t_old[7] == {}, t_old[7])
t_sc, _r, _l = tt.track_series(cen_mv, max_dist=30, max_gap=3,
                               gap_dist_scale=1.0, predict_motion=False)
check('a cutoff that grows with the gap recovers it', t_sc[7] == {9: 1}, t_sc[7])
t_pr, _r, _l = tt.track_series(cen_mv, max_dist=30, max_gap=3,
                               gap_dist_scale=0.0, predict_motion=True)
check('motion prediction alone also recovers it', t_pr[7] == {9: 1}, t_pr[7])
t_new, _r, lin_mv = tt.track_series(cen_mv, max_dist=30, max_gap=3)
check('both together (the defaults) recover it', t_new[7] == {9: 1}, t_new[7])
check('and the track is not marked lost',
      lin_mv[1]['end_reason'] == 'series_end', lin_mv[1])
rec7 = [r for r in _r if r['timepoint'] == 7][0]
check('the record shows a 3-frame gap was bridged', rec7['gap'] == 3, rec7)

# prediction must not manufacture links
cen_far = {1: {1: (100.0, 100.0)}, 2: {1: (100.0, 112.0)}, 3: {},
           4: {9: (400.0, 400.0)}}
t_far, _r, _l = tt.track_series(cen_far, max_dist=30, max_gap=3)
check('an unrelated cell far off the predicted path is not linked',
      t_far[4] == {}, t_far[4])

# a stationary cell must be unaffected by prediction
cen_st = {1: {1: (100.0, 100.0)}, 2: {1: (100.0, 101.0)}, 3: {},
          4: {9: (100.0, 102.0)}}
check('a stationary cell still reconnects',
      tt.track_series(cen_st, max_dist=30, max_gap=2)[0][4] == {9: 1})

# the scale is a dial, not a hardcoding
cen_j = {1: {1: (100.0, 100.0)}, 2: {}, 3: {9: (100.0, 145.0)}}
check('gap_dist_scale=0 keeps the strict single-frame cutoff',
      tt.track_series(cen_j, max_dist=30, max_gap=2, gap_dist_scale=0.0,
                      predict_motion=False)[0][3] == {})
check('gap_dist_scale=1 admits the same jump after one missed frame',
      tt.track_series(cen_j, max_dist=30, max_gap=2, gap_dist_scale=1.0,
                      predict_motion=False)[0][3] == {9: 1})

# a consecutive-frame match must keep the ordinary cutoff
cen_adj = {1: {1: (100.0, 100.0)}, 2: {9: (100.0, 145.0)}}
check('no gap means no extra distance allowance',
      tt.track_series(cen_adj, max_dist=30, max_gap=3)[0][2] == {})

# ------------------------------------------------------------- gap closing --
print('\n--- gap closing ---')
# a cell that vanishes for one frame and comes back in the same place
cen_gap = {
    1: {1: (10.0, 10.0)},
    2: {},                       # not segmented this frame
    3: {5: (11.0, 10.0)},        # back, 1 px from where it was
}
tracks, _r, lin = tt.track_series(cen_gap, max_dist=5, max_gap=0)
check('without gap closing a one-frame dropout ends the track',
      tracks[3] == {}, tracks[3])
check('and the track is marked lost', lin[1]['end_reason'] == 'lost', lin[1])

tracks, recs, lin = tt.track_series(cen_gap, max_dist=5, max_gap=1)
check('with max_gap=1 the track is reconnected', tracks[3] == {5: 1}, tracks[3])
check('reconnected track runs to the end',
      lin[1]['end_reason'] == 'series_end' and lin[1]['end_tp'] == 3, lin[1])
rec3 = [r for r in recs if r['timepoint'] == 3][0]
check('the record notes the gap it bridged', rec3['gap'] == 1, rec3)

# a gap longer than max_gap must NOT be bridged
cen_gap2 = {1: {1: (10.0, 10.0)}, 2: {}, 3: {}, 4: {7: (11.0, 10.0)}}
tracks, _r, _l = tt.track_series(cen_gap2, max_dist=5, max_gap=1)
check('a 2-frame gap is not bridged when max_gap=1', tracks[4] == {}, tracks[4])
tracks, _r, _l = tt.track_series(cen_gap2, max_dist=5, max_gap=2)
check('the same gap is bridged when max_gap=2', tracks[4] == {7: 1}, tracks[4])

# gap closing must not let a cell jump further than max_dist
cen_gap3 = {1: {1: (10.0, 10.0)}, 2: {}, 3: {5: (200.0, 200.0)}}
tracks, _r, _l = tt.track_series(cen_gap3, max_dist=5, max_gap=2)
check('distance is still enforced across a gap', tracks[3] == {}, tracks[3])

# division still works with gap closing enabled
cen_div = {1: {1: (100.0, 100.0)}, 2: {1: (100.0, 101.0)},
           3: {5: (100.0, 93.0), 6: (100.0, 109.0)}}
ar_div = {1: {1: 400}, 2: {1: 400}, 3: {5: 200, 6: 200}}
tracks, _r, lin = tt.track_series(cen_div, max_dist=25, areas_per_tp=ar_div,
                                  allow_division=True, max_gap=2)
dau = sorted(tracks[3].values())
check('division still detected with gap closing on', len(dau) == 2, tracks[3])
check('parent still marked divided, not lost',
      lin[1]['end_reason'] == 'divided', lin[1])

# ------------------------------------------------------------ wide table ----
print('\n--- wide table ---')
measurements = [
    {'track_id': 1, 'timepoint': 1, 'Ef_cell': 0.30, 'area_cell': 100},
    {'track_id': 2, 'timepoint': 1, 'Ef_cell': 0.32, 'area_cell': 110},
    {'track_id': 3, 'timepoint': 1, 'Ef_cell': 0.34, 'area_cell': 120},
    {'track_id': 1, 'timepoint': 2, 'Ef_cell': 0.35, 'area_cell': 105},
    {'track_id': 2, 'timepoint': 2, 'Ef_cell': 0.36, 'area_cell': 115},
]
meta = {'construct': 'GOOSE1', 'well': 'A1', 'tile': 1}
wide = tt.assemble_wide(measurements, meta, timepoints=[1, 2])
wide = tt.order_columns(wide)
check('one row per track', len(wide) == 3, len(wide))
check('complete track flagged True',
      bool(wide.loc[wide.track_id == 1, 'complete_track'].iloc[0]))
check('incomplete track flagged False',
      not bool(wide.loc[wide.track_id == 3, 'complete_track'].iloc[0]))
check('n_timepoints correct for the short track',
      int(wide.loc[wide.track_id == 3, 'n_timepoints'].iloc[0]) == 1)
check('per-timepoint columns present',
      {'Ef_cell_t1', 'Ef_cell_t2', 'area_cell_t1'}.issubset(wide.columns))
check('delta computed against timepoint 1',
      np.isclose(wide.loc[wide.track_id == 1, 'Ef_cell_delta_t2'].iloc[0], 0.05))
check('no delta column for a track missing t2',
      pd.isna(wide.loc[wide.track_id == 3, 'Ef_cell_delta_t2'].iloc[0]))
check('missing timepoint leaves NaN not a crash',
      pd.isna(wide.loc[wide.track_id == 3, 'Ef_cell_t2'].iloc[0]))
check('metadata columns lead',
      list(wide.columns)[:4] == ['construct', 'well', 'tile', 'track_id'],
      list(wide.columns)[:4])

lin_demo = {1: {'parent_track_id': None, 'root_track_id': 1, 'generation': 0,
                'start_tp': 1, 'end_tp': 2, 'end_reason': 'divided'},
            2: {'parent_track_id': 1, 'root_track_id': 1, 'generation': 1,
                'start_tp': 2, 'end_tp': 2, 'end_reason': 'series_end'},
            3: {'parent_track_id': None, 'root_track_id': 3, 'generation': 0,
                'start_tp': 1, 'end_tp': 1, 'end_reason': 'lost'}}
wide2 = tt.order_columns(tt.assemble_wide(measurements, meta, timepoints=[1, 2],
                                          lineage=lin_demo))
check('lineage columns present in the wide table',
      {'parent_track_id', 'root_track_id', 'generation',
       'end_reason'}.issubset(wide2.columns))
check('divided parent is distinguishable from a lost track',
      wide2.loc[wide2.track_id == 1, 'end_reason'].iloc[0] == 'divided'
      and wide2.loc[wide2.track_id == 3, 'end_reason'].iloc[0] == 'lost')
check('generation recorded for the daughter',
      int(wide2.loc[wide2.track_id == 2, 'generation'].iloc[0]) == 1)

# ----------------------------------------------------------- measurement ----
print('\n--- measurement ---')
H = W = 40
img = np.zeros((H, W, tt.N_CHANNELS), dtype=np.uint16)
mask_cell = np.zeros((H, W), int)
mask_cell[10:30, 10:30] = 1
mask_nuc = np.zeros((H, W), int)
mask_nuc[16:24, 16:24] = 1
img[10:30, 10:30, tt.CH_DONOR] = 1000
img[10:30, 10:30, tt.CH_ACCEPTOR] = 800
img[10:30, 10:30, tt.CH_MNG] = 500
res = tt.measure_cell(img, mask_cell, mask_cell, mask_nuc, 1)
check('measurement returns a dict', isinstance(res, dict))
check('cell area correct', res['area_cell'] == 400, res['area_cell'])
check('nucleus area correct', res['area_nuc'] == 64, res['area_nuc'])
check('cytoplasm is cell minus nucleus', res['area_cyto'] == 400 - 64, res['area_cyto'])
expected_corr = 800 - tt.BLEED_DIRECT * 500 - tt.BLEED_DONOR * 1000
check('bleedthrough correction matches the original formula',
      np.isclose(res['A_cell_corr'], expected_corr), res['A_cell_corr'])
check('Ef matches A_corr/(A_corr+D)',
      np.isclose(res['Ef_cell'], expected_corr / (expected_corr + 1000)))
check('quantile columns present',
      all(f'directA_Q{i}_cyto' in res for i in range(1, 11)))
check('cell with no nucleus returns None',
      tt.measure_cell(img, mask_cell, mask_cell, np.zeros((H, W), int), 1) is None)

# background subtraction should behave like the original on a simple case
print('\n--- background subtraction ---')
flat = np.full((20, 20), 150, dtype=np.uint16)
m2 = np.ones((20, 20), int)
out_img, out_mask = tt.background_subtract(flat, m2, 102, 24)
check('background subtracted', out_img[0, 0] == 48, out_img[0, 0])
check('mask untouched when above the noise floor', np.all(out_mask == 1))
dim = np.full((20, 20), 110, dtype=np.uint16)
out_img, out_mask = tt.background_subtract(dim, m2, 102, 24)
check('mask zeroed where signal is at the noise floor', np.all(out_mask == 0))

print('\n' + ('ALL TESTS PASSED' if not FAILS else f'{len(FAILS)} FAILURE(S): {FAILS}'))
raise SystemExit(1 if FAILS else 0)
