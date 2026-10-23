"""Tests for the 'staged' matcher, the division refractory period and the
healed-split pruning rule -- run with: python test_staged_tracking.py

Each section is one of the failure modes seen on real tiles:
  1. a real division followed as the parent carrying on, the other daughter lost
  2. a cell that has just divided "dividing" again when Cellpose splits it
  3. that fake split healing, and the healed nucleus handed to a fake daughter
"""

import timelapse_tracking as tt

FAILS = []


def check(name, cond, detail=''):
    if cond:
        print(f'  PASS  {name}')
    else:
        print(f'  FAIL  {name}   {detail}')
        FAILS.append(name)


# the thresholds from the current command line in CLAUDE.md
LOOSE = dict(min_daughter_frac=0.30, max_daughter_frac=1.10,
             min_sum_frac=0.80, max_sum_frac=2.20, min_symmetry=0.35)
KW = dict(max_dist=30, allow_division=True, division_max_dist=60)


def series(n, fn):
    cen, ar = {}, {}
    for t in range(1, n + 1):
        cen[t], ar[t] = {}, {}
        for lab, (rc, a) in fn(t).items():
            cen[t][lab] = rc
            ar[t][lab] = a
    return cen, ar


# ------------------------------------------------------------------------ 1 --
print('\n--- 1. a daughter that sits nearer a neighbour is still found ---')
# P at col 100 divides at t3 into col 82 and col 124.  The col-124 daughter
# is 24 px from P but only 23 px from neighbour N at col 147, so under the old
# rule it claims N, collides with N's own cell, and both are dropped -- while
# P carries on as the col-82 daughter as if nothing happened.
def steal(t):
    out = {2: ((100.0, 147.0), 3000)}
    if t <= 2:
        out[1] = ((100.0, 100.0), 3000)
    else:
        out[1] = ((100.0, 82.0 - (t - 3)), 1500)
        out[3] = ((100.0, 124.0 + (t - 3) * 0.5), 1500)
    return out
cen, ar = series(6, steal)
tr_old, _r, lin_old = tt.track_series(cen, areas_per_tp=ar, method='nearest',
                                      division_thresholds=LOOSE, **KW)
check('old rule: P is never marked divided (the reported failure)',
      lin_old[1]['end_reason'] != 'divided', lin_old[1])
check('old rule: N is lost to the collision',
      2 not in tr_old[3].values(), tr_old[3])
tr, recs, lin = tt.track_series(cen, areas_per_tp=ar, method='staged',
                                division_thresholds=LOOSE, **KW)
check('staged: P divides at t3', lin[1]['end_reason'] == 'divided'
      and lin[1]['end_tp'] == 2, lin[1])
check('staged: two daughters run to the end',
      sorted(lin[k]['parent_track_id'] for k in tr[6].values()
             if lin[k]['parent_track_id'] is not None) == [1, 1], tr[6])
check('staged: N keeps its track throughout',
      all(tr[t].get(2) == 2 for t in tr), {t: tr[t] for t in tr})

print('\n--- 1b. both daughters beyond --max-dist ---')
def wide(t):
    if t <= 2:
        return {1: ((0.0, 0.0), 3000)}
    return {1: ((0.0, -40.0), 1500), 2: ((0.0, 40.0), 1500)}
cen, ar = series(4, wide)
tr, _r, lin = tt.track_series(cen, areas_per_tp=ar, method='staged',
                              division_thresholds=LOOSE, **KW)
check('a parent that held no cell can still divide into two leftovers',
      lin[1]['end_reason'] == 'divided' and len(tr[4]) == 2, (lin, tr[4]))

print('\n--- 1c. a late second daughter is still a division ---')
# B is not segmented at t3 (P carries on as A for one frame) and appears at
# t4.  The midpoint test is judged against P's position before it split.
def late(t):
    if t <= 2:
        return {1: ((0.0, 0.0), 3000)}
    if t == 3:
        return {1: ((0.0, -18.0), 1500)}
    return {1: ((0.0, -20.0 - t), 1500), 2: ((0.0, 20.0 + t), 1500)}
cen, ar = series(6, late)
tr, _r, lin = tt.track_series(cen, areas_per_tp=ar, method='staged',
                              division_thresholds=dict(LOOSE,
                                                       max_midpoint_ratio=0.6),
                              **KW)
check('division accepted when the sibling turns up a frame late',
      lin[1]['end_reason'] == 'divided' and len(tr[6]) == 2, lin[1])

print('\n--- 1d. a cell carrying on next to a newcomer is NOT a division ---')
# With the loose area limits a whole cell (1.0x) plus a whole neighbour (1.0x)
# passes every area test.  Only geometry separates it: P has not moved, so it
# is not between the two "daughters".
def newcomer(t):
    out = {1: ((0.0, 0.0), 3000)}
    if t >= 3:
        out[2] = ((0.0, 30.0), 3000)
    return out
cen, ar = series(5, newcomer)
_t, _r, lin_nomid = tt.track_series(cen, areas_per_tp=ar, method='staged',
                                    division_thresholds=LOOSE, **KW)
check('without the midpoint test it is wrongly accepted',
      lin_nomid[1]['end_reason'] == 'divided', lin_nomid[1])
rej = []
tr, _r, lin = tt.track_series(cen, areas_per_tp=ar, method='staged',
                              division_thresholds=dict(LOOSE,
                                                       max_midpoint_ratio=0.6),
                              division_rejects=rej, **KW)
check('with it, P carries on undivided',
      lin[1]['end_reason'] != 'divided' and all(tr[t].get(1) == 1 for t in tr),
      lin[1])
check('and the rejection is reported with its reason',
      any(r['reason'] == 'midpoint' for r in rej), rej)

print('\n--- staged with divisions off is plain one-to-one matching ---')
cen, ar = series(6, steal)
a = tt.track_series(cen, areas_per_tp=ar, max_dist=30, method='staged')[0]
b = tt.track_series(cen, areas_per_tp=ar, max_dist=30, method='hungarian')[0]
check('same tracks as hungarian', a == b, (a, b))

# ------------------------------------------------------------------------ 2 --
print('\n--- 2. a fresh daughter cannot divide again ---')
# Real division at t3.  At t6-t9 Cellpose splits daughter A into two halves
# (four frames, so --min-division-overlap 3 alone would believe it), then it
# heals at t10.
def resplit(t):
    if t <= 2:
        return {1: ((0.0, 0.0), 3000)}
    out = {2: ((0.0, 40.0), 1500)}
    if 6 <= t <= 9:
        out[1] = ((0.0, -52.0), 750)
        out[3] = ((0.0, -28.0), 750)
    else:
        out[1] = ((0.0, -40.0), 1500)
    return out
cen, ar = series(12, resplit)
th = dict(LOOSE, max_midpoint_ratio=0.6)
_t, _r, lin_free = tt.track_series(cen, areas_per_tp=ar, method='staged',
                                   division_thresholds=th, **KW)
gen2 = [k for k, v in lin_free.items() if v['generation'] == 2]
check('without a refractory period the split becomes a second division',
      len(gen2) == 2, lin_free)
tr, recs, lin = tt.track_series(cen, areas_per_tp=ar, method='staged',
                                division_thresholds=th,
                                min_division_interval=24, **KW)
check('with it, nobody reaches generation 2',
      max(v['generation'] for v in lin.values()) == 1, lin)
daughter_a = tr[3][1]
check('daughter A is one continuous track through the split and the heal',
      all(daughter_a in tr[t].values() for t in range(3, 13)),
      {t: tr[t] for t in tr})
check('the fragment is dropped as refractory',
      any(r['drop_reason'] == 'refractory' for r in recs if r['dropped']),
      [r['drop_reason'] for r in recs if r['dropped']])
check('nearest rule honours the refractory period too',
      max(v['generation'] for v in tt.track_series(
          cen, areas_per_tp=ar, method='nearest', division_thresholds=th,
          min_division_interval=24, **KW)[2].values()) == 1)

# ------------------------------------------------------------------------ 3 --
print('\n--- 3. a split that lasts, then heals, is reverted ---')
# A seed cell (no refractory protection) splits for 4 frames at t3-t6 and
# heals at t7.  The healed nucleus goes to one fake daughter; the other is
# lost.  Overlap 4 >= 3, so only the heal rule can catch it.
def heal(t):
    if t <= 2 or t >= 7:
        return {1: ((0.0, 0.0), 3000)}
    return {1: ((0.0, -12.0), 1500), 2: ((0.0, 12.0), 1500)}
cen, ar = series(10, heal)
tr, _r, lin = tt.track_series(cen, areas_per_tp=ar, method='staged',
                              division_thresholds=th, max_gap=2, **KW)
check('the split is first accepted as a division',
      lin[1]['end_reason'] == 'divided', lin[1])
_t, _l, rev_off = tt.prune_spurious_divisions(
    {t: dict(m) for t, m in tr.items()}, {k: dict(v) for k, v in lin.items()},
    areas_per_tp=ar, min_overlap=3)
check('the overlap rule alone keeps it (4 frames >= 3)', rev_off == [], rev_off)
tr, lin, rev = tt.prune_spurious_divisions(tr, lin, areas_per_tp=ar,
                                           min_overlap=3, heal_window=8)
check('the heal rule reverts it', len(rev) == 1 and rev[0]['reason'] == 'healed',
      rev)
check('one continuous track again', all(tr[t] == {1: 1} or
                                        list(tr[t].values()) == [1]
                                        for t in tr), {t: tr[t] for t in tr})

print('\n--- 3b. a real division with one daughter lost is NOT reverted ---')
# the survivor stays half-size, which is what a real daughter looks like
def real_lost(t):
    if t <= 2:
        return {1: ((0.0, 0.0), 3000)}
    out = {1: ((0.0, -20.0 - t), 1500)}
    if t <= 6:
        out[2] = ((0.0, 20.0 + t), 1500)
    return out
cen, ar = series(14, real_lost)
tr, _r, lin = tt.track_series(cen, areas_per_tp=ar, method='staged',
                              division_thresholds=th, max_gap=2, **KW)
tr, lin, rev = tt.prune_spurious_divisions(tr, lin, areas_per_tp=ar,
                                           min_overlap=3, heal_window=8)
check('kept', rev == [] and lin[1]['end_reason'] == 'divided', (rev, lin[1]))

print('\n' + ('ALL TESTS PASSED' if not FAILS else
              f'{len(FAILS)} FAILURE(S): {FAILS}'))
raise SystemExit(1 if FAILS else 0)
