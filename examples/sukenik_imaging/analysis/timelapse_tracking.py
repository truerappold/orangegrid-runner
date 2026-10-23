"""
Shared logic for the timelapse versions of analyzeDirPerturbBGST.

Deliberately imports nothing from imagej / cellpose / pystackreg, so the parts
that carry the real risk — filename parsing, frame-to-frame tracking and the
wide-table assembly — can be unit tested without a GPU or a JVM.  The two driver
scripts (analyzeDirTimelapseBGST.py, maskPrint_timelapse.py) supply the imaging.

File convention
---------------
Each filename prefix is a *block* of one or more timepoints, and a block's .vsi
may itself hold many frames (SizeT > 1):

    A1-1_DAPI.vsi      -> block 0 (no prefix), typically the single basal frame
    1A1-1_DAPI.vsi     -> block 1, typically the whole timelapse (e.g. SizeT=20)
    2A1-1_DAPI.vsi     -> block 2, if a second acquisition block exists

Blocks are concatenated in numerical order and the frames inside each block in
acquisition order, giving global timepoints 1..N.  So a basal image plus a
20-frame timelapse is 21 timepoints: t1 = basal, t2..t21 = the stack.  How many
frames a block holds can only be read from the file, so the split is:
`discover_blocks` here works on filenames, and timelapse_imaging.BlockReader
opens each block and reports how many frames it holds.

Tracking
--------
Timepoint 1 defines the tracks.  For every later timepoint each cell is matched
to the nearest *nuclear* centroid among the cells of the previous timepoint that
still hold a live track.  A cell is dropped when

  * its nearest previous centroid is further than `max_dist` pixels away, or
  * two or more cells in this timepoint claim the same previous cell.

A dropped cell ends its track; earlier timepoints of that track are kept and the
track is flagged as incomplete (`complete_track`).

Division
--------
Two daughters of a dividing cell are both nearest to the parent, so without
special handling the collision rule drops both.  With `allow_division=True` a
two-way collision is accepted as a division when it also passes area tests
(each daughter a plausible fraction of the parent, the two roughly symmetric).
The parent track then ends with `end_reason='divided'` and two new tracks begin,
carrying `parent_track_id`, `root_track_id` and `generation` so a lineage can be
reassembled.  Three-way and larger collisions are still dropped — those are
segmentation noise, not mitosis.

The area tests are the only thing separating a real division from one cell being
over-segmented into two fragments, and they are not decisive on their own.  What
does most of the work is that every tracked cell must contain a nucleus: an
over-segmented cell usually puts the nucleus wholly in one fragment, so the other
fragment never becomes a tracking candidate at all.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# image layout — one 4-channel plane per timepoint
# --------------------------------------------------------------------------- #
CH_DONOR = 0      # FRET donor (mTQ2), camera 1
CH_ACCEPTOR = 1   # FRET acceptor (mNG emission under donor excitation), camera 2
CH_MNG = 2        # direct mNG, camera 2
CH_DAPI = 3       # DAPI, camera 1
N_CHANNELS = 4

CHANNEL_FILES = ['FRET, FRET', 'mNG', 'DAPI']

# bleedthrough coefficients, from analyzeDirPerturbBGST.py
BLEED_DIRECT = 0.070
BLEED_DONOR = 0.4163

# flat background / noise threshold per channel, from analyzeDirPerturbBGST.py
BG_SUBTRACT = {
    CH_DONOR: (102, 24),
    CH_ACCEPTOR: (105, 24),
    CH_MNG: (101, 20),
}

VSI_RE = re.compile(
    r'^(?P<prefix>\d*)(?P<well>[A-Za-z]+\d+)-(?P<tile>\d+)_(?P<channel>.+)\.vsi$')


# --------------------------------------------------------------------------- #
# 1. filenames -> (well, tile, timepoint, channel)
# --------------------------------------------------------------------------- #
def parse_vsi_name(basename):
    """'2A1-7_DAPI.vsi' -> {'well': 'A1', 'tile': 7, 'block': 2, ...}

    Returns None for anything that does not fit the pattern.  Parsing is
    unambiguous because a well label always starts with a letter, so the leading
    run of digits can only be the block prefix.
    """
    m = VSI_RE.match(basename)
    if m is None:
        return None
    prefix = m.group('prefix')
    return {
        'well': m.group('well'),
        'tile': int(m.group('tile')),
        'block': 0 if prefix == '' else int(prefix),
        'channel': m.group('channel'),
        'prefix': prefix,
    }


def block_prefix(block):
    """Inverse of the convention: block 0 -> '', block 2 -> '2'."""
    if block < 0:
        raise ValueError(f'block must be >= 0, got {block}')
    return '' if block == 0 else str(block)


def build_filename(pics_dir, well, tile, block, channel):
    return os.path.join(
        pics_dir, f'{block_prefix(block)}{well}-{tile}_{channel}.vsi')


def discover_blocks(filelist, channels=CHANNEL_FILES):
    """Map (well, tile) -> ordered list of blocks that have every channel.

    Also returns human-readable problems: blocks missing channels, and gaps in
    the block numbering.  A gap is reported and the series truncated rather than
    bridged — silently concatenating across a missing block would inflate
    centroid distances at that join and quietly corrupt the matching.
    """
    seen = defaultdict(set)            # (well, tile, block) -> {channel}
    for path in filelist:
        info = parse_vsi_name(os.path.basename(path))
        if info is None:
            continue
        seen[(info['well'], info['tile'], info['block'])].add(info['channel'])

    complete = defaultdict(list)
    problems = []
    for (well, tile, block), chans in sorted(seen.items()):
        missing = [c for c in channels if c not in chans]
        if missing:
            problems.append(
                f'{well} tile {tile} block {block}: missing channel(s) '
                f'{", ".join(missing)} — block skipped')
        else:
            complete[(well, tile)].append(block)

    series = {}
    for key, blocks in complete.items():
        blocks = sorted(blocks)
        if blocks and blocks[0] != 0:
            problems.append(
                f'{key[0]} tile {key[1]}: no unprefixed block — tile skipped')
            continue
        gaps = [b for b in range(max(blocks) + 1) if b not in blocks] if blocks else []
        if gaps:
            problems.append(
                f'{key[0]} tile {key[1]}: gap at block(s) {gaps} — '
                f'truncating after block {min(gaps) - 1}')
            blocks = [b for b in blocks if b < min(gaps)]
        if blocks:
            series[key] = blocks
    return series, problems


# --------------------------------------------------------------------------- #
# 2. per-timepoint mask bookkeeping
# --------------------------------------------------------------------------- #
def centroids_by_label(mask):
    """{label: (row, col)} for every non-zero label, via a vectorised mean."""
    labels = mask.ravel()
    nz = labels > 0
    if not np.any(nz):
        return {}
    labels = labels[nz]
    rows, cols = np.nonzero(mask)
    n = int(labels.max()) + 1
    count = np.bincount(labels, minlength=n).astype(float)
    sum_r = np.bincount(labels, weights=rows, minlength=n)
    sum_c = np.bincount(labels, weights=cols, minlength=n)
    out = {}
    for lab in np.unique(labels):
        if count[lab] > 0:
            out[int(lab)] = (sum_r[lab] / count[lab], sum_c[lab] / count[lab])
    return out


# --------------------------------------------------------------------------- #
# 3. tracking
# --------------------------------------------------------------------------- #
# Division acceptance thresholds, as fractions of the parent's cell area.
DIVISION_DEFAULTS = dict(
    min_daughter_frac=0.25,   # each daughter at least this much of the parent
    max_daughter_frac=0.85,   # ... and no more than this
    min_sum_frac=0.60,        # the two together, versus the parent
    max_sum_frac=1.60,
    min_symmetry=0.40,        # smaller daughter / larger daughter
    # Distance from the daughters' midpoint to where the parent was, divided
    # by half the daughter-daughter separation.  ~0 for a real division (the
    # parent sits between its daughters); ~1 when the "division" is really the
    # parent carrying on in place with an unrelated cell next to it.  None
    # skips the test.
    max_midpoint_ratio=None,
)


def match_timepoint(prev_centroids, curr_centroids, max_dist,
                    method='nearest', prev_areas=None, curr_areas=None,
                    allow_division=False, division_max_dist=None,
                    division_thresholds=None, division_rejects=None,
                    nearest_out=None, prev_max_dist=None,
                    division_blocked=None, prev_anchors=None,
                    prev_overlap=None, link_overlap=0.3,
                    division_relaxed=None, relaxed_max_dist=None):
    """Match cells of one timepoint to the previous timepoint's cells.

    prev_centroids / curr_centroids: {label: (row, col)}.  Only cells with a
    live track should appear in `prev_centroids`.

    Returns (assignment, distances, dropped, divisions) where
        assignment  {curr_label: prev_label} for accepted matches
        distances   {curr_label: distance to its nearest prev centroid}
                    — recorded for every current cell, matched or not, so the
                    cutoff can be retuned later without re-segmenting
        dropped     {curr_label: 'distance' | 'collision' | 'no_previous_cells'}
        divisions   {prev_label: [daughter_a, daughter_b]} accepted divisions

    With allow_division, a two-way collision is tested as a division: both
    daughters within division_max_dist (defaults to max_dist) of the parent, and
    the area ratios inside division_thresholds.  Failing any test, both are
    dropped as 'collision' exactly as before.

    method 'nearest'  : each cell takes its nearest previous centroid; if two or
                        more cells claim the same one, all of them are dropped.
    method 'hungarian': globally optimal one-to-one assignment under the same
                        distance cutoff.  Cannot produce collisions, so it keeps
                        cells that 'nearest' would discard.
    method 'staged'   : decisions in a fixed order of confidence -- see
                        _staged_divisions.  (1) one-to-one assignment as in
                        'hungarian'; (2) only cells left over after that are
                        considered as the missing half of a division, either
                        partnered with the cell the parent already holds or
                        as a pair for a parent that held none; (3) whatever
                        is still left is dropped.

    division_blocked: prev labels not allowed to divide this frame (a daughter
    that divided too recently to do it again).  Under 'nearest' a second
    claimant on a blocked parent is dropped as 'refractory' and the closer one
    keeps the track, rather than both being lost.
    prev_anchors: {prev_label: [recent (row, col)]} for the midpoint test, so a
    division detected a frame or two late is still judged against where the
    parent was before it split.

    'staged' only:
    prev_overlap: {prev_label: {curr_label: overlap fraction}} between masks.
    A pair overlapping by at least `link_overlap` may link however far apart
    the centroids are, and overlap lowers the cost of a link.  This is what
    keeps a flat cell linked while it rounds up for mitosis: its centroid
    jumps tens of px but the round cell sits inside its old footprint.
    division_relaxed: prev labels flagged as mitotic (round and shrunken
    recently).  They are searched for daughters out to relaxed_max_dist and
    their division skips the area-fraction tests -- symmetry and the midpoint
    test still apply.
    """
    curr_labels = sorted(curr_centroids)
    prev_labels = sorted(prev_centroids)

    thresholds = dict(DIVISION_DEFAULTS)
    if division_thresholds:
        thresholds.update(division_thresholds)
    div_max = max_dist if division_max_dist is None else division_max_dist

    if not prev_labels:
        return ({}, {c: np.inf for c in curr_labels},
                {c: 'no_previous_cells' for c in curr_labels}, {})
    if not curr_labels:
        return {}, {}, {}, {}

    # Each previous cell may carry its own cutoff: a track rejoining after a
    # gap has had more time to move, so holding it to the single-frame limit
    # would reject the very re-links gap closing exists to make.
    per_prev = np.array([(prev_max_dist or {}).get(p, max_dist)
                         for p in prev_labels], float)

    curr_xy = np.array([curr_centroids[c] for c in curr_labels], float)
    prev_xy = np.array([prev_centroids[p] for p in prev_labels], float)
    # (n_curr, n_prev) euclidean distance matrix
    dmat = np.linalg.norm(curr_xy[:, None, :] - prev_xy[None, :, :], axis=2)

    nearest_idx = np.argmin(dmat, axis=1)
    nearest_dist = dmat[np.arange(len(curr_labels)), nearest_idx]
    distances = {c: float(d) for c, d in zip(curr_labels, nearest_dist)}
    if nearest_out is not None:
        # Recorded for *every* current cell including the ones about to be
        # dropped.  Without this a dropped row has no parent, and the whole
        # point of the match-distance file -- re-deciding the cutoffs without
        # paying for Cellpose again -- does not survive to the collision rule.
        nearest_out.update({c: prev_labels[i]
                            for c, i in zip(curr_labels, nearest_idx)})

    dropped = {}
    if method == 'hungarian':
        from scipy.optimize import linear_sum_assignment
        # forbid pairings beyond the cutoff by making them prohibitively costly
        cost = np.where(dmat <= per_prev[None, :], dmat, 1e9)
        rows, cols = linear_sum_assignment(cost)
        assignment = {}
        matched_curr = set()
        for r, c in zip(rows, cols):
            if dmat[r, c] <= per_prev[c]:
                assignment[curr_labels[r]] = prev_labels[c]
                matched_curr.add(curr_labels[r])
        for c in curr_labels:
            if c not in matched_curr:
                dropped[c] = 'distance'
        # Division detection is defined for the 'nearest' rule only; the
        # Hungarian assignment is one-to-one by construction, so a one-to-two
        # parent/daughter relation has no representation in it.
        return assignment, distances, dropped, {}

    if method == 'staged':
        from scipy.optimize import linear_sum_assignment
        ovm = np.zeros_like(dmat)
        if prev_overlap:
            cidx = {c: i for i, c in enumerate(curr_labels)}
            for j, p in enumerate(prev_labels):
                for c, f in (prev_overlap.get(p) or {}).items():
                    if c in cidx:
                        ovm[cidx[c], j] = f
        allowed = (dmat <= per_prev[None, :]) | (ovm >= link_overlap)
        cost = np.where(allowed, dmat * (1.0 - np.clip(ovm, 0, 1)), 1e9)
        rows, cols = linear_sum_assignment(cost)
        assignment = {curr_labels[r]: prev_labels[c]
                      for r, c in zip(rows, cols) if allowed[r, c]}
        divisions, refractory = {}, set()
        if allow_division:
            divisions, refractory = _staged_divisions(
                assignment, curr_labels, prev_labels, dmat, nearest_idx,
                prev_centroids, curr_centroids, prev_areas, curr_areas,
                div_max, thresholds, division_blocked or set(),
                prev_anchors or {}, division_rejects, ovm=ovm,
                relaxed=division_relaxed or set(),
                relaxed_max=relaxed_max_dist or 1.5 * div_max)
        for i, c in enumerate(curr_labels):
            if c in assignment or any(c in ds for ds in divisions.values()):
                continue
            if c in refractory:
                dropped[c] = 'refractory'
            elif nearest_dist[i] > per_prev[nearest_idx[i]] and not (
                    ovm[i] >= link_overlap).any():
                dropped[c] = 'distance'
            else:
                # in range, but the one-to-one assignment gave that track to
                # a better candidate
                dropped[c] = 'collision'
        return assignment, distances, dropped, divisions

    # 'nearest' — the rule as specified
    #
    # Candidate daughters need a WIDER radius than ordinary matches.  A cell
    # creeps a few px between frames, but at mitosis each daughter centroid
    # lands roughly half a nucleus away from where the parent's was -- further
    # than any interphase cell moves.  Gating candidates at max_dist therefore
    # hides every division: the two-claimants-on-one-parent signature can never
    # form, so the division test is never even consulted.  Cells admitted only
    # by the wider radius are tracked in `wide_only` and are kept *solely* as
    # half of an accepted division; on any other outcome they drop as
    # 'distance', exactly as before.
    provisional, wide_only = {}, set()
    for i, c in enumerate(curr_labels):
        lim = per_prev[nearest_idx[i]]
        search = max(lim, div_max) if allow_division else lim
        if nearest_dist[i] > search:
            dropped[c] = 'distance'
            continue
        provisional[c] = prev_labels[nearest_idx[i]]
        if nearest_dist[i] > lim:
            wide_only.add(c)

    claims = defaultdict(list)
    for c, p in provisional.items():
        claims[p].append(c)

    assignment = {}
    divisions = {}
    for p, cs in claims.items():
        if len(cs) == 1:
            if cs[0] in wide_only:
                dropped[cs[0]] = 'distance'
            else:
                assignment[cs[0]] = p
            continue
        if division_blocked and p in division_blocked:
            near = [c for c in cs if c not in wide_only]
            keep = min(near, key=lambda c: distances[c]) if near else None
            for c in cs:
                if c == keep:
                    assignment[c] = p
                else:
                    dropped[c] = 'refractory'
            continue
        if allow_division and len(cs) == 2:
            ok, reason, detail = _is_division(
                p, cs, prev_centroids, curr_centroids,
                prev_areas, curr_areas, div_max, thresholds,
                anchors=(prev_anchors or {}).get(p))
            if ok:
                divisions[p] = sorted(cs)
                continue
            if division_rejects is not None:
                division_rejects.append({'parent': p, 'daughters': sorted(cs),
                                         'reason': reason, **detail})
        for c in cs:
            dropped[c] = 'distance' if c in wide_only else 'collision'
    return assignment, distances, dropped, divisions


def _staged_divisions(assignment, curr_labels, prev_labels, dmat, nearest_idx,
                      prev_centroids, curr_centroids, prev_areas, curr_areas,
                      div_max, thr, blocked, anchors, division_rejects,
                      ovm=None, relaxed=(), relaxed_max=None, near_overlap=0.1):
    """Stage 2 of the 'staged' matcher: find divisions among leftover cells.

    Stage 1 has already given every track at most one cell, so no cell is lost
    to a collision and no neighbour can steal a daughter.  What remains are
    cells no track wanted.  A division shows up as one of two patterns:

      held + free   the track kept one daughter (it was within --max-dist) and
                    the other daughter is a leftover cell nearby.  This is the
                    case the old two-claimants rule missed whenever the second
                    daughter happened to sit nearer a neighbour: the parent
                    just carried on as one daughter and the other was dropped.
      free + free   both daughters landed beyond --max-dist (typical: each is
                    ~half a nucleus from the parent) so the track held nothing.

    Every candidate pair must pass _is_division.  Accepted pairs are taken
    best-first so a leftover cell is never given to two parents.  "Best" is
    how centrally the parent sits between the two daughters, not how close
    they are: a leftover daughter is often about as close to a neighbour as
    to its own parent, but only its own parent lies between it and its
    sibling.  Tracks in `blocked` cannot divide; leftover
    cells that would have been their daughter are returned as `refractory`.

    A leftover cell counts as near a parent if it is within div_max (or
    relaxed_max for a parent in `relaxed`) or overlaps the parent's mask by
    near_overlap.  Parents in `relaxed` were flagged as mitotic, so their
    division is expected: the area-fraction tests are skipped for them, and
    a pair of two leftovers may replace the cell stage 1 gave them.
    """
    from itertools import combinations

    cidx = {c: i for i, c in enumerate(curr_labels)}
    held = {p: c for c, p in assignment.items()}
    free = [c for c in curr_labels if c not in assignment]
    options, refractory, best_reject = [], set(), {}
    thr_relaxed = dict(thr, min_daughter_frac=0.0, max_daughter_frac=np.inf,
                       min_sum_frac=0.0, max_sum_frac=np.inf)
    for j, p in enumerate(prev_labels):
        lim = relaxed_max if p in relaxed else div_max
        near = [c for c in free if dmat[cidx[c], j] <= lim
                or (ovm is not None and ovm[cidx[c], j] >= near_overlap)]
        if not near:
            continue
        if p in blocked:
            refractory.update(c for c in near if nearest_idx[cidx[c]] == j)
            continue
        if p in held:
            pairs = [(held[p], u) for u in near]
            # A mitotic parent's one-to-one match is not trusted to be a
            # daughter: at cytokinesis the closest object is often the
            # bridge between the two.  Let two leftovers outbid it.
            if p in relaxed:
                pairs += list(combinations(near, 2))
        else:
            pairs = list(combinations(near, 2))
        passed = False
        # A mitotic parent is judged only against where it was while round:
        # older positions belong to the flat cell, whose centroid can sit
        # half a cell away from where it rounded up and divided.
        p_anchors = None if p in relaxed else anchors.get(p)
        for d1, d2 in pairs:
            score = (_midpoint_ratio(curr_centroids[d1], curr_centroids[d2],
                                     list(p_anchors or [])
                                     + [prev_centroids[p]]),
                     float(dmat[cidx[d1], j] + dmat[cidx[d2], j]))
            # distance was settled by `near` (which may admit on overlap);
            # the held cell was admitted by stage 1 on the same terms
            ok, reason, detail = _is_division(
                p, [d1, d2], prev_centroids, curr_centroids, prev_areas,
                curr_areas, np.inf, thr_relaxed if p in relaxed else thr,
                anchors=p_anchors)
            if ok:
                options.append((score, p, d1, d2))
                passed = True
            elif (division_rejects is not None
                  and any(nearest_idx[cidx[d]] == j for d in (d1, d2)
                          if d != held.get(p))
                  and (p not in best_reject or score < best_reject[p][0])):
                best_reject[p] = (score, {'parent': p,
                                          'daughters': sorted([d1, d2]),
                                          'reason': reason, **detail})
        if passed:
            best_reject.pop(p, None)

    divisions, used = {}, set()
    for _score, p, d1, d2 in sorted(options):
        if p in divisions or {d1, d2} & used:
            continue
        divisions[p] = sorted([d1, d2])
        used |= {d1, d2}
        if held.get(p) is not None:
            assignment.pop(held[p], None)
    if division_rejects is not None:
        division_rejects.extend(rj for p, (_s, rj) in sorted(best_reject.items())
                                if p not in divisions)
    return divisions, refractory - used


def _midpoint_ratio(ca, cb, refs):
    """Offset of the daughters' midpoint from the nearest parent position,
    over half their separation.  0 = parent exactly between them."""
    mid = ((ca[0] + cb[0]) / 2, (ca[1] + cb[1]) / 2)
    half = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1])) / 2
    off = min(float(np.hypot(mid[0] - r[0], mid[1] - r[1])) for r in refs)
    return off / half if half > 0 else np.inf


def _is_division(parent, daughters, prev_centroids, curr_centroids,
                 prev_areas, curr_areas, div_max, thr, anchors=None):
    """Does this two-way collision look like mitosis rather than noise?

    Returns (accepted, reason, detail).  `reason` names the test that rejected
    the candidate -- 'distance', 'midpoint', 'daughter_frac', 'sum_frac',
    'symmetry', 'no_area' -- and `detail` carries the numbers that failed it.  A rejected
    division is otherwise indistinguishable from a tracking collision, so
    without this a real mitosis that misses a threshold just disappears and
    there is nothing to tune against.
    """
    a, b = daughters
    pc = prev_centroids[parent]
    dists = [float(np.hypot(curr_centroids[d][0] - pc[0],
                            curr_centroids[d][1] - pc[1])) for d in (a, b)]
    if max(dists) > div_max:
        return False, 'distance', {'daughter_dists': [round(d, 1) for d in dists],
                                   'limit': div_max}
    if thr.get('max_midpoint_ratio') is not None:
        ratio = _midpoint_ratio(curr_centroids[a], curr_centroids[b],
                                list(anchors or []) + [pc])
        if ratio > thr['max_midpoint_ratio']:
            return False, 'midpoint', {'midpoint_ratio': round(ratio, 2),
                                       'limit': thr['max_midpoint_ratio']}
    if not prev_areas or not curr_areas:
        # No areas supplied: accept on geometry alone.  Permissive, so only do
        # this knowingly -- the area tests are what exclude over-segmentation.
        return True, '', {}
    pa = prev_areas.get(parent)
    da, db = curr_areas.get(a), curr_areas.get(b)
    if not pa or not da or not db:
        return False, 'no_area', {}
    fracs = [da / pa, db / pa]
    if not all(thr['min_daughter_frac'] <= f <= thr['max_daughter_frac']
               for f in fracs):
        return False, 'daughter_frac', {
            'daughter_fracs': [round(f, 2) for f in fracs],
            'limits': [thr['min_daughter_frac'], thr['max_daughter_frac']]}
    total = (da + db) / pa
    if not (thr['min_sum_frac'] <= total <= thr['max_sum_frac']):
        return False, 'sum_frac', {
            'sum_frac': round(total, 2),
            'limits': [thr['min_sum_frac'], thr['max_sum_frac']]}
    symmetry = min(da, db) / max(da, db)
    if symmetry < thr['min_symmetry']:
        return False, 'symmetry', {'symmetry': round(symmetry, 2),
                                   'limit': thr['min_symmetry']}
    return True, '', {}


def track_series(centroids_per_tp, max_dist, method='nearest',
                 first_step_max_dist=None, areas_per_tp=None,
                 allow_division=False, division_max_dist=None,
                 division_thresholds=None, max_gap=0,
                 division_rejects=None, gap_dist_scale=1.0,
                 predict_motion=True, area_history=5,
                 min_division_interval=0, anchor_history=3,
                 overlaps=None, link_overlap=0.3, circularity_per_tp=None,
                 mitotic_circularity=None, mitotic_area_frac=0.6,
                 mitosis_window=8, single_daughter_frac=0.75,
                 mitotic_division_max_dist=None, round_hysteresis=0.10,
                 mitotic_evidence_per_tp=None, division_requires_mitosis=False):
    """Assign track ids across a whole series, optionally following divisions.

    centroids_per_tp: {timepoint: {cell_label: (row, col)}}, timepoint 1 first.
    areas_per_tp:     {timepoint: {cell_label: area}} -- needed for the division
                      area tests; without it divisions are accepted on geometry
                      alone, which is permissive.

    first_step_max_dist overrides max_dist for the t1->t2 step only.

    max_gap lets a track survive frames in which its cell was not segmented at
    all: with max_gap=2 a track last seen at t5 can still be claimed at t6, t7
    or t8.  This matters whenever a cell can legitimately disappear from the
    mask -- a PIP-FUCCI cell segmented on fluorescence goes dark for a few
    frames in early S -- because otherwise one missed frame ends the track
    permanently and the cell is never seen again.  max_gap=0 reproduces the
    strict frame-to-frame behaviour.

    Matching is done against *track* positions rather than previous-frame
    labels, so a track carries its last known centroid across a gap.

    min_division_interval: a track born from a division cannot itself divide
    until it is this many frames old.  A real cell cycle is many hours, so a
    "division" a few frames after the last one is Cellpose splitting a nucleus
    in two -- and accepting it is what lets the tracker then hand the healed
    nucleus to whichever fake daughter happens to be nearest.  Seed tracks
    (generation 0) have unknown age and are never blocked.

    anchor_history: how many recent positions each track keeps for the
    division midpoint test, so a division whose second daughter only shows up
    a frame or two late is judged against where the parent was before it split.

    overlaps: {(tp_prev, tp): {(label_prev, label_curr): overlap fraction}}
    from the masks ('staged' only).  See match_timepoint: a pair overlapping by
    link_overlap may link at any distance.

    Mitosis flag (mitotic_circularity not None, needs circularity_per_tp):
    a track whose cell is at least mitotic_circularity round AND has shrunk to
    at most mitotic_area_frac of its own pre-mitotic median area is flagged
    mitotic.  For mitosis_window frames after its last round frame:
      * its division is searched for out to mitotic_division_max_dist
        (default 1.5x division_max_dist) and skips the area-fraction tests;
      * if it flattens out again (circularity below the threshold minus
        round_hysteresis) while still at most single_daughter_frac of the
        pre-mitotic area, it is a cell that divided with only one daughter
        found.  The parent track ends there and the cell continues under a new
        daughter id (division_kind 'single'), so it is no longer counted as
        the parent.  If the missing sibling turns up nearby within the window
        it is adopted as the second daughter ('late_sibling').
    A flagged cell that flattens back to full size simply loses the flag.

    mitotic_evidence_per_tp: {tp: {label: bool}} extra evidence a round cell
    is really mitotic (dual mode: the brightfield halo of a refractile rounded
    cell).  When given, a cell is flagged only if it is round, shrunk AND has
    the evidence.
    division_requires_mitosis: only flagged tracks may divide.  On brightfield
    every real division seen was preceded by a round, haloed cell, while the
    false ones were a flat cell Cellpose cut in half for a frame; extra
    objects next to an unflagged track are dropped as 'not_mitotic'.

    Returns
        tracks   {timepoint: {cell_label: track_id}}
        records  one dict per cell per timepoint, with the match distance, the
                 drop reason, and 'event' in {'seed', 'matched', 'division',
                 'dropped'}
        lineage  {track_id: {parent_track_id, root_track_id, generation,
                 start_tp, end_tp, end_reason, division_kind}} where
                 end_reason is 'divided' | 'lost' | 'series_end' and
                 division_kind (daughters only) 'pair' | 'single' |
                 'late_sibling'
    """
    timepoints = sorted(centroids_per_tp)
    tracks = {}
    records = []
    lineage = {}
    if not timepoints:
        return tracks, records, lineage

    areas_per_tp = areas_per_tp or {}
    circ_per_tp = circularity_per_tp or {}
    overlaps = overlaps or {}
    div_max = max_dist if division_max_dist is None else division_max_dist
    mit_max = mitotic_division_max_dist or 1.5 * div_max
    thresholds = dict(DIVISION_DEFAULTS)
    thresholds.update(division_thresholds or {})

    def area_of(tp, lab):
        return areas_per_tp.get(tp, {}).get(lab)

    def circ_of(tp, lab):
        c = circ_per_tp.get(tp, {}).get(lab)
        return c if c is not None and np.isfinite(c) else None

    def ref_area(ot):
        return float(np.median(ot['hist'])) if ot['hist'] else ot['area']

    def is_round(tp, lab, ref):
        if mitotic_circularity is None:
            return False
        if mitotic_evidence_per_tp is not None and \
                not mitotic_evidence_per_tp.get(tp, {}).get(lab, False):
            return False
        c, a = circ_of(tp, lab), area_of(tp, lab)
        return bool(c is not None and c >= mitotic_circularity and a and ref
                    and a <= mitotic_area_frac * ref)

    open_tracks = {}

    def open_track(tid, tp, lab, cen):
        a = area_of(tp, lab)
        open_tracks[tid] = {'tp': tp, 'cen': cen, 'area': a, 'vel': (0.0, 0.0),
                            'hist': [a] if a else [], 'pos': [cen],
                            'label': lab, 'mitotic_tp': None}

    def record(tp, lab, tid, event, dist=np.nan, prev=np.nan, parent=np.nan,
               gap=0, **extra):
        cen = centroids_per_tp[tp][lab]
        rec = {'timepoint': tp, 'cell_label': lab, 'track_id': tid,
               'match_dist': dist, 'matched_prev_label': prev,
               'dropped': False, 'drop_reason': '', 'event': event,
               'parent_track_id': parent, 'gap': gap,
               'centroid_row': cen[0], 'centroid_col': cen[1]}
        rec.update(extra)
        records.append(rec)

    def start_daughter(tid, parent_tid, tp, lab, kind):
        lineage[tid] = {
            'parent_track_id': parent_tid,
            'root_track_id': lineage[parent_tid]['root_track_id'],
            'generation': lineage[parent_tid]['generation'] + 1,
            'start_tp': tp, 'end_tp': tp, 'end_reason': 'series_end',
            'division_kind': kind}
        open_track(tid, tp, lab, centroids_per_tp[tp][lab])
        tracks[tp][lab] = tid

    t0 = timepoints[0]
    tracks[t0] = {lab: int(lab) for lab in sorted(centroids_per_tp[t0])}
    next_id = (max(tracks[t0].values()) + 1) if tracks[t0] else 1
    for lab, tid in tracks[t0].items():
        lineage[tid] = {'parent_track_id': None, 'root_track_id': tid,
                        'generation': 0, 'start_tp': t0, 'end_tp': t0,
                        'end_reason': 'series_end'}
        open_track(tid, t0, lab, centroids_per_tp[t0][lab])
        record(t0, lab, tid, 'seed', parent=np.nan)

    for step, tp in enumerate(timepoints[1:]):
        # Candidates are tracks last seen recently enough to bridge the gap.
        # Two things have to change for a track that is *in* a gap:
        #
        #   position -- it is matched against where it was last SEEN, but the
        #     cell kept moving while it was invisible.  Carrying its last
        #     velocity forward puts the search where the cell actually is.
        #   cutoff -- it has had k frames to move, not one, so holding it to
        #     the single-frame limit rejects exactly the re-links that gap
        #     closing exists to make.
        cands, cand_max, cand_ov = {}, {}, {}
        for tid, ot in open_tracks.items():
            elapsed = tp - ot['tp']
            if not (0 < elapsed <= max_gap + 1):
                continue
            if predict_motion and elapsed > 1:
                cands[tid] = (ot['cen'][0] + ot['vel'][0] * elapsed,
                              ot['cen'][1] + ot['vel'][1] * elapsed)
            else:
                cands[tid] = ot['cen']
            cand_max[tid] = max_dist * (1.0 + gap_dist_scale * (elapsed - 1))
            ov = overlaps.get((ot['tp'], tp))
            if ov:
                cand_ov[tid] = {lc: f for (lp, lc), f in ov.items()
                                if lp == ot['label']}
        # The division area tests compare each daughter to the parent's recent
        # median, not its last frame: a dividing cell condenses, so the frame
        # before mitosis is the smallest it ever is.  The median also ignores
        # the occasional frame where Cellpose merged a neighbour in.  Round
        # frames are kept out of the history (below), so a mitotic cell is
        # still compared with what it was before it rounded up.
        cand_areas = {}
        for tid in cands:
            ra = ref_area(open_tracks[tid])
            if ra:
                cand_areas[tid] = ra
        curr = centroids_per_tp[tp]
        step_max = (first_step_max_dist
                    if step == 0 and first_step_max_dist is not None else max_dist)
        if step == 0 and first_step_max_dist is not None:
            cand_max = {tid: first_step_max_dist for tid in cands}
        blocked = {tid for tid in cands
                   if lineage[tid]['parent_track_id'] is not None
                   and tp - lineage[tid]['start_tp'] < min_division_interval}
        mitotic = {tid for tid in cands
                   if open_tracks[tid]['mitotic_tp'] is not None
                   and tp - open_tracks[tid]['mitotic_tp'] <= mitosis_window}
        blocked -= mitotic
        not_mitotic = (set(cands) - mitotic - blocked
                       if division_requires_mitosis else set())
        anchors = {tid: open_tracks[tid]['pos'] for tid in cands}

        step_rejects = [] if division_rejects is not None else None
        nearest = {}
        assignment, distances, dropped, divisions = match_timepoint(
            cands, curr, step_max, method=method,
            prev_areas=cand_areas or None,
            curr_areas=areas_per_tp.get(tp),
            allow_division=allow_division, division_max_dist=division_max_dist,
            division_thresholds=division_thresholds,
            division_rejects=step_rejects, nearest_out=nearest,
            prev_max_dist=cand_max, division_blocked=blocked | not_mitotic,
            prev_anchors=anchors, prev_overlap=cand_ov,
            link_overlap=link_overlap, division_relaxed=mitotic,
            relaxed_max_dist=mit_max)

        for lab, why in list(dropped.items()):
            if why == 'refractory' and nearest.get(lab) in not_mitotic:
                dropped[lab] = 'not_mitotic'

        if step_rejects:
            for rj in step_rejects:
                rj['timepoint'] = tp
                rj['mitotic'] = rj['parent'] in mitotic
            division_rejects.extend(step_rejects)

        tracks[tp] = {}

        for lab, tid in assignment.items():
            ot = open_tracks[tid]
            gap = tp - ot['tp'] - 1
            elapsed = tp - ot['tp']
            ref = ref_area(ot)
            new_area = area_of(tp, lab)
            round_now = is_round(tp, lab, ref)
            c_now = circ_of(tp, lab)

            # one daughter found: a flagged cell has flattened out at
            # daughter size without a second daughter ever being seen.  It
            # must also have lost the mitotic evidence: a still-refractile
            # cell whose outline briefly caught a protrusion is still in
            # mitosis, and calling it a daughter dates the division early.
            still_mitotic = (mitotic_evidence_per_tp is not None and
                             mitotic_evidence_per_tp.get(tp, {}).get(lab, False))
            if (allow_division and ot['mitotic_tp'] is not None
                    and not still_mitotic
                    and tid not in blocked
                    and tp - ot['mitotic_tp'] <= mitosis_window
                    and not round_now and c_now is not None
                    and c_now < mitotic_circularity - round_hysteresis
                    and new_area and ref
                    and new_area <= single_daughter_frac * ref):
                lineage[tid]['end_tp'] = ot['tp']
                lineage[tid]['end_reason'] = 'divided'
                open_tracks.pop(tid)
                new = next_id
                next_id += 1
                start_daughter(new, tid, tp, lab, 'single')
                lineage[new]['sibling_missing'] = True
                record(tp, lab, new, 'division',
                       dist=distances.get(lab, np.nan), prev=tid, parent=tid,
                       division_kind='single')
                continue

            if (ot['mitotic_tp'] is not None
                    and tp - ot['mitotic_tp'] > mitosis_window):
                ot['mitotic_tp'] = None
            if round_now:
                ot['mitotic_tp'] = tp
            elif ot['mitotic_tp'] is not None and new_area and ref \
                    and new_area > single_daughter_frac * ref \
                    and c_now is not None \
                    and c_now < mitotic_circularity - round_hysteresis:
                ot['mitotic_tp'] = None          # flattened back, full size
            hist = ot['hist']
            if new_area and not round_now and ot['mitotic_tp'] is None:
                hist = (hist + [new_area])[-area_history:]
            vel = ((curr[lab][0] - ot['cen'][0]) / elapsed,
                   (curr[lab][1] - ot['cen'][1]) / elapsed)
            tracks[tp][lab] = tid
            lineage[tid]['end_tp'] = tp
            ot.update(tp=tp, cen=curr[lab], area=new_area, vel=vel, hist=hist,
                      pos=(ot['pos'] + [curr[lab]])[-anchor_history:],
                      label=lab)
            record(tp, lab, tid, 'matched', dist=distances.get(lab, np.nan),
                   prev=tid, gap=gap, mitotic=ot['mitotic_tp'] == tp)

        for parent_tid, daughters in divisions.items():
            lineage[parent_tid]['end_tp'] = open_tracks[parent_tid]['tp']
            lineage[parent_tid]['end_reason'] = 'divided'
            open_tracks.pop(parent_tid, None)
            for lab in daughters:
                tid = next_id
                next_id += 1
                start_daughter(tid, parent_tid, tp, lab, 'pair')
                record(tp, lab, tid, 'division',
                       dist=distances.get(lab, np.nan), prev=parent_tid,
                       parent=parent_tid, division_kind='pair',
                       mitotic=parent_tid in mitotic)

        # A daughter born without its sibling may still find it: a leftover
        # cell next to it, of similar size, within the mitosis window.
        orphans = {tid for tid, info in lineage.items()
                   if info.get('sibling_missing') and tid in open_tracks
                   and open_tracks[tid]['tp'] == tp
                   and tp - info['start_tp'] <= mitosis_window}
        if orphans and allow_division:
            free = [lab for lab in sorted(curr) if lab not in tracks[tp]]
            opts = []
            for tid in orphans:
                sib_lab = open_tracks[tid]['label']
                sa = area_of(tp, sib_lab)
                for lab in free:
                    d = float(np.hypot(curr[lab][0] - curr[sib_lab][0],
                                       curr[lab][1] - curr[sib_lab][1]))
                    a = area_of(tp, lab)
                    sym = (min(a, sa) / max(a, sa)) if a and sa else 1.0
                    if d <= mit_max and sym >= thresholds['min_symmetry']:
                        opts.append((d, tid, lab))
            used_t, used_l = set(), set()
            for d, tid, lab in sorted(opts):
                if tid in used_t or lab in used_l:
                    continue
                used_t.add(tid)
                used_l.add(lab)
                parent_tid = lineage[tid]['parent_track_id']
                new = next_id
                next_id += 1
                start_daughter(new, parent_tid, tp, lab, 'late_sibling')
                lineage[tid]['sibling_missing'] = False
                record(tp, lab, new, 'division', dist=d, prev=parent_tid,
                       parent=parent_tid, division_kind='late_sibling')

        for lab in sorted(curr):
            if lab in tracks[tp]:
                continue
            cen = curr[lab]
            records.append({'timepoint': tp, 'cell_label': lab,
                            'track_id': np.nan,
                            'match_dist': distances.get(lab, np.nan),
                            'matched_prev_label': nearest.get(lab, np.nan),
                            'dropped': True,
                            'drop_reason': dropped.get(lab, 'unmatched'),
                            'event': 'dropped', 'parent_track_id': np.nan,
                            'gap': np.nan,
                            'centroid_row': cen[0], 'centroid_col': cen[1]})

        # Retire tracks that can no longer be bridged.
        for tid, ot in list(open_tracks.items()):
            if (tp - ot['tp']) > max_gap:
                if lineage[tid]['end_reason'] == 'series_end':
                    lineage[tid]['end_reason'] = 'lost'
                open_tracks.pop(tid)

    last = timepoints[-1]
    for tid in tracks.get(last, {}).values():
        lineage[tid]['end_reason'] = 'series_end'
        lineage[tid]['end_tp'] = last
    return tracks, records, lineage


def prune_spurious_divisions(tracks, lineage, areas_per_tp=None,
                             min_overlap=3, log=None, heal_window=0,
                             heal_frac=0.75, heal_frames=3):
    """Undo "divisions" that are really one nucleus Cellpose split in half.

    A fragmented nucleus is indistinguishable from mitosis on area alone --
    two objects, each about half the parent, summing to roughly one.  Every
    geometric test passes.  What separates them is what happens NEXT: real
    daughters go on existing side by side, while a fragment heals within a
    frame or two and from then on there is only one object, which the tracker
    hands alternately to the two daughter ids.

    So the test is persistence: count the frames on which both daughters are
    present at the same time.  Below `min_overlap` the division is reverted and
    both daughters are merged back into the parent, which restores one
    continuous track.

    Measured on a 96-frame tile this separates cleanly -- spurious splits
    coexist for 1-2 frames, real divisions for 4 to 67, with nothing in
    between.

    Divisions close to the end of the series are judged against the frames that
    actually remain, so a genuine division at t95 is not thrown away for
    failing to prove itself over three frames that do not exist.

    A split can also outlast min_overlap and then heal.  The signature is then:
    one daughter is 'lost' within `heal_window` frames of the division, and
    in the next `heal_frames` frames the surviving daughter is back to at least
    `heal_frac` of the parent's median area -- one whole nucleus, not half.  A
    real daughter whose sibling was merely lost by the tracker is still about
    half the parent, so it is left alone.  heal_window=0 disables this.

    Returns (tracks, lineage, reverted) with `reverted` one dict per undone
    division.  Both inputs are modified in place as well.
    """
    timepoints = sorted(tracks)
    if not timepoints:
        return tracks, lineage, []

    presence = defaultdict(set)
    for tp, mapping in tracks.items():
        for tid in mapping.values():
            presence[tid].add(tp)

    children = defaultdict(list)
    for tid, info in lineage.items():
        par = info.get('parent_track_id')
        if par is not None:
            children[par].append(tid)

    def area_of(tp, label):
        if not areas_per_tp:
            return 0
        return areas_per_tp.get(tp, {}).get(label, 0) or 0

    def track_areas(tid, tps):
        out = []
        for tp in tps:
            for lab, t in tracks.get(tp, {}).items():
                if t == tid and area_of(tp, lab):
                    out.append(area_of(tp, lab))
        return out

    def healed(parent, a, b, t_div):
        """One daughter lost soon after the split, the other whole again."""
        if not heal_window or not areas_per_tp:
            return None
        par_areas = track_areas(parent, sorted(presence[parent]))
        if not par_areas:
            return None
        par_med = float(np.median(par_areas))
        for gone, kept in ((a, b), (b, a)):
            if (lineage[gone].get('end_reason') != 'lost' or children.get(gone)
                    or not presence[gone]
                    or max(presence[gone]) - t_div >= heal_window):
                continue
            after = sorted(t for t in presence[kept]
                           if t > max(presence[gone]))[:heal_frames]
            kept_areas = track_areas(kept, after)
            if kept_areas and np.median(kept_areas) >= heal_frac * par_med:
                return round(float(np.median(kept_areas)) / par_med, 2)
        return None

    # Latest first, so a daughter that itself divided is already resolved by
    # the time we consider merging it away.
    order = sorted((info.get('start_tp', 0), tid) for tid, info in lineage.items()
                   if lineage[tid].get('end_reason') == 'divided')
    reverted = []
    for _start, parent in reversed(order):
        kids = [k for k in children.get(parent, []) if k in lineage]
        if len(kids) != 2:
            continue
        a, b = kids
        t_div = min(lineage[a]['start_tp'], lineage[b]['start_tp'])
        overlap = len(presence[a] & presence[b])
        remaining = sum(1 for t in timepoints if t >= t_div)
        need = min(min_overlap, remaining)
        heal_ratio = None
        if overlap >= need:
            heal_ratio = healed(parent, a, b, t_div)
            if heal_ratio is None:
                continue

        # Merge both daughters back into the parent.  Where both hold a label
        # in the same frame, the larger object is the real nucleus and the
        # other is the fragment.
        for tp in sorted(presence[a] | presence[b]):
            labs = [lab for lab, tid in tracks.get(tp, {}).items() if tid in (a, b)]
            if len(labs) > 1:
                labs.sort(key=lambda L: area_of(tp, L), reverse=True)
                for extra in labs[1:]:
                    del tracks[tp][extra]
            if labs:
                tracks[tp][labs[0]] = parent
                presence[parent].add(tp)

        for grandchild in children.get(a, []) + children.get(b, []):
            if grandchild in lineage:
                lineage[grandchild]['parent_track_id'] = parent
                children[parent].append(grandchild)

        last_kid = max((a, b), key=lambda k: len(presence[k]))
        lineage[parent]['end_tp'] = max(presence[parent]) if presence[parent] \
            else lineage[parent]['end_tp']
        lineage[parent]['end_reason'] = (
            'divided' if any(g in lineage for g in children.get(parent, [])
                             if g not in (a, b))
            else lineage[last_kid]['end_reason'])
        reverted.append({'parent_track_id': parent, 'daughters': [a, b],
                         'timepoint': t_div, 'overlap_frames': overlap,
                         'required': need,
                         'reason': 'healed' if heal_ratio is not None
                         else 'overlap', 'healed_area_frac': heal_ratio})
        for k in (a, b):
            lineage.pop(k, None)
            presence.pop(k, None)
        children[parent] = [g for g in children.get(parent, []) if g not in (a, b)]

    # generation and root have to be rebuilt: merging changes depth
    for tid, info in lineage.items():
        depth, root, seen = 0, tid, set()
        while True:
            par = lineage.get(root, {}).get('parent_track_id')
            if par is None or par not in lineage or par in seen:
                break
            seen.add(root)
            root = par
            depth += 1
        info['generation'] = depth
        info['root_track_id'] = root

    if log and reverted:
        log(f'reverted {len(reverted)} spurious division(s):')
        for r in reverted[:8]:
            log('   ' + describe_reverted(r))
        if len(reverted) > 8:
            log(f'   ... and {len(reverted) - 8} more')
    return tracks, lineage, reverted


def describe_reverted(r):
    """One line for a reverted division, whichever rule undid it."""
    head = f't{r["timepoint"]} track {r["parent_track_id"]} -> {r["daughters"]}  '
    if r.get('reason') == 'healed':
        return (head + f'healed: one daughter lost, the other back to '
                f'{r["healed_area_frac"]}x parent area')
    return (head + f'coexisted {r["overlap_frames"]} frame(s), '
            f'needed {r["required"]}')


# --------------------------------------------------------------------------- #
# 4. measurement
# --------------------------------------------------------------------------- #
def background_subtract(img, mask, background, threshold):
    """Flat background subtraction; zero out mask pixels near the noise floor.

    Same behaviour as backgroundSubtract() in the original script, rewritten to
    do one pass over the whole image instead of one per cell — the per-cell loop
    dilated an image-sized array for every label, which at ~100 cells per tile
    and T timepoints dominated the runtime.
    """
    import skimage as ski

    tmp = img.astype(np.int32) - background
    tmp[tmp < 1] = 1
    noise = (tmp <= threshold) & (mask > 0)
    if np.any(noise):
        dilated = ski.morphology.binary_dilation(
            noise, footprint=ski.morphology.footprint_rectangle((5, 5)))
        mask_corrected = mask.copy()
        mask_corrected[dilated] = 0
    else:
        mask_corrected = mask.copy()
    return tmp.astype(np.uint16), mask_corrected


def _remove_small_holes(binary, area_threshold):
    """skimage renamed area_threshold -> max_size in 0.26; support both."""
    import skimage as ski

    try:
        return ski.morphology.remove_small_holes(binary, max_size=area_threshold)
    except TypeError:
        return ski.morphology.remove_small_holes(binary, area_threshold=area_threshold)


def refine_nuclei(mask_cell, dapi_binary, area_threshold=200):
    """Nucleus mask labelled by the containing cell: largest blob, holes filled.

    Same as the per-cell loop in the original script.
    """
    import skimage as ski

    mask_nuc = (dapi_binary > 0).astype(mask_cell.dtype) * mask_cell
    for cell_id in np.unique(mask_cell):
        if cell_id == 0:
            continue
        cell_dapi = (mask_nuc == cell_id)
        if not np.any(cell_dapi):
            continue
        labeled = ski.measure.label(cell_dapi)
        regions = ski.measure.regionprops(labeled)
        if not regions:
            continue
        largest = max(regions, key=lambda r: r.area)
        keep = _remove_small_holes(labeled == largest.label, area_threshold)
        mask_nuc[mask_cell == cell_id] = 0
        mask_nuc[keep] = cell_id
    return mask_nuc


def quantile_means(values, n_bins=10):
    """Mean intensity within each of n_bins intensity quantiles."""
    if values.size == 0:
        return [np.nan] * n_bins
    edges = np.percentile(values, np.linspace(0, 100, n_bins + 1))
    out = []
    for i in range(n_bins):
        if i == n_bins - 1:
            sel = (values >= edges[i]) & (values <= edges[i + 1])
        else:
            sel = (values >= edges[i]) & (values < edges[i + 1])
        out.append(float(np.mean(values[sel])) if np.any(sel) else np.nan)
    return out


def measure_cell(image_tp, mask_cell_orig, mask_cell, mask_nuc, label):
    """Every per-cell quantity the original script computed, for one timepoint.

    image_tp is [H, W, 4] for this timepoint, already registered and background
    subtracted.  Returns a flat dict with no timepoint suffix; the caller adds it.
    """
    import skimage as ski

    sel_cell_original = mask_cell_orig == label
    sel_cell = mask_cell == label
    sel_nuc = mask_nuc == label
    sel_cyto = np.bitwise_xor(sel_cell, sel_nuc)

    if not np.any(sel_nuc) or not np.any(sel_cyto) or not np.any(sel_cell):
        return None

    D, A, dA = image_tp[:, :, CH_DONOR], image_tp[:, :, CH_ACCEPTOR], image_tp[:, :, CH_MNG]
    out = {}

    out['area_cell_original'] = int(np.sum(sel_cell_original))
    out['area_cell'] = int(np.sum(sel_cell))
    out['area_cyto'] = int(np.sum(sel_cyto))
    out['area_nuc'] = int(np.sum(sel_nuc))

    for region, sel in (('cell', sel_cell), ('cyto', sel_cyto), ('nuc', sel_nuc)):
        d, a, da = D[sel], A[sel], dA[sel]
        out[f'D_{region}'] = float(np.mean(d))
        out[f'A_{region}'] = float(np.mean(a))
        out[f'directA_{region}'] = float(np.mean(da))
        out[f'D_{region}_max'] = float(np.max(d))
        out[f'A_{region}_max'] = float(np.max(a))
        out[f'directA_{region}_max'] = float(np.max(da))
        a_corr = out[f'A_{region}'] - BLEED_DIRECT * out[f'directA_{region}'] \
            - BLEED_DONOR * out[f'D_{region}']
        out[f'A_{region}_corr'] = a_corr
        denom = a_corr + out[f'D_{region}']
        out[f'Ef_{region}'] = a_corr / denom if denom != 0 else np.nan

    perim = ski.measure.perimeter(sel_cell_original)
    out['circ_cell_original'] = (
        4 * np.pi * out['area_cell_original'] / perim ** 2) if perim > 0 else np.nan

    for region, sel in (('cyto', sel_cyto), ('nuc', sel_nuc)):
        for i, v in enumerate(quantile_means(dA[sel]), start=1):
            out[f'directA_Q{i}_{region}'] = v
    return out


# --------------------------------------------------------------------------- #
# 5. wide table assembly
# --------------------------------------------------------------------------- #
def assemble_wide(measurements, meta, timepoints, lineage=None,
                  ef_delta_keys=('Ef_cell', 'Ef_cyto', 'Ef_nuc')):
    """One row per track, measurements suffixed _t1 .. _tN.

    measurements: flat list of dicts, each carrying at least 'track_id' and
                  'timepoint' alongside the measured parameters.  A flat list is
                  used rather than a nested dict because a nested dict is easy
                  to build the wrong way round, and the failure is silent.
    meta:         dict of columns shared by every track (construct, well, tile)

    Adds n_timepoints / complete_track / first_timepoint / last_timepoint, and
    Ef deltas relative to the track's own first timepoint for each later one.

    With `lineage` from track_series, also adds parent_track_id / root_track_id /
    generation / end_reason.  end_reason matters for filtering: a parent track
    that ends because it divided is not a failure, so `complete_track` alone
    would mislabel it.  Deltas are taken against the track's own first
    timepoint, which for a daughter is the division, not the start of the series.
    """
    all_t = sorted(timepoints)
    per_track = defaultdict(dict)
    for rec in measurements:
        rec = dict(rec)
        track_id = rec.pop('track_id')
        tp = rec.pop('timepoint')
        per_track[track_id][tp] = rec

    rows = []
    for track_id in sorted(per_track):
        by_tp = per_track[track_id]
        present = sorted(by_tp)
        row = dict(meta)
        row['track_id'] = track_id
        row['n_timepoints'] = len(present)
        row['first_timepoint'] = present[0] if present else np.nan
        row['last_timepoint'] = present[-1] if present else np.nan
        row['complete_track'] = len(present) == len(all_t)
        if lineage and track_id in lineage:
            info = lineage[track_id]
            row['parent_track_id'] = info['parent_track_id']
            row['root_track_id'] = info['root_track_id']
            row['generation'] = info['generation']
            row['end_reason'] = info['end_reason']
        for t in all_t:
            vals = by_tp.get(t)
            if vals is None:
                continue
            for k, v in vals.items():
                row[f'{k}_t{t}'] = v
        base_tp = present[0] if present else None
        for key in ef_delta_keys:
            base = by_tp.get(base_tp, {}).get(key) if base_tp is not None else None
            if base is None or (isinstance(base, float) and np.isnan(base)):
                continue
            for t in present[1:]:
                vals = by_tp.get(t)
                if vals is not None and key in vals:
                    row[f'{key}_delta_t{t}'] = vals[key] - base
        rows.append(row)
    return pd.DataFrame(rows)


def order_columns(df, meta_cols=('construct', 'well', 'tile', 'track_id',
                                 'parent_track_id', 'root_track_id',
                                 'generation', 'end_reason',
                                 'n_timepoints', 'complete_track',
                                 'first_timepoint', 'last_timepoint')):
    """Metadata first, then measurements grouped by parameter and ordered by t."""
    lead = [c for c in meta_cols if c in df.columns]
    rest = [c for c in df.columns if c not in lead]

    def sort_key(col):
        m = re.match(r'^(?P<base>.+)_t(?P<t>\d+)$', col)
        if m:
            return (m.group('base'), int(m.group('t')))
        return (col, -1)

    return df[lead + sorted(rest, key=sort_key)]
