"""Compare the example's combined CSV with the reference result.

    python3 check_results.py results/fret_example_cells.csv

The reference (expected/cells_df_BGSTperturbq.csv) is what the original,
whole-plate script analyzeDirPerturbBGST.py produced for these four tiles on
the lab workstation (RTX 2000 Ada GPU).

Cellpose's outlines can differ by a pixel here and there between GPU
generations, so the check allows for that: the same tiles, nearly the same
number of cells, and per-well mean FRET changes that agree closely.  A GPU of
the reference's generation (an L40S on OrangeGrid) gives identical numbers.
Only the standard library is used, so this runs with any python3.
"""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, 'expected', 'cells_df_BGSTperturbq.csv')
KEY = ('well', 'tile', 'cell_idx')
MEASURE = 'Ef_cell_delta'      # FRET efficiency after minus before, whole cell
CELL_COUNT_TOLERANCE = 0.05    # fraction of a tile's cells
MEAN_TOLERANCE = 0.005         # in Ef; cell-to-cell SD within a well is ~0.03


def read(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return {tuple(r[k] for k in KEY): r for r in rows}


def by(rows, *fields):
    out = {}
    for r in rows.values():
        out.setdefault(tuple(r[f] for f in fields), []).append(r)
    return out


def mean(rows):
    vals = [float(r[MEASURE]) for r in rows if r[MEASURE] not in ('', 'nan')]
    return sum(vals) / len(vals) if vals else float('nan')


def same(a, b):
    """Equal as text, or as numbers (1.0 vs 1)."""
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    got, ref = read(sys.argv[1]), read(REFERENCE)
    problems = []

    tiles_got, tiles_ref = by(got, 'well', 'tile'), by(ref, 'well', 'tile')
    print(f'{"tile":8s} {"cells":>14s}')
    for t in sorted(tiles_ref):
        n_ref, n_got = len(tiles_ref[t]), len(tiles_got.get(t, []))
        print(f'{"-".join(t):8s} {n_got:5d} (ref {n_ref:3d})')
        if abs(n_got - n_ref) > max(1, CELL_COUNT_TOLERANCE * n_ref):
            problems.append(f'tile {"-".join(t)}: {n_got} cells, reference {n_ref}')
    extra = sorted(set(tiles_got) - set(tiles_ref))
    if extra:
        problems.append(f'tiles not in the reference: {extra}')

    print(f'\n{"well":5s} {"construct":10s} {"mean " + MEASURE:>22s}')
    wells_got, wells_ref = by(got, 'well'), by(ref, 'well')
    for w in sorted(wells_ref):
        m_got, m_ref = mean(wells_got.get(w, [])), mean(wells_ref[w])
        print(f'{w[0]:5s} {wells_ref[w][0]["construct"]:10s} '
              f'{m_got:+.4f} (ref {m_ref:+.4f})')
        if not abs(m_got - m_ref) <= MEAN_TOLERANCE:
            problems.append(f'well {w[0]}: mean {MEASURE} {m_got:+.4f}, '
                            f'reference {m_ref:+.4f}')

    common = set(got) & set(ref)
    identical = set(got) == set(ref) and all(
        same(got[k].get(c), ref[k][c]) for k in common for c in ref[k] if c)
    print()
    if problems:
        print('FAIL')
        for p in problems:
            print(f'  {p}')
        return 1
    if identical:
        print('PASS: identical to the reference, value for value')
    else:
        print('PASS: matches the reference within GPU-to-GPU differences '
              '(cell outlines differ by a few pixels on another GPU generation)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
