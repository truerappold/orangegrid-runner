"""Join per-job CSV outputs into one file.

Streams the files line by line with the csv module: values are copied as
text, never parsed as numbers, so every value is exactly what the job wrote
(re-reading floats with pandas' default parser can change the last digit),
and memory use stays small however large the result.  Columns are the union
over all files in first-seen order; a file lacking a column leaves it blank.
"""

from __future__ import annotations

import csv
import glob
import os
import sys


def find_csvs(root, pattern, exclude=None):
    files = glob.glob(os.path.join(root, '**', pattern), recursive=True)
    ex = os.path.abspath(exclude) if exclude else None
    return sorted(f for f in files if os.path.abspath(f) != ex
                  and not os.path.basename(f).startswith('.'))


def combine(root, pattern, out, job_column='ogrun_job', log=print):
    csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
    files = find_csvs(root, pattern, exclude=out)
    if not files:
        raise SystemExit(f'no files matching {pattern!r} under {root}')
    header, seen, empty = [], set(), []
    for f in files:
        with open(f, newline='') as fh:
            row = next(csv.reader(fh), None)
        if row is None:
            empty.append(f)
            continue
        for c in row:
            if c not in seen:
                seen.add(c)
                header.append(c)
    cols = ([job_column] if job_column and job_column not in seen else []) + header
    n_rows = 0
    tmp = out + '.part'
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(tmp, 'w', newline='') as fo:
        w = csv.writer(fo)
        w.writerow(cols)
        for f in files:
            if f in empty:
                continue
            job = os.path.relpath(os.path.dirname(f), root)
            with open(f, newline='') as fh:
                r = csv.reader(fh)
                fh_header = next(r)
                pos = {c: i for i, c in enumerate(fh_header)}
                for row in r:
                    vals = []
                    for c in cols:
                        if c == job_column and c not in pos:
                            vals.append(job)
                        else:
                            i = pos.get(c)
                            vals.append(row[i] if i is not None and i < len(row) else '')
                    w.writerow(vals)
                    n_rows += 1
    os.replace(tmp, out)
    log(f'{len(files) - len(empty)} file(s), {n_rows} rows, {len(cols)} columns -> {out}'
        + (f'  ({len(empty)} empty file(s) skipped)' if empty else ''))
    return n_rows
