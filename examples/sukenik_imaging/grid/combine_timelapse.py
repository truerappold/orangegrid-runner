"""Join per-tile timelapse-FRET results into the original's two output files.

    python combine_timelapse.py RESULTS_DIR [--into OUT_DIR]

RESULTS_DIR is the --output of the ogrun run; every <well>-<tile>_BGSTtimelapse.csv
and <well>-<tile>_match_distances.csv under it is joined, tiles in (well, tile)
order, into OUT_DIR/cells_df_BGSTtimelapse.csv (columns ordered as the original:
metadata first, then each measurement by timepoint) and
OUT_DIR/timelapse_match_distances.csv.  Floats are read back exactly
(float_precision='round_trip'), so the result is byte-identical to the original
script's output.
"""
import argparse
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('CODE_DIR', os.path.join(HERE, '..', 'analysis')))
import timelapse_tracking as tt  # noqa: E402


def key(path):
    w, t = os.path.basename(path).split('_')[0].rsplit('-', 1)
    return (w, int(t))


def read_all(files):
    parts = []
    for f in sorted(files, key=key):
        try:
            df = pd.read_csv(f, float_precision='round_trip', low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if len(df):
            parts.append(df)
    return parts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('results_dir')
    ap.add_argument('--into', default=None, help='default: RESULTS_DIR')
    a = ap.parse_args()
    out = a.into or a.results_dir
    os.makedirs(out, exist_ok=True)
    wide = read_all(glob.glob(os.path.join(a.results_dir, '**', '*_BGSTtimelapse.csv'),
                              recursive=True))
    if wide:
        cells = tt.order_columns(pd.concat(wide, ignore_index=True))
        cells.to_csv(os.path.join(out, 'cells_df_BGSTtimelapse.csv'), index=False)
        print(f'{len(wide)} tile(s), {len(cells)} tracks -> {out}/cells_df_BGSTtimelapse.csv')
    rec = read_all(glob.glob(os.path.join(a.results_dir, '**', '*_match_distances.csv'),
                             recursive=True))
    if rec:
        pd.concat(rec, ignore_index=True).to_csv(
            os.path.join(out, 'timelapse_match_distances.csv'), index=False)
        print(f'match records -> {out}/timelapse_match_distances.csv')


if __name__ == '__main__':
    main()
