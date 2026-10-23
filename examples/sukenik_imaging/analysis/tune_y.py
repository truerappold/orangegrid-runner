"""
Sweep the tracking cutoff y offline, against distances already computed.

Cellpose is the expensive part of the pipeline, and it does not depend on y at
all — so there is no reason to re-segment when trying a different cutoff. Both
driver scripts write `timelapse_match_distances.csv` containing, for every cell
at every timepoint, the distance to its nearest nuclear centroid in the previous
timepoint. This reads that file and reports what each candidate y would keep.

    python tune_y.py path/to/timelapse_match_distances.csv
    python tune_y.py path/to/timelapse_match_distances.csv --plot out.png

What to look for: raising y should sharply increase cells kept up to a point and
then flatten. The flattening is where true matches are exhausted and further
increases only admit spurious ones — the collision count usually starts climbing
there too. Pick y at the knee, then confirm on a couple of tiles with
maskPrint_timelapse.py.

Caveat: this re-thresholds the *distance* rule exactly, but it can only
approximate the collision rule, because which cells collide depends on which
survived the distance cut at the previous timepoint, and that cascades. Counts
here are therefore a close guide, not the exact result — the driver is the
authority.
"""

import argparse

import numpy as np
import pandas as pd


def sweep(df, cutoffs, dist_col='match_dist'):
    """For each candidate y, count what the distance rule keeps and what collides."""
    later = df[df['timepoint'] > df['timepoint'].min()].copy()
    rows = []
    for y in cutoffs:
        within = later[later[dist_col] <= y]
        # approximate the collision rule: cells sharing a nearest previous cell
        if 'matched_prev_label' in within.columns:
            grp = within.dropna(subset=['matched_prev_label']).groupby(
                ['well', 'tile', 'timepoint', 'matched_prev_label']).size()
            n_collide = int(grp[grp > 1].sum())
        else:
            n_collide = np.nan
        rows.append({
            'y_px': y,
            'cells_within_cutoff': len(within),
            'pct_of_cells': 100 * len(within) / max(len(later), 1),
            'approx_collisions': n_collide,
            'approx_kept': len(within) - (n_collide if np.isfinite(n_collide) else 0),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description='Sweep the tracking cutoff y offline.')
    ap.add_argument('distances_csv')
    ap.add_argument('--min', type=float, default=1.0)
    ap.add_argument('--max', type=float, default=80.0)
    ap.add_argument('--step', type=float, default=2.0)
    ap.add_argument('--use-cell-centroid', action='store_true',
                    help='sweep on the whole-cell centroid distance instead of '
                         'the nuclear one, to compare which separates better')
    ap.add_argument('--plot', default=None, help='write a PNG of the sweep here')
    args = ap.parse_args()

    df = pd.read_csv(args.distances_csv)
    dist_col = 'match_dist_cell' if args.use_cell_centroid else 'match_dist'
    if dist_col not in df.columns:
        raise SystemExit(f'{dist_col} not in {args.distances_csv}')
    df = df.replace([np.inf, -np.inf], np.nan)

    later = df[df['timepoint'] > df['timepoint'].min()]
    d = later[dist_col].dropna()
    print(f'{len(df)} records, {len(later)} of them matchable '
          f'(timepoint 2 onward), {len(d)} with a finite distance')
    if d.empty:
        raise SystemExit('No finite distances to sweep.')
    print(f'distance percentiles (px): ' + '  '.join(
        f'{p}%={d.quantile(p / 100):.1f}' for p in (5, 25, 50, 75, 90, 95, 99)))

    # Per-step breakdown.  The basal frame and the first timelapse frame are
    # usually separated by a much longer interval than consecutive frames inside
    # the stack (~110 s vs ~35 s in the example data), so the t1->t2 step can
    # drift further than every step after it.  If the t2 row below stands out,
    # a single global y is the wrong model — see --first-step-max-dist.
    per_step = (later.groupby('timepoint')[dist_col]
                .agg(n='size', median='median', p90=lambda v: v.quantile(.90),
                     p99=lambda v: v.quantile(.99)).round(1))
    print('\nper-step distance to the previous timepoint (px):')
    print(per_step.to_string())
    if len(per_step) > 1:
        first, rest = per_step.iloc[0], per_step.iloc[1:]
        if np.isfinite(first['p90']) and first['p90'] > 1.5 * rest['p90'].median():
            print(f"  NOTE: the first step drifts ~"
                  f"{first['p90'] / max(rest['p90'].median(), 1e-9):.1f}x further "
                  f"than later steps — consider --first-step-max-dist on the drivers")

    cutoffs = np.arange(args.min, args.max + args.step, args.step)
    table = sweep(df, cutoffs, dist_col=dist_col)
    print()
    print(table.to_string(index=False, float_format=lambda v: f'{v:.1f}'))

    # the knee: where the gain per extra pixel of y falls below 10% of its peak
    gain = np.gradient(table['approx_kept'].values, table['y_px'].values)
    if np.nanmax(gain) > 0:
        knee_idx = int(np.argmax(gain < 0.1 * np.nanmax(gain)))
        print(f'\nsuggested y ~ {table["y_px"].iloc[knee_idx]:.0f} px '
              f'(where added cells per extra px drops below 10% of peak)')
        print('confirm it on a couple of tiles with maskPrint_timelapse.py '
              'before running the whole plate')

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(table['y_px'], table['approx_kept'], 'o-', color='#2471a3',
                label='cells kept (approx)')
        ax.plot(table['y_px'], table['approx_collisions'], 's-', color='#c0392b',
                label='collisions (approx)')
        ax.set_xlabel('y, cutoff on centroid movement (px)')
        ax.set_ylabel('cells')
        ax.legend(frameon=False)
        ax.grid(alpha=.15)
        ax.set_title('Raising y past the knee admits spurious matches, not real ones')
        fig.tight_layout()
        fig.savefig(args.plot, dpi=180)
        print(f'wrote {args.plot}')


if __name__ == '__main__':
    main()
