#!/bin/bash
# Run the FRET snapshot example: one job per well, then combine.
#
#   bash run_example.sh grid     # on the OrangeGrid login node: submit 2 GPU jobs
#   bash run_example.sh local    # on any machine with a GPU: run the same 2 jobs here
#   bash run_example.sh status   # how the grid jobs are doing
#   bash run_example.sh combine  # once they're done: join the per-tile CSVs and check them
#
# CONDA_ENV (default fucci_grid) is the env from ../sukenik_imaging/setup.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OGRUN="$HERE/../../bin/ogrun"          # works without installing ogrun
CONDA_ENV=${CONDA_ENV:-fucci_grid}
MODE=${1:-}
cd "$HERE"                             # all paths below are relative to this folder

if [ ! -f data/fret_snapshot_example/seqs.csv ] && { [ "$MODE" = grid ] || [ "$MODE" = local ]; }; then
    echo "no images yet: run  bash get_data.sh  first"
    exit 1
fi

submit() {   # the whole ogrun call; "$@" adds --no-submit for local runs
    "$OGRUN" submit --name fret_example \
        --inputs "data/fret_snapshot_example/Pics/*.vsi" \
        --group-by "(?P<well>[A-Z]+\d+)-\d+_" --group-size 12 \
        --output results/fret_example \
        --conda-env "$CONDA_ENV" --gpu \
        --gpu-check "python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)'" \
        --cpus 4 --memory 16GB --env JAVA_MEM=8g \
        --command "python ../sukenik_imaging/grid/analyzeDirPerturbBGST_grid.py data/fret_snapshot_example --well {well} --output-dir {output}" \
        "$@"
}

case "$MODE" in
    grid)
        submit
        echo "watch with  bash run_example.sh status ; when it says all done:  bash run_example.sh combine"
        ;;
    local)
        submit --no-submit
        for well in B4 G1; do
            "$OGRUN" run-local ogrun_runs/fret_example --job "$well"
        done
        bash "$HERE/run_example.sh" combine
        ;;
    status)
        "$OGRUN" status ogrun_runs/fret_example
        ;;
    combine)
        "$OGRUN" combine results/fret_example --pattern "*_BGSTperturbq.csv" \
            --no-job-column --into results/fret_example_cells.csv
        python3 check_results.py results/fret_example_cells.csv
        ;;
    *)
        echo "usage: bash run_example.sh grid | local | status | combine"
        exit 2
        ;;
esac
