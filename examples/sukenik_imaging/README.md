# Sukenik lab imaging pipelines on OrangeGrid

Three microscopy analyses, each run through `ogrun` with one GPU job per
well/tile. They read Olympus `.vsi` files through pyimagej (Fiji/Bio-Formats)
and segment with Cellpose-SAM on the GPU.

| pipeline | per-job script | one job = | validated |
|---|---|---|---|
| **NeoPIP-FUCCI** cell-cycle timelapse | `analysis/analyzeDirFUCCI.py` | one well/tile, 96 frames | grid vs lab-machine run of the same tile: same 153 tracks and 48 divisions, values equal to ~1e-13 |
| **FRET snapshot** (before/after) | `grid/analyzeDirPerturbBGST_grid.py` | one well | byte-identical CSVs to the original `analysis/analyzeDirPerturbBGST.py` |
| **FRET timelapse** | `grid/analyzeDirTimelapseBGST_grid.py` | one well/tile | byte-identical CSVs to the original `analysis/analyzeDirTimelapseBGST.py` |

`analysis/` holds the lab's analysis code (all `test_*.py` files run without a
GPU: `cd analysis && for t in test_*.py; do python $t; done`). `grid/` holds the
optimized per-job versions of the two FRET scripts, and their tests against the
originals.

## Data layout

Each experiment is one folder with a `Pics/` subfolder of `.vsi` files and,
optionally, a `seqs.csv` mapping well to construct:

```
data/MY_EXPERIMENT/
  seqs.csv                         well,construct
  Pics/A9-1_BF Cam 1.vsi           header ...
  Pics/_A9-1_BF Cam 1_/stack1/...  ... and its pixels: copy these folders too
```

## One-time setup (login node)

```bash
cd ~/orangegrid-runner/examples/sukenik_imaging/setup
conda env create -f environment.yml   # 10-20 min; jgo/scyjava are pinned (see the file)
conda activate fucci_grid
python prefetch.py                    # downloads Fiji (~600 MB) + Cellpose weights (~1.2 GB)
```

`prefetch.py` matters because compute nodes may have no internet. Jobs then find
both downloads in your (shared) home directory.

In all commands below, `$EX` is this folder and the commands run from the
folder that contains `data/`:
```bash
EX=~/orangegrid-runner/examples/sukenik_imaging
GPU_CHECK="python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)'"
```

## NeoPIP-FUCCI

One job per well/tile. The regex groups the three channel files and their
three pixel folders, so `--group-size 6` skips incomplete tiles:

```bash
ogrun submit --name fucci_MYEXP \
  --inputs "data/MYEXP/Pics/*" \
  --group-by "(?P<well>[A-Z]+\d+)-(?P<tile>\d+)_" --group-size 6 \
  --output results/fucci_MYEXP \
  --conda-env fucci_grid --gpu --gpu-check "$GPU_CHECK" \
  --cpus 4 --memory 24GB --env JAVA_MEM=8g \
  --command "python $EX/analysis/analyzeDirFUCCI.py data/MYEXP --well {well} --tile {tile} --segment-on dual --max-dist 30 --division-max-dist 60 --max-gap 6 --daughter-frac 0.30 1.10 --daughter-sum-frac 0.80 2.20 --daughter-symmetry 0.35 --fill-gaps --reuse-masks --output-dir {output}"

ogrun combine results/fucci_MYEXP --pattern "*_fucci.csv" --into results/fucci_MYEXP_all.csv
```

- **Speed:** about 1 h per tile on a mid-range GPU, faster on the grid's A40/A100s.
- **Resuming:** `--reuse-masks` makes an evicted job continue from the frames it
  already segmented (each tile's masks are cached in `{output}/<tile>_masks/`).
- **A quick test:** add `--max-frames 5` to `--command` and `--limit 1` to `submit`.
- **Output:** one row per cell per frame. Shape columns describe the brightfield
  cell; `mcherry_*`/`mvenus_*` are the nucleus, segmented or carried into frames
  where it is too dim (`nuc_source`). There are lineage columns, and
  `mitosis_no_division` flags tracks that looked mitotic but never divided,
  usually a missed daughter, so check or exclude those. `max_off_path_px` /
  `off_path_px` mark tracking slips.
- **Diagnostic plots** (lineage paths, outlines over time, MSD) come from
  `analysis/diagnose_tracks.py <tile>_fucci.csv --masks-dir <tile>_masks`.
- **Phase calls** use `PHASE_THRESHOLDS` in `analysis/fucci_pipeline.py`. Tune them
  with `analysis/reclassify_fucci.py`; no re-run is needed.

## FRET snapshot (analyzeDirPerturbBGST)

Before (`A2-1_*.vsi`) and after (`1A2-1_*.vsi`) images per tile. One job per well:

```bash
ogrun submit --name fret_MYEXP \
  --inputs "data/MYEXP/Pics/*.vsi" --group-by "(?P<well>[A-Z]+\d+)-\d+_" \
  --output results/fret_MYEXP \
  --conda-env fucci_grid --gpu --gpu-check "$GPU_CHECK" --cpus 4 --memory 16GB --env JAVA_MEM=8g \
  --command "python $EX/grid/analyzeDirPerturbBGST_grid.py data/MYEXP --well {well} --output-dir {output}"

ogrun combine results/fret_MYEXP --pattern "*_BGSTperturbq.csv" --no-job-column \
  --into results/fret_MYEXP/cells_df_BGSTperturbq.csv
```
With `--no-job-column` the combined file has the original script's layout.

## FRET timelapse (analyzeDirTimelapseBGST)

One job per well/tile. Add any option of the original script to `--command`:

```bash
ogrun submit --name fretTL_MYEXP \
  --inputs "data/MYEXP/Pics/*.vsi" --group-by "(?P<well>[A-Z]+\d+)-(?P<tile>\d+)_" \
  --output results/fretTL_MYEXP \
  --conda-env fucci_grid --gpu --gpu-check "$GPU_CHECK" --cpus 4 --memory 16GB --env JAVA_MEM=8g \
  --command "python $EX/grid/analyzeDirTimelapseBGST_grid.py data/MYEXP --well {well} --tile {tile} --output-dir {output} --max-dist 25"

python $EX/grid/combine_timelapse.py results/fretTL_MYEXP
```

`combine_timelapse.py` writes `cells_df_BGSTtimelapse.csv` and
`timelapse_match_distances.csv`, with the original's column order.

## Notes from running these

- **GPU memory:** Cellpose with batch 32 needs ~12 GB, which is why
  `--gpu-memory-mb` defaults to 16000.
- **Version pins:** unpinned conda installs `jgo` 3.x, whose Maven resolver can't
  find a Fiji dependency (`woolz:JWlz`). `environment.yml` pins the versions that
  work.
- **Out-of-focus glow:** bright, out-of-focus fluorescence can hide dim daughter
  cells from both segmentations. Those cases surface as `mitosis_no_division`
  tracks.
