# fret_snapshot: a real imaging analysis you can run

Four tiles of real microscopy data, with the lab's FRET analysis run on them as
grid jobs. Use it to see how an imaging pipeline runs under `ogrun`, and to
check your setup before running your own plate.

- **Data:** wells B4 and G1, tiles 1 and 2, from one 96-well plate of cells
  expressing FRET sensors (constructs GOOSE15 and GOOSE12). Each tile was
  imaged before and after a perturbation, in three channels: FRET (donor and
  acceptor), mNG (direct acceptor) and DAPI. That's 24 Olympus `.vsi` files,
  ~200 MB to download.
- **Analysis:** [`analyzeDirPerturbBGST_grid.py`](../sukenik_imaging/grid/analyzeDirPerturbBGST_grid.py),
  the grid version of the lab script `analyzeDirPerturbBGST.py`. Per tile it
  registers the images, finds cells with Cellpose on the GPU, subtracts
  background, finds nuclei from DAPI, and measures each cell's FRET efficiency
  before and after.
- **Jobs:** one per well (2 jobs of 2 tiles each). Once a job starts it takes about a
  minute: ~20-30 s per tile, plus loading Fiji and Cellpose. Waiting in the queue
  for a free GPU usually adds a few minutes.
- **Result:** one row per cell (174 cells, 97 columns), checked against the
  original script's output for the same tiles.

[MAKING_A_SCRIPT_GRID_READY.md](MAKING_A_SCRIPT_GRID_READY.md) explains,
step by step, how the original script was changed to run on the grid, and how to
do the same to your own script.

## Run it

**Once:** create the conda environment and download Fiji and the Cellpose
model on the login node (~20 minutes). This is the same environment as
[`../sukenik_imaging`](../sukenik_imaging):
```bash
cd ~/orangegrid-runner/examples/sukenik_imaging/setup
conda env create -f environment.yml
conda activate fucci_grid
python prefetch.py
```

**Then,** in this folder:
```bash
cd ~/orangegrid-runner/examples/fret_snapshot
bash get_data.sh              # download the 4 tiles (~200 MB) into data/
bash run_example.sh grid      # submit the 2 GPU jobs
bash run_example.sh status    # repeat until it says "all done" (usually 5-10 minutes)
bash run_example.sh combine   # join the results and check them
```

The last step ends with
```
PASS: matches the reference within GPU-to-GPU differences ...
```
or, if your jobs ran on the same GPU generation as the reference machine,
`PASS: identical to the reference, value for value`.

**Without the grid:** on any Linux computer with an NVIDIA GPU and the conda
environment, `bash run_example.sh local` runs the same two jobs one after the
other on that computer, then combines and checks them. If your environment has
another name, set it first: `CONDA_ENV=myenv bash run_example.sh local`.

## What happens

`run_example.sh` is one `ogrun submit` command; open it to see the options.
The important ones:

| option | what it does here |
|---|---|
| `--inputs "data/fret_snapshot_example/Pics/*.vsi"` | the image files |
| `--group-by "(?P<well>[A-Z]+\d+)-\d+_"` | groups files by well: `B4-1_DAPI.vsi` and `1B4-2_mNG.vsi` (the "after" image) both belong to job `B4`, and `{well}` becomes `B4` |
| `--group-size 12` | a well is complete with 12 files (2 tiles × 3 channels × before/after); incomplete wells get no job |
| `--command "python ... --well {well} --output-dir {output}"` | what each job runs |
| `--gpu --gpu-check ...` | asks for a GPU, and checks that CUDA works before starting |
| `--conda-env fucci_grid --env JAVA_MEM=8g --memory 16GB` | the environment, and a memory cap for Java |

Files it makes:
```
ogrun_runs/fret_example/       the HTCondor files and logs/B4.out, logs/G1.out
results/fret_example/B4/       B4-1_BGSTperturbq.csv, B4-2_BGSTperturbq.csv
results/fret_example/G1/       G1-1_..., G1-2_...
results/fret_example_cells.csv all cells, in the original script's layout
```

To look at one job's log: `cat ogrun_runs/fret_example/logs/B4.out`. To run
again from scratch, delete `results/` and `ogrun_runs/`.

## The check

`check_results.py` compares your combined CSV with
`expected/cells_df_BGSTperturbq.csv`, which the original, unmodified script
produced for these tiles on the lab workstation:

| tile | cells |
|---|---|
| B4-1 | 39 |
| B4-2 | 41 |
| G1-1 | 44 |
| G1-2 | 50 |

| well | construct | mean `Ef_cell_delta` (FRET efficiency after − before) |
|---|---|---|
| B4 | GOOSE15 | −0.0317 |
| G1 | GOOSE12 | −0.0005 |

Different GPU generations make Cellpose draw some cell outlines a pixel
differently, so on most grid GPUs a few values differ in their last digits. The
check allows for that (cell counts within 5%, well means within 0.005) and says
which case you got. On the lab workstation, and on the grid's L40S GPUs (the same
generation), the output is identical to the reference.

## Using it as a template

To run the same analysis on your own plate, copy the `ogrun submit` command from
`run_example.sh` and change the two paths: `--inputs` to your `Pics/*.vsi` and
the experiment folder in `--command`. A plate usually has 9 tiles per well, so
use `--group-size 54`, or leave the option out to run every well whatever it has.
The full commands for all three lab pipelines are in
[`../sukenik_imaging/README.md`](../sukenik_imaging/README.md).
