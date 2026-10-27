# Making a script grid-ready

A worked example: the lab's FRET snapshot script
[`analyzeDirPerturbBGST.py`](../sukenik_imaging/analysis/analyzeDirPerturbBGST.py)
and its grid version,
[`analyzeDirPerturbBGST_grid.py`](../sukenik_imaging/grid/analyzeDirPerturbBGST_grid.py).
Both give the same numbers. The grid version is just organised so that many
copies of it can run at once.

## Why a workstation script needs changes

The original is written for one computer. It walks through every well and
tile of a plate in one long run, and writes one CSV at the end:

```bash
python analyzeDirPerturbBGST.py /path/to/experiment/
```

On OrangeGrid you'd rather run 96 copies at once, one per well, on 96
different GPUs. Three things about the grid shape the changes:

- **Every job is a separate run of your script** on some machine you don't choose.
  The jobs don't know about each other.
- **A job can be stopped at any moment and started again from the beginning**,
  on another machine. Part of the grid is borrowed desktops, and when their
  owner comes back your job is evicted. That's normal, and it happens a lot.
- **Nobody is watching.** A job that hangs, silently does the wrong thing, or
  needs to download something from the internet just fails, often hours later.

The changes below follow from those three points. Items 1-3 are the ones every
script needs; the rest prevent the failures we actually ran into.

---

## 1. Process one piece per run, chosen on the command line

**Before:** the script decides what to do, which is everything.
```python
wells = list(set([os.path.basename(a).split('-')[0] for a in filelist]))
...
for well in wells:
    for tile in tiles:
        ...
```

**After:** the command line says which well (and optionally which tiles) to
analyse. Without `--well` it still does every well, so the same script still
works on the workstation.
```python
ap.add_argument('--well', help='well to process (e.g. A2); default all')
ap.add_argument('--tiles', type=int, nargs='*', default=None)
...
wells = [args.well] if args.well else sorted(tiles)
```

`ogrun` then makes one job per well and fills in `{well}`:
```bash
--command "python analyzeDirPerturbBGST_grid.py data/EXP --well {well} --output-dir {output}"
```

**Choosing the size of a piece:** a well (9 tiles at ~30 s each, so ~5 minutes) is
a good job size. A job much shorter than that spends most of its time starting up
(activating conda and loading Fiji and Cellpose takes a minute or more). A job of
many hours loses a lot of work when it's evicted.

## 2. Write each piece's result to its own file, in a folder you're given

**Before:** one file for the whole plate, written at the very end, into the
data folder:
```python
out_csv = os.path.join(dir, 'cells_df_BGSTperturbq.csv')
cells_df.to_csv(out_csv)
```
On the grid this breaks in two ways. 96 jobs writing the same file overwrite
each other. And if anything fails at well 90, wells 1-89 are lost too, because
nothing was saved yet.

**After:** one CSV per tile, in the folder named by `--output-dir`:
```python
def tile_csv(out_dir, well, tile):
    return os.path.join(out_dir, f'{well}-{tile}_BGSTperturbq.csv')
```
Afterwards, `ogrun combine` joins them into a file with the original's layout:
```bash
ogrun combine results/EXP --pattern "*_BGSTperturbq.csv" --no-job-column --into cells_df_BGSTperturbq.csv
```

## 3. Make being interrupted harmless

Two small habits make eviction cost nothing but time.

**Write to a temporary name, then rename.** A job killed mid-write then
leaves a `.part` file, never a half-written CSV that looks finished:
```python
tmp = path + '.part'
df.to_csv(tmp)
os.replace(tmp, path)
```

**Skip what's already done**, so a restarted job carries on where it stopped:
```python
todo = [(w, t) for w in wells for t in (args.tiles or tiles[w])
        if args.force or not os.path.exists(tile_csv(out_dir, w, t))]
```

`ogrun` adds the same at the job level. It marks a job done only when your
script exits with 0, and `ogrun resubmit` reruns only the jobs that aren't done.

## 4. Find the inputs; don't assume them

**Before:**
```python
tiles = np.arange(1,10)                                   # every well has tiles 1-9
filename = dir+'Pics/'+well+'-'+str(tile)+'_'+channel+'.vsi'   # dir must end in "/"
```
The example data in this folder has only tiles 1 and 2, so the original would
stop with an error when it tries to open tile 3. On a real plate, a single
skipped or misnamed tile does the same thing, and takes the whole run down with it.

**After:** the tiles come from the file names, and incomplete tiles are reported
and skipped instead of crashing the run:
```python
NAME_RE = re.compile(r'^(?P<after>1?)(?P<well>[A-Za-z]+\d+)-(?P<tile>\d+)_(?P<ch>.+)\.vsi$')
...
for p in glob.glob(os.path.join(pics_dir(exp_dir), '*.vsi')):
    m = NAME_RE.match(os.path.basename(p))
```
Use `os.path.join` for paths, so `data/EXP` and `data/EXP/` both work.

A `--list` option that prints what it found, without loading anything heavy, is
a cheap way to check a dataset before submitting 96 jobs:
```bash
python analyzeDirPerturbBGST_grid.py data/EXP --list
```

## 5. Load heavy things once, after reading the arguments, and fail loudly

**Before:** Cellpose is loaded when the file is imported (line 16), and again in
`__main__`. If loading fails, the error is printed and the script carries on, then
crashes later with a confusing message:
```python
model = models.CellposeModel(gpu=True)        # at import time
...
try:
    model = models.CellposeModel(gpu=True)
except:
    print("Error initializing Cellpose model. ...")
```

**After:** loaded once, inside `main()`, only after it's clear there is work to
do (so `--list`, or a job whose tiles are all done, takes a second, not two
minutes). Let errors stop the job: a non-zero exit is what tells `ogrun` and
HTCondor that the job failed.
```python
from cellpose import models
model = models.CellposeModel(gpu=True)
print(f'cellpose on GPU: {model.gpu}')
```

## 6. No internet, and a memory limit, on the compute nodes

- **Downloads:** `imagej.init('sc.fiji:fiji:2.14.0')` downloads Fiji (~600 MB)
  the first time, and Cellpose downloads its model (~1.2 GB). Compute nodes may have
  no internet, so do both once on the login node with
  [`setup/prefetch.py`](../sukenik_imaging/setup/prefetch.py). Your home folder
  is shared with every node, so the jobs find them there.
- **Java memory:** the JVM that ImageJ starts sizes itself to the whole machine,
  which on a shared node is far more than your job asked for. HTCondor may then
  kill the job. Cap it from an environment variable:
  ```python
  mem = os.environ.get('JAVA_MEM')
  if mem:
      scyjava.config.add_option(f'-Xmx{mem}')
  ```
  and pass `--env JAVA_MEM=8g --memory 16GB` to `ogrun`.

## 7. Check the GPU really works

If CUDA doesn't work on a node, Cellpose doesn't fail; it quietly runs on the CPU,
many times slower. `ogrun`'s `--gpu-check` runs a test before your script and
exits with code 3 if there's no usable GPU, and HTCondor retries the job
on another machine:
```bash
--gpu-check "python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)'"
```

## 8. Keep logs readable

Progress bars (`tqdm`, Cellpose's `progress=True`) redraw themselves hundreds of
times, which in a log file becomes thousands of lines. Turn them off and print
one line per finished piece instead:
```python
print(f'{w}-{t}: {len(rows)} cells, {time.time() - t0:.0f} s', flush=True)
```
`flush=True` makes the line appear in the job's `.out` file straight away.

## 9. Prove you didn't change the science

Run the old and new versions on the same tile on the same machine, and compare
the CSVs value by value. For this script they're identical: the same 97 columns
in the same order, and every value equal. Do this on the **same** machine: on a
different GPU generation, Cellpose's outlines can differ by a pixel here and
there (see the main README's notes on GPUs).
[`grid/test_fret_grid.py`](../sukenik_imaging/grid/test_fret_grid.py) keeps a
quick version of that check that runs without a GPU.

---

## Optional: making it faster

None of this is needed to run on the grid, but it shortens each job.
**Measure first**: time each step on one tile, and change only what is slow.
In this script, the per-cell steps grow with the number of cells and dominate
on crowded tiles:

- **Work on a crop around each cell instead of the full 2304 × 2304 frame.**
  The original dilates and indexes whole frames once per cell. The grid
  version cuts out each cell's bounding box, padded by the dilation's reach, so
  every pixel decision is the same (`background_subtract`, `clean_dapi`,
  `measure_cells`).
- **Build the table once.** `pd.concat` inside the per-cell loop copies the
  growing table for every cell. Collecting rows in a list and making one
  DataFrame at the end takes the same time no matter how many cells there are.

Keep the science untouched while you do this: the same order of operations, the
same parameters, and the check from step 9 after every change.

## The pattern, for any script

```python
def main():
    args = parse_args()                       # --well/--tiles, --output-dir, --force, --list
    pieces = find_pieces(args.directory)      # from the files, not assumed
    if args.list:
        print(pieces); return
    todo = [p for p in chosen(pieces, args)
            if args.force or not os.path.exists(out_path(args.output_dir, p))]
    if not todo:
        return                                # nothing left: exits 0, counts as done
    model = load_heavy_things()               # once, after deciding there's work
    for p in todo:
        rows = analyse(model, p)              # the original science, unchanged
        write_atomically(rows, out_path(args.output_dir, p))
```

Then `ogrun submit --inputs ... --group-by ... --command "python your_script.py ... {well} {output}"`
runs it on the grid. Try it on the example in this folder first: see
[README.md](README.md).
