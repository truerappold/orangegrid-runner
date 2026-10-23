# orangegrid-runner (`ogrun`)

Run **any** analysis script over **many** input files on Syracuse University's
[OrangeGrid](https://researchcomputing.syr.edu/resources/orange-grid/) (HTCondor),
from one command line:

```bash
ogrun submit --name myrun \
    --inputs "data/*.tif" \
    --output results/myrun \
    --command "python analyze.py {input} --out {output}"
```

That makes one job per input file, writes the HTCondor files, and submits
them. Every job runs your command on a grid node with `{input}` and `{output}`
filled in. Your script needs no changes, as long as it can be told on the command
line what to read and where to write.

`ogrun` handles the parts of running on a shared grid that are easy to get wrong:

- **Resubmitting** only redoes jobs that failed or never ran. A job counts as done
  only when its command exited successfully.
- **No double submission.** Jobs that are already queued or running are never submitted twice.
- **Evictions** (OrangeGrid reclaims scavenged desktops) and failed attempts are
  retried automatically.
- **GPU jobs** check the GPU works before starting; a broken node means a retry
  elsewhere, not hours on the CPU. A known-bad GPU node is excluded by default.
- **conda** environments are activated the way HTCondor jobs need.
- **Uploads in progress:** `--min-age` skips inputs that are still being copied.
- **`ogrun status`** shows done / running / idle / held / failed, with the hold reason or exit code.
- **`ogrun combine`** joins per-job CSVs into one, copying values exactly (no float rounding).

`ogrun` itself is plain Python (3.8+, standard library only), so it runs on the
login node with no setup beyond cloning this repository.

Contents: [Install](#install) · [Quick test](#quick-test) ·
[Writing the command](#writing-the-command) · [Grouping files into jobs](#grouping-files-into-jobs) ·
[Resources and GPUs](#resources-and-gpus) · [Testing before a big run](#testing-before-a-big-run) ·
[Watching, fixing, resubmitting](#watching-fixing-resubmitting) · [Collecting results](#collecting-results) ·
[Moving data on and off the grid](#moving-data-on-and-off-the-grid) · [Software environments](#software-environments) ·
[Troubleshooting](#troubleshooting) · [Examples](#examples) · [How it works](#how-it-works)

---

## Install

You need an OrangeGrid account (ask [Research Computing](https://researchcomputing.syr.edu/contact/)),
the SU VPN when off campus, and your login node, `its-og-loginN.syr.edu`.

On the login node:
```bash
git clone https://github.com/truerappold/orangegrid-runner.git ~/orangegrid-runner
echo 'export PATH="$HOME/orangegrid-runner/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
ogrun --version
```
(`pip install --user -e ~/orangegrid-runner` works too and gives the same `ogrun` command.)

---

## Quick test

> **This is only a test that the tool works on the grid; it is not a real
> analysis.** [`examples/quick_test`](examples/quick_test) has three tiny text
> files (a sentence or two each) and a 15-line script that counts their words.
> No images, GPU or conda needed, so in two minutes you know the grid side
> works before trying your own script. For a real pipeline, see
> [`examples/sukenik_imaging`](examples/sukenik_imaging).

```bash
cd ~/orangegrid-runner
ogrun submit --name quick_test \
    --inputs "examples/quick_test/data/*.txt" \
    --output results/quick_test \
    --command "python examples/quick_test/word_count.py {input} {output}"
ogrun status ogrun_runs/quick_test            # until everything is "done"
ogrun combine results/quick_test --pattern counts.csv --into results/quick_test_all.csv
```

What happened:

1. `submit` found 3 inputs and made 3 jobs, named after the files (`cells`, `fox`, `hamlet`).
2. It wrote the run's files to `ogrun_runs/quick_test/` and submitted them.
3. Each job ran `word_count.py` on its file, writing to `results/quick_test/<job>/`.
4. `combine` joined the three `counts.csv` files, adding an `ogrun_job` column.

---

## Writing the command

`--command` is your command line with placeholders. It runs once per job, in
`--workdir` (by default the folder you ran `ogrun` from), so relative paths work.

| placeholder | becomes |
|---|---|
| `{input}` | the job's input path (with grouping: its first file) |
| `{inputs}` | all of the job's input paths, space-separated |
| `{output}` | the job's own output folder, `<--output>/<job name>/` (created for you) |
| `{name}` | the job name: the input's name without extension, or the `--group-by` match |
| `{stem}`, `{dir}` | the first input's file name without extension / its folder |
| `{index}` | 0, 1, 2, ... |
| `{anything}` | a named group from `--group-by`, e.g. `{well}` |

**Don't put quotes around placeholders.** Values are quoted for you, so a path
like `A9-1_BF Cam 1.vsi` with spaces just works:
`--command "python a.py {input}"`, **not** `"python a.py '{input}'"`.

Write the whole `--command` in one pair of double quotes, and use single quotes
inside it if your command needs them. Run `ogrun submit ... --dry-run` to see the
exact commands that would run.

If your script writes into its current folder instead of taking an output path,
`cd` into the job's folder first:
`--command "cd {output} && python $PWD/analyze.py {input}"`.

---

## Grouping files into jobs

By default each input path is one job. An input can be a folder, too:
`--inputs "data/*/"` gives one job per folder.

When one job needs several files (all channels of one image, both ends of a
sequencing read), `--group-by` takes a regular expression. Files whose names
give the same match go into one job:

```bash
# A9-1_BF Cam 1.vsi, A9-1_mCherry.vsi, A9-1_mVenus.vsi  ->  one job "A9-1"
--inputs "Pics/*.vsi" --group-by "(?P<well>[A-Z]+\d+)-(?P<tile>\d+)_" --group-size 3
```

- **Named groups become placeholders and the job name.** `(?P<well>...)` gives
  `{well}`, and the name is the groups joined by `-` (here `A9-1`).
- **A regex without named groups** uses its captured text as the name.
- **`--group-size N`** skips groups that don't have exactly N files (incomplete
  data), with a note. Files that don't match the regex are ignored, also with a note.
- **Hidden files** (names starting with `.`: rsync temporaries, `.DS_Store`) are
  never inputs.

If your script takes a folder plus an identifier rather than file paths, use the
groups only to make the jobs and pass the identifiers:
`--command "python analyze.py data/EXP --well {well} --tile {tile} --out {output}"`.

`--inputs` accepts several patterns, and `**` recurses (`"data/**/*.csv"`).
`--inputs-file list.txt` reads one path per line.

---

## Resources and GPUs

| option | default | meaning |
|---|---|---|
| `--cpus` | 1 | cores; more only helps if your script uses them |
| `--memory` | 4GB | RAM per job. Check the real peak afterwards in `ogrun_runs/<run>/logs/<job>.log` (`tail -n 9`) |
| `--disk` | 2GB | local scratch; your files live on the shared home, so this rarely matters |
| `--gpu` | off | request one GPU, check it works before running (exit 3 + retry if not) |
| `--gpu-check CMD` | `nvidia-smi -L` | the check, e.g. `"python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)'"` |
| `--gpu-memory-mb` | 16000 | minimum GPU memory; 0 = any |
| `--cuda-driver` | 12.0 | minimum CUDA driver version |
| `--max-retries` | 3 | retries after a failure |
| `--exclude-machines` | `CRUSH-NODE-10-5-5-5` | machines never to use |
| `--requirements EXPR` | | any extra HTCondor requirement |
| `--env KEY=VALUE ...` | | environment variables for the job |
| `--conda-env NAME` | | conda env activated first (see [Software environments](#software-environments)) |

GPUs are the scarcest resource on the grid, so request one only if your script
uses it.

---

## Testing before a big run

1. **`--dry-run`** shows how many jobs, and the first few exact commands.
   Nothing is written or submitted.
2. **`--no-submit`, then `ogrun run-local RUN_DIR --job NAME`** runs one job on the
   current machine through the very same `run_job.sh` the grid uses. On the login
   node, keep this to a few minutes (the
   [Good Neighbor Policy](https://docs.syr.edu/ResearchComputing/good-neighbor-policy.html)).
3. **`--limit 2`** submits just two jobs to the grid. Check their output, then run
   the same `submit` without `--limit`: the two finished ones are skipped.

Many analysis scripts have their own "small test" option (e.g. a frame limit).
Put it in `--command` for the test run, then submit again without it. Delete the
test outputs first, or use `--redo`, because those jobs are already done.

---

## Watching, fixing, resubmitting

```bash
ogrun status ogrun_runs/myrun      # counts per state, and why held/failed jobs are
condor_q                           # HTCondor's view of your queue
tail -f ogrun_runs/myrun/logs/<job>.out
```

| state | meaning | what to do |
|---|---|---|
| idle | waiting for a free machine | normal, especially for GPU jobs. If it stays idle for hours: `condor_q -better-analyze <id>` |
| running | on a node now | |
| held | HTCondor stopped it; `status` prints the reason | fix it, then `condor_rm <id>` and resubmit |
| failed | the command exited non-zero | read `logs/<job>.err` / `.out`, fix, resubmit |
| done | finished successfully (`<output>/<job>/.ogrun_done` exists) | |

**Resubmit:** `ogrun resubmit ogrun_runs/myrun` reruns the same `submit` with the
saved options. It re-scans the inputs, so new files become jobs, and it submits only
jobs that are neither done nor queued. `--redo` reruns finished jobs too.

**Cancel:** `condor_rm <cluster id>` (shown at submit) or `condor_rm $USER` (all of yours).

---

## Collecting results

Every job writes into its own `<--output>/<job name>/` folder. To join CSVs:

```bash
ogrun combine results/myrun --pattern "*_summary.csv" --into results/myrun_all.csv
```

- **Exact text:** values are copied as text, so numbers come out exactly as the jobs
  wrote them.
- **Column union:** columns are the union over all files.
- **Job column:** an `ogrun_job` column records each row's job (`--no-job-column` leaves it out).

It streams, so a combined file of many GB is fine. For any other format, the
per-job folders are yours to process however you like.

---

## Moving data on and off the grid

Your grid home directory, about 4 TB with `quota -s` showing the exact amount,
is visible from every node, so jobs read and write there directly. You copy data
in and results out with `rsync`, **run on your own computer** (the one that has
the files), not on the grid:

```bash
# upload: one line, no backslash line breaks (a "\" followed by a space breaks the command)
rsync -avP "/path/to/MY DATA/" netid@its-og-loginN.syr.edu:data/my_data/

# download results
rsync -avP netid@its-og-loginN.syr.edu:orangegrid-runner/results/myrun ~/Downloads/
```

- **Interrupted?** Run the same `rsync` again; it continues where it stopped.
- **Long uploads:** set up an SSH key once, so retries don't wait for a password,
  then let it retry by itself:
  ```bash
  ssh-keygen -t ed25519                          # Enter at every question: no passphrase
  ssh-copy-id netid@its-og-loginN.syr.edu        # your password, one last time
  until rsync -avP "SRC" netid@its-og-loginN.syr.edu:DEST; do sleep 60; done
  ```
  On a Mac, prefix `caffeinate -i` so it doesn't sleep. A wired campus connection
  is usually much faster than VPN over Wi-Fi.
- **Folder-based formats:** some formats keep the pixels in a folder next to a
  small header file (Olympus `.vsi` → `_name_/`). Copy the folders too.
- **Starting before the upload finishes:** `--min-age 600` skips jobs whose inputs
  changed in the last 10 minutes. Make the glob also cover any data folders, so
  they are checked too.
- **Clean up:** delete raw data on the grid once its results are safe.

---

## Software environments

Jobs run non-interactively, so `ogrun` activates conda for them
(`--conda-env NAME`; conda is found automatically, or set `--conda-base`). Set the
environment up once on the login node:

```bash
# Miniconda, if you don't have it:
#   https://docs.conda.io/en/latest/miniconda.html  (install to ~/miniconda3)
~/miniconda3/bin/conda init bash && source ~/.bashrc
conda env create -f environment.yml
```

- **Pin versions.** Unpinned, conda picks the newest release of everything, which
  may not behave like the versions you tested locally (see
  [Troubleshooting](#troubleshooting)).
- **No internet on nodes.** Compute nodes may have no internet access. Anything your
  code downloads on first use, such as model weights, Maven or Java libraries, or
  NLTK data, must be downloaded once on the login node first. Your home directory
  is shared, so jobs then find it.
- **Containers:** HTCondor on OrangeGrid can also run them. See Research
  Computing's [docs](https://docs.syr.edu/ResearchComputing/) if conda isn't enough.

---

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `conda: command not found` after login | `~/miniconda3/bin/conda init bash`, then `source ~/.bashrc`. If it recurs on every login, add `[ -f ~/.bashrc ] && . ~/.bashrc` to `~/.bash_profile` |
| `ModuleNotFoundError` when *you* run a script | the env isn't active (prompt says `(base)`): `conda activate NAME` |
| `ssh: connect ... Operation timed out` | VPN not connected (or just dropped) |
| `rsync` errors with `No such file` paths that look like your computer's | you ran it on the grid; run it on the computer that has the files |
| `No space left on device` during upload | check which disk: a broken command line can make `rsync` copy locally. On the grid: `quota -s` |
| job **held**: `Permission denied` | a script isn't executable. `ogrun` runs everything through bash, so this only happens with your own submit files |
| job **failed**, exit 3 | the GPU check failed on that node. It is retried elsewhere; if every attempt fails, check `--gpu-check` |
| all jobs **idle** for hours | `condor_q -better-analyze <id>`: if no machine matches, relax `--gpu-memory-mb` / `--memory` |
| job fails instantly | read `ogrun_runs/<run>/logs/<job>.err`; usually a path or the env |
| a library fetches things at start-up and fails | no internet on nodes; pre-download on the login node |
| results differ slightly between runs | different GPU models can differ in the last digits. Compare against your local run |
| `jgo`/`scyjava` errors, `Artifact ... not found` (pyimagej) | pin `jgo=1.0.6 scyjava=1.12.0` (see `examples/sukenik_imaging/setup/environment.yml`) |

More help: `condor_q -better-analyze <id>` for idle jobs, `condor_history -l <id>`
for finished ones, and **researchcomputing@syr.edu**.

---

## Examples

- [`examples/quick_test`](examples/quick_test): **not a real analysis**, just a
  two-minute check that `ogrun` works on the grid (counts the words in three
  tiny text files; CPU only). Run it once before your own data.
- [`examples/sukenik_imaging`](examples/sukenik_imaging): three real
  microscopy pipelines used in the Sukenik lab. Each was validated by running the
  same images through the original script and comparing results; see that
  folder's README.
  - **NeoPIP-FUCCI cell-cycle timelapse:** Cellpose on brightfield plus fluorescence,
    tracking with divisions, GPU, one job per well/tile. Used for a full 96-well
    plate: 192 tiles, ~500 GB of images.
  - **FRET snapshot (before/after):** an optimized version of
    `analyzeDirPerturbBGST.py`, with byte-identical output.
  - **FRET timelapse:** an optimized version of `analyzeDirTimelapseBGST.py`, with
    byte-identical output.

---

## How it works

```
ogrun submit
  find inputs ──► group into jobs ──► fill in --command per job
       │
       ▼
  ogrun_runs/<run>/
    run.json        options (for resubmit)
    jobs.tsv        every job: index, name, output folder, command
    queue.txt       the jobs of this submission (not done, not queued)
    run_job.sh      what a node runs for one job
    job.sub         HTCondor submit file: executable = /bin/bash run_job.sh <index>
    logs/<job>.out .err .log
       │ condor_submit
       ▼
  on a grid node, run_job.sh:
    skip if <output>/<job>/.ogrun_done exists
    activate conda ─► check GPU (exit 3 → retry elsewhere) ─► cd workdir
    run the command ─► exit 0 ? write .ogrun_done : leave it unmarked
```

A job's state comes from its done marker plus its HTCondor event log. That's how
`status` can tell idle, running, held and failed apart, and how `resubmit` avoids
duplicates.

---

## License

MIT -- free for anyone to use, copy, change and share, for any purpose; see
[LICENSE](LICENSE).

## Acknowledgements

Built on the patterns in the Sukenik lab's
[IntroResearchComputing](https://github.com/sukeniklab/IntroResearchComputing)
guide. Please acknowledge the Syracuse University HTC Campus Grid and NSF award
ACI-1341006 in publications that use OrangeGrid.
