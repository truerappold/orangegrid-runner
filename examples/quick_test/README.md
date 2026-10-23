# quick_test

**This is not a real analysis -- it only checks that `ogrun` works on the
grid.** The three files in `data/` are a sentence or two of text each
(`fox.txt`, `hamlet.txt`, `cells.txt`), and `word_count.py` counts their words.
That stands in for any real script: it takes an input file and an output
folder on the command line, which is all `ogrun` needs. No images, GPU or
conda, so it runs in a minute or two; if it works, the grid side is fine and
you can move on to your own script (see `../sukenik_imaging` for a real one).

From the repository root, on the OrangeGrid login node:

```bash
ogrun submit --name quick_test \
    --inputs "examples/quick_test/data/*.txt" \
    --output results/quick_test \
    --command "python examples/quick_test/word_count.py {input} {output}" \
    --dry-run                      # look first; then run it again without --dry-run
ogrun status ogrun_runs/quick_test
ogrun combine results/quick_test --pattern counts.csv --into results/quick_test_all.csv
```

You get three jobs (`cells`, `fox`, `hamlet`), each writing
`results/quick_test/<job>/counts.csv`, and one combined CSV. To test without
the grid at all:

```bash
ogrun submit ... --no-submit                    # same options, minus --dry-run
ogrun run-local ogrun_runs/quick_test --job fox
```
