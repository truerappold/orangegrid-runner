# hello_world

Counts the words in each text file in `data/`. It stands in for any real analysis:
`word_count.py INPUT OUTPUT_DIR` reads one file and writes `OUTPUT_DIR/counts.csv`.

From the repository root, on the OrangeGrid login node:

```bash
ogrun submit --name hello \
    --inputs "examples/hello_world/data/*.txt" \
    --output results/hello \
    --command "python examples/hello_world/word_count.py {input} {output}" \
    --dry-run                      # look first; then run it again without --dry-run
ogrun status ogrun_runs/hello
ogrun combine results/hello --pattern counts.csv --into results/hello_all.csv
```

Three jobs, CPU only, done in a minute or two once they start. To test without
the grid at all:

```bash
ogrun submit ... --no-submit                    # same options, minus --dry-run
ogrun run-local ogrun_runs/hello --job fox
```
