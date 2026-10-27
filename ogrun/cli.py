"""ogrun -- run any analysis script over many inputs on OrangeGrid (HTCondor).

    ogrun submit    --name NAME --inputs GLOB --output DIR --command TEMPLATE [...]
    ogrun status    RUN_DIR
    ogrun resubmit  RUN_DIR            (same options; skips done and queued jobs)
    ogrun run-local RUN_DIR --job NAME (one job here, exactly as on the grid)
    ogrun combine   DIR --pattern "*.csv" --into ALL.csv

See README.md for the full guide.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

from . import __version__
from . import condor, jobs as J
from .combine import combine

SUBMIT_KEYS = ('name', 'inputs', 'inputs_file', 'group_by', 'group_size',
               'command', 'output', 'workdir', 'run_dir', 'conda_env',
               'conda_base', 'gpu', 'gpu_check', 'gpu_memory_mb', 'cuda_driver',
               'cpus', 'memory', 'disk', 'max_retries', 'exclude_machines',
               'requirements', 'env', 'min_age')


def default_conda_base():
    exe = os.environ.get('CONDA_EXE')
    if exe:
        return os.path.dirname(os.path.dirname(exe))
    for d in ('~/miniconda3', '~/anaconda3', '~/miniforge3'):
        if os.path.isdir(os.path.expanduser(d)):
            return os.path.expanduser(d)
    return ''


def add_submit_options(ap):
    g = ap.add_argument_group('what to run')
    g.add_argument('--name', required=True,
                   help='name of this run (no spaces); its files go in '
                        'ogrun_runs/NAME unless --run-dir')
    g.add_argument('--inputs', nargs='*', default=[],
                   help='glob pattern(s), quoted, e.g. "data/*.tif" or '
                        '"data/**/*.csv"; files or folders. One job per match '
                        'unless --group-by')
    g.add_argument('--inputs-file', help='text file listing inputs, one per line')
    g.add_argument('--group-by', help='regex searched in each input file name; '
                   'files with the same match form one job. Named groups '
                   '(?P<well>...) become {well} placeholders')
    g.add_argument('--group-size', type=int,
                   help='skip groups that do not have exactly this many files')
    g.add_argument('--command', required=True,
                   help='command template, quoted, run once per job. '
                        'Placeholders: {input} {inputs} {output} {name} {stem} '
                        '{dir} {index} + --group-by names. Do not put quotes '
                        'around them; values are quoted for you')
    g.add_argument('--output', required=True,
                   help='output root; each job gets OUTPUT/<job name>/ as {output}')
    g.add_argument('--workdir', default=None,
                   help='folder the command runs in (default: the current one); '
                        'relative paths in --command/--inputs are from here')
    g.add_argument('--run-dir', default=None)
    g = ap.add_argument_group('environment')
    g.add_argument('--conda-env', help='conda env activated before the command')
    g.add_argument('--conda-base', default=None,
                   help='conda install folder (default: auto, e.g. ~/miniconda3)')
    g.add_argument('--env', nargs='*', default=[], metavar='KEY=VALUE',
                   help='extra environment variables for the job')
    g = ap.add_argument_group('resources')
    g.add_argument('--gpu', action='store_true', help='request one GPU')
    g.add_argument('--gpu-check', default=None,
                   help='command that must succeed on the node before running '
                        '(default with --gpu: nvidia-smi -L). E.g. "python -c '
                        "'import torch,sys; sys.exit(not torch.cuda.is_available())'\"")
    g.add_argument('--gpu-memory-mb', type=int, default=16000,
                   help='minimum GPU memory with --gpu (default 16000; 0 = any)')
    g.add_argument('--cuda-driver', default='12.0',
                   help='minimum CUDA driver version with --gpu (default 12.0)')
    g.add_argument('--cpus', type=int, default=1)
    g.add_argument('--memory', default='4GB', help='e.g. 16GB (default 4GB)')
    g.add_argument('--disk', default='2GB', help='local scratch (default 2GB)')
    g.add_argument('--max-retries', type=int, default=3)
    g.add_argument('--exclude-machines', nargs='*', default=list(condor.DEFAULT_EXCLUDE),
                   help='machines never to use (default: %(default)s)')
    g.add_argument('--requirements', help='extra HTCondor requirements expression')
    g = ap.add_argument_group('which jobs')
    g.add_argument('--limit', type=int, default=0,
                   help='submit at most N jobs -- for a first test')
    g.add_argument('--only', nargs='*', help='only these job names')
    g.add_argument('--redo', action='store_true',
                   help='also rerun jobs that are already done')
    g.add_argument('--min-age', type=int, default=0, metavar='SECONDS',
                   help='skip jobs whose inputs changed in the last N seconds '
                        '(an upload still in progress)')
    g.add_argument('--dry-run', action='store_true',
                   help='show the jobs and commands, write nothing, submit nothing')
    g.add_argument('--no-submit', action='store_true',
                   help='write the run files but do not call condor_submit')


def config_from_args(a):
    workdir = os.path.abspath(a.workdir or os.getcwd())
    out = a.output if os.path.isabs(a.output) else os.path.join(workdir, a.output)
    run_dir = a.run_dir or os.path.join(workdir, 'ogrun_runs', a.name)
    run_dir = os.path.abspath(run_dir)
    env = {}
    for kv in a.env or []:
        if '=' not in kv:
            raise J.JobError(f'--env {kv!r}: expected KEY=VALUE')
        k, v = kv.split('=', 1)
        env[k] = v
    gpu_check = a.gpu_check
    if a.gpu and gpu_check is None:
        gpu_check = 'nvidia-smi -L > /dev/null'
    return dict(name=J.sanitize(a.name), inputs=a.inputs, inputs_file=a.inputs_file,
                group_by=a.group_by, group_size=a.group_size, command=a.command,
                output=os.path.abspath(out), workdir=workdir, run_dir=run_dir,
                conda_env=a.conda_env,
                conda_base=a.conda_base or (default_conda_base() if a.conda_env else ''),
                gpu=a.gpu, gpu_check=gpu_check or '',
                gpu_memory_mb=a.gpu_memory_mb if a.gpu else 0,
                cuda_driver=a.cuda_driver, cpus=a.cpus, memory=a.memory,
                disk=a.disk, max_retries=a.max_retries,
                exclude_machines=a.exclude_machines, requirements=a.requirements,
                env=env, min_age=a.min_age, version=__version__)


def plan(cfg):
    """(jobs, warnings) for a run configuration."""
    paths = J.find_inputs(cfg['inputs'], cfg.get('inputs_file'), cfg['workdir'])
    groups, warnings = J.group_inputs(paths, cfg.get('group_by'), cfg.get('group_size'))
    if not groups:
        raise J.JobError('no jobs: the inputs matched, but no complete group was found'
                         if paths else 'no inputs found')
    return J.build_jobs(cfg['command'], groups, cfg['output']), warnings


def do_submit(cfg, a):
    jobs, warnings = plan(cfg)
    for w in warnings[:20]:
        print(f'  note: {w}')
    if len(warnings) > 20:
        print(f'  ... and {len(warnings) - 20} more notes')
    run_dir = cfg['run_dir']
    counts = {}
    todo = []
    for j in jobs:
        state, _d = condor.job_state(run_dir, j)
        if a.only and j.name not in a.only:
            continue
        if state == 'done' and not a.redo:
            counts['already done'] = counts.get('already done', 0) + 1
            continue
        if condor.in_queue(state):
            counts[f'already {state}'] = counts.get(f'already {state}', 0) + 1
            continue
        if J.still_changing(j, cfg.get('min_age')):
            counts['inputs still changing'] = counts.get('inputs still changing', 0) + 1
            continue
        todo.append(j)
    if a.limit:
        todo = todo[:a.limit]
    cfg = dict(cfg, force=bool(a.redo))
    print(f'{len(jobs)} job(s) in run "{cfg["name"]}"; {len(todo)} to submit'
          + (' (' + ', '.join(f'{v} {k}' for k, v in counts.items()) + ')'
             if counts else ''))
    if a.dry_run:
        for j in todo[:5]:
            print(f'\n  [{j.index}] {j.name}\n      inputs : {len(j.inputs)} file(s), '
                  f'first {j.inputs[0]}\n      output : {j.output}\n      command: {j.command}')
        if len(todo) > 5:
            print(f'\n  ... and {len(todo) - 5} more')
        print('\n(dry run: nothing written or submitted)')
        return 0
    condor.write_run(run_dir, jobs, cfg, todo)
    print(f'run files: {run_dir}')
    if not todo:
        print('nothing to submit')
        return 0
    if a.no_submit:
        print('not submitted (--no-submit).  Test one job here with:\n'
              f'  ogrun run-local {run_dir} --job {todo[0].name}\n'
              f'or submit with:  condor_submit {run_dir}/job.sub')
        return 0
    cluster, out = condor.condor_submit(run_dir)
    print(out.splitlines()[-1] if out else '')
    print(f'watch it:  ogrun status {run_dir}')
    return 0


def cmd_submit(a):
    return do_submit(config_from_args(a), a)


def load_run(run_dir):
    p = os.path.join(run_dir, 'run.json')
    if not os.path.exists(p):
        raise SystemExit(f'{run_dir} is not an ogrun run directory (no run.json)')
    with open(p) as f:
        return json.load(f)


def run_jobs(run_dir, cfg):
    jobs = []
    with open(os.path.join(run_dir, 'jobs.tsv')) as f:
        for line in f:
            idx, name, out, cmd = line.rstrip('\n').split('\t', 3)
            jobs.append(J.Job(int(idx), name, [], {}, out, cmd))
    return jobs


def cmd_resubmit(a):
    run_dir = os.path.abspath(a.run_dir)
    cfg = load_run(run_dir)
    cfg = {k: cfg[k] for k in SUBMIT_KEYS if k in cfg}
    return do_submit(cfg, a)


def cmd_status(a):
    run_dir = os.path.abspath(a.run_dir)
    cfg = load_run(run_dir)
    states = {}
    problems = []
    for j in run_jobs(run_dir, cfg):
        s, d = condor.job_state(run_dir, j)
        states.setdefault(s, []).append(j.name)
        if s in ('failed', 'held'):
            problems.append((j.name, s, d))
    total = sum(len(v) for v in states.values())
    order = ['done', 'running', 'idle', 'held', 'failed', 'not submitted']
    print(f'run "{cfg["name"]}": {total} job(s)')
    for s in order:
        if s in states:
            print(f'  {s:14s} {len(states[s]):6d}')
    if problems:
        print('\nproblems (see logs/<name>.out and .err):')
        for n, s, d in problems[:a.max_list]:
            print(f'  {n}: {s}  {d}')
        if len(problems) > a.max_list:
            print(f'  ... and {len(problems) - a.max_list} more')
        print(f'\nto retry failed ones:  ogrun resubmit {run_dir}'
              + ('\nheld jobs: fix the cause, then  condor_release <id>  or '
                 'condor_rm <id>  and resubmit' if 'held' in states else ''))
    elif states.get('done') and len(states['done']) == total:
        print('\nall done.')
    return 0


def cmd_run_local(a):
    run_dir = os.path.abspath(a.run_dir)
    cfg = load_run(run_dir)
    js = run_jobs(run_dir, cfg)
    pick = [j for j in js if (a.job is None or j.name == a.job)][:1]
    if not pick:
        raise SystemExit(f'no job named {a.job!r} in {run_dir}')
    j = pick[0]
    print(f'running job {j.index} ({j.name}) on this machine -- on a login node '
          f'keep this to short tests (Good Neighbor Policy)', flush=True)
    env = dict(os.environ)
    for k, v in (cfg.get('env') or {}).items():
        env[k] = v
    if a.force:
        env['OGRUN_FORCE'] = '1'
    return subprocess.call(['bash', os.path.join(run_dir, 'run_job.sh'), str(j.index)],
                           env=env)


def cmd_combine(a):
    out = a.into or os.path.join(a.dir, 'combined.csv')
    combine(a.dir, a.pattern, out, job_column=None if a.no_job_column else 'ogrun_job')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog='ogrun', description=__doc__.split('\n')[1])
    ap.add_argument('--version', action='version', version=f'ogrun {__version__}')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('submit', help='make jobs from inputs and submit them')
    add_submit_options(p)
    p.set_defaults(fn=cmd_submit)

    p = sub.add_parser('resubmit', help='submit again with a run\'s saved options '
                       '(picks up new inputs; skips done and queued jobs)')
    p.add_argument('run_dir')
    for flag in ('--redo', '--dry-run', '--no-submit'):
        p.add_argument(flag, action='store_true')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--only', nargs='*')
    p.set_defaults(fn=cmd_resubmit)

    p = sub.add_parser('status', help='done / running / idle / held / failed per job')
    p.add_argument('run_dir')
    p.add_argument('--max-list', type=int, default=20)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser('run-local', help='run one job on this machine, as on the grid')
    p.add_argument('run_dir')
    p.add_argument('--job', help='job name (default: the first)')
    p.add_argument('--force', action='store_true', help='run even if already done')
    p.set_defaults(fn=cmd_run_local)

    p = sub.add_parser('combine', help='join per-job CSVs into one (values copied exactly)')
    p.add_argument('dir', help='output root of a run')
    p.add_argument('--pattern', default='*.csv', help='file pattern (default *.csv)')
    p.add_argument('--into', help='combined file (default DIR/combined.csv)')
    p.add_argument('--no-job-column', action='store_true',
                   help='do not add the ogrun_job column (the job each row came from)')
    p.set_defaults(fn=cmd_combine)

    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except (J.JobError, ValueError, RuntimeError) as exc:
        print(f'ogrun: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
