"""Tests for ogrun -- no HTCondor needed.  Run:  python -m unittest discover tests"""

import csv
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from ogrun import condor, jobs as J  # noqa: E402
from ogrun.cli import main  # noqa: E402
from ogrun.combine import combine  # noqa: E402


def run_cli(*argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(list(argv))
    return code, buf.getvalue()


class Tmp(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix='ogrun_test_')
        self.old = os.getcwd()
        os.chdir(self.d)

    def tearDown(self):
        os.chdir(self.old)
        shutil.rmtree(self.d)

    def touch(self, rel, text='x'):
        p = os.path.join(self.d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w') as f:
            f.write(text)
        return p


class Grouping(Tmp):
    def test_one_job_per_file(self):
        for n in ('a.txt', 'b.txt'):
            self.touch(f'data/{n}')
        paths = J.find_inputs(['data/*.txt'], workdir=self.d)
        groups, warn = J.group_inputs(paths)
        self.assertEqual([g[1] for g in groups], ['a', 'b'])
        self.assertEqual(warn, [])

    def test_hidden_files_ignored(self):
        self.touch('data/a.txt')
        self.touch('data/.a.txt.Xyz12')          # an rsync temporary
        self.assertEqual(len(J.find_inputs(['data/*'], workdir=self.d)), 1)

    def test_microscope_tiles_grouped_and_incomplete_skipped(self):
        # three channels per tile, file names with spaces, an 'after' prefix
        for tile in ('A9-1', 'A9-2'):
            for ch in ('BF Cam 1', 'mCherry', 'mVenus'):
                self.touch(f'Pics/{tile}_{ch}.vsi')
        self.touch('Pics/B3-1_BF Cam 1.vsi')     # incomplete tile
        paths = J.find_inputs(['Pics/*.vsi'], workdir=self.d)
        groups, warn = J.group_inputs(paths, r'(?P<well>[A-Z]+\d+)-(?P<tile>\d+)_', 3)
        self.assertEqual([g[1] for g in groups], ['A9-1', 'A9-2'])
        self.assertEqual(groups[0][0], {'well': 'A9', 'tile': '1'})
        self.assertTrue(any('B3-1' in w for w in warn))

    def test_unnamed_group_and_no_match(self):
        for n in ('s1_r1.csv', 's1_r2.csv', 's2_r1.csv', 'notes.md'):
            self.touch(f'd/{n}')
        paths = J.find_inputs(['d/*'], workdir=self.d)
        groups, warn = J.group_inputs(paths, r'(s\d+)_')
        self.assertEqual([(g[1], len(g[2])) for g in groups], [('s1', 2), ('s2', 1)])
        self.assertTrue(any('did not match' in w for w in warn))

    def test_glob_that_matches_nothing_is_an_error(self):
        with self.assertRaises(J.JobError):
            J.find_inputs(['nothing/*.tif'], workdir=self.d)


class Rendering(unittest.TestCase):
    def test_values_are_shell_quoted(self):
        cmd = J.render('python a.py {input} --out {output}',
                       {'input': '/x/A9-1_BF Cam 1.vsi', 'output': '/r/A9-1'})
        self.assertEqual(cmd, "python a.py '/x/A9-1_BF Cam 1.vsi' --out /r/A9-1")

    def test_unknown_placeholder_lists_the_available_ones(self):
        with self.assertRaises(J.JobError) as e:
            J.render('python a.py {inpt}', {'input': 'x'})
        self.assertIn('{input}', str(e.exception))

    def test_duplicate_names_rejected(self):
        groups = [({}, 'a', ['/1/a.txt']), ({}, 'a', ['/2/a.txt'])]
        with self.assertRaises(J.JobError):
            J.build_jobs('echo {input}', groups, '/out')

    def test_inputs_placeholder_lists_every_file(self):
        jobs = J.build_jobs('cat {inputs}', [({}, 'g', ['/d/a b.txt', '/d/c.txt'])], '/o')
        self.assertEqual(jobs[0].command, "cat '/d/a b.txt' /d/c.txt")


class EndToEnd(Tmp):
    """Write a run without submitting, then run its jobs here with run_job.sh."""

    def make_run(self, *extra):
        for n, text in (('fox', 'the fox and the dog'), ('cat', 'a cat')):
            self.touch(f'data/{n}.txt', text)
        self.touch('count.py', 'import sys,os\n'
                   'os.makedirs(sys.argv[2], exist_ok=True)\n'
                   'n = len(open(sys.argv[1]).read().split())\n'
                   'open(os.path.join(sys.argv[2], "n.csv"), "w").write(f"file,n\\n{sys.argv[1]},{n}\\n")\n'
                   'sys.exit(1 if "cat" in sys.argv[1] and os.environ.get("FAIL_CAT") else 0)\n')
        return run_cli('submit', '--name', 'wc', '--inputs', 'data/*.txt',
                       '--output', 'results', '--no-submit',
                       '--command', f'{sys.executable} count.py {{input}} {{output}}',
                       *extra)

    def test_full_cycle(self):
        code, out = self.make_run()
        self.assertEqual(code, 0, out)
        run_dir = os.path.join(self.d, 'ogrun_runs', 'wc')
        for f in ('job.sub', 'jobs.tsv', 'queue.txt', 'run_job.sh', 'run.json'):
            self.assertTrue(os.path.exists(os.path.join(run_dir, f)), f)
        sub = open(os.path.join(run_dir, 'job.sub')).read()
        self.assertIn('executable           = /bin/bash', sub)
        self.assertIn('CRUSH-NODE-10-5-5-5', sub)
        self.assertNotIn('request_gpus', sub)
        self.assertEqual(len(open(os.path.join(run_dir, 'queue.txt')).read().split()), 2)

        # run both jobs exactly as a grid node would
        for i in (0, 1):
            r = subprocess.run(['bash', os.path.join(run_dir, 'run_job.sh'), str(i)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for name in ('cat', 'fox'):
            self.assertTrue(os.path.exists(f'results/{name}/n.csv'))
            self.assertTrue(os.path.exists(f'results/{name}/{condor.DONE_MARKER}'))

        # a finished job is not run again
        r = subprocess.run(['bash', os.path.join(run_dir, 'run_job.sh'), '0'],
                           capture_output=True, text=True)
        self.assertIn('already done', r.stdout)

        # resubmitting finds nothing to do
        code, out = run_cli('resubmit', run_dir, '--no-submit')
        self.assertIn('0 to submit', out)
        self.assertIn('2 already done', out)

        # status
        code, out = run_cli('status', run_dir)
        self.assertIn('all done', out)

        # combine keeps values as text and adds the job column
        combine('results', 'n.csv', 'all.csv', log=lambda *a: None)
        rows = list(csv.DictReader(open('all.csv')))
        self.assertEqual(sorted(r['ogrun_job'] for r in rows), ['cat', 'fox'])
        self.assertEqual({r['ogrun_job']: r['n'] for r in rows}, {'cat': '2', 'fox': '5'})

    def test_failed_job_gets_no_marker_and_is_resubmitted(self):
        self.make_run('--env', 'FAIL_CAT=1')
        run_dir = os.path.join(self.d, 'ogrun_runs', 'wc')
        env = dict(os.environ, FAIL_CAT='1')
        idx = {line.split('\t')[1]: line.split('\t')[0]
               for line in open(os.path.join(run_dir, 'jobs.tsv'))}
        r = subprocess.run(['bash', os.path.join(run_dir, 'run_job.sh'), idx['cat']],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 1)
        self.assertFalse(os.path.exists(f'results/cat/{condor.DONE_MARKER}'))
        # simulate HTCondor's log of that failed attempt
        with open(os.path.join(run_dir, 'logs', 'cat.log'), 'w') as f:
            f.write('000 (5.000.000) 2026-10-20 10:00:00 Job submitted\n...\n'
                    '001 (5.000.000) 2026-10-20 10:00:10 Job executing\n...\n'
                    '005 (5.000.000) 2026-10-20 10:00:20 Job terminated.\n'
                    '\t(1) Normal termination (return value 1)\n...\n')
        code, out = run_cli('status', run_dir)
        self.assertIn('failed', out)
        self.assertIn('exit code 1', out)
        code, out = run_cli('resubmit', run_dir, '--no-submit')
        self.assertIn('2 to submit', out)    # the failed one and the never-run one

    def test_queued_jobs_are_not_submitted_twice(self):
        self.make_run()
        run_dir = os.path.join(self.d, 'ogrun_runs', 'wc')
        with open(os.path.join(run_dir, 'logs', 'fox.log'), 'w') as f:
            f.write('000 (6.000.000) 2026-10-20 10:00:00 Job submitted\n...\n'
                    '001 (6.000.000) 2026-10-20 10:00:10 Job executing\n...\n')
        code, out = run_cli('resubmit', run_dir, '--no-submit')
        self.assertIn('1 to submit', out)
        self.assertIn('1 already running', out)

    def test_held_job_reported_with_reason(self):
        self.make_run()
        run_dir = os.path.join(self.d, 'ogrun_runs', 'wc')
        with open(os.path.join(run_dir, 'logs', 'fox.log'), 'w') as f:
            f.write('000 (7.000.000) 2026-10-20 10:00:00 Job submitted\n...\n'
                    "012 (7.000.000) 2026-10-20 10:00:05 Job was held.\n"
                    "\tFailed to execute 'x': Permission denied\n\tCode 6 Subcode 13\n...\n")
        code, out = run_cli('status', run_dir)
        self.assertIn('held', out)
        self.assertIn('Permission denied', out)

    def test_gpu_job_description(self):
        self.make_run('--gpu', '--conda-env', 'myenv', '--conda-base', '/opt/conda',
                      '--memory', '24GB')
        sub = open(os.path.join(self.d, 'ogrun_runs', 'wc', 'job.sub')).read()
        self.assertIn('+request_gpus        = 1', sub)
        self.assertIn('TotalGPUS > 0', sub)
        self.assertIn('CUDAGlobalMemoryMb >= 16000', sub)
        self.assertIn('request_memory       = 24GB', sub)
        sh = open(os.path.join(self.d, 'ogrun_runs', 'wc', 'run_job.sh')).read()
        self.assertIn('/opt/conda/bin/conda shell.bash hook', sh)
        self.assertIn('nvidia-smi -L', sh)

    def test_dry_run_writes_nothing(self):
        for n in ('a', 'b'):
            self.touch(f'data/{n}.txt')
        code, out = run_cli('submit', '--name', 'dry', '--inputs', 'data/*.txt',
                            '--output', 'res', '--command', 'echo {input}', '--dry-run')
        self.assertEqual(code, 0)
        self.assertIn('command: echo', out)
        self.assertFalse(os.path.exists('ogrun_runs'))

    def test_inputs_still_uploading_are_skipped(self):
        self.touch('data/old.txt')
        old = time.time() - 3600
        os.utime('data/old.txt', (old, old))
        self.touch('data/new.txt')
        code, out = run_cli('submit', '--name', 'up', '--inputs', 'data/*.txt',
                            '--output', 'res', '--command', 'echo {input}',
                            '--min-age', '600', '--no-submit')
        self.assertIn('1 to submit', out)
        self.assertIn('inputs still changing', out)

    def test_path_with_space_rejected_clearly(self):
        os.makedirs('my runs')
        self.touch('data/a.txt')
        code, out = run_cli('submit', '--name', 'x', '--inputs', 'data/*.txt',
                            '--output', 'res', '--command', 'echo {input}',
                            '--run-dir', 'my runs/x', '--no-submit')
        self.assertEqual(code, 1)


class Combining(Tmp):
    def test_union_of_columns_and_exact_text(self):
        self.touch('r/a/x.csv', 'id,v\n1,0.1234567890123456789\n')
        self.touch('r/b/x.csv', 'id,w\n2,3\n')
        combine('r', 'x.csv', 'r/all.csv', log=lambda *a: None)
        rows = list(csv.reader(open('r/all.csv')))
        self.assertEqual(rows[0], ['ogrun_job', 'id', 'v', 'w'])
        self.assertEqual(rows[1], ['a', '1', '0.1234567890123456789', ''])
        self.assertEqual(rows[2], ['b', '2', '', '3'])


if __name__ == '__main__':
    unittest.main()
