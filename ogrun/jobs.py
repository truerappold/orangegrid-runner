"""Turning input files into jobs: finding them, grouping them, filling in the
command template.

A *job* is one run of the user's command.  By default every input path (file
or folder) is one job.  With --group-by, input files whose names share the
same regex match become one job -- e.g. the three channel files of one
microscope tile -- and the regex's named groups become placeholders.
"""

from __future__ import annotations

import glob
import os
import re
import shlex
import time

PLACEHOLDER_RE = re.compile(r'\{(\w+)\}')
NAME_OK = re.compile(r'[^A-Za-z0-9._-]+')

# placeholders every job has; regex named groups are added on top
BUILTIN_FIELDS = ('input', 'inputs', 'name', 'output', 'stem', 'dir', 'index')


class JobError(ValueError):
    """A problem with how the jobs were specified (shown to the user as is)."""


class Job:
    def __init__(self, index, name, inputs, fields, output, command):
        self.index = index
        self.name = name
        self.inputs = inputs
        self.fields = fields
        self.output = output
        self.command = command

    def __repr__(self):
        return f'Job({self.index}, {self.name!r})'


def sanitize(name):
    """A name safe for file names and HTCondor job lists."""
    out = NAME_OK.sub('_', name).strip('_.')
    return out or 'job'


def stem(path):
    base = os.path.basename(os.path.normpath(path))
    return os.path.splitext(base)[0] if os.path.isfile(path) else base


def find_inputs(patterns=(), inputs_file=None, workdir='.'):
    """Sorted, de-duplicated paths matching the glob patterns (relative to
    workdir; ** recurses) plus any listed one per line in inputs_file."""
    found = []
    for pat in patterns or ():
        full = pat if os.path.isabs(pat) else os.path.join(workdir, pat)
        matches = glob.glob(full, recursive=True)
        if not matches:
            raise JobError(f'--inputs {pat!r} matched nothing (searched from {workdir})')
        found += matches
    if inputs_file:
        with open(inputs_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    p = line if os.path.isabs(line) else os.path.join(workdir, line)
                    if not os.path.exists(p):
                        raise JobError(f'{inputs_file}: {line!r} does not exist')
                    found.append(p)
    # hidden files are rsync temporaries, editor backups, .DS_Store, ...
    found = [p for p in found
             if not os.path.basename(os.path.normpath(p)).startswith('.')]
    return sorted(set(os.path.abspath(p) for p in found))


def group_inputs(paths, group_by=None, group_size=None):
    """[(fields, name, files)] and a list of human-readable warnings.

    No group_by: one group per path.  With group_by (a regex searched in each
    file's base name): files with the same match form one group; named groups
    become fields, and the name is their values joined by '-' (or the whole
    match if the regex has no groups).  group_size, if given, drops groups
    that do not have exactly that many files -- incomplete data.
    """
    warnings = []
    if not group_by:
        return [({}, sanitize(stem(p)), [p]) for p in paths], warnings
    try:
        rx = re.compile(group_by)
    except re.error as exc:
        raise JobError(f'--group-by is not a valid regex: {exc}')
    groups, order, unmatched = {}, [], 0
    for p in paths:
        m = rx.search(os.path.basename(os.path.normpath(p)))
        if not m:
            unmatched += 1
            continue
        fields = {k: v for k, v in m.groupdict().items() if v is not None}
        if fields:
            key = tuple(sorted(fields.items()))
            name = '-'.join(fields[k] for k in rx.groupindex if k in fields)
        else:
            vals = [g for g in m.groups() if g is not None] or [m.group(0)]
            key, name = tuple(vals), '-'.join(vals)
        if key not in groups:
            groups[key] = (fields, sanitize(name), [])
            order.append(key)
        groups[key][2].append(p)
    if unmatched:
        warnings.append(f'{unmatched} input(s) did not match --group-by and '
                        f'were ignored')
    out = []
    for key in order:
        fields, name, files = groups[key]
        if group_size and len(files) != group_size:
            warnings.append(f'{name}: {len(files)} file(s), expected '
                            f'{group_size} -- skipped as incomplete')
            continue
        out.append((fields, name, sorted(files)))
    return out, warnings


def render(template, fields):
    """Fill {placeholders}.  Values are shell-quoted, so write {input}, not
    "{input}" -- paths with spaces then work without any extra quoting."""
    missing = sorted({m.group(1) for m in PLACEHOLDER_RE.finditer(template)}
                     - set(fields))
    if missing:
        raise JobError(f'unknown placeholder(s) in --command: '
                       + ', '.join('{' + m + '}' for m in missing)
                       + f'.  Available: '
                       + ', '.join('{' + k + '}' for k in sorted(fields)))

    def repl(m):
        v = fields[m.group(1)]
        return v if m.group(1) == 'inputs' else shlex.quote(str(v))
    return PLACEHOLDER_RE.sub(repl, template)


def build_jobs(command, groups, output_root):
    """Jobs from groups: each gets its own output folder <output_root>/<name>."""
    names = [g[1] for g in groups]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise JobError('several jobs would get the same name: '
                       + ', '.join(dupes[:5])
                       + ' -- use --group-by so each job has a unique name')
    jobs = []
    for i, (fields, name, files) in enumerate(groups):
        out = os.path.join(output_root, name)
        f = dict(fields)
        f.update(input=files[0], inputs=' '.join(shlex.quote(x) for x in files),
                 name=name, output=out, stem=stem(files[0]),
                 dir=os.path.dirname(files[0]), index=str(i))
        cmd = render(command, f)
        if '\t' in cmd or '\n' in cmd:
            raise JobError('the command must be a single line without tabs')
        jobs.append(Job(i, name, files, f, out, cmd))
    return jobs


def newest_mtime(paths):
    newest = 0.0
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for f in files:
                    newest = max(newest, os.path.getmtime(os.path.join(root, f)))
        elif os.path.exists(p):
            newest = max(newest, os.path.getmtime(p))
    return newest


def still_changing(job, min_age_s, now=None):
    """True if any input changed in the last min_age_s seconds (still uploading)."""
    if not min_age_s:
        return False
    now = time.time() if now is None else now
    return now - newest_mtime(job.inputs) < min_age_s
