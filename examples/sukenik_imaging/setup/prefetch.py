"""Download everything the jobs would otherwise fetch from the internet.

    python prefetch.py            # on the login node, once, after creating the env

The analysis downloads two things the first time it runs:
  * Fiji 2.14.0 and Bio-Formats (~600 MB of Java libraries), into ~/.jgo
  * the Cellpose-SAM model weights (~1.2 GB), into ~/.cellpose/models
Compute nodes may have no internet access, and 100 jobs fetching 2 GB each
at once would be bad manners anyway.  Your home directory is shared with every
compute node, so fetching once here means every job finds them in place.

Uses the network and some memory for a minute or two, no GPU -- fine on the
login node under the Good Neighbor Policy (a short setup step, not compute).
Run it with the env active:  conda activate fucci_grid && python prefetch.py
"""

import os
import sys
import time


def main():
    t0 = time.time()
    print('1/2  Fiji + Bio-Formats -> ~/.jgo')
    import imagej
    ij = imagej.init('sc.fiji:fiji:2.14.0')
    print(f'     ImageJ {ij.getVersion()} ready ({time.time() - t0:.0f} s)')

    print('2/2  Cellpose-SAM weights -> ~/.cellpose/models')
    t1 = time.time()
    from cellpose import models
    models.CellposeModel(gpu=False)          # CPU here: only the download matters
    print(f'     model ready ({time.time() - t1:.0f} s)')

    for d in ('~/.jgo', '~/.cellpose/models'):
        p = os.path.expanduser(d)
        size = sum(os.path.getsize(os.path.join(r, f))
                   for r, _d, fs in os.walk(p) for f in fs)
        print(f'     {d}: {size / 1e9:.2f} GB')
    print('done -- jobs can now start ImageJ and Cellpose without internet')


if __name__ == '__main__':
    sys.exit(main())
