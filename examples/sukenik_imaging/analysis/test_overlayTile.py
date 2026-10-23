"""Checks for overlayTile.py that do not need a JVM or any .vsi files."""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overlayTile as ot

fails = []
def check(name, cond, extra=''):
    print(f'  {"PASS" if cond else "FAIL"}  {name}' + ('' if cond else f'   [{extra}]'))
    if not cond:
        fails.append(name)

print('--- file discovery ---')
d = tempfile.mkdtemp()
for n in ('A1-1_mCitrineNeoPIPFUCCI.vsi', 'A1-1_mCherryNeoPIPFUCCI.vsi',
          'A1-1_BF Cam 1.vsi', 'A10-2_mVenusNeoPIPFUCCI.vsi',
          'A10-2_mCherryNeoPIPFUCCI.vsi', 'B3-1_mCherryNeoPIPFUCCI.vsi',
          'notes.txt', 'A1-1_mCitrineNeoPIPFUCCI.vsi.bak'):
    open(os.path.join(d, n), 'w').close()
f = ot.scan(d)
check('finds the tiles', set(f) == {('A1', 1), ('A10', 2), ('B3', 1)}, set(f))
check('ignores non-vsi and backup files', len(f[('A1', 1)]) == 3, f[('A1', 1)])
check('mCitrine picked as green',
      ot.pick_channel(f[('A1', 1)], ot.GREEN_HINTS) == 'mCitrineNeoPIPFUCCI')
check('mVenus also picked as green',
      ot.pick_channel(f[('A10', 2)], ot.GREEN_HINTS) == 'mVenusNeoPIPFUCCI')
check('mCherry picked as red',
      ot.pick_channel(f[('A1', 1)], ot.RED_HINTS) == 'mCherryNeoPIPFUCCI')
check('BF is not mistaken for a marker channel',
      'BF' not in str(ot.pick_channel(f[('A1', 1)], ot.GREEN_HINTS)))
check('a tile missing green reports None',
      ot.pick_channel(f[('B3', 1)], ot.GREEN_HINTS) is None)
check('well sort is numeric not lexical (A10 after A1, not after A1 by string)',
      sorted(f, key=ot.sort_key) == [('A1', 1), ('A10', 2), ('B3', 1)])
try:
    ot.pick_channel(f[('A1', 1)], ot.GREEN_HINTS, override='nope')
    check('bad override raises', False)
except SystemExit as e:
    check('bad override raises and lists what is there', 'mCitrine' in str(e))

print('\n--- contrast is fixed across the series ---')
stack = np.zeros((8, 2, 64, 64), np.uint16)
stack[:, 0] = 300
stack[:, 1] = 800
stack[:, 0, 10:20, 10:20] = np.arange(8, dtype=np.uint16)[:, None, None] * 500 + 2000
stack[:, 1, 40:50, 40:50] = 9000
lim = ot.stretch_limits(stack, 1, 99.8)
check('limits are per channel', len(lim) == 2, lim)
rgb0, rgb7 = ot.to_rgb(stack[0], lim), ot.to_rgb(stack[7], lim)
check('a cell that gets brighter looks brighter (no per-frame rescale)',
      rgb7[15, 15, 1] > rgb0[15, 15, 1], (rgb0[15, 15, 1], rgb7[15, 15, 1]))

print('\n--- colour mapping ---')
fr = np.zeros((2, 8, 8), np.uint16)
fr[0, 0, 0] = 65535
fr[1, 1, 1] = 65535
fr[:, 2, 2] = 65535
r = ot.to_rgb(fr, [(0, 65535), (0, 65535)])
check('green channel is pure green', tuple(r[0, 0]) == (0, 255, 0), r[0, 0])
check('red channel is magenta', tuple(r[1, 1]) == (255, 0, 255), r[1, 1])
check('overlap reads white', tuple(r[2, 2]) == (255, 255, 255), r[2, 2])
check('empty is black', tuple(r[5, 5]) == (0, 0, 0), r[5, 5])
check('output is uint8 HWC', r.dtype == np.uint8 and r.shape == (8, 8, 3), r.shape)

print('\n--- tiff ---')
import tifffile
p = os.path.join(d, 'o.tif')
ot.write_tif(p, stack, 'mCit', 'mChe', pixel_size_um=0.325, frame_interval_min=15)
with tifffile.TiffFile(p) as tf:
    md = tf.imagej_metadata
    check('imagej hyperstack', tf.is_imagej)
    check('composite mode', md.get('mode') == 'composite', md.get('mode'))
    check('two LUTs', len(md.get('LUTs', [])) == 2)
    check('frame interval stored in seconds', md.get('finterval') == 900.0)
    a = tf.asarray()
    check('shape preserved TCYX', a.shape == stack.shape, a.shape)
    check('raw counts preserved, not rescaled', (a == stack).all())

print('\n--- ffmpeg detection ---')
exe = ot.ffmpeg_exe()
check('detection returns a path or None, never raises',
      exe is None or os.path.exists(exe), exe)
print(f'      (this machine: {exe or "no ffmpeg -> GIF fallback"})')

print('\n--- downscaling the movie ---')
big = [np.zeros((2048, 2048, 3), np.uint8) for _ in range(3)]
_d, _f = ot._downscale(big, 1024)
check('2048 shrinks to the 1024 default', _d[0].shape[:2] == (1024, 1024))
check('and reports the factor it actually used', abs(_f - 0.5) < 1e-9, _f)
check('every frame shrinks to the same size', len({f.shape for f in _d}) == 1)
check('max_px=0 disables shrinking',
      ot._downscale(big, 0)[0][0].shape[:2] == (2048, 2048))
check('max_px=0 reports a factor of 1', ot._downscale(big, 0)[1] == 1.0)
check('frames already smaller are left alone',
      ot._downscale([np.zeros((300, 400, 3), np.uint8)], 1024)[0][0].shape[:2]
      == (300, 400))
check('non-square keeps its aspect ratio',
      ot._downscale([np.zeros((2048, 1024, 3), np.uint8)], 512)[0][0].shape[:2]
      == (512, 256))

print('\n--- video ---')
lab = [f'{i + 1}/8' for i in range(8)]
quiet = lambda *a, **k: None
g = ot.write_video(os.path.join(d, 'v.gif'), stack, lim, 6, labels=lab, log=quiet)
check('gif written', os.path.getsize(g) > 0, g)
m = ot.write_video(os.path.join(d, 'v.mp4'), stack, lim, 6, labels=lab, log=quiet)
check('mp4 request always yields a playable file (mp4, or gif if no ffmpeg)',
      os.path.exists(m) and os.path.getsize(m) > 0, m)
# size actually matters here: a 96-frame 2048px GIF runs to ~550 MB, which is
# why the movie is downscaled by default while the TIFF stays full resolution
rng = np.random.default_rng(0)
tex = rng.integers(800, 6000, size=(4, 2, 512, 512), dtype=np.uint16)
tl = ot.stretch_limits(tex, 1, 99.8)
s_small = os.path.getsize(ot.write_video(os.path.join(d, 's.gif'), tex, tl, 8,
                                         max_px=128, log=quiet))
s_full = os.path.getsize(ot.write_video(os.path.join(d, 'f.gif'), tex, tl, 8,
                                        max_px=0, log=quiet))
check('downscaling really does shrink the file', s_full > 4 * s_small,
      (s_small, s_full))

odd = np.zeros((4, 2, 65, 63), np.uint16)
odd[:, 0] = 1000
o = ot.write_video(os.path.join(d, 'odd.mp4'), odd,
                   ot.stretch_limits(odd, 1, 99.8), 6, log=quiet)
check('odd pixel dimensions handled (mp4 encoders need even)',
      os.path.getsize(o) > 0, o)
check('timestamp works without a guaranteed system font',
      ot.stamp(np.zeros((64, 64, 3), np.uint8), '1/8').shape == (64, 64, 3))

print('\n--- scale bar ---')
from PIL import Image

def _longest_white_run(img_rgb, frac=0.12):
    """Longest CONTIGUOUS pure-white run in the bottom-left region.

    Contiguous, not min-to-max span: saturated image pixels are also pure
    white, and a span measurement silently swallows them and reports a bar
    three times its real length.
    """
    hh, ww = img_rgb.shape[:2]
    band = img_rgb[int(hh * (1 - frac)):, :int(ww * 0.6)]
    wh = (band == 255).all(2)
    best = 0
    for row in wh:
        run = 0
        for v in row:
            run = run + 1 if v else 0
            best = max(best, run)
    return best

check('0.325 um/px over 2048 px picks a 100 um bar',
      ot.nice_bar_um(0.325, 2048) == 100, ot.nice_bar_um(0.325, 2048))
check('the same sample shown at 1024 px still picks 100 um',
      ot.nice_bar_um(0.65, 1024) == 100)
check('the length is always a round number',
      all(ot.nice_bar_um(q, 1024) in ot.NICE_BAR_UM
          for q in (0.05, 0.1, 0.325, 0.65, 1.3, 6.5)))
check('the bar never exceeds ~20% of the width',
      all(ot.nice_bar_um(q, 1024) / q <= 0.21 * 1024 for q in (0.05, 0.325, 1.3)))
check('a tiny pixel size still yields the smallest bar, not zero',
      ot.nice_bar_um(0.0001, 100) == 1)

# A dim image so nothing in the data can reach pure white and be mistaken
# for the bar.
_dim = (np.random.default_rng(0)
        .integers(0, 1000, size=(4, 2, 2048, 2048))).astype(np.uint16)
_dl = [(0.0, 4000.0), (0.0, 4000.0)]
PX = 0.325
for _mx in (0, 1024, 512):
    _p = ot.write_video(os.path.join(d, f'bar{_mx}.gif'), _dim, _dl, 6,
                        labels=[f'{i + 1}/4' for i in range(4)], max_px=_mx,
                        pixel_size_um=PX, log=quiet)
    _a = np.asarray(Image.open(_p).convert('RGB'))
    _w = _a.shape[1]
    _eff = PX * (2048 / _w)
    _meas = _longest_white_run(_a) * _eff
    _want = ot.nice_bar_um(_eff, _w)
    check(f'at {_w}px the bar is as long as its label says '
          f'({_meas:.0f} vs {_want:g} um)', abs(_meas - _want) <= 2 * _eff + 2,
          (_meas, _want))
_p = ot.write_video(os.path.join(d, 'bar50.gif'), _dim, _dl, 6, max_px=1024,
                    pixel_size_um=PX, bar_um=50, log=quiet)
check('an explicit bar length is honoured',
      abs(_longest_white_run(np.asarray(Image.open(_p).convert('RGB')))
          * PX * 2 - 50) <= 2 * PX * 2 + 2)
_p = ot.write_video(os.path.join(d, 'barq.gif'), _dim, _dl, 6, max_px=512,
                    pixel_size_um=PX, log=quiet)
_g2 = Image.open(_p)
_runs = []
for _i in range(4):
    _g2.seek(_i)
    _runs.append(_longest_white_run(np.asarray(_g2.convert('RGB'))))
check('the bar is identical and pure white in every frame',
      len(set(_runs)) == 1, _runs)
_p = ot.write_video(os.path.join(d, 'barnone.gif'), _dim, _dl, 6, max_px=256,
                    pixel_size_um=None, log=quiet)
check('no bar is drawn when the pixel size is unknown',
      _longest_white_run(np.asarray(Image.open(_p).convert('RGB'))) < 4)

print('\n--- gif palette stability ---')
# A GIF holds 256 colours.  With a per-frame adaptive palette, the nearest
# available colour to white changes as the field swings green -> magenta, and
# the white timestamp visibly cycles colour through the movie.
swing = np.zeros((8, 2, 256, 256), np.uint16)
_rng = np.random.default_rng(0)
for _i in range(8):
    swing[_i, 0] = _rng.integers(0, int(60000 * (1 - _i / 7)) + 1, (256, 256))
    swing[_i, 1] = _rng.integers(0, int(60000 * (_i / 7)) + 1, (256, 256))
swl = ot.stretch_limits(swing, 1, 99.8)
swlab = [f'{i + 1}/8  {i * 0.25:.2f} h' for i in range(8)]
gp = ot.write_video(os.path.join(d, 'swing.gif'), swing, swl, 8,
                    labels=swlab, max_px=0, log=quiet)
_g = Image.open(gp)
_cols = []
for _i in range(8):
    _g.seek(_i)
    _a = np.asarray(_g.convert('RGB'))[:40, :140]
    _cols.append(tuple(int(v) for v in _a.reshape(-1, 3)[_a.sum(2).argmax()]))
check('the timestamp is the same colour in every frame',
      len(set(_cols)) == 1, _cols)
check('and that colour is pure white, not a palette approximation',
      _cols[0] == (255, 255, 255), _cols[0])
_g.seek(0)
_p0 = _g.getpalette()
check('frame 0 carries the global palette with white in the last slot',
      _p0 is not None and _p0[-3:] == [255, 255, 255], _p0[-3:] if _p0 else None)
_locals = []
for _i in range(1, 8):
    _g.seek(_i)
    _locals.append(_g.getpalette())
check('later frames carry no local palette, so they inherit the global one',
      all(x is None for x in _locals),
      [i + 1 for i, x in enumerate(_locals) if x is not None])
# the property that matters: colours decode close to the source in every frame
_errs = []
for _i in (0, 7):
    _g.seek(_i)
    _errs.append(float(np.abs(np.asarray(_g.convert('RGB')).astype(int)
                              - ot.to_rgb(swing[_i], swl).astype(int)).mean()))
check('quantisation error stays small and even across the movie',
      max(_errs) < 5 and abs(_errs[0] - _errs[1]) < 3, _errs)

print()
print('ALL TESTS PASSED' if not fails else f'{len(fails)} FAILURE(S): {fails}')
sys.exit(1 if fails else 0)
