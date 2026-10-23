"""
Quick two-colour overlay of one tile's timelapse: mCitrine + mCherry.

No Cellpose, no tracking, no GPU -- this only reads the two fluorescence .vsi
files for one well/tile and writes them out together.  It is deliberately
standalone (it does not import fucci_pipeline / fucci_imaging) so it can be
dropped anywhere on the workstation on its own.

Three kinds of output, any combination:

  --tif    an ImageJ composite hyperstack (TCYX, 16-bit, green + magenta).
           This is the one to keep: it preserves the raw counts, so you can
           still adjust each channel's brightness/contrast separately in Fiji
           and make montages from it.  Written by default.
  --video  an mp4 (or .gif) for showing people.  8-bit and contrast-stretched,
           so it is for looking at, never for measuring.
  --project  a single frame with all timepoints collapsed (max or mean).

Contrast for the video is computed once over the whole series rather than per
frame, so a cell getting brighter actually looks brighter instead of the
display rescaling under it every frame.

    python overlayTile.py <image_dir> --well A1 --tile 1
    python overlayTile.py <image_dir> --well A1 --tile 1 --video A1-1.mp4 --fps 8
    python overlayTile.py <image_dir> --list
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

# Substrings used to work out which file is which channel, tried in order.
# mCitrine and mVenus are both YFP variants and the two experiments name them
# differently, so both are accepted rather than hardcoding either.
GREEN_HINTS = ('mcitrine', 'mcit', 'mvenus', 'venus', 'citrine', 'yfp')
RED_HINTS = ('mcherry', 'cherry', 'rfp', 'mscarlet')

NAME_RE = re.compile(r'^(?P<well>[A-H]\d{1,2})-(?P<tile>\d+)_(?P<channel>.+)\.vsi$')


# --------------------------------------------------------------------------- #
# file discovery
# --------------------------------------------------------------------------- #
def resolve_dir(path, subdir='Pics'):
    """Accept either the experiment folder or the folder of .vsi files itself.

    analyzeDirFUCCI.py takes <experiment>/ and looks inside <experiment>/Pics,
    so being handed the same path here and getting 'no files found' is a trap
    worth closing rather than documenting.
    """
    inner = os.path.join(path, subdir)
    if os.path.isdir(inner) and not any(f.endswith('.vsi')
                                        for f in os.listdir(path)):
        return inner
    return path


def scan(image_dir):
    """{(well, tile): {channel_suffix: path}} for every .vsi in the folder."""
    out = {}
    for entry in sorted(os.listdir(image_dir)):
        m = NAME_RE.match(entry)
        if not m:
            continue
        key = (m.group('well'), int(m.group('tile')))
        out.setdefault(key, {})[m.group('channel')] = os.path.join(image_dir, entry)
    return out


def pick_channel(channels, hints, override=None):
    """Choose a channel suffix by substring hint, or take an exact override."""
    if override:
        if override not in channels:
            raise SystemExit(
                f'channel {override!r} not found. Present: '
                + ', '.join(sorted(channels)))
        return override
    for hint in hints:
        for name in sorted(channels):
            if hint in name.lower():
                return name
    return None


def sort_key(key):
    well, tile = key
    return (well[0], int(well[1:]), tile)


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def axis_index(ds):
    """{'X':0,'Y':1,'Time':2,...}.

    ImageJ2 drops singleton axes, so a C=1/Z=1 file has no Channel or Z axis at
    all.  Indexing by position would therefore silently read the wrong plane on
    some files and the right one on others -- read the labels instead.
    """
    out = {}
    try:
        for i in range(int(ds.numDimensions())):
            out[str(ds.axis(i).type().getLabel())] = i
    except Exception:
        ndim = int(ds.numDimensions())
        names = ['X', 'Y'] + (['Time'] if ndim >= 3 else [])
        out = {n: i for i, n in enumerate(names[:ndim])}
    return out


def n_frames(ds):
    a = axis_index(ds)
    return int(ds.dimension(a['Time'])) if 'Time' in a else 1


def plane_slice(ds, frame):
    a = axis_index(ds)
    idx = [slice(None)] * int(ds.numDimensions())
    if 'Time' in a:
        idx[a['Time']] = frame
    elif frame not in (0, None):
        raise ValueError(f'frame {frame} requested but this file has no Time axis')
    return tuple(idx)


def detect_pixel_size(ds):
    """Micrometres per pixel along X, from the file's own calibration.

    Returns None rather than guessing: a wrong pixel size makes a scale bar
    that is confidently incorrect, which is worse than no scale bar at all.
    Every accessor is tried defensively because the ImageJ2 calibration API
    differs between versions and an uncalibrated file simply has none.
    """
    try:
        a = axis_index(ds)
        if 'X' not in a:
            return None
        i = a['X']
        scale = None
        for get in (lambda: float(ds.averageScale(i)),
                    lambda: float(ds.axis(i).calibratedValue(1)
                                  - ds.axis(i).calibratedValue(0))):
            try:
                scale = get()
                break
            except Exception:
                continue
        if not scale or not np.isfinite(scale) or scale <= 0:
            return None
        unit = ''
        try:
            unit = str(ds.axis(i).unit() or '').strip().lower()
        except Exception:
            pass
        factor = {'um': 1.0, '\u00b5m': 1.0, 'micron': 1.0, 'microns': 1.0,
                  'micrometer': 1.0, 'micrometre': 1.0,
                  'nm': 1e-3, 'mm': 1e3, 'cm': 1e4, 'm': 1e6}.get(unit)
        if factor is None:
            # No unit, or one we do not recognise.  A scale of exactly 1.0 is
            # what an uncalibrated file reports, so treat that as "unknown"
            # instead of claiming 1 um per pixel.
            if abs(scale - 1.0) < 1e-9:
                return None
            factor = 1.0
        return scale * factor
    except Exception:
        return None


def release(ds):
    """net.imagej.Dataset has no close(); dispose the backing img if it has one."""
    try:
        img = ds.getImgPlus().getImg()
        if hasattr(img, 'dispose'):
            img.dispose()
    except Exception:
        pass


def read_pair(ij, green_path, red_path, max_frames=0, log=print):
    """([T, 2, H, W] uint16, pixel_size_um_or_None) -- ch0 green, ch1 red."""
    g = ij.io().open(green_path)
    r = ij.io().open(red_path)
    try:
        px = detect_pixel_size(g)
        ng, nr = n_frames(g), n_frames(r)
        if ng != nr:
            raise SystemExit(
                f'the two channels disagree on frame count ({ng} vs {nr}); '
                f'they cannot be paired into timepoints')
        n = ng if not max_frames else min(ng, max_frames)
        log(f'  {n} of {ng} timepoints')
        frames = []
        for i in range(n):
            gi = None if ng == 1 else i
            planes = [
                np.asarray(ij.py.from_java(g[plane_slice(g, gi)])),
                np.asarray(ij.py.from_java(r[plane_slice(r, gi)])),
            ]
            frames.append(np.stack(planes).astype(np.uint16))
            if (i + 1) % 10 == 0 or i + 1 == n:
                log(f'    read {i + 1}/{n}')
        return np.stack(frames), px
    finally:
        release(g)
        release(r)


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def ramp_lut(r, g, b):
    ramp = np.arange(256, dtype=np.float64)
    return np.stack([(ramp * r).astype(np.uint8),
                     (ramp * g).astype(np.uint8),
                     (ramp * b).astype(np.uint8)])


def write_tif(path, data, green_name, red_name, pixel_size_um=None,
              frame_interval_min=None):
    """ImageJ composite hyperstack, 16-bit, raw counts preserved."""
    import tifffile

    meta = {'axes': 'TCYX', 'mode': 'composite',
            'LUTs': [ramp_lut(0, 1, 0), ramp_lut(1, 0, 1)],
            'Labels': [green_name, red_name]}
    kw = {}
    if pixel_size_um:
        kw['resolution'] = (1 / pixel_size_um, 1 / pixel_size_um)
        meta['unit'] = 'um'
    if frame_interval_min:
        meta['finterval'] = frame_interval_min * 60.0
        meta['tunit'] = 's'
    tifffile.imwrite(path, data, imagej=True, photometric='minisblack',
                     metadata=meta, **kw)


def stretch_limits(stack, lo_pct, hi_pct, max_sample=12):
    """Per-channel display limits from the WHOLE series, not per frame.

    Rescaling each frame independently makes a cell that is genuinely getting
    brighter look constant, and makes the background pulse -- which is exactly
    the artefact you would then try to interpret.  One set of limits for the
    whole movie keeps real intensity changes visible.
    """
    t = stack.shape[0]
    picks = np.unique(np.linspace(0, t - 1, min(t, max_sample)).astype(int))
    sample = stack[picks]
    return [(float(np.percentile(sample[:, c], lo_pct)),
             float(np.percentile(sample[:, c], hi_pct)))
            for c in range(stack.shape[1])]


def to_rgb(frame, limits):
    """[2, H, W] uint16 -> [H, W, 3] uint8, green + magenta, overlap white."""
    out = np.zeros(frame.shape[1:] + (3,), np.float32)
    for c, (lo, hi) in enumerate(limits):
        v = np.clip((frame[c].astype(np.float32) - lo) / max(hi - lo, 1e-9), 0, 1)
        if c == 0:
            out[:, :, 1] += v                      # green
        else:
            out[:, :, 0] += v                      # magenta = red + blue
            out[:, :, 2] += v
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _label_font(height):
    from PIL import ImageFont

    size = max(16, height // 40)
    for cand in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                 '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
                 '/Library/Fonts/Arial.ttf'):
        if os.path.exists(cand):
            try:
                return ImageFont.truetype(cand, size), size
            except Exception:
                pass
    try:
        return ImageFont.load_default(), size
    except Exception:
        return None, size


NICE_BAR_UM = (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000)


def nice_bar_um(pixel_size_um, width_px, target_frac=0.20):
    """A round scale-bar length covering roughly target_frac of the width.

    Round numbers only -- a bar labelled "137 um" is harder to read off than
    one labelled "100 um", and the point of a scale bar is to be read at a
    glance.
    """
    target = pixel_size_um * width_px * target_frac
    usable = [v for v in NICE_BAR_UM if v <= target]
    return usable[-1] if usable else NICE_BAR_UM[0]


def annotate(img, text=None, pixel_size_um=None, bar_um=None, fill=None):
    """Draw the timestamp (top-left) and scale bar (bottom-left) onto a PIL image.

    Works on both RGB and palette images: `fill` is an (r, g, b) tuple for the
    mp4 path and a palette index for the GIF path, where pure white has to be
    addressed by index to survive quantisation.
    """
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    w, h = img.size
    font, size = _label_font(h)
    if fill is None:
        fill = (255, 255, 255)
    pad = max(size // 2, 4)

    if text:
        draw.text((pad, pad), text, fill=fill, font=font)

    if pixel_size_um and bar_um:
        bar_px = bar_um / pixel_size_um
        if bar_px >= 4:
            thick = max(size // 4, 3)
            x0 = pad
            y0 = h - pad - thick
            draw.rectangle([x0, y0, x0 + bar_px, y0 + thick], fill=fill)
            label = f'{bar_um:g} \u00b5m'
            try:
                tw = draw.textlength(label, font=font)
            except Exception:
                tw = len(label) * size * 0.6
            draw.text((x0 + max((bar_px - tw) / 2, 0), y0 - size - pad // 2),
                      label, fill=fill, font=font)
    return img


def stamp(rgb, text, pixel_size_um=None, bar_um=None):
    """Annotate an RGB frame in place of the old text-only helper."""
    from PIL import Image

    return np.asarray(annotate(Image.fromarray(rgb), text,
                               pixel_size_um, bar_um))


def ffmpeg_exe():
    """An ffmpeg binary from imageio-ffmpeg or from PATH, else None.

    Checked up front rather than by catching an exception from imageio: with no
    ffmpeg plugin installed, imageio happily hands a .mp4 path to its *TIFF*
    writer and the failure surfaces as
    "TiffWriter.write() got an unexpected keyword argument 'fps'", which says
    nothing about the actual problem.
    """
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    from shutil import which
    return which('ffmpeg')


def _downscale(frames, max_px):
    """Shrink so the long edge is at most max_px; returns (frames, factor).

    0 disables.

    A 2048x2048 movie is far more than a movie needs, and at 96 frames a
    full-size GIF runs to hundreds of MB.  The TIFF keeps full resolution --
    this only affects what you hand to someone to watch.
    """
    if not max_px:
        return frames, 1.0
    h, w = frames[0].shape[:2]
    scale = max_px / max(h, w)
    if scale >= 1:
        return frames, 1.0
    from PIL import Image
    size = (max(int(round(w * scale)), 2), max(int(round(h * scale)), 2))
    out = [np.asarray(Image.fromarray(f).resize(size, Image.LANCZOS))
           for f in frames]
    # the realised factor, not the requested one -- rounding to whole pixels
    # moves it slightly, and the scale bar is computed from it
    return out, size[0] / w


def _write_mp4(exe, path, frames, fps):
    """Pipe raw RGB straight into ffmpeg -- no extra Python dependency."""
    import subprocess

    h, w = frames[0].shape[:2]
    # yuv420p requires even dimensions
    h, w = h - h % 2, w - w % 2
    frames = [f[:h, :w] for f in frames]
    cmd = [exe, '-y', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
           '-r', str(fps), '-i', '-',
           '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
           '-pix_fmt', 'yuv420p', path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for f in frames:
            proc.stdin.write(np.ascontiguousarray(f).tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode(errors='replace')
    if proc.wait() != 0:
        raise RuntimeError(err.strip() or 'ffmpeg failed')


def _to_gif_frames(rgb, labels=None, pixel_size_um=None, bar_um=None):
    """Quantise every frame against ONE shared palette, with white reserved.

    A GIF holds 256 colours.  Left to itself PIL builds a fresh adaptive
    palette per frame, so as the field swings green to magenta the nearest
    available colour to white changes and a pure-white timestamp comes back
    tinted differently in every frame -- it visibly cycles.  Worse, the image
    colours drift too, which reads as intensity change that is not there.

    So: one palette derived from a sample of frames, index 255 forced to pure
    white, and the label drawn *after* quantisation straight onto that index,
    which makes it exactly white by construction rather than by luck.
    """
    from PIL import Image

    picks = np.unique(np.linspace(0, len(rgb) - 1, min(len(rgb), 8)).astype(int))
    sample = Image.fromarray(np.concatenate([rgb[i] for i in picks], axis=0))
    base = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    pal = base.getpalette()[:255 * 3] + [255, 255, 255]
    base.putpalette(pal)

    out = []
    for i, frame in enumerate(rgb):
        q = Image.fromarray(frame).quantize(palette=base,
                                            dither=Image.Dither.NONE)
        # fill=255 is the palette index reserved above, not a grey level
        annotate(q, labels[i] if labels else None, pixel_size_um, bar_um,
                 fill=255)
        out.append(q)
    return out


def write_video(path, stack, limits, fps, labels=None, max_px=1024,
                pixel_size_um=None, bar_um=None, log=print):
    """mp4 if ffmpeg is anywhere to be found, otherwise an animated GIF."""
    rgb = [to_rgb(stack[i], limits) for i in range(stack.shape[0])]
    rgb, factor = _downscale(rgb, max_px)
    h, w = rgb[0].shape[:2]

    # Shrinking the movie makes every pixel cover more sample, so the scale
    # bar is sized from the DOWNSCALED pixel size.  Getting this wrong is the
    # classic way a scale bar ends up silently off by 2x.
    px = (pixel_size_um / factor) if pixel_size_um else None
    if px and bar_um is None:
        bar_um = nice_bar_um(px, w)
    if px:
        log(f'  scale bar: {bar_um:g} um = {bar_um / px:.0f} px '
            f'at {px:.4f} um/px')

    if not path.lower().endswith('.gif'):
        exe = ffmpeg_exe()
        if exe:
            # mp4 is true colour, so the label can simply be drawn in white
            stamped = [stamp(f, labels[i] if labels else None, px, bar_um)
                       for i, f in enumerate(rgb)]
            _write_mp4(exe, path, stamped, fps)
            log(f'  wrote {path}  ({len(rgb)} frames, {w}x{h}, {fps} fps)')
            return path
        path = os.path.splitext(path)[0] + '.gif'
        log('  no ffmpeg available, writing a GIF instead')
        log('    for mp4: pip install imageio-ffmpeg   (or conda install ffmpeg)')

    imgs = _to_gif_frames(rgb, labels, pixel_size_um=px, bar_um=bar_um)
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / max(fps, 1)), loop=0, optimize=False)
    mb = os.path.getsize(path) / 1e6
    log(f'  wrote {path}  ({len(rgb)} frames, {w}x{h}, {fps} fps, {mb:.0f} MB)')
    return path


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description='Two-colour overlay of one tile: mCitrine + mCherry.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('image_dir', help='folder holding the .vsi files')
    ap.add_argument('--well')
    ap.add_argument('--tile', type=int)
    ap.add_argument('--list', action='store_true',
                    help='list the tiles and channels found, then exit')
    ap.add_argument('--pics-subdir', default='Pics',
                    help='if image_dir has no .vsi files but does have this '
                         'subfolder, use the subfolder')
    ap.add_argument('--output-dir', default=None,
                    help='default: neoPIP_FUCCI_outputs/ beside this script')
    ap.add_argument('--green-channel', default=None,
                    help='exact channel suffix to use as green, if the guess '
                         'from the filename is wrong')
    ap.add_argument('--red-channel', default=None)
    ap.add_argument('--max-frames', type=int, default=0, help='0 = all')
    ap.add_argument('--no-tif', action='store_true',
                    help='skip the 16-bit composite hyperstack')
    ap.add_argument('--video', nargs='?', const='auto', default=None,
                    metavar='PATH',
                    help='also write a movie; .mp4 (default) or .gif. '
                         'Pass the flag alone to name it after the tile')
    ap.add_argument('--fps', type=float, default=6.0)
    ap.add_argument('--video-max-px', type=int, default=1024, metavar='PX',
                    help='shrink the movie so its long edge is at most this '
                         '(the TIFF is always full resolution); 0 = no shrink')
    ap.add_argument('--no-timestamp', action='store_true',
                    help='do not burn the frame/elapsed time into the video')
    ap.add_argument('--frame-interval', type=float, default=15.0,
                    metavar='MIN', help='minutes per frame, for the timestamp')
    ap.add_argument('--project', choices=('max', 'mean'), default=None,
                    help='also write one frame with all timepoints collapsed')
    ap.add_argument('--clip', type=float, nargs=2, default=(1.0, 99.8),
                    metavar=('LO', 'HI'),
                    help='display percentiles for the video and projection '
                         '(the TIFF is always raw)')
    ap.add_argument('--pixel-size', type=float, default=None, metavar='UM',
                    help='micrometres per pixel. Read from the .vsi metadata '
                         'when present; pass this to override it, or to supply '
                         'it when the file carries no calibration')
    ap.add_argument('--no-scale-bar', action='store_true')
    ap.add_argument('--scale-bar-um', type=float, default=None, metavar='UM',
                    help='force the bar length instead of picking a round '
                         'number near 20%% of the image width')
    args = ap.parse_args()

    if not os.path.isdir(args.image_dir):
        raise SystemExit(f'not a folder: {args.image_dir}')

    image_dir = resolve_dir(args.image_dir, args.pics_subdir)
    if image_dir != args.image_dir:
        print(f'using {image_dir}')

    found = scan(image_dir)
    if not found:
        raise SystemExit(f'no files matching <WELL>-<TILE>_<CHANNEL>.vsi in '
                         f'{image_dir}')

    if args.list:
        print(f'{len(found)} tile(s) in {image_dir}\n')
        for key in sorted(found, key=sort_key):
            chans = found[key]
            g = pick_channel(chans, GREEN_HINTS)
            r = pick_channel(chans, RED_HINTS)
            flag = '' if (g and r) else '   <-- missing a channel'
            print(f'  {key[0]}-{key[1]}  green={g}  red={r}{flag}')
            print(f'      all channels: {", ".join(sorted(chans))}')
        return

    if not args.well or args.tile is None:
        raise SystemExit('--well and --tile are required (or use --list)')

    key = (args.well, args.tile)
    if key not in found:
        raise SystemExit(f'{args.well}-{args.tile} not found. '
                         f'Run with --list to see what is there.')
    chans = found[key]

    green = pick_channel(chans, GREEN_HINTS, args.green_channel)
    red = pick_channel(chans, RED_HINTS, args.red_channel)
    if not green or not red:
        raise SystemExit(
            f'could not identify both channels for {args.well}-{args.tile}.\n'
            f'  green={green}  red={red}\n'
            f'  present: {", ".join(sorted(chans))}\n'
            f'  pass --green-channel / --red-channel with the exact suffix.')

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'neoPIP_FUCCI_outputs')
    os.makedirs(out_dir, exist_ok=True)
    stem = f'{args.well}-{args.tile}'

    print(f'{stem}:  green = {green}   magenta = {red}')
    print('starting the JVM (first run in a session takes a minute)...')
    import imagej
    ij = imagej.init('sc.fiji:fiji:2.14.0')

    stack, px_detected = read_pair(ij, chans[green], chans[red],
                                   max_frames=args.max_frames)
    t, c, h, w = stack.shape
    print(f'  stack {t} frames x {c} channels x {h} x {w}')

    pixel_size = args.pixel_size or px_detected
    if args.pixel_size and px_detected:
        print(f'  pixel size {args.pixel_size} um/px (overriding the '
              f'{px_detected:.4f} in the file)')
    elif px_detected:
        print(f'  pixel size {px_detected:.4f} um/px (from the .vsi metadata)')
    elif args.pixel_size:
        print(f'  pixel size {args.pixel_size} um/px (given on the command line)')
    else:
        print('  no pixel size: the file carries no calibration and '
              '--pixel-size was not given, so no scale bar will be drawn')

    if not args.no_tif:
        p = os.path.join(out_dir, f'{stem}_overlay.tif')
        write_tif(p, stack, green, red, pixel_size_um=pixel_size,
                  frame_interval_min=args.frame_interval)
        print(f'  wrote {p}')
        print('    open with File > Open in Fiji; it arrives as a composite '
              'with the frame slider already set')

    limits = stretch_limits(stack, args.clip[0], args.clip[1])
    print('  display range (whole series): '
          + '  '.join(f'{n}=[{lo:.0f}, {hi:.0f}]'
                      for n, (lo, hi) in zip((green, red), limits)))

    if args.video:
        path = args.video
        if path == 'auto':
            path = os.path.join(out_dir, f'{stem}_overlay.mp4')
        elif not os.path.isabs(path):
            path = os.path.join(out_dir, path)
        labels = None
        if not args.no_timestamp:
            labels = [f'{i + 1:3d}/{t}   {i * args.frame_interval / 60:5.2f} h'
                      for i in range(t)]
        write_video(path, stack, limits, args.fps, labels=labels,
                    max_px=args.video_max_px,
                    pixel_size_um=None if args.no_scale_bar else pixel_size,
                    bar_um=args.scale_bar_um)

    if args.project:
        import tifffile
        fn = np.max if args.project == 'max' else np.mean
        proj = fn(stack.astype(np.float32), axis=0)
        p = os.path.join(out_dir, f'{stem}_{args.project}proj.tif')
        write_tif(p, proj.astype(np.uint16)[None], green, red,
                  pixel_size_um=pixel_size)
        rgb = to_rgb(proj.astype(np.uint16), limits)
        if pixel_size and not args.no_scale_bar:
            rgb = stamp(rgb, None, pixel_size,
                        args.scale_bar_um or nice_bar_um(pixel_size,
                                                         rgb.shape[1]))
        from PIL import Image
        png = os.path.join(out_dir, f'{stem}_{args.project}proj.png')
        Image.fromarray(rgb).save(png)
        print(f'  wrote {p}')
        print(f'  wrote {png}')

    print('done.')


if __name__ == '__main__':
    main()
