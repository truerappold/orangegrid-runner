"""
Write one image per timelapse frame with cell and nucleus outlines drawn on.

Unlike maskPrint_timelapse.py (which makes a single multi-panel PNG for judging
the tracking at a glance), this writes a separate full-resolution file per frame,
so the series can be opened as a stack, scrubbed through, or dropped into a
figure one frame at a time.

    python maskOutline_timelapse.py <dir> <out_dir> --pics-subdir pics --max-dist 25

Output per frame:  <out_dir>/<well>_tile<N>_t<NN>.tif   (ImageJ 24-bit RGB)
    - background   the direct mNG channel, contrast stretched (or DAPI/donor/black)
    - red outline  cell mask from Cellpose
    - cyan outline nuclear mask from DAPI
    - amber outline  segmented but not tracked (dropped by the distance or
                   collision rule) -- drawn so you can see what was discarded
    - white number the track id, at the nuclear centroid

A NOTE ON FORMAT: .vsi cannot be written.  It is a proprietary Olympus/Evident
container and Bio-Formats ships a CellSensReader with no corresponding writer,
so no tool -- Fiji included -- can produce one.

Frames are written as ImageJ-format 24-bit RGB TIFFs rather than OME-TIFF.  OME
has no "single RGB image" concept, only SizeC, so an RGB OME-TIFF declares
SizeC=3 and Fiji imports three greyscale planes per frame -- one per colour --
which looks like triplicate frames differing only in how the outlines render.
--format ome restores that behaviour if something needs the OME metadata.

--save-labels additionally writes the raw 16-bit label masks beside each frame,
which is what you want if anything downstream needs to re-measure rather than
just look.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

import skimage as ski
from skimage.segmentation import find_boundaries

import timelapse_tracking as tt
import timelapse_imaging as ti

# imagej / cellpose / pystackreg are imported inside main() rather than here, so
# the rendering functions above can be exercised without a JVM or a GPU.

CELL_COLOR = (230, 60, 50)        # red
NUC_COLOR = (60, 200, 230)        # cyan
DROPPED_COLOR = (255, 190, 0)     # amber -- grey disappeared against bright cells
TEXT_COLOR = (255, 255, 255)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
# raw image channels that can be included, with a default LUT colour each
IMAGE_CHANNELS = {
    'mng':      (tt.CH_MNG,      (0.0, 1.0, 0.0)),   # green
    'dapi':     (tt.CH_DAPI,     (0.0, 0.4, 1.0)),   # blue
    'donor':    (tt.CH_DONOR,    (1.0, 0.0, 1.0)),   # magenta
    'acceptor': (tt.CH_ACCEPTOR, (1.0, 1.0, 0.0)),   # yellow
}


def stretch(img, low=20.0, high=99.9):
    """Percentile contrast stretch to uint8, guarded against a flat image.

    Defaults suit sparse fields.  These tiles are roughly 2-3% cells, so a
    conventional 1-99.5 stretch is set almost entirely by background: on the
    example tile it put white at the 99.5th percentile (836 for mNG against a
    maximum of 24050) and saturated half the nuclei.  Putting black near the
    background level and white at 99.9 keeps the cells on scale and renders the
    background as black, which is what you want under an overlay.
    """
    img = np.asarray(img, float)
    lo, hi = np.percentile(img, [low, high])
    if hi <= lo:
        return np.zeros(img.shape, np.uint8)
    return (np.clip((img - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def outline(mask, width=1):
    """Boolean outline of every label, thickened to `width` pixels.

    Thickening dilates outward, so a 2-px outline spills one pixel beyond the
    mask edge. That is what you want visually -- the rim reads as sitting on the
    boundary rather than eating into the cell.
    """
    b = find_boundaries(mask, mode='inner')
    if width > 1:
        fp = np.ones((width, width), bool)
        try:                       # skimage renamed this in 0.26
            b = ski.morphology.dilation(b, footprint=fp)
        except (AttributeError, TypeError):
            b = ski.morphology.binary_dilation(b, footprint=fp)
    return b


def _font(size):
    from PIL import ImageFont
    try:
        from matplotlib import font_manager
        path = font_manager.findfont('DejaVu Sans')
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _background_layer(plane, mask_shape, background):
    if background == 'black':
        return np.zeros(mask_shape, np.uint8)
    ch = {'mng': tt.CH_MNG, 'dapi': tt.CH_DAPI, 'donor': tt.CH_DONOR}[background]
    return stretch(plane[:, :, ch])


def _text_layer(shape, tracks_this_tp, centroids, font_size, label_source):
    """White numbers on black, as its own 8-bit layer."""
    from PIL import Image, ImageDraw
    img = Image.new('L', (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(img)
    font = _font(font_size)
    for label, track_id in sorted(tracks_this_tp.items()):
        if label not in centroids:
            continue
        r, c = centroids[label]
        draw.text((c, r), str(track_id if label_source == 'track' else label),
                  fill=255, font=font, anchor='mm')
    return np.asarray(img)


def render_layers(plane, mask_cell, mask_nuc, tracks_this_tp, centroids,
                  image_channels=('mng', 'dapi'), line_width=2, font_size=28,
                  label_source='track', dropped_labels=(),
                  stretch_low=20.0, stretch_high=99.9):
    """The overlay as separate 8-bit layers, for a multichannel composite.

    Returns (names, colors, stack) with stack shaped [C, H, W].  Keeping the
    outlines in their own channels is what makes a LUT meaningful: in the
    flattened RGB the greyscale background sits in all three colour channels, so
    splitting it gives you three copies of the image with different outlines
    burned in, not separable layers.

    image_channels picks which raw channels to include, each as its own layer
    with its own LUT -- so mNG and DAPI can be overlaid and windowed separately,
    and either can be toggled off.  Nucleus outlines default to magenta rather
    than cyan here, because cyan is hard to tell from green mNG over blue DAPI.

    Image layers keep their original 16-bit values.  These fields are only 2-3%
    cells, so any fixed display stretch is set almost entirely by background and
    is easy to get wrong; leaving the data raw means the mistake is fixable with
    the Brightness/Contrast sliders rather than another Cellpose run.
    """
    tracked = (np.isin(mask_cell, list(tracks_this_tp)) if tracks_this_tp
               else np.zeros(mask_cell.shape, bool))
    drop_mask = (np.where(np.isin(mask_cell, list(dropped_labels)), mask_cell, 0)
                 if dropped_labels else np.zeros_like(mask_cell))
    ON = np.uint16(65535)
    layers = []
    for cname in image_channels:
        if cname not in IMAGE_CHANNELS:
            raise ValueError(f'unknown image channel {cname!r}; choose from '
                             f'{sorted(IMAGE_CHANNELS)}')
        ch, colour = IMAGE_CHANNELS[cname]
        # raw 16-bit, NOT stretched: Brightness/Contrast in Fiji then has the
        # full range to work with.  Baking an 8-bit stretch in here would mean
        # re-running Cellpose every time the display turned out wrong.
        layers.append((cname, colour, plane[:, :, ch].astype(np.uint16)))
    mark = lambda b: np.where(b, ON, np.uint16(0))
    layers += [
        ('cells', (0.9, 0.24, 0.20),
         mark(outline(np.where(tracked, mask_cell, 0), line_width))),
        ('nuclei', (1.0, 0.0, 1.0), mark(outline(mask_nuc, line_width))),
        ('untracked', (1.0, 0.75, 0.0), mark(outline(drop_mask, line_width))),
        ('labels', (1.0, 1.0, 1.0),
         (_text_layer(mask_cell.shape, tracks_this_tp, centroids, font_size,
                      label_source).astype(np.uint16) * 257)),
    ]
    names = [n for n, _c, _l in layers]
    colors = [c for _n, c, _l in layers]
    stack = np.stack([l for _n, _c, l in layers])
    return names, colors, stack


def render_frame(plane, mask_cell, mask_nuc, tracks_this_tp, centroids,
                 background='mng', line_width=2, font_size=28,
                 label_source='track', dropped_labels=()):
    """RGB uint8 image at native resolution with outlines and labels burned in."""
    from PIL import Image, ImageDraw

    base = _background_layer(plane, mask_cell.shape, background)
    rgb = np.dstack([base] * 3)

    tracked = (np.isin(mask_cell, list(tracks_this_tp)) if tracks_this_tp
               else np.zeros(mask_cell.shape, bool))
    if dropped_labels:
        drop_mask = np.where(np.isin(mask_cell, list(dropped_labels)), mask_cell, 0)
        rgb[outline(drop_mask, line_width)] = DROPPED_COLOR
    rgb[outline(np.where(tracked, mask_cell, 0), line_width)] = CELL_COLOR
    rgb[outline(mask_nuc, line_width)] = NUC_COLOR

    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    font = _font(font_size)
    for label, track_id in sorted(tracks_this_tp.items()):
        if label not in centroids:
            continue
        r, c = centroids[label]
        draw.text((c, r), str(track_id if label_source == 'track' else label),
                  fill=TEXT_COLOR, font=font, anchor='mm',
                  stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(img)


def write_composite(path, names, colors, stack, pixel_size_um=None):
    """Multichannel ImageJ composite with a LUT preset per channel.

    Fiji opens this in composite mode with the channels already coloured, so
    Brightness/Contrast and the Channels Tool act per layer and the settings
    propagate across the whole series.
    """
    import tifffile

    ramp = np.arange(256, dtype=np.float64)
    luts = [np.stack([(ramp * r).astype(np.uint8),
                      (ramp * g).astype(np.uint8),
                      (ramp * b).astype(np.uint8)]) for r, g, b in colors]
    meta = {'axes': 'CYX', 'mode': 'composite', 'unit': 'um',
            'LUTs': luts, 'Labels': names}
    kw = {}
    if pixel_size_um:
        kw['resolution'] = (1 / pixel_size_um, 1 / pixel_size_um)
    tifffile.imwrite(path, stack, imagej=True, metadata=meta, **kw)
    return f'imagej composite, {len(names)} channels'


def write_tiff(path, rgb, pixel_size_um=None, fmt='imagej'):
    """Write the RGB frame as a TIFF that Fiji opens as ONE colour image.

    Default is an ImageJ-format 24-bit RGB TIFF, not OME-TIFF, and that matters.
    The OME data model has no notion of "this is one RGB image" -- it only has
    SizeC -- so an RGB OME-TIFF is declared SizeC=3 and Bio-Formats hands Fiji
    three separate greyscale planes per frame, one per colour. Importing the
    series then gives three times the frames, differing only in how each outline
    colour renders. An ImageJ RGB TIFF has a real 24-bit colour type, so one
    file is one plane.

    fmt='ome' restores OME-TIFF for anything downstream that needs the OME
    metadata and can cope with the channel split.
    """
    try:
        import tifffile
    except ImportError:
        from PIL import Image
        Image.fromarray(rgb).save(path)
        return 'tiff via Pillow (tifffile not installed, no calibration)'

    if fmt == 'ome':
        kw = {}
        if pixel_size_um:
            kw['metadata'] = {'PhysicalSizeX': pixel_size_um,
                              'PhysicalSizeY': pixel_size_um,
                              'PhysicalSizeXUnit': 'um',
                              'PhysicalSizeYUnit': 'um'}
        tifffile.imwrite(path, rgb, photometric='rgb', ome=True, **kw)
        return 'ome-tiff (Fiji shows 3 channels per frame)'

    kw = dict(imagej=True, photometric='rgb', metadata={'unit': 'um'})
    if pixel_size_um:
        kw['resolution'] = (1 / pixel_size_um, 1 / pixel_size_um)
    tifffile.imwrite(path, rgb, **kw)
    return 'imagej rgb tiff'


def write_series(path, frames, names=None, colors=None, pixel_size_um=None):
    """Write a whole tile as ONE ImageJ hyperstack, so Fiji needs no import step.

    Importing 21 separate 5-channel files as an image sequence gives Fiji 105
    flat slices with no idea which axis is which; Make Composite then defaults
    to 3 channels and shreds the series.  A single file carries frames= and
    channels= in its ImageJ header, so it opens with the right sliders already.

    frames: list of [C,H,W] arrays (composite) or [H,W,3] arrays (RGB).
    """
    import tifffile

    data = np.stack(frames)
    kw = {}
    if pixel_size_um:
        kw['resolution'] = (1 / pixel_size_um, 1 / pixel_size_um)

    if data.ndim == 4 and data.shape[-1] == 3:          # RGB series
        tifffile.imwrite(path, data, imagej=True, photometric='rgb',
                         metadata={'axes': 'TYXS', 'unit': 'um'}, **kw)
        return f'imagej rgb hyperstack, {data.shape[0]} frames'

    ramp = np.arange(256, dtype=np.float64)
    luts = [np.stack([(ramp * r).astype(np.uint8),
                      (ramp * g).astype(np.uint8),
                      (ramp * b).astype(np.uint8)]) for r, g, b in colors]
    tifffile.imwrite(path, data, imagej=True, photometric='minisblack',
                     metadata={'axes': 'TCYX', 'mode': 'composite',
                               'unit': 'um', 'LUTs': luts, 'Labels': names},
                     **kw)
    return (f'imagej composite hyperstack, {data.shape[0]} frames x '
            f'{data.shape[1]} channels')


def write_labels(path, mask_cell, mask_nuc):
    """Raw label masks as a 2-plane 16-bit TIFF: plane 0 cells, plane 1 nuclei."""
    stack = np.stack([mask_cell.astype(np.uint16), mask_nuc.astype(np.uint16)])
    try:
        import tifffile
        tifffile.imwrite(path, stack, imagej=True, photometric='minisblack',
                         metadata={'axes': 'CYX', 'mode': 'composite',
                                   'Labels': ['cell_labels', 'nucleus_labels']})
    except ImportError:
        from PIL import Image
        Image.fromarray(stack[0]).save(path.replace('.tif', '_cells.tif'))
        Image.fromarray(stack[1]).save(path.replace('.tif', '_nuclei.tif'))


# --------------------------------------------------------------------------- #
def main(dir, out_dir, args, ij, model):
    import tqdm
    from pystackreg import StackReg

    sr = StackReg(StackReg.RIGID_BODY)
    os.makedirs(out_dir, exist_ok=True)

    try:
        seqs = pd.read_csv(os.path.join(dir, 'seqs.csv'), index_col=0)
    except FileNotFoundError:
        print(f'Error: seqs.csv not found in {dir}.')
        return

    pics_dir = os.path.join(dir, args.pics_subdir) if args.pics_subdir else dir
    filelist = glob.glob(os.path.join(pics_dir, '*.vsi'))
    if not filelist:
        print(f'No VSI files found in {pics_dir}.')
        return

    series, problems = tt.discover_blocks(filelist)
    for p in problems[:20]:
        print(f'  {p}')

    keys = sorted(k for k in series if k[0] in seqs.index)
    if args.wells:
        keys = [k for k in keys if k[0] in args.wells]
    if args.tiles:
        keys = [k for k in keys if k[1] in args.tiles]
    if not keys:
        print('No well/tile series selected.')
        return
    print(f'{len(keys)} well/tile series selected')

    cellpose_kwargs = dict(flow_threshold=args.flow_threshold,
                           cellprob_threshold=args.cellprob_threshold,
                           tile_norm_blocksize=args.tile_norm_blocksize)

    image_channels = [c.strip() for c in args.image_channels.split(',') if c.strip()]
    bad = [c for c in image_channels if c not in IMAGE_CHANNELS]
    if bad:
        raise SystemExit(f'unknown --image-channels {bad}; choose from '
                         f'{sorted(IMAGE_CHANNELS)}')

    n_written = 0
    pbar = tqdm.tqdm(total=len(keys), desc='Rendering series', ncols=150)
    for well, tile in keys:
        blocks = series[(well, tile)]
        print(f'on well {well} tile {tile} (blocks {blocks})')
        try:
            res = ti.process_tile(
                ij, sr, model, pics_dir, well, tile, blocks, cellpose_kwargs,
                max_dist=args.max_dist, match_method=args.match_method,
                first_step_max_dist=args.first_step_max_dist,
                allow_division=args.track_divisions,
                division_max_dist=args.division_max_dist,
                hole_area=args.hole_area, dapi_mode=args.dapi_threshold,
                dapi_sigma=args.dapi_sigma, dapi_contrast=args.dapi_contrast, do_background=not args.no_background,
                keep_masks=True, log=print)
        except Exception as exc:
            print(f'  ERROR on {well} tile {tile}: {type(exc).__name__}: {exc}')
            pbar.update(1)
            continue

        dropped_by_tp = {}
        for rec in res['records']:
            if rec['dropped']:
                dropped_by_tp.setdefault(rec['timepoint'], []).append(
                    rec['cell_label'])

        # Re-read each frame to render on: process_tile keeps the masks but
        # discards the pixels as it goes, so the background image has to be
        # pulled again.  Cheap next to Cellpose, and it keeps peak memory at one
        # frame instead of the whole series.
        tp = 0
        donor_ref = None
        last_fmt = None
        series_frames = []
        series_names = series_colors = None
        for block in blocks:
            with ti.BlockReader(ij, pics_dir, well, tile, block) as reader:
                for frame in range(reader.n_frames):
                    tp += 1
                    if tp not in res['masks']:
                        continue
                    plane = reader.frame(frame)
                    plane = ti.register_to_reference(
                        plane, None if tp == 1 else donor_ref, sr)
                    if tp == 1:
                        donor_ref = plane[:, :, tt.CH_DONOR].copy()

                    _orig, mask_cell, mask_nuc = res['masks'][tp]
                    rgb = render_frame(
                        plane, mask_cell, mask_nuc,
                        res['tracks'].get(tp, {}),
                        res['nuc_centroids'].get(tp, {}),
                        background=args.background, line_width=args.line_width,
                        font_size=args.font_size, label_source=args.label_source,
                        dropped_labels=dropped_by_tp.get(tp, ()))

                    name = f'{well}_tile{tile}_t{tp:02d}'
                    ext = '.ome.tif' if args.format == 'ome' else '.tif'
                    if args.format == 'composite':
                        nm, cols, stack = render_layers(
                            plane, mask_cell, mask_nuc,
                            res['tracks'].get(tp, {}),
                            res['nuc_centroids'].get(tp, {}),
                            image_channels=image_channels,
                            line_width=args.line_width,
                            font_size=args.font_size,
                            label_source=args.label_source,
                            dropped_labels=dropped_by_tp.get(tp, ()),
                            stretch_low=args.stretch_low,
                            stretch_high=args.stretch_high)
                        series_names, series_colors = nm, cols
                        if args.single_stack:
                            series_frames.append(stack)
                            fmt = 'buffered'
                        else:
                            fmt = write_composite(
                                os.path.join(out_dir, name + ext), nm, cols,
                                stack, pixel_size_um=args.pixel_size)
                    elif args.single_stack:
                        series_frames.append(rgb)
                        fmt = 'buffered'
                    else:
                        fmt = write_tiff(os.path.join(out_dir, name + ext), rgb,
                                         pixel_size_um=args.pixel_size,
                                         fmt=args.format)
                    if args.save_labels:
                        write_labels(
                            os.path.join(out_dir, name + '_labels.tif'),
                            mask_cell, mask_nuc)
                    n_written += 1
                    last_fmt = fmt
                    del plane
        if args.single_stack and series_frames:
            stack_path = os.path.join(out_dir, f'{well}_tile{tile}.tif')
            last_fmt = write_series(stack_path, series_frames,
                                    names=series_names, colors=series_colors,
                                    pixel_size_um=args.pixel_size)
            series_frames.clear()
        print(f'  wrote {tp} frame(s)'
              + (f' as {last_fmt}' if tp else ' (nothing to render)'))
        pbar.update(1)
    pbar.close()
    print(f'\n{n_written} frame image(s) written to {out_dir}')
    print('Open the series in Fiji with File > Import > Image Sequence to '
          'scrub through it as a stack.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Write one outlined image per timelapse frame.')
    parser.add_argument('directory')
    parser.add_argument('output_dir')
    parser.add_argument('--wells', nargs='*', default=None)
    parser.add_argument('--tiles', nargs='*', type=int, default=None)
    parser.add_argument('--max-dist', type=float, default=25.0)
    parser.add_argument('--match-method', choices=['nearest', 'hungarian'],
                        default='nearest')
    parser.add_argument('--first-step-max-dist', type=float, default=None)
    parser.add_argument('--track-divisions', action='store_true')
    parser.add_argument('--division-max-dist', type=float, default=None)
    parser.add_argument('--background', choices=['mng', 'dapi', 'donor', 'black'],
                        default='mng', help='what to draw the outlines on top of')
    parser.add_argument('--label-source', choices=['track', 'cell'],
                        default='track',
                        help="'track' numbers stay with a cell across the whole "
                             "series; 'cell' is the raw per-frame Cellpose label, "
                             "which is re-derived every frame and so changes "
                             "meaning between frames (default track)")
    parser.add_argument('--line-width', type=int, default=2)
    parser.add_argument('--font-size', type=int, default=28)
    parser.add_argument('--pixel-size', type=float, default=0.325,
                        help='um per pixel, written into the OME metadata '
                             '(default 0.325)')
    parser.add_argument('--format', choices=['imagej', 'composite', 'ome'],
                        default='imagej',
                        help="'imagej' writes a 24-bit RGB TIFF that Fiji opens "
                             "as one colour image per frame; 'ome' writes "
                             "OME-TIFF, which Bio-Formats splits into 3 "
                             "greyscale channels per frame. 'composite' writes "
                             "5 real channels (background / cells / nuclei / "
                             "untracked / labels) with LUTs preset, so each "
                             "layer can be coloured and toggled independently "
                             "(default imagej)")
    parser.add_argument('--image-channels', default='mng,dapi',
                        help='raw channels to include as separate layers in a '
                             'composite, comma separated, from mng/dapi/donor/'
                             'acceptor. Each gets its own LUT so they can be '
                             'overlaid and windowed independently '
                             '(default mng,dapi). Ignored by --format imagej, '
                             'which uses the single --background channel')
    parser.add_argument('--stretch-low', type=float, default=20.0,
                        help='percentile mapped to black (default 20)')
    parser.add_argument('--stretch-high', type=float, default=99.9,
                        help='percentile mapped to white (default 99.9)')
    parser.add_argument('--single-stack', action='store_true',
                        help='write ONE hyperstack per tile instead of one file '
                             'per frame. Fiji then opens it with the channel and '
                             'frame sliders already correct, with no Image '
                             'Sequence import and no Make Composite step. '
                             'Buffers the series in memory first: about 0.6 GB '
                             'for 21 frames x 5 channels at 2304x2304')
    parser.add_argument('--save-labels', action='store_true',
                        help='also write the raw 16-bit label masks per frame')
    parser.add_argument('--flow-threshold', type=float, default=0.4)
    parser.add_argument('--cellprob-threshold', type=float, default=0.0)
    parser.add_argument('--tile-norm-blocksize', type=int, default=0)
    parser.add_argument('--hole-area', type=int, default=200)
    parser.add_argument('--dapi-threshold', choices=['per-cell', 'global'],
                        default='per-cell',
                        help='per-cell thresholds DAPI inside each Cellpose mask '
                             'separately. A single global Otsu cannot cope with '
                             "the field's dynamic range and shaves dim nuclei "
                             'down to their bright cores (default per-cell)')
    parser.add_argument('--dapi-sigma', type=float, default=2.0,
                        help='Gaussian smoothing on DAPI before thresholding; 0 '
                             'disables. Without it the threshold cuts through '
                             'pixel noise and the outlines come out ragged '
                             '(default 2.0)')
    parser.add_argument('--dapi-contrast', type=float, default=1.5,
                        help='reject a cell as having no nucleus unless its '
                             'bright DAPI median exceeds its dim median by this '
                             'factor (default 1.5)')
    parser.add_argument('--pics-subdir', default='Pics')
    parser.add_argument('--no-background', action='store_true')
    args = parser.parse_args()

    import imagej
    from cellpose import models, io

    io.logger_setup()
    model = None
    try:
        model = models.CellposeModel(gpu=True)
    except Exception as exc:
        print(f'Error initializing Cellpose model: {exc}')
    if not model:
        raise SystemExit('Cellpose model could not be initialized. Exiting.')
    try:
        ij = imagej.init('sc.fiji:fiji:2.14.0')
    except Exception as exc:
        raise SystemExit(f'Error initializing ImageJ: {exc}')
    print(f'ImageJ version: {ij.getVersion()}')
    main(args.directory, args.output_dir, args, ij, model)
