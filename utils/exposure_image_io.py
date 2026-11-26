"""CPU image/track I/O for exposure inspection, without importing the renderer."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
import importlib.util
import math
import struct

import numpy as np
from PIL import Image, ImageCms

from utils.exposure_balance import srgb_to_linear


@dataclass
class Frame:
    image_id: int
    camera_id: int
    name: str
    width: int
    height: int
    xy: np.ndarray
    tracks: np.ndarray


def checked_name(name):
    """COLMAP names must stay inside the selected image/output directories."""
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError(f'Unsafe COLMAP image name: {name!r}')
    return path.as_posix()


def _read(stream, count):
    data = stream.read(count)
    if len(data) != count:
        raise ValueError('Truncated COLMAP binary file')
    return data


def camera_sizes(model):
    model = Path(model)
    binary = model / 'cameras.bin'
    cameras = {}
    if binary.is_file() and (model / 'images.bin').is_file():
        # Load only the lightweight parser, not scene.__init__ (which needs CUDA).
        path = Path(__file__).resolve().parents[1] / 'scene' / 'colmap_loader.py'
        spec = importlib.util.spec_from_file_location('_exposure_colmap', path)
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        for camera_id, camera in loader.read_intrinsics_binary(binary).items():
            cameras[camera_id] = (camera.width, camera.height)
        return cameras, 'bin'
    with (model / 'cameras.txt').open() as stream:
        for line in stream:
            if line.strip() and not line.lstrip().startswith('#'):
                row = line.split()
                cameras[int(row[0])] = (int(row[2]), int(row[3]))
    return cameras, 'txt'


def load_frames(model, max_samples=1500, seed=0):
    """Retain a bounded sample per image; never load all keypoints at once."""
    cameras, extension = camera_sizes(model)
    frames = []

    def append(image_id, camera_id, name, xy, tracks):
        name = checked_name(name)
        width, height = cameras[camera_id]
        good = np.flatnonzero((tracks >= 0) & np.isfinite(xy).all(axis=1))
        if len(good) > max_samples:
            rng = np.random.default_rng(np.random.SeedSequence([seed, image_id]))
            good = rng.choice(good, max_samples, replace=False)
        # Advanced indexing releases the full binary buffer after this frame.
        frames.append(Frame(image_id, camera_id, name, width, height,
                            xy[good].copy(), tracks[good].copy()))

    path = Path(model) / ('images.' + extension)
    if extension == 'bin':
        dtype = np.dtype([('xy', '<f8', (2,)), ('track', '<i8')])
        with path.open('rb') as stream:
            count, = struct.unpack('<Q', _read(stream, 8))
            for _ in range(count):
                header = struct.unpack('<idddddddi', _read(stream, 64))
                name = bytearray()
                while True:
                    char = _read(stream, 1)
                    if char == b'\0':
                        break
                    name.extend(char)
                n_points, = struct.unpack('<Q', _read(stream, 8))
                points = np.frombuffer(_read(stream, 24 * n_points), dtype=dtype)
                append(header[0], header[-1], name.decode('utf-8'),
                       points['xy'], points['track'])
    else:
        with path.open() as stream:
            while True:
                header = stream.readline()
                if not header:
                    break
                if not header.strip() or header.lstrip().startswith('#'):
                    continue
                row = header.rstrip('\r\n').split(maxsplit=9)
                if len(row) != 10:
                    raise ValueError(f'Invalid COLMAP image header: {header[:100]}')
                observations = stream.readline()
                if observations == '':
                    raise ValueError('Missing COLMAP POINTS2D line')
                parts = observations.split()
                if len(parts) % 3:
                    raise ValueError('Invalid COLMAP POINTS2D line')
                xy = np.asarray(list(zip(map(float, parts[0::3]),
                                         map(float, parts[1::3]))), dtype=float).reshape(-1, 2)
                tracks = np.asarray([int(x) for x in parts[2::3]], dtype=np.int64)
                append(int(row[0]), int(row[8]), row[9], xy, tracks)
    frames.sort(key=lambda item: item.name)
    if len({frame.name for frame in frames}) != len(frames):
        raise ValueError('COLMAP has duplicate image names')
    return frames


def ordered_map(function, items, workers):
    """Bound outstanding image decodes rather than queuing every full image."""
    from collections import deque
    with ThreadPoolExecutor(max_workers=workers) as pool:
        iterator, pending = iter(items), deque()
        for _ in range(workers * 2):
            item = next(iterator, None)
            if item is None:
                break
            pending.append(pool.submit(function, item))
        while pending:
            yield pending.popleft().result()
            item = next(iterator, None)
            if item is not None:
                pending.append(pool.submit(function, item))


def _exif_values(image):
    tags = {33434: 'ExposureTime', 33437: 'FNumber', 34855: 'ISO',
            34850: 'ExposureProgram', 41986: 'ExposureMode',
            37380: 'ExposureBiasValue'}
    exif = image.getexif()
    values = dict(exif)
    if 34665 in exif:
        try:
            values.update(exif.get_ifd(34665))
        except (KeyError, TypeError, ValueError):
            pass
    result = {}
    for tag, name in tags.items():
        value = values.get(tag)
        if value is not None:
            try:
                number = float(value)
                if math.isfinite(number):
                    result[name] = number
            except (TypeError, ValueError, ZeroDivisionError):
                pass
    if all(result.get(key, 0) > 0 for key in ('ExposureTime', 'ISO', 'FNumber')):
        result['exposure_proxy_ev'] = math.log2(
            result['ExposureTime'] * result['ISO'] / result['FNumber'] ** 2)
    return result


def read_rgb(path):
    """Decode into sRGB without silently changing COLMAP pixel orientation."""
    with Image.open(path) as image:
        if image.getexif().get(274, 1) not in (1, None):
            raise ValueError(f'EXIF-rotated image needs an explicitly aligned COLMAP model: {path}')
        exif = _exif_values(image)
        profile = image.info.get('icc_profile')
        if profile:
            image = ImageCms.profileToProfile(image, BytesIO(profile),
                ImageCms.createProfile('sRGB'), outputMode='RGB')
        else:
            image = image.convert('RGB')
        return np.asarray(image, dtype=np.uint8).copy(), exif


def sample_frame(frame, image_root, patch_radius=1, allow_resized=False):
    rgb, exif = read_rgb(Path(image_root) / frame.name)
    height, width = rgb.shape[:2]
    resized = (width, height) != (frame.width, frame.height)
    if resized and not allow_resized:
        raise ValueError(f'{frame.name}: image {width}x{height} differs from COLMAP '
                         f'{frame.width}x{frame.height}; use --allow-resized only for '
                         'a known full-frame resize, never a crop')
    sx, sy = width / frame.width, height / frame.height
    if resized and abs(sx / sy - 1) > .005:
        raise ValueError(f'{frame.name}: image aspect ratio differs from COLMAP')
    # COLMAP pixel centers begin at (0.5,0.5); array centers begin at (0,0).
    xy = np.rint(frame.xy * [sx, sy] - .5).astype(np.int64)
    r = patch_radius
    valid = ((xy[:, 0] >= r) & (xy[:, 0] < width-r) &
             (xy[:, 1] >= r) & (xy[:, 1] < height-r))
    xy, tracks = xy[valid], frame.tracks[valid]
    offsets = np.asarray([(x, y) for y in range(-r, r+1)
                          for x in range(-r, r+1)], dtype=int)
    positions = xy[:, None, :] + offsets[None, :, :]
    patches = rgb[positions[:, :, 1], positions[:, :, 0]]
    linear = srgb_to_linear(patches.astype(np.float64) / 255.)
    luminance = np.median(linear @ np.array([.2126, .7152, .0722]), axis=1)
    saturation = (patches.max(axis=2) >= 250).mean(axis=1)
    valid = (luminance >= .01) & (luminance <= .8) & (saturation < .2)
    return dict(tracks=tracks[valid], log_luminance=np.log2(luminance[valid]),
                width=width, height=height, exif=exif, resized=resized,
                sampled_observations=len(frame.tracks), valid_observations=int(valid.sum()))


def save_rgb(rgb, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    image = Image.fromarray(rgb)
    if suffix in ('.jpg', '.jpeg'):
        image.save(path, quality=100, subsampling=0)
    elif suffix == '.png':
        image.save(path, compress_level=3)
    elif suffix in ('.tif', '.tiff'):
        image.save(path, compression='tiff_lzw')
    else:
        raise ValueError(f'Unsupported output image extension: {suffix}')


def save_thumbnail(rgb, path, width=640):
    image = Image.fromarray(rgb)
    image.thumbnail((width, width), Image.Resampling.LANCZOS)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=90, subsampling=0)
