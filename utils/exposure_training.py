"""CPU validation and provenance for training on an exposure-normalized export."""
import hashlib
import json
from pathlib import Path

from PIL import Image


def _inside(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _absolute_path(value, field):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f'Exposure metadata {field} must be an absolute path')
    return Path(value).resolve()


def validate_exposure_export(exposure_dir, source):
    """Return the corrected RGB directory and a JSON-serializable provenance record.

    This reads image headers, not all image pixels. It checks a completed export,
    exact registered-name coverage, and unchanged dimensions. It does not judge
    whether the photometric correction improved the scene; fit warnings remain
    visible in the plan and are saved with each run.
    """
    from scripts.view_comparisons import read_colmap_records

    directory = Path(exposure_dir).expanduser().resolve()
    source = Path(source).expanduser().resolve()
    contents = {'status.json': (directory / 'status.json').read_bytes()}
    status = json.loads(contents['status.json'])
    if not isinstance(status, dict) or status.get('status') != 'complete':
        raise ValueError(f'Exposure export is not complete: {directory / "status.json"}')
    contents.update({name: (directory / name).read_bytes()
                     for name in ('summary.json', 'frames.json')})
    summary, frames = (json.loads(contents[name]) for name in ('summary.json', 'frames.json'))
    if not isinstance(summary, dict) or not isinstance(frames, list):
        raise ValueError('Exposure summary must be an object and frames must be a list')
    if _absolute_path(summary.get('source'), 'source') != source:
        raise ValueError(f'Exposure source does not match the training source: {source}')
    original = _absolute_path(summary.get('images'), 'images')
    corrected = (directory / 'corrected').resolve()
    if not corrected.is_dir():
        raise FileNotFoundError(f'Corrected image directory not found: {corrected}')
    if not _inside(corrected, directory) or _inside(corrected, original):
        raise ValueError('Corrected images must be a separate tree inside the exposure export')

    records = read_colmap_records(source)
    count = len(records)
    for label, value in (('status', status.get('registered_images')),
                         ('summary', summary.get('registered_images'))):
        if type(value) is not int or value != count:
            raise ValueError(f'Exposure {label} image count {value!r} does not match COLMAP ({count})')
    by_name = {}
    for frame in frames:
        name = frame.get('name') if isinstance(frame, dict) else None
        if (not isinstance(name, str) or not name or Path(name).is_absolute()
                or '..' in Path(name).parts or Path(name).as_posix() != name):
            raise ValueError(f'Unsafe or invalid exposure frame name: {name!r}')
        if name in by_name:
            raise ValueError(f'Duplicate exposure frame: {name}')
        if frame.get('corrected_rel') != (Path('corrected') / name).as_posix():
            raise ValueError(f'Exposure corrected path does not match its frame: {name}')
        by_name[name] = frame
    missing = sorted(set(records) - set(by_name))
    extra = sorted(set(by_name) - set(records))
    if missing or extra:
        raise ValueError(f'Exposure frames differ from COLMAP: missing={missing[:5]}, extra={extra[:5]}')

    for name, frame in by_name.items():
        image_path = corrected / name
        original_path = original / name
        if not _inside(image_path.resolve(), corrected):
            raise ValueError(f'Corrected frame resolves outside the corrected image tree: {name}')
        for path in (original_path, image_path):
            if not path.is_file():
                raise FileNotFoundError(f'Registered image not found: {path}')
        if image_path.samefile(original_path):
            raise ValueError(f'Corrected frame points to the original image: {name}')
        dimensions = (frame.get('width'), frame.get('height'))
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError(f'Invalid exposure frame dimensions: {name}')
        with Image.open(original_path) as before, Image.open(image_path) as after:
            if before.size != dimensions or after.size != dimensions:
                raise ValueError(f'Exposure image dimensions differ for {name}: '
                                 f'original={before.size}, corrected={after.size}, manifest={dimensions}')

    warnings = summary.get('warnings', [])
    if not isinstance(warnings, list) or any(not isinstance(w, str) for w in warnings):
        raise ValueError('Exposure warnings must be a list of strings')
    csv = directory / 'exposure.csv'
    if csv.is_file():
        contents[csv.name] = csv.read_bytes()
    provenance = dict(
        directory=str(directory), source=str(source), original_images=str(original),
        corrected_images=str(corrected), status='complete', registered_images=count,
        supported_images=summary.get('supported_images'),
        unsupported_images=summary.get('unsupported_images'),
        capped_images=summary.get('capped_images'),
        applied_correction_ev_range=summary.get('applied_correction_ev_range'),
        heldout_applied=summary.get('heldout_applied'), warnings=warnings,
        metadata_sha256={name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
        validation='Exact registered filenames and original/export dimensions checked using image headers. '
                   'Metadata hashes identify the export; full image pixels were not hashed.')
    return corrected, provenance


def snapshot_exposure_metadata(provenance, destination):
    """Freeze the validated small metadata files without copying the RGB dataset."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name, expected_hash in provenance['metadata_sha256'].items():
        data = (Path(provenance['directory']) / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError(f'Exposure metadata changed after validation: {name}')
        (destination / name).write_bytes(data)
