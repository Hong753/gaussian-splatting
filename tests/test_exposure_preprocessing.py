"""End-to-end CPU checks with a synthetic, known-exposure COLMAP fixture."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.normalize_image_exposure import main
from utils.exposure_balance import linear_to_srgb
from utils.exposure_image_io import checked_name, load_frames, sample_frame


def fixture(root):
    model = root / 'sparse' / '0'
    images = root / 'images'
    model.mkdir(parents=True)
    images.mkdir()
    width, height = 200, 160
    (model / 'cameras.txt').write_text(f'1 PINHOLE {width} {height} 140 140 100 80\n')
    centers = np.asarray([(x, y) for y in range(10, 151, 14)
                          for x in range(10, 191, 15)], dtype=int)
    rng = np.random.default_rng(4)
    radiances = rng.uniform(.04, .28, len(centers))
    radiance = np.full((height, width, 3), .15)
    for (x, y), value in zip(centers, radiances):
        radiance[y-3:y+4, x-3:x+4] = value
    exposures = [-.7, .2, .5]
    lines = ['# Synthetic images, not Orange4F data.']
    for index, ev in enumerate(exposures):
        name = f'{index:05d}.png'
        rgb = np.rint(linear_to_srgb(radiance * 2.**ev) * 255).astype(np.uint8)
        Image.fromarray(rgb).save(images / name)
        lines.append(f'{index+1} 1 0 0 0 0 0 0 1 {name}')
        lines.append(' '.join(f'{x+.5} {y+.5} {2**54+j}'
                              for j, (x, y) in enumerate(centers)))
    (model / 'images.txt').write_text('\n'.join(lines)+'\n')
    return model, exposures


class ExposurePreprocessingTests(unittest.TestCase):
    def test_exports_known_exposure_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'scene', root / 'review'
            _, expected = fixture(source)
            before = {p.relative_to(source): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in source.rglob('*') if p.is_file()}
            argv = ['--source', str(source), '--output', str(output), '--workers', '2',
                    '--samples-per-frame', '1000', '--min-observations', '20', '--seed', '7']
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(argv), 0)
            records = json.loads((output / 'frames.json').read_text())
            np.testing.assert_allclose([r['estimated_exposure_ev'] for r in records], expected, atol=.025)
            corrected = [np.asarray(Image.open(output / 'corrected' / r['name'])).astype(float)
                         for r in records]
            self.assertLess(np.abs(corrected[0] - corrected[2]).mean(), 1.2)
            self.assertEqual(corrected[0].shape, (160, 200, 3))
            after = {p.relative_to(source): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            for filename in ('index.html', 'preview.html', 'contact_sheet.jpg', 'exposure.csv'):
                self.assertTrue((output / filename).is_file())
            self.assertIn('data:image/jpeg;base64,', (output / 'preview.html').read_text())
            summary = json.loads((output / 'summary.json').read_text())
            validation = summary['heldout_applied']
            self.assertLess(validation['median_abs_deviation_after_ev'],
                            validation['median_abs_deviation_before_ev'])
            with self.assertRaises(FileExistsError):
                main(argv)
            # Capped gains must be explicit, and validation must use applied gains.
            capped_output = root / 'capped'
            argv[argv.index('--output')+1] = str(capped_output)
            with redirect_stdout(io.StringIO()):
                main(argv + ['--max-correction-ev', '.25'])
            capped = json.loads((capped_output / 'frames.json').read_text())
            self.assertAlmostEqual(capped[0]['correction_ev'], .25)
            self.assertIn('capped', capped[0]['status'])
            capped_summary = json.loads((capped_output / 'summary.json').read_text())
            self.assertGreater(capped_summary['heldout_applied']['median_abs_deviation_after_ev'],
                               summary['heldout_applied']['median_abs_deviation_after_ev'])

    def test_binary_reader_preserves_track_ids_and_pixel_centers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'cameras.bin').write_bytes(struct.pack('<QiiQQdddd', 1, 1, 1,
                30, 20, 20., 20., 15., 10.))
            data = struct.pack('<Qidddddddi', 1, 1, 1., 0., 0., 0., 0., 0., 0., 1)
            data += b'frame.png\0' + struct.pack('<Qddq', 1, 10.5, 8.5, 2**54+1)
            (root / 'images.bin').write_bytes(data)
            frame, = load_frames(root)
            self.assertEqual(int(frame.tracks[0]), 2**54+1)
            rgb = np.zeros((20, 30, 3), dtype=np.uint8)
            rgb[8, 10] = 128
            Image.fromarray(rgb).save(root / 'frame.png')
            result = sample_frame(frame, root, patch_radius=0)
            self.assertEqual(result['valid_observations'], 1)

    def test_rejects_unsafe_names_and_unknown_image_resizing(self):
        for name in ('../bad.jpg', '/tmp/a.jpg', 'a/../../b.jpg', r'a\b.jpg'):
            with self.assertRaises(ValueError):
                checked_name(name)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'scene'
            model, _ = fixture(source)
            frames = load_frames(model)
            path = source / 'images' / frames[0].name
            with Image.open(path) as image:
                resized = image.resize((100, 80))
            resized.save(path)
            with self.assertRaisesRegex(ValueError, 'differs from COLMAP'):
                sample_frame(frames[0], source / 'images')
            result = sample_frame(frames[0], source / 'images', allow_resized=True)
            self.assertTrue(result['resized'])
            self.assertGreater(result['valid_observations'], 20)


if __name__ == '__main__':
    unittest.main()
