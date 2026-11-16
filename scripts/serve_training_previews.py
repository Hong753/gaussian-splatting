#!/usr/bin/env python3
"""Serve a training preview directory; no model or GPU is loaded."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def preview_directory(model=None, root=None, variant='inference'):
    subdirectory = 'debug_previews_training' if variant == 'training' else 'debug_previews'
    if model is not None:
        folder = Path(model).expanduser().resolve() / subdirectory
    else:
        candidates = list(Path(root).expanduser().resolve().glob(f'*/{subdirectory}/history.json'))
        if not candidates:
            raise FileNotFoundError('No training previews yet. Start the updated trainer and wait for its debug preview path.')
        folder = max(candidates, key=lambda path: path.stat().st_mtime_ns).parent
    if not (folder / 'index.html').is_file() or not (folder / 'history.json').is_file():
        raise FileNotFoundError(f'No training preview page in {folder}')
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', type=Path, help='Exact training model directory')
    group.add_argument('--root', type=Path, help='Select the most recently updated preview run under this output root')
    parser.add_argument('--port', type=int, default=8082)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--variant', choices=('inference','training'), default='inference',
                        help='Training variant shows mesh culling ON for train-only culling runs')
    args = parser.parse_args()
    folder = preview_directory(args.model, args.root, args.variant)
    server = ThreadingHTTPServer((args.host, args.port), partial(SimpleHTTPRequestHandler, directory=str(folder)))
    print(f'Preview model: {folder.parent}\nServing {folder}\n'
          f'Open http://<server-ip>:{server.server_port}/ (Ctrl-C to stop the preview server).', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
