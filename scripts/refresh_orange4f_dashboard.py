#!/usr/bin/env python3
"""Add upstream Scaffold previews to an existing Orange4F suite homepage."""
import argparse
import fcntl
from html import escape
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.checkpoint_snapshots import atomic_write

BEGIN = '<!-- upstream Scaffold previews -->'
END = '<!-- end upstream Scaffold previews -->'
UPSTREAM = re.compile(r'scaffold_upstream_(fixed|adaptive)_[a-zA-Z0-9_-]+')
DEFAULTS = ('scaffold_upstream_fixed_v1', 'scaffold_upstream_adaptive_v1')

POLL = """<script>
async function refreshPreview(row) {
  try {
    const url = row.dataset.preview;
    const response = await fetch(url + '/latest.json', {cache: 'no-store'});
    if (!response.ok) return;
    const snapshot = await response.json();
    if (!Number.isInteger(snapshot.iteration)) return;
    row.querySelector('a').href = url + '/index.html';
    row.querySelector('span').textContent = 'Latest preview at step ' + snapshot.iteration.toLocaleString();
  } catch {}
}
function refreshUpstream() {
  for (const row of document.querySelectorAll('[data-preview]')) refreshPreview(row);
}
refreshUpstream();
setInterval(refreshUpstream, 15000);
</script>"""


def status(model):
    folder = model/'debug_previews'
    if (folder/'index.html').is_file():
        try:
            latest = json.loads((folder/'latest.json').read_text())
        except (FileNotFoundError, ValueError):
            latest = None
        if isinstance(latest, dict) and isinstance(latest.get('iteration'), int):
            return True, f"Latest preview at step {latest['iteration']:,}"
        return True, 'Preview page ready, waiting for first snapshot'
    if (model/'upstream.json').is_file():
        return False, 'Prepared, no preview saved yet'
    return False, 'No preview saved yet'


def refresh(root, *, native=None):
    root = Path(root)
    index = root/'index.html'
    if not (index.is_file() or (root/'suite.json').is_file()):
        raise FileNotFoundError(f'Orange4F suite missing at {root}. Use the directory served on port 8098.')
    # The two upstream jobs and the native Scaffold launcher can publish concurrently.
    with (root/'.dashboard.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if index.is_file():
            old = index.read_text()
        else:
            runs = json.loads((root/'suite.json').read_text())['runs']
            links = ''.join(f'<li>GPU {escape(str(r["gpu"]))}: <a href="{escape(r["name"])}/debug_previews/index.html">{escape(r["name"])}</a></li>'
                            for r in runs)
            old = '<!doctype html><meta charset="utf-8"><h1>orange4f · stride 3 · AA off</h1><ul>'+links+'</ul>'
        before, marker, rest = old.partition(BEGIN)
        if marker:
            _, end, after = rest.partition(END)
            if not end:
                raise ValueError(f'Incomplete upstream section in {index}')
        else:
            before, closing, tail = old.partition('</body>')
            after = closing + tail
        # Recover older native previews even when their launchers never linked them.
        native_runs = {page.parent.parent.name: None for page in root.glob('scaffold_*/debug_previews/index.html')
                       if not UPSTREAM.fullmatch(page.parent.parent.name)}
        if native:
            native_runs[native[0]] = native[1]
        for name, gpu in sorted(native_runs.items()):
            link = escape(name, quote=True) + '/debug_previews/index.html'
            if f'href="{link}"' in before + after:
                continue
            if gpu is None:
                plan = root/name/'periodic_evaluation/plan.json'
                if plan.is_file():
                    gpu = json.loads(plan.read_text()).get('gpu')
            prefix = f'GPU {escape(str(gpu))}: ' if gpu is not None else ''
            before += f'<p>{prefix}<a href="{link}">{escape(name)}</a></p>'

        names = set(DEFAULTS)
        for path in root.glob('scaffold_upstream_*'):
            name = path.name.removesuffix('.lock') if path.is_file() else path.name
            if UPSTREAM.fullmatch(name):
                names.add(name)
        states = {}
        rows = []
        for name in [*DEFAULTS, *sorted(names - set(DEFAULTS))]:
            ready, label = status(root/name)
            states[name] = label
            url = name + '/debug_previews'
            href = f' href="{url}/index.html"' if ready else ''
            rows.append(f'<li data-preview="{url}"><a{href}>{name}</a> · <span>{label}</span></li>')
        content = before + BEGIN + '<h2>Upstream Scaffold-GS</h2><ul>' + ''.join(rows) + '</ul>' + POLL + END + after
        if content != old:
            atomic_write(index, lambda path: path.write_text(content))
    return states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parent = Path(os.environ.get('ORANGE4F_OUTPUT_ROOT', '/data/3dgs_outputs/orange4f/3dgs_finetuning'))
    parser.add_argument('--suite', type=Path, default=parent/'stride3_noaa_v1')
    args = parser.parse_args()
    for name, state in refresh(args.suite).items():
        print(f'{name}: {state}')
    print(f'Updated {args.suite/"index.html"}')


if __name__ == '__main__':
    main()
