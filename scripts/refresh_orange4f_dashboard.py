#!/usr/bin/env python3
"""Publish Scaffold and one-view training previews on the Orange4F suite homepage."""
import argparse
import fcntl
from html import escape
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.checkpoint_snapshots import atomic_write

BEGIN = '<!-- upstream Scaffold previews -->'
END = '<!-- end upstream Scaffold previews -->'
UPSTREAM = re.compile(r'scaffold_upstream_(fixed|adaptive)_[a-zA-Z0-9_-]+')
DEFAULTS = ('scaffold_upstream_fixed_v1', 'scaffold_upstream_adaptive_v1')

POLL = """<script>
const viewerLink = document.querySelector('a[data-scaffold-viewer-port]');
if (viewerLink) {
  const url = new URL('/', location.href);
  url.port = viewerLink.dataset.scaffoldViewerPort;
  viewerLink.href = url.href;
}
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


def preview_row(model, url, states):
    ready, label = status(model)
    states[model.name] = label
    href = f' href="{url}/index.html"' if ready else ''
    return f'<li data-preview="{url}"><a{href}>{escape(model.name)}</a> · <span>{label}</span></li>'


def one_view_section(root, states):
    rows = []
    for plan_path in sorted(root.parent.glob('*/detail_plan.json')):
        try:
            plan = json.loads(plan_path.read_text())
        except (OSError, ValueError) as exc:
            print(f'Skipping unreadable detail plan {plan_path}: {exc}', file=sys.stderr)
            continue
        if not plan.get('probe_keyframes') or Path(plan['baseline']).parent.resolve() != root:
            continue
        model = plan_path.parent
        # Only publish the preview tree. The trainer may create or rotate it after registration.
        alias = root/'one_view'/model.name
        alias.parent.mkdir(exist_ok=True)
        preview = model/'debug_previews'
        try:
            alias.symlink_to(os.path.relpath(preview, alias.parent), target_is_directory=True)
        except FileExistsError:
            if not alias.is_symlink() or alias.resolve() != preview.resolve():
                print(f'Skipping one-view preview: another entry exists at {alias}', file=sys.stderr)
                continue
        rows.append(preview_row(model, 'one_view/' + quote(model.name, safe=''), states))
    if not rows:
        return ''
    return '<h2>One-view training previews</h2><ul>' + ''.join(rows) + '</ul>'


def refresh(root, *, native=None, viewer_port=None):
    root = Path(root).resolve()
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
        rows = [preview_row(root/name, name + '/debug_previews', states)
                for name in [*DEFAULTS, *sorted(names - set(DEFAULTS))]]
        probes = one_view_section(root, states)
        previous_port = re.search(r'<a data-scaffold-viewer-port="(\d+)">', old)
        port = viewer_port
        if port is None and previous_port:
            port = int(previous_port[1])
        viewer = ''
        if port is not None:
            if not 1 <= port <= 65535:
                raise ValueError('Invalid Scaffold viewer port')
            viewer = f'<p><a data-scaffold-viewer-port="{port}">Open Scaffold 3D viewer</a></p>'
        floor_report = 'floor_neighbors_09882_v2/index.html'
        if not (root/floor_report).is_file():
            floor_report = 'floor_neighbors_09882_v1/index.html'
        floor_link = (f'<p><a href="{floor_report}">Nearby floor photos · 09882.jpg</a></p>'
                      if (root/floor_report).is_file() else '')
        footprint = 'floor_splat_audit_v1/index.html'
        if (root/footprint).is_file():
            floor_link += f'<p><a href="{footprint}">3DGS floor footprint audit</a></p>'
        weighted_probe = 'one_view_floor_weight_v1/index.html'
        if (root/weighted_probe).is_file():
            floor_link += f'<p><a href="{weighted_probe}">One-view floor weighting · 09882.jpg</a></p>'
        content = before + BEGIN + viewer + probes + floor_link + '<h2>Upstream Scaffold-GS</h2><ul>' + ''.join(rows) + '</ul>' + POLL + END + after
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
