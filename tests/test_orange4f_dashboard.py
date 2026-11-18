import json
from concurrent.futures import ThreadPoolExecutor

from scripts.refresh_orange4f_dashboard import DEFAULTS, refresh


def test_repair_existing_live_suite_without_touching_models(tmp_path):
    index = tmp_path/'index.html'
    original = '<html><body><p>GPU 0: <a href="scaffold_indexed_anchors_v1/debug_previews/index.html">indexed</a></p>'
    index.write_text(original + '</body></html>')
    fixed, adaptive = DEFAULTS
    (tmp_path/(adaptive+'.lock')).touch()
    custom = 'scaffold_upstream_fixed_trial2'
    (tmp_path/(custom+'.lock')).touch()
    folder = tmp_path/fixed/'debug_previews'
    folder.mkdir(parents=True)
    (folder/'index.html').write_text('live')
    (folder/'latest.json').write_text(json.dumps(dict(iteration=2500)))
    control = tmp_path/'scaffold_indexed-only_anchors_v1'
    (control/'debug_previews').mkdir(parents=True)
    (control/'debug_previews/index.html').touch()
    states = refresh(tmp_path)
    repaired = index.read_text()
    assert repaired.count('href="scaffold_indexed-only_anchors_v1/debug_previews/index.html"') == 1
    assert repaired.startswith(original) and repaired.endswith('</body></html>')
    assert states[fixed] == 'Latest preview at step 2,500'
    assert states[adaptive] == states[custom] == 'No preview saved yet'
    assert f'href="{fixed}/debug_previews/index.html"' in repaired
    assert f'href="{adaptive}/debug_previews/index.html"' not in repaired
    assert f'data-preview="{adaptive}/debug_previews"' in repaired
    assert not (tmp_path/adaptive).exists() and not (tmp_path/custom).exists()
    refresh(tmp_path)
    assert index.read_text() == repaired
    # A previously absent run becomes available after its first snapshot.
    new = tmp_path/adaptive/'debug_previews'
    new.mkdir(parents=True)
    (new/'index.html').touch()
    (new/'latest.json').write_text('{"iteration":1}')
    assert refresh(tmp_path)[adaptive] == 'Latest preview at step 1'


def test_concurrent_launchers_keep_every_link(tmp_path):
    (tmp_path/'index.html').write_text('<h1>Orange4F</h1>')
    runs = [(f'scaffold_run{i}', i) for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda r: refresh(tmp_path, native=r), runs))
    html = (tmp_path/'index.html').read_text()
    for name, _ in runs:
        assert html.count(f'href="{name}/debug_previews/index.html"') == 1
    assert html.count('<h2>Upstream Scaffold-GS</h2>') == 1


def test_viewer_port_persists_across_training_refreshes(tmp_path):
    index = tmp_path/'index.html'
    index.write_text('<html><body>Training</body></html>')
    refresh(tmp_path)
    assert '<a data-scaffold-viewer-port=' not in index.read_text()
    refresh(tmp_path, viewer_port=8099)
    refresh(tmp_path, native=('scaffold_example', 1))
    assert index.read_text().count('<a data-scaffold-viewer-port="8099">') == 1
    refresh(tmp_path, viewer_port=8100)
    assert '8099' not in index.read_text()
    assert '<a data-scaffold-viewer-port="8100">' in index.read_text()
