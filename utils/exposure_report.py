"""Offline visual reports for exposure-only image preprocessing.

All displayed pixels come from the supplied original/corrected thumbnails.
The portable preview embeds its images and does not require a web server.
"""
from __future__ import annotations

import base64
import html
import json
import math
import mimetypes
from pathlib import Path, PurePosixPath
from urllib.parse import quote


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_json(value):
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _safe_json(value.tolist())
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _json_script(value):
    return (json.dumps(_safe_json(value), ensure_ascii=False, allow_nan=False)
            .replace("&", "\\u0026").replace("<", "\\u003c")
            .replace(">", "\\u003e").replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029"))


def _relative(value):
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not str(path) or str(path) == ".":
        raise ValueError(f"Report asset must be a nonempty relative path: {value!r}")
    return path


def _asset_url(value):
    return quote(str(_relative(value)), safe="/")


def _embedded_image(output, relative):
    path = output.joinpath(*_relative(relative).parts)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    if mime not in ("image/jpeg", "image/png", "image/webp"):
        raise ValueError(f"Unsupported preview image type: {path.name}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _select_examples(records, count):
    """Mix sequence coverage, both gain extremes and abrupt gain changes."""
    n = len(records)
    count = min(n, max(1, int(count)))
    if count == n:
        return list(range(n))
    gains = [_number(r.get("correction_ev")) or 0.0 for r in records]
    ranked = sorted(range(n), key=lambda i: gains[i])
    abrupt = sorted(range(1, n), key=lambda i: abs(gains[i] - gains[i - 1]), reverse=True)
    selected = set()
    # Leave room for exposure extremes and change points after uniform samples.
    uniform = max(2, count // 2)
    for k in range(uniform):
        selected.add(round(k * (n - 1) / max(1, uniform - 1)))
    candidates = []
    for k in range(n):
        candidates.extend((ranked[k], ranked[-1 - k]))
        if k < len(abrupt):
            candidates.extend((abrupt[k] - 1, abrupt[k]))
    for index in candidates:
        if len(selected) >= count:
            break
        selected.add(index)
    return sorted(selected)[:count]


def _exposure_curve(records, selected):
    width, height, pad = 1100, 180, 46
    values = [_number(r.get("correction_ev")) or 0.0 for r in records]
    span = max(.25, math.ceil(max(map(abs, values), default=0.0) * 4) / 4)
    top, bottom = 24, 144
    def xy(index, value):
        x = pad + (width - 2 * pad) * index / max(1, len(records) - 1)
        y = (top + bottom) / 2 - value / span * (bottom - top) / 2
        return x, y
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             'aria-label="Applied exposure correction in EV across frame order">']
    for value in (-span, 0, span):
        _, y = xy(0, value)
        parts.append(f'<line x1="{pad}" y1="{y:.2f}" x2="{width-pad}" y2="{y:.2f}" '
                     'stroke="#d4dde7" stroke-width="1"/>')
        parts.append(f'<text x="{pad-7}" y="{y+4:.2f}" text-anchor="end" '
                     f'fill="#526174" font-size="12">{value:+.2f}</text>')
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in (xy(i, v) for i, v in enumerate(values)))
    parts.append(f'<polyline points="{points}" fill="none" stroke="#147d75" stroke-width="1.8"/>')
    for index in selected:
        x, y = xy(index, values[index])
        title = html.escape(f"{records[index]['name']}: {values[index]:+.3f} EV")
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#147d75"><title>{title}</title></circle>')
    parts.extend((f'<text x="{pad}" y="171" fill="#526174" font-size="12">Frame 1</text>',
                  f'<text x="{width-pad}" y="171" text-anchor="end" fill="#526174" font-size="12">Frame {len(records):,}</text>',
                  '</svg>'))
    return "".join(parts)


_STYLE = """
:root{color-scheme:light;--ink:#172333;--muted:#5b6879;--line:#d9e1e9;--accent:#147d75}
*{box-sizing:border-box}body{margin:0;background:#f3f6f9;color:var(--ink);font:15px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1500px;margin:0 auto;padding:28px 24px 56px}h1{font-size:32px;line-height:1.15;letter-spacing:-.6px;margin:6px 0 12px}h2{font-size:19px;margin:0 0 10px}p{margin:8px 0}.eyebrow{font-size:12px;letter-spacing:1.8px;text-transform:uppercase;font-weight:750;color:var(--accent)}.muted{color:var(--muted)}
.panel{background:white;border:1px solid var(--line);border-radius:12px;padding:20px;margin-top:18px}.stats{display:flex;gap:14px;flex-wrap:wrap;margin:20px 0}.stat{background:white;border:1px solid var(--line);border-radius:10px;padding:12px 18px;min-width:150px}.stat strong{display:block;font-size:24px}.stat span{color:var(--muted);font-size:12px}.notices{border-left:4px solid #c58926;background:#fff7e8;padding:12px 18px;margin:16px 0}.notices ul{margin:0;padding-left:20px}
.controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:14px}button,input,select{font:inherit;border:1px solid #bac7d5;border-radius:7px;padding:8px 12px;background:white;color:var(--ink)}button{cursor:pointer}button:hover{border-color:var(--accent);background:#edf8f5}button:disabled{opacity:.4;cursor:default}input{min-width:200px;flex:1}select{max-width:420px;min-width:210px;flex:1}a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}.count{font-size:13px;color:var(--muted);white-space:nowrap}
.frame-title{font-size:19px;overflow-wrap:anywhere;margin:0 0 5px}.frame-meta{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}.badge{font-size:12px;background:#edf2f6;border-radius:4px;padding:4px 8px}.badge.main{background:#e6f4f0;color:#155f54}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}.pair figure{margin:0;min-width:0;background:#0e131c;border-radius:7px;overflow:hidden}.pair figcaption{padding:9px 12px;color:white;font-size:13px;display:flex;justify-content:space-between}.pair figcaption a{color:#a7e5d9}.pair img{width:100%;height:auto;display:block;object-fit:contain;min-height:80px}.detail{font-size:13px;color:var(--muted);margin-top:12px}.detail span{display:inline-block;margin-right:18px}.curve svg{width:100%;max-height:240px;display:block}.overview{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}.example{padding:0;text-align:left;overflow:hidden;border:1px solid var(--line);border-radius:8px;background:white}.example-pair{display:grid;grid-template-columns:1fr 1fr;gap:2px;background:#182130}.example img{width:100%;display:block}.example-label{padding:9px 10px;font-size:12px;overflow-wrap:anywhere}.example-label strong{display:block}.example-label span{color:var(--muted)}details{margin-top:16px}summary{cursor:pointer;font-weight:650}pre{overflow:auto;font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;background:#edf2f6;padding:16px;border-radius:8px;max-height:420px}.foot{font-size:12px;color:var(--muted);margin-top:24px}.empty{padding:40px;text-align:center;color:var(--muted)}
@media(max-width:760px){main{padding:20px 12px}.panel{padding:13px}.pair{grid-template-columns:1fr}.stats{gap:8px}.stat{padding:10px 12px;min-width:130px;flex:1}h1{font-size:28px}.controls input,.controls select{min-width:100%;max-width:none}.overview{grid-template-columns:repeat(2,minmax(0,1fr))}}
"""


_SCRIPT = """
const data=JSON.parse(document.getElementById('frames-data').textContent);
const selected=JSON.parse(document.getElementById('examples-data').textContent);
let indices=data.map((_,i)=>i), current=0, full=false;
const $=id=>document.getElementById(id);
const number=(v,d=3)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(d):'—';
const pct=v=>typeof v==='number'&&Number.isFinite(v)?(100*v).toFixed(3)+'%':'—';
const signed=v=>typeof v==='number'&&Number.isFinite(v)?(v>=0?'+':'')+v.toFixed(3):'—';
function badge(text,primary=false){const x=document.createElement('span');x.className='badge'+(primary?' main':'');x.textContent=text;return x;}
function render(){
  const has=indices.length>0;$('viewer').hidden=!has;$('empty').hidden=has;
  $('prev').disabled=!has||indices.indexOf(current)<=0;$('next').disabled=!has||indices.indexOf(current)>=indices.length-1;
  if(!has){$('counter').textContent='No matching frames';return;}
  if(!indices.includes(current))current=indices[0];const r=data[current];$('frame-select').value=String(current);
  $('counter').textContent=(indices.indexOf(current)+1)+' / '+indices.length+' shown';$('frame-name').textContent=r.name;
  const meta=$('frame-meta');meta.replaceChildren(badge('Correction '+signed(r.correction_ev)+' EV',true),badge('Linear gain ×'+number(r.gain)),badge(r.status||'Status not supplied'),badge((r.fit_observations??0)+' fit observations'));
  $('before').src=full&&r.original_url?r.original_url:r.before_url;$('after').src=full&&r.corrected_url?r.corrected_url:r.after_url;
  $('before').alt='Original '+r.name;$('after').alt='Exposure-corrected '+r.name;
  for(const [id,key] of [['original-link','original_url'],['corrected-link','corrected_url']]){const link=$(id);link.hidden=!r[key];if(r[key])link.href=r[key];}
  $('full-resolution').hidden=!r.original_url||!r.corrected_url;$('full-resolution').textContent=full?'Use preview images':'Load full resolution';
  $('detail').textContent='Estimated relative exposure: '+signed(r.estimated_exposure_ev)+' EV · Overlap group: '+(r.component??'—')+' · Fit residual: '+number(r.residual_median_ev)+' EV · Clipped pixels: '+pct(r.clip_before)+' → '+pct(r.clip_after)+' · Newly clipped: '+pct(r.newly_clipped_fraction);
  $('exif').textContent=JSON.stringify(r.exif||{},null,2);
}
function options(){const select=$('frame-select');select.replaceChildren();for(const i of indices){const o=document.createElement('option');o.value=i;o.textContent=data[i].name;select.append(o);}if(!indices.includes(current))current=indices[0]??0;render();}
$('frame-select').addEventListener('change',e=>{current=Number(e.target.value);render();});
$('search').addEventListener('input',e=>{const term=e.target.value.toLocaleLowerCase();indices=data.map((_,i)=>i).filter(i=>data[i].name.toLocaleLowerCase().includes(term));options();});
function move(step){const next=indices.indexOf(current)+step;if(next>=0&&next<indices.length){current=indices[next];render();}}
$('prev').addEventListener('click',()=>move(-1));$('next').addEventListener('click',()=>move(1));
$('full-resolution').addEventListener('click',()=>{full=!full;render();});
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;if(e.key==='ArrowLeft'){e.preventDefault();move(-1);}if(e.key==='ArrowRight'){e.preventDefault();move(1);}});
const grid=$('overview');for(const i of selected){const r=data[i],button=document.createElement('button');button.type='button';button.className='example';const pair=document.createElement('div');pair.className='example-pair';for(const [url,label] of [[r.before_url,'Original'],[r.after_url,'Corrected']]){const im=document.createElement('img');im.src=url;im.alt=label+' '+r.name;im.loading='lazy';pair.append(im);}const label=document.createElement('div');label.className='example-label';const title=document.createElement('strong');title.textContent=r.name;const extra=document.createElement('span');extra.textContent=signed(r.correction_ev)+' EV · ×'+number(r.gain)+' · '+(r.status||'');label.append(title,extra);button.append(pair,label);button.addEventListener('click',()=>{$('search').value='';indices=data.map((_,j)=>j);current=i;options();$('frame-panel').scrollIntoView({behavior:'smooth',block:'start'});});grid.append(button);}
const requested=new URLSearchParams(location.hash.slice(1)).get('frame');if(requested){const match=data.findIndex(r=>r.name===requested);if(match>=0)current=match;}
options();
"""


def _page(records, summary, curve, selected, total_frames, portable):
    values = [_number(r.get("correction_ev")) or 0.0 for r in records]
    changed = sum(abs(v) > 1e-6 for v in values)
    maximum = max(map(abs, values), default=0.0)
    mode = (f"Portable preview: {len(records):,} selected frames from {total_frames:,}. Images are embedded; "
            "this file works by itself. Full-resolution images remain in the output folder.") if portable else (
            f"Browse all {len(records):,} frames. Use the arrow keys to move between frames, or search by filename.")
    warnings = summary.get("warnings", [])
    if isinstance(warnings, str):
        warnings = [warnings]
    notices = ""
    if warnings:
        notices = '<aside class="notices"><ul>'+"".join(f"<li>{html.escape(str(w))}</li>" for w in warnings)+"</ul></aside>"
    links = '<a href="preview.html">Portable preview</a> · <a href="contact_sheet.jpg">Contact sheet</a>' if not portable else "Embedded images use the same display scale for each original/corrected pair."
    audit = html.escape(json.dumps(_safe_json(summary), indent=2, ensure_ascii=False, allow_nan=False))
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exposure review</title><style>{_STYLE}</style></head>
<body><main><header><div class="eyebrow">Image preprocessing review</div><h1>Original and corrected frames</h1><p class="muted">One exposure gain per frame. Positive EV brightens; negative EV darkens.</p><p class="muted">{html.escape(mode)}</p></header>
<div class="stats"><div class="stat"><strong>{total_frames:,}</strong><span>Source frames</span></div><div class="stat"><strong>{len(records):,}</strong><span>Frames in this gallery</span></div><div class="stat"><strong>{changed:,}</strong><span>Nonzero corrections in this gallery</span></div><div class="stat"><strong>{maximum:.3f} EV</strong><span>Largest correction in this gallery</span></div></div>
{notices}
<section class="panel curve"><h2>Applied correction across frame order</h2><p class="muted">Dots identify representative examples. This plot shows applied corrections, including any limits.</p>{curve}</section>
<section class="panel" id="frame-panel"><div class="controls"><button id="prev" type="button">← Previous</button><button id="next" type="button">Next →</button><input id="search" type="search" placeholder="Search filenames" aria-label="Search filenames"><select id="frame-select" aria-label="Select a frame"></select><span id="counter" class="count"></span></div>
<div id="empty" class="empty" hidden>No filenames match your search.</div><div id="viewer"><h2 id="frame-name" class="frame-title"></h2><div id="frame-meta" class="frame-meta"></div><div class="pair"><figure><figcaption><span>Original</span><a id="original-link" target="_blank" rel="noopener">Open full image ↗</a></figcaption><img id="before" alt="Original frame"></figure><figure><figcaption><span>Exposure corrected</span><a id="corrected-link" target="_blank" rel="noopener">Open full image ↗</a></figcaption><img id="after" alt="Corrected frame"></figure></div><p id="detail" class="detail"></p><button type="button" id="full-resolution">Load full resolution</button><details><summary>Camera metadata for this frame</summary><pre id="exif"></pre></details></div></section>
<section class="panel"><h2>Representative frames</h2><p class="muted">Original on the left, corrected on the right. Examples cover the sequence, strong corrections and abrupt changes.</p><div id="overview" class="overview"></div></section>
<section class="panel"><h2>Files and measurement details</h2><p>{links}</p><details><summary>Exposure estimation and validation report</summary><pre>{audit}</pre></details></section>
<p class="foot">Exposure correction changes brightness; it cannot restore information lost to saturation or motion blur. Review local detail and repeated scene surfaces, not only whole-image brightness.</p>
<script id="frames-data" type="application/json">{_json_script(records)}</script><script id="examples-data" type="application/json">{_json_script(selected)}</script><script>{_SCRIPT}</script></main></body></html>'''


def _contact_sheet(output, records):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    indices = _select_examples(records, 12)
    tile_width, image_height, margin, row_height = 580, 290, 22, 344
    width = tile_width * 2 + margin * 3
    sheet = Image.new("RGB", (width, 100 + row_height * len(indices)), "#f3f6f9")
    draw = ImageDraw.Draw(sheet)
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 27)
        label_font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        title_font = label_font = ImageFont.load_default()
    draw.text((margin, 16), "Exposure review - original / corrected", fill="#172333", font=title_font)
    draw.text((margin, 57), f"{len(indices)} representative frames from {len(records):,}; identical framing and display scale", fill="#5b6879", font=label_font)
    for row, index in enumerate(indices):
        r = records[index]
        top = 100 + row * row_height
        ev, gain = _number(r.get("correction_ev")), _number(r.get("gain"))
        info = f"{r['name']}  |  {ev:+.3f} EV  |  gain {gain:.3f}" if ev is not None and gain is not None else str(r['name'])
        if len(info) > 110:
            info = info[:107] + "..."
        draw.text((margin, top), info, fill="#172333", font=label_font)
        for column, key in enumerate(("thumbnail_before", "thumbnail_after")):
            left, y = margin + column * (tile_width + margin), top + 29
            draw.rectangle((left, y, left + tile_width, y + image_height), fill="#111721")
            with Image.open(output.joinpath(*_relative(r[key]).parts)) as source:
                tile = ImageOps.contain(source.convert("RGB"), (tile_width, image_height), method=Image.Resampling.LANCZOS)
                sheet.paste(tile, (left + (tile_width - tile.width) // 2, y + (image_height - tile.height) // 2))
    sheet.save(output / "contact_sheet.jpg", quality=92, subsampling=0)


def write_report(output: Path, records: list[dict], summary: dict, preview_count=24) -> None:
    """Write index.html, portable preview.html and an actual-image contact sheet.

    ``records`` follow the caller's frame order. Asset paths are relative to
    ``output``; original files may be symlinked by the preprocessing command.
    No image data is invented, aligned, retouched or generated by this module.
    """
    output = Path(output)
    if not records:
        raise ValueError("Cannot build an exposure report without image records")
    if int(preview_count) < 1:
        raise ValueError("preview_count must be positive")
    output.mkdir(parents=True, exist_ok=True)
    indices = _select_examples(records, preview_count)
    curve = _exposure_curve(records, indices)
    full = []
    for record in records:
        item = _safe_json(dict(record))
        item["before_url"] = _asset_url(record["thumbnail_before"])
        item["after_url"] = _asset_url(record["thumbnail_after"])
        item["original_url"] = _asset_url(record["original_rel"])
        item["corrected_url"] = _asset_url(record["corrected_rel"])
        full.append(item)
    (output / "index.html").write_text(_page(full, summary, curve, indices, len(records), False), encoding="utf-8")
    embedded = []
    for index in indices:
        item = dict(full[index])
        item["before_url"] = _embedded_image(output, records[index]["thumbnail_before"])
        item["after_url"] = _embedded_image(output, records[index]["thumbnail_after"])
        item["original_url"] = item["corrected_url"] = None
        embedded.append(item)
    (output / "preview.html").write_text(_page(embedded, summary, curve, list(range(len(embedded))), len(records), True), encoding="utf-8")
    _contact_sheet(output, records)
