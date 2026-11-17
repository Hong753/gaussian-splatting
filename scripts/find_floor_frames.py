#!/usr/bin/env python3
"""Rank regional RGB views and build a browser crop picker for the floor audit."""
import argparse
import json
from pathlib import Path
import shlex
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.keyframe_pipeline.common import image_path
from utils.region_loss import mask_path


def best_window(gray, mask):
    """Prefer textured rectangles supported by the regional mask in the lower image."""
    height, width = gray.shape
    corners = cv2.goodFeaturesToTrack(gray, 600, .01, 4, mask=mask.astype(np.uint8))
    texture = np.zeros(gray.shape, np.uint8)
    if corners is not None:
        x, y = corners.reshape(-1, 2).astype(int).T
        texture[y, x] = 1
    support = cv2.integral(mask.astype(np.uint8))
    features = cv2.integral(texture)

    def count(integral, x, y, w, h):
        return integral[y+h, x+w] - integral[y, x+w] - integral[y+h, x] + integral[y, x]

    best = (0., None)
    for fraction in (.20, .30):
        w, h = max(2, round(width * fraction)), max(2, round(height * fraction))
        for y in range(round(.42 * height), height - h + 1, max(1, h // 5)):
            for x in range(0, width - w + 1, max(1, w // 5)):
                coverage = count(support, x, y, w, h) / (w * h)
                corner_count = count(features, x, y, w, h)
                if coverage < .45 or corner_count < 5:
                    continue
                score = (.75 * coverage + .25 * min(1., corner_count / 25)) * (1 + .04 * y / height)
                if score > best[0]:
                    best = score, [x, y, w, h]
    return best


def ranked_frames(region, selection):
    ranked = []
    for row in selection:
        name = row["name"]
        path = image_path(region, "images", name)
        with Image.open(path) as raw:
            width, height = raw.size
            preview = raw.convert("L")
            preview.thumbnail((512, 512), Image.Resampling.LANCZOS)
        with Image.open(mask_path(region / "masks", name)) as raw_mask:
            mask = np.asarray(raw_mask.convert("L").resize(preview.size, Image.Resampling.NEAREST)) > 127
        score, window = best_window(np.asarray(preview), mask)
        roi = None
        if window is not None:
            x, y, w, h = window
            left, top = round(x * width / preview.width), round(y * height / preview.height)
            right, bottom = round((x + w) * width / preview.width), round((y + h) * height / preview.height)
            roi = [left, top, right - left, bottom - top]
            if min(roi[2:]) < 64:
                roi, score = None, 0.
        ranked.append(dict(name=name, quoted_name=shlex.quote(name), split=row["split"],
                           size=[width, height], roi=roi, score=score))
    return sorted(ranked, key=lambda row: (row["roi"] is None, row["split"] != "train", -row["score"], row["name"]))


PAGE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Floor frame picker</title>
<style>
body{font:15px system-ui,sans-serif;color:#e6edf3;background:#111820;margin:24px}
main{display:grid;grid-template-columns:minmax(230px,1fr) minmax(0,3fr);gap:20px}
h1{font-size:24px;margin-top:0}p{color:#b9c7d4}
button{font:inherit;color:inherit;background:#1d2b37;border:1px solid #405161;border-radius:6px;padding:8px;cursor:pointer}
button[aria-pressed="true"]{border-color:#68d6cc;background:#254144}
#frames{display:grid;gap:8px;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));max-height:78vh;overflow:auto}
#frames button{text-align:left;padding:4px;overflow-wrap:anywhere}#frames img{display:block;width:100%;border-radius:4px}
figure{position:relative;width:fit-content;max-width:100%;margin:10px 0;line-height:0;touch-action:none;cursor:crosshair}
figure img{display:block;max-width:100%;max-height:65vh;user-select:none}
[hidden]{display:none!important}
#mask,#box{position:absolute;pointer-events:none}#mask{inset:0;width:100%;height:100%}
#box{border:2px solid #5df3e0;box-sizing:border-box}
textarea{box-sizing:border-box;width:100%;min-height:82px;color:inherit;background:#18232d;border:1px solid #405161;padding:8px}
@media(max-width:760px){main{display:flex;flex-direction:column-reverse}#frames{max-height:30vh}}
</style>
<h1>Floor frame picker</h1>
<main>
 <nav id="frames" aria-label="Candidate frames"></nav>
 <section>
  <p>Choose a sharp floor seam inside the region. Drag on the image to adjust the crop.</p>
  <b id="frame"></b>
  <label><input type="checkbox" id="show-mask"> Show region mask</label>
  <figure id="preview">
   <img id="image" alt="Selected source RGB" draggable="false">
   <img id="mask" alt="" hidden>
   <span id="box" hidden></span>
  </figure>
  <p id="coordinates" aria-live="polite"></p>
  <textarea id="command" readonly aria-label="Floor blur audit command"></textarea>
  <button id="copy" type="button">Copy audit command</button>
 </section>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById("data").textContent);
const byId = id => document.getElementById(id);
const image = byId("image"), mask = byId("mask"), box = byId("box"), preview = byId("preview");
const buttons = [];
let current, roi, start = null;

function refresh() {
  box.hidden = !roi;
  byId("command").value = "";
  byId("copy").disabled = true;
  byId("copy").textContent = "Copy audit command";
  if (!roi) {
    byId("coordinates").textContent = "Drag a rectangle to choose the floor crop.";
    return;
  }
  const [x,y,w,h] = roi, [width,height] = current.size;
  box.style.left = (100*x/width) + "%";
  box.style.top = (100*y/height) + "%";
  box.style.width = (100*w/width) + "%";
  box.style.height = (100*h/height) + "%";
  const valid = Math.min(w,h) >= 64;
  byId("coordinates").textContent = "X Y WIDTH HEIGHT  " + roi.join(" ") +
      (valid ? "" : "  (at least 64 px per side)");
  if (!valid) return;
  byId("command").value = data.prefix + " --reference " + current.quoted_name +
      " --roi " + roi.join(" ") + " --output " + data.outputPrefix + '$(date +%Y%m%d_%H%M%S)';
  byId("copy").disabled = false;
}

function select(index) {
  current = data.rows[index];
  roi = current.roi;
  byId("frame").textContent = current.name + " · " + current.split;
  image.src = "frames/" + index + ".jpg";
  mask.src = "frames/" + index + ".png";
  buttons.forEach((button,i) => button.setAttribute("aria-pressed", i === index));
  refresh();
}

for (const [index,row] of data.rows.entries()) {
  const button = document.createElement("button");
  const thumb = document.createElement("img");
  thumb.loading = "lazy";
  thumb.src = "frames/" + index + "_thumb.jpg";
  thumb.alt = "";
  button.append(thumb, row.name + " · " + row.split);
  button.addEventListener("click", () => select(index));
  byId("frames").append(button);
  buttons.push(button);
}
function pixel(event) {
  const rect = image.getBoundingClientRect();
  return current.size.map((size,axis) => {
    const f = axis ? (event.clientY - rect.top)/rect.height : (event.clientX - rect.left)/rect.width;
    return Math.round(Math.max(0, Math.min(1, f)) * size);
  });
}
preview.addEventListener("pointerdown", event => {
  if (!image.complete || !image.naturalWidth || event.button !== 0) return;
  event.preventDefault();
  start = pixel(event);
  preview.setPointerCapture(event.pointerId);
});
function updateDrag(event) {
  if (!start) return;
  const [x,y] = pixel(event), [a,b] = start;
  roi = [Math.min(a,x), Math.min(b,y), Math.abs(x-a), Math.abs(y-b)];
  refresh();
}
preview.addEventListener("pointermove", updateDrag);
preview.addEventListener("pointerup", event => { updateDrag(event); start = null; });
preview.addEventListener("pointercancel", () => { start = null; });
byId("show-mask").addEventListener("change", event => { mask.hidden = !event.target.checked; });
byId("copy").addEventListener("click", async () => {
  const command = byId("command");
  command.select();
  try {
    await navigator.clipboard.writeText(command.value);
    byId("copy").textContent = "Copied";
  } catch {
    byId("copy").textContent = "Press Ctrl+C to copy";
  }
});
select(0);
</script>
</html>
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", type=Path, required=True)
    parser.add_argument("--keyframes", type=Path, help="Optional full-run keyframe manifest for the generated command")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=36, help="Max frames in the gallery; 0 includes every regional frame")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    region = args.region.expanduser().resolve()
    output = args.output.expanduser().resolve()
    selection = json.loads((region / "selection.json").read_text())["selection"]
    if not selection:
        parser.error("The regional export has no images")
    if output.exists():
        parser.error("Choose a new --output directory")
    rows = ranked_frames(region, selection)
    if args.limit:
        rows = rows[:args.limit]
    images = output / "frames"
    images.mkdir(parents=True)
    for index, row in enumerate(rows):
        with Image.open(image_path(region, "images", row["name"])) as raw:
            preview = raw.convert("RGB")
            preview.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        preview.save(images / f"{index}.jpg", quality=91)
        preview.thumbnail((360, 360), Image.Resampling.LANCZOS)
        preview.save(images / f"{index}_thumb.jpg", quality=85)
        with Image.open(mask_path(region / "masks", row["name"])) as raw:
            signal = raw.convert("L").point(lambda value: 64 if value > 127 else 0)
            overlay = Image.new("RGBA", signal.size, (46, 236, 203))
            overlay.putalpha(signal)
            overlay.save(images / f"{index}.png")
    command = ["python", "scripts/diagnose_floor_blur.py", "--region", str(region)]
    if args.keyframes:
        command.extend(["--keyframes", str(args.keyframes.expanduser().resolve())])
    data = dict(rows=rows, prefix=shlex.join(command),
                outputPrefix=shlex.quote(str(output.parent / "floor_blur_audit_")))
    encoded = json.dumps(data).replace("<", "\\u003c")
    (output / "index.html").write_text(PAGE.replace("__DATA__", encoded, 1), encoding="utf-8")
    print("Leading candidates and suggested X Y WIDTH HEIGHT")
    for row in rows[:8]:
        roi = " ".join(map(str, row["roi"])) if row["roi"] else "draw a crop"
        print(f"{row['name']:>18}  {row['split']:>5}  {roi}")
    print(f"{len(rows)}/{len(selection)} images in {output / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
