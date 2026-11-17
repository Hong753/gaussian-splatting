import json

import cv2
import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from scripts.diagnose_floor_blur import main


@pytest.mark.parametrize("group_args", [[], ["--group", "nearby", "near.png", "soft.png",
                                            "--group", "cross", "shifted.png"]])
def test_floor_diagnostic_separates_displacement_from_source_blur(tmp_path, group_args):
    width, height, f, distance = 320, 240, 256, 5
    roi = [85, 60, 150, 120]
    model = tmp_path / "sparse" / "0"
    model.mkdir(parents=True)
    (tmp_path / "images").mkdir()
    (model / "cameras.txt").write_text(f"1 PINHOLE {width} {height} {f} {f} {width / 2} {height / 2}\n")

    rng = np.random.default_rng(17)
    texture = rng.integers(25, 230, (height, width), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), .6)
    # The images have translated camera viewpoints. Only "shifted" has a
    # deliberate error in its recorded projection, orthogonal to the yaw axis.
    frames = [("ref.png", (0., 0.), (0., 0.), False),
              ("near.png", (20., -10.), (0., 0.), False),
              ("soft.png", (12., 14.), (0., 0.), True),
              ("shifted.png", (-24., -10.), (4.4, -2.2), False)]
    lines, refined_lines = [], []
    for index, (name, offset, error, blur) in enumerate(frames):
        actual = np.array(offset)
        saved = (actual - error) * distance / f
        lines.append(f"{index + 1} 1 0 0 0 {saved[0]} {saved[1]} 0 1 {name}\n\n")
        refined = actual * distance / f
        refined_lines.append(f"{index + 1} 1 0 0 0 {refined[0]} {refined[1]} 0 1 {name}\n\n")
        warp = np.column_stack([np.eye(2), actual])
        pixels = cv2.warpAffine(texture, warp, (width, height))
        if blur:
            pixels = cv2.GaussianBlur(pixels, (7, 7), 1.05)
        assert cv2.imwrite(str(tmp_path / "images" / name), pixels)
    (model / "images.txt").write_text("".join(lines))
    refined_model = tmp_path / "refined"
    refined_model.mkdir()
    (refined_model / "cameras.txt").write_bytes((model / "cameras.txt").read_bytes())
    (refined_model / "images.txt").write_text("".join(refined_lines))

    x, y, w, h = roi
    xy = rng.uniform([x, y], [x + w, y + h], (1800, 2))
    depth = rng.normal(distance, .002, len(xy))
    depth[:180] -= rng.uniform(.1, .5, 180)
    xyz = (xy + .5 - np.array([width / 2, height / 2])) * depth[:, None] / f
    vertices = np.array(list(zip(xyz[:, 0], xyz[:, 1], depth)), dtype=[(axis, "f4") for axis in "xyz"])
    PlyData([PlyElement.describe(vertices, "vertex")]).write(tmp_path / "input.ply")

    # Exercise the real regional export metadata route as well as explicit groups.
    region = tmp_path / "region"
    region.mkdir()
    (region / "input.ply").symlink_to(tmp_path / "input.ply")
    keyframes = tmp_path / "keyframes.json"
    keyframes.write_text(json.dumps(dict(selected_images=[row[0] for row in frames])))
    (region / "selection.json").write_text(json.dumps(dict(
        source=str(tmp_path), original_keyframes=str(keyframes),
        selection=[dict(name=row[0], split="train") for row in frames[:3]])))

    output = tmp_path / "audit"
    assert main(["--region", str(region), "--reference", "ref.png", "--roi", *map(str, roi),
                 "--alternate-model", str(refined_model), "--output", str(output), *group_args]) == 0
    report = json.loads((output / "report.json").read_text())
    rows = {row["name"]: row for row in report["views"]}
    assert report["plane"]["inliers"] == pytest.approx(1620, abs=5)
    assert report["plane"]["covered_roi_cells"] == 16
    assert rows["near.png"]["shift_xy_px"] == pytest.approx([0, 0], abs=.25)
    assert rows["soft.png"]["shift_xy_px"] == pytest.approx([0, 0], abs=.3)
    assert rows["shifted.png"]["shift_xy_px"] == pytest.approx([4.4, -2.2], abs=.25)
    assert rows["shifted.png"]["p90_after_translation_px"] < .4
    cross = report["composites"]["cross" if group_args else "Additional full-scene training views"]
    assert cross["corrected"]["gradient_agreement"] > cross["uncorrected"]["gradient_agreement"] + .15
    assert cross["corrected"]["gradient_agreement"] > .95
    assert (output / "index.html").is_file()
    assert (output / rows["shifted.png"]["overlay"]).is_file()

    assert rows["soft.png"]["aligned_gradients"]["edge_strength"] < .65
    assert rows["near.png"]["aligned_gradients"]["edge_strength"] > .95
    paired = {row["name"]: row for row in report["pose_comparison"]}
    assert paired["shifted.png"]["original_median_px"] == pytest.approx(np.hypot(4.4, 2.2), abs=.25)
    assert paired["shifted.png"]["alternate_median_px"] < .25
    assert (output / "alternate" / "index.html").is_file()
