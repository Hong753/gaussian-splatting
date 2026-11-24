import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from scripts import orange4f_photo_detail as photo


def render_fixture(width, height, focal, center_x, angle, contrast):
    """Analytic textured wall at z=3 with a closer occluder at z=1.5."""
    y, x = np.indices((height, width))
    rx, ry = (x-(width-1)/2)/focal, (y-(height-1)/2)/focal
    dx = np.cos(angle)*rx + np.sin(angle)
    dz = np.cos(angle) - np.sin(angle)*rx
    front = (np.abs(center_x + 1.5*dx/dz) < .18) & (np.abs(1.5*ry/dz) < .4)
    z = np.where(front, 1.5, 3.) / dz
    world_x, world_y = center_x + z*dx, z*ry
    stripe = 40*np.sin(45*world_x + 31*world_y) + 21*np.sin(61*world_y)
    red = np.where(front, 160., 94.) + contrast*stripe
    rgb = np.stack((red, red*.8, 180-red*.4), axis=-1).round().astype(np.uint8)
    return rgb, z.astype(np.float32)


def test_fixed_pose_two_held_out_targets(tmp_path):
    w, h, f = 180, 120, 146.
    preview = tmp_path/"model/debug_previews"
    cache = tmp_path/"priors"
    priors = cache/"frames"
    sparse = tmp_path/"sparse"
    images = tmp_path/"images"
    for d in (preview, priors, sparse, images):
        d.mkdir(parents=True)
    (cache/"manifest.json").write_text("{}")
    (sparse/"cameras.txt").write_text(f"1 PINHOLE {w} {h} {f} {f} {w/2} {h/2}\n")
    rows, poses, previews = [], [], []
    k = np.array([[f/2, 0, (w/2-1)/2], [0, f/2, (h/2-1)/2], [0, 0, 1]])
    for i, (cx, angle) in enumerate([(-.16, .015), (.14, -.018), (-.25, .04), (.24, -.03),
                                     (0., .005), (.045, -.008)]):
        name = f"frame_{i}.png"
        train = i < 4
        rows.append(dict(name=name, staged_name=name, split="train" if train else "test",
                         width=w, height=h, intrinsics=[f, f, w/2, h/2]))
        rgb, _ = render_fixture(w, h, f, cx, angle, 1.)
        Image.fromarray(rgb).save(images/name)
        poses.append(f"{i+1} {np.cos(angle/2)} 0 {-np.sin(angle/2)} 0 "
                     f"{-cx*np.cos(angle)} 0 {-cx*np.sin(angle)} 1 {name}\n\n")
        if train:
            _, depth = render_fixture(w//2, h//2, f/2, cx, angle, 1.)
            np.savez(priors/(Path(name).stem+".npz"), depth=depth, valid=np.ones_like(depth, bool),
                     intrinsic=k)
        else:
            base, _ = render_fixture(w, h, f, cx, angle, .20)
            Image.fromarray(base).save(preview/name)
            previews.append(dict(name=name, split="held-out", render=name))
    (preview/"latest.json").write_text(json.dumps(dict(iteration=7000, frames=previews)))
    (sparse/"images.txt").write_text("".join(poses))
    prior = dict(cache=str(cache), cache_sha256=hashlib.sha256(b"{}").hexdigest())
    manifest = dict(settings=dict(geometry_priors=prior, colmap_model=str(sparse), iterations=30000),
                    frames=rows)
    (tmp_path/"split_manifest.json").write_text(json.dumps(manifest))
    args = SimpleNamespace(run=tmp_path, all_held_out=False)
    photo.run(args)
    result = tmp_path/"photo_detail_preview_7000"
    metrics = json.loads((result/"results.json").read_text())
    predictions = {}
    for row in metrics["frames"]:
        assert .5 < row["coverage"] < 1 and row["photo_psnr"] > row["baseline_psnr"] + 2
        assert row["photo_detail_rmse"] < row["baseline_detail_rmse"]
        assert set(row["sources"]) == {r["name"] for r in rows[:4]}
        name = row["frame"]
        stem = Path(name).stem
        render = photo.read_rgb(result/(stem+"_render.png"))
        support = np.asarray(Image.open(result/(stem+"_support.png")))
        base = photo.read_rgb(preview/name)
        assert np.array_equal(render[~support], base[~support])
        predictions[name] = render
        # Replace held-out RGB and rerender with the same sources and baseline.
        Image.fromarray(np.zeros_like(base)).save(images/name)
    photo.run(SimpleNamespace(run=tmp_path, all_held_out=False, inspect_warps=True))
    diagnostics = tmp_path/"photo_warps_preview_7000"
    audited = json.loads((diagnostics/"results.json").read_text())
    for target in audited["frames"]:
        assert len(target["warps"]) == 4
        for warp in target["warps"]:
            assert warp["source"] in {r["name"] for r in rows[:4]}
            assert 0 < warp["valid_pixels"] < w*h
            # The target photograph was made black above. It has zero edges.
            assert warp["valid_edge_pixels"] == 0 and warp["valid_edge_warp_psnr"] is None
            mask = np.asarray(Image.open(diagnostics/warp["validity_mask"]))
            assert mask.sum() == warp["valid_pixels"]
    result.rename(tmp_path/"first")
    photo.run(args)
    frames = tmp_path/"frames"
    frames.mkdir()
    for name, previous in predictions.items():
        assert np.array_equal(photo.read_rgb(result/(Path(name).stem+"_render.png")), previous)
        (frames/name).symlink_to(preview/name)
    photo.run(SimpleNamespace(run=tmp_path, all_held_out=True))
    full = tmp_path/"photo_detail_all_held_out_30000"
    assert all(np.array_equal(photo.read_rgb(full/(Path(name).stem+"_render.png")), previous)
               for name, previous in predictions.items())


def test_unsupported_and_occluded_texels_keep_baseline():
    image = np.full((12, 12, 3), 100, dtype=np.uint8)
    z = np.ones((12, 12), np.float32)
    valid = np.ones_like(z, bool)
    # A nearer, untrusted layer vetoes two agreeing rear colors.
    near = (z, ~valid, image, np.zeros_like(image, np.int16))
    rear = (z*2, valid, image, np.full_like(image, 8, np.int16))
    for warps in ([], [rear], [near, rear, rear]):
        result, supported = photo.transfer(image, warps)
        assert not supported.any()
        assert np.array_equal(image, result)
