"""Exercise fixed scene construction and novel-camera rendering without a GPU."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter, map_coordinates

from scripts import orange4f_textured_scene as world
from scripts.indoor_triangles.world_texture import WorldMesh, raster


W, H, F = 192, 144, 140.
K = np.array([[F, 0, (W-1)/2], [0, F, (H-1)/2], [0, 0, 1.]])


def pose(center, angle):
    a = np.deg2rad(angle)
    r = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return dict(width=W, height=H, R=r.tolist(), T=(-r.T@center).tolist(),
                intrinsics=[F, F, (W-1)/2, (H-1)/2])


def photograph(p):
    # Independent analytic evaluator. Only RGB fitting photos enter stereo.
    cam = world.path_camera(p)
    y, x = np.indices((H, W))
    d = np.stack([(x-K[0, 2])/F, (y-K[1, 2])/F, np.ones(x.shape)], axis=-1)@cam.rotation
    distances = []
    for z in (1.6, 3.1):
        distances.append((z-cam.center[2])/d[..., 2])
    front = cam.center + d*distances[0][..., None]
    close = (abs(front[..., 0]-.18)<.28) & (abs(front[..., 1]-.02)<.38)
    xyz = cam.center+d*np.where(close, *distances)[..., None]
    xx, yy = xyz[..., 0], xyz[..., 1]
    tex = 30*np.sin(44*xx+15*yy) + 17*np.cos(17*xx-33*yy)
    red = np.where(close, 183., 98.)+tex+16*np.sin(5*yy)
    return np.stack([red, 108+tex*.6+30*np.sin(11*xx), 165-.45*red], axis=-1).round().astype(np.uint8)


def stereo(images, centers, i):
    y, x = np.indices((H, W), np.float32)
    others = sorted((j for j in range(len(images)) if j != i), key=lambda j: abs(centers[i]-centers[j]))[:2]
    depths = np.linspace(1/4., 1/1.15, 65)**-1
    costs = []
    dst = images[i].astype(np.float32)
    for z in depths:
        errors = []
        for j in others:
            u = x + (centers[i]-centers[j])*F/z
            warped = np.stack([map_coordinates(images[j][..., c], [y, u], order=1, mode="nearest")
                               for c in range(3)], axis=-1)
            e = np.abs(dst-warped).mean(axis=2)
            e[(u<1) | (u>=W-1)] = 255
            errors.append(uniform_filter(e, size=5))
        costs.append(np.minimum(*errors))
    costs = np.array(costs)
    return depths[np.argmin(costs, axis=0)].astype(np.float32), costs.min(0)<9


def test_standalone_random_path(tmp_path):
    run = tmp_path/"capture"
    sparse, photos, cache = run/"sparse", run/"images", run/"cache"
    for d in (sparse, photos, cache/"frames"):
        d.mkdir(parents=True)
    centers = [-.42, -.28, -.14, 0., .14, .28, .42]
    fit = [pose(np.array([cx, 0., 0.]), 0.) for cx in centers]
    frames = [pose(np.array([x, .15+.07*np.cos(t), .22+.06*np.sin(t)]), 8*np.sin(t))
              for t, x in zip(np.linspace(0, 2*np.pi, 17), np.linspace(-.3, .3, 17))]
    images = [photograph(p) for p in fit]
    (cache/"manifest.json").write_text("{}")
    (sparse/"cameras.txt").write_text(f"1 PINHOLE {W} {H} {F} {F} {W/2} {H/2}\n")
    rows, poses = [], []
    for i, p in enumerate(fit + [frames[j] for j in (2, 8, 13)]):
        name = f"test_{i}.png"
        training = i < len(fit)
        if training:
            z, mask = stereo(images, centers, i)
            np.savez(cache/"frames"/(Path(name).stem+".npz"), depth=z, valid=mask, intrinsic=K)
        rgb = images[i] if training else photograph(p)
        Image.fromarray(rgb).save(photos/name)
        # Source model uses world-to-camera quaternions.
        from utils.read_write_model import rotmat2qvec
        q = rotmat2qvec(np.array(p["R"]).T)
        poses.append(f"{i+1} " + " ".join(map(str, [*q, *p["T"]])) + f" 1 {name}\n\n")
        rows.append(dict(name=name, staged_name=name, width=W, height=H, intrinsics=p["intrinsics"],
                         split="train" if training else "test"))
    (sparse/"images.txt").write_text("".join(poses))
    manifest = dict(frames=rows, settings=dict(colmap_model=str(sparse), geometry_priors=dict(
        cache=str(cache), cache_sha256=hashlib.sha256(b"{}").hexdigest())))
    (run/"split_manifest.json").write_text(json.dumps(manifest))
    traj = tmp_path/"off_path.json"
    traj.write_text(json.dumps(dict(fps=12, cameras=frames)))
    args = SimpleNamespace(run=run, scene=tmp_path/"export", trajectory=traj, voxel=.055,
                           far=5., all_held_out=True)
    world.build(args)
    world.evaluate(args)
    scores = json.loads((args.scene/"held_out_3/scores.json").read_text())
    print("OFF-PATH", json.dumps(scores))
    assert min(v["nearest_fit"] for v in scores["views"]) > .2
    assert min(v["textured"] for v in scores["views"]) > .6
    assert scores["mean_full_psnr"] > 15
    # Verify the model is self-contained. No paired target, source RGB, depths or poses remain at the build paths.
    run.rename(tmp_path/"removed_capture")
    world.render_path(args)
    paths = list(args.scene.glob("path_*/render.mp4"))
    assert len(paths) == 1 and paths[0].stat().st_size > 1000
    coverage = json.loads((paths[0].parent/"coverage.json").read_text())
    assert len(coverage) == len(frames)
    assert (args.scene/"surface.obj").stat().st_size > 0 and (args.scene/"surface.mtl").is_file()
    # Novel target facing away from the observed room sees no surface.
    mesh = WorldMesh(args.scene)
    _, hit, _ = mesh.render(world.path_camera(pose(np.array([0., 0., .2]), 180)))
    assert not hit.any()


def test_perspective_depth_and_near_plane():
    camera = world.path_camera(pose(np.zeros(3), 0))
    xyz = np.array([[-1., -1., 1.], [1., -1., 2.], [0., 1., 3.]], np.float32)
    ids, z, bary = raster(xyz, np.array([[0, 1, 2]]), camera)
    y, x = H//2, W//2
    uv = bary[y, x]
    hit = (1-uv.sum())*xyz[0]+uv@xyz[1:]
    assert ids[y, x] == 0 and abs(z[y, x]-hit[2]) < 1e-5
    assert np.allclose(hit[:2]/hit[2]*F+K[:2, 2], [x, y], atol=1e-4)
    xyz[0, 2] = -.4
    ids, z, _ = raster(xyz, np.array([[0, 1, 2], [0, 1, 2]]), camera)
    assert (ids == 0).any() and not (ids == 1).any() and z.min() >= .02
