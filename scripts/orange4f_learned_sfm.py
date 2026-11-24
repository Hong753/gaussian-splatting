"""Full-sequence learned matching, scan-anchored SfM and transactional sparse replacement."""

import argparse
import ctypes
import errno
import fcntl
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from uuid import uuid4

from scripts.keyframes import registered_names
from scripts.view_comparisons import read_colmap_records
from utils.checkpoint_snapshots import atomic_json, atomic_write, digest
from utils.colmap_paths import colmap_model_dir


MARKER = "superglue_lidar.json"
BINARIES = ("cameras.bin", "images.bin", "points3D.bin")


def pin_consumers(scene):
    # The previous 3DGS runner made file symlinks. Detach those from the
    # mutable scene name while their targets still refer to the SIFT version.
    for name in BINARIES:
        source = scene / "sparse/0" / name
        for path in scene.parent.glob(f"*/scene/sparse/0/{name}"):
            if not path.is_symlink() or path.resolve() != source.resolve():
                continue

            def pin(tmp):
                try:
                    os.link(source.resolve(strict=True), tmp)
                except OSError as e:
                    if e.errno != errno.EXDEV:
                        raise
                    shutil.copy2(source, tmp)

            atomic_write(path, pin)
            print(f"Pinned existing model input {path}", flush=True)


def publish(scene, candidate):
    with (scene / ".sparse-activation.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        exchange_sparse(scene, candidate)


def exchange_sparse(scene, candidate):
    live = scene / "sparse"
    backup = scene / "sparse_sift_backup"
    if live.is_symlink() or not (live / "0/images.bin").is_file():
        raise ValueError(f"Unexpected live sparse directory: {live}")
    if (live / MARKER).exists():
        return
    if backup.exists():
        raise FileExistsError(f"Reserved backup path already exists: {backup}")
    pin_consumers(scene)
    # Linux renameat2 exchanges entire nonempty directories in one syscall.
    candidate.rename(backup)
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        if libc.renameat2(-100, os.fsencode(live), -100, os.fsencode(backup), 2) != 0:
            n = ctypes.get_errno()
            raise OSError(n, os.strerror(n), str(live))
    except BaseException:
        if not (live / MARKER).exists():
            backup.rename(candidate)
        raise
    print(f"New model: {live / '0'}\nSIFT backup: {backup / '0'}", flush=True)


def configuration(args, records, cloud, points):
    from hashlib import sha256

    h = sha256()
    for name in sorted(records):
        st = Path(records[name]["image_path"]).stat()
        h.update(repr((name, st.st_size, st.st_mtime_ns)).encode())
    cal = colmap_model_dir(args.original)
    scan_hash = digest(cloud)
    return dict(
        schema=2,
        registered=len(records),
        image_fingerprint=h.hexdigest(),
        original=str(args.original),
        calibrated_inputs={
            p.name: digest(p)
            for prefix in ("images", "cameras")
            for p in [next(q for q in [cal / (prefix + ".bin"), cal / (prefix + ".txt")] if q.exists())]
        },
        scan=str(cloud),
        scan_sha256=scan_hash,
        tracks=str(points) if points else None,
        tracks_sha256=(scan_hash if points == cloud else digest(points)) if points else None,
        max_size=1600,
        max_keypoints=2048,
        frontend="SuperPoint + indoor SuperGlue ddcf11f",
        mapper="Spirula v2026.11.12",
        metric_refinement="stored registered scan pixels when available, otherwise local scan planes",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("check", "matches", "mapped", "publish"))
    for key in ("scene", "original", "work", "lidar", "repository", "model"):
        p.add_argument("--" + key, type=Path, required=key in ("scene", "original", "work"))
    args = p.parse_args()
    scene, work = args.scene.resolve(), args.work.resolve()
    args.original = args.original.resolve()
    if (
        work != scene / "superpoint_superglue_lidar"
        or scene == args.original
        or (scene / "images").resolve() != (args.original / "images").resolve()
        or Path((scene / "spirula-input.txt").read_text().strip()).resolve()
        != (args.original / "images").resolve()
    ):
        p.error("Expected a separate Spirula scene linked to the original capture")
    from scripts.orange4f_lidar_tracks import scan_inputs

    cloud, points = scan_inputs(args.original, args.lidar)
    records = read_colmap_records(args.original)
    expected = set(registered_names(scene))
    if set(records) != expected:
        raise ValueError("Input frame names differ between the original dataset and the SIFT run")
    work.mkdir(exist_ok=True)
    path = work / "inputs.json"
    if args.stage == "check":
        import torch
        import numpy as np
        from scripts.orange4f_lidar_tracks import collect

        if not torch.cuda.is_available():
            raise RuntimeError("The selected GPU is unavailable to PyTorch in 3dgs")
        config = configuration(args, records, cloud, points)
        cache = work / "scan_pixels.npz"
        previous = json.loads(path.read_text()) if path.exists() else {}
        # The learned matches and mapped RGB geometry only depend on the image
        # inputs. A previous scan configuration can reuse both of these caches.
        image_keys = (
            "registered",
            "image_fingerprint",
            "original",
            "calibrated_inputs",
            "max_size",
            "max_keypoints",
            "frontend",
            "mapper",
        )
        if previous and any(previous.get(k) != config[k] for k in image_keys):
            raise ValueError(f"RGB inputs changed since preparation: {path}")
        if previous != config or not cache.is_file():
            data, stats = collect(args.original, cloud, points, records)

            def write_cache(tmp):
                with tmp.open("wb") as stream:
                    np.savez(stream, **data, stats=json.dumps(stats))

            atomic_write(cache, write_cache)
            atomic_json(path, config)
        else:
            with np.load(cache, allow_pickle=False) as data:
                stats = json.loads(data["stats"].item())
        print(
            f"Original scan {cloud} ({stats['scan_points']:,} points), "
            f"{stats.get('usable_pixel_links', 0):,} usable stored scan-to-pixel observations across "
            f"{stats.get('images_with_links', 0):,}/{len(records):,} cameras. "
            f"Sampled {stats['sampled_pixel_links']:,} observations.",
            flush=True,
        )
        if points is not None:
            print(
                f"Track source {points}: {stats.get('raw_forward_links', 0):,} image-to-point links, "
                f"{stats.get('raw_reverse_links', 0):,} point-to-image links.",
                flush=True,
            )
        if not stats["sampled_pixel_links"]:
            print(
                "No usable stored pixel links. Using the XYZ-only local-plane refinement.",
                flush=True,
            )
        print(
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, {torch.cuda.get_device_name(0)}",
            flush=True,
        )
    elif args.stage == "matches":
        from scripts.orange4f_learned_matches import build

        build(work, args.original / "images", records, args.repository, "cuda")
    elif args.stage == "mapped":
        # Require one model covering the same capture before changing the active sparse folder.
        for folder in sorted(args.model.iterdir()):
            if not (folder / "images.bin").is_file():
                continue
            import pycolmap

            m = pycolmap.Reconstruction(str(folder))
            if {m.image(i).name for i in m.reg_image_ids()} == expected and m.num_points3D() >= 4:
                atomic_write(
                    work / "mapped.ok",
                    lambda tmp: tmp.write_text(str(folder.resolve())),
                )
                return
        raise RuntimeError(
            f"No single learned model registered all {len(expected)} input views: {args.model}. SIFT is still active."
        )
    else:
        from scripts.orange4f_scan_bundle import refine
        import pycolmap

        candidate = scene / "sparse_sift_backup"
        if candidate.exists() and not (scene / "sparse" / MARKER).is_file():
            # A crash between the staging rename and the exchange leaves a fully
            # validated candidate here. Verify identity before using it again.
            info = json.loads((candidate / MARKER).read_text())
            if info["inputs"] != json.loads(path.read_text()) or info["binary_sha256"] != {
                f: digest(candidate / "0" / f) for f in BINARIES
            }:
                raise ValueError(f"Conflicting backup: {candidate}")
            restored = work / ("recovered-" + uuid4().hex)
            candidate.rename(restored)
            publish(scene, restored)
            return
        with TemporaryDirectory(dir=work) as tmp:
            candidate = Path(tmp) / "sparse"
            candidate.mkdir()
            import numpy as np

            with np.load(work / "scan_pixels.npz", allow_pickle=False) as data:
                report = refine(
                    Path((work / "mapped.ok").read_text()),
                    cloud,
                    records,
                    candidate / "0",
                    data,
                )
                report["source_observations"] = json.loads(data["stats"].item())
            final = pycolmap.Reconstruction(str(candidate / "0"))
            final.check()
            if {final.image(i).name for i in final.reg_image_ids()} != expected:
                raise ValueError("Incomplete final sparse model")
            atomic_json(
                candidate / MARKER,
                dict(
                    inputs=json.loads(path.read_text()),
                    bundle=report,
                    binary_sha256={f: digest(candidate / "0" / f) for f in BINARIES},
                ),
            )
            publish(scene, candidate)


if __name__ == "__main__":
    main()
