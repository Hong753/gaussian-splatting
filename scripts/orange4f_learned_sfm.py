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


def configuration(args, records, cloud):
    from hashlib import sha256

    h = sha256()
    for name in sorted(records):
        st = Path(records[name]["image_path"]).stat()
        h.update(repr((name, st.st_size, st.st_mtime_ns)).encode())
    cal = colmap_model_dir(args.original)
    return dict(
        schema=1,
        registered=len(records),
        image_fingerprint=h.hexdigest(),
        original=str(args.original),
        calibrated_inputs={
            p.name: digest(p)
            for prefix in ("images", "cameras")
            for p in [next(q for q in [cal / (prefix + ".bin"), cal / (prefix + ".txt")] if q.exists())]
        },
        scan=str(cloud),
        scan_sha256=digest(cloud),
        max_size=1600,
        max_keypoints=2048,
        frontend="SuperPoint + indoor SuperGlue ddcf11f",
        mapper="Spirula v2026.11.12",
        metric_refinement="fixed scan-projected landmarks, variable image poses and remaining landmarks",
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
    cloud = (args.lidar or args.original / "input.ply").expanduser().resolve()
    if not cloud.is_file():
        p.error(
            f"Missing registered metric LiDAR cloud at {cloud}. Set ORANGE4F_LIDAR_PLY to the scan in the original camera frame."
        )
    if not args.lidar and any(
        q.is_file() and q.samefile(cloud)
        for q in [colmap_model_dir(args.original) / "points3D.ply", scene / "sparse/0/points3D.ply"]
    ):
        p.error(
            "The default cloud aliases a sparse reconstruction. Set ORANGE4F_LIDAR_PLY to the verified LiDAR PLY."
        )
    records = read_colmap_records(args.original)
    expected = set(registered_names(scene))
    if set(records) != expected:
        raise ValueError("Input frame names differ between the original dataset and the SIFT run")
    work.mkdir(exist_ok=True)
    path = work / "inputs.json"
    if args.stage == "check":
        import torch
        from plyfile import PlyData

        if not torch.cuda.is_available():
            raise RuntimeError("The selected GPU is unavailable to PyTorch in 3dgs")
        vertices = PlyData.read(cloud, mmap="r")["vertex"]
        if not set("xyz") <= set(vertices.data.dtype.names) or "f_dc_0" in vertices.data.dtype.names:
            raise ValueError("Expected a registered LiDAR scan, not Gaussian coefficients")
        config = configuration(args, records, cloud)
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError(f"Inputs changed since preparation: {path}. Use a fresh cache directory.")
        atomic_json(path, config)
        print(
            f"Registered scan {cloud} ({len(vertices.data):,} points), {len(records):,} RGBs, "
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
                atomic_write(work / "mapped.ok", lambda tmp: tmp.write_text(str(folder.resolve())))
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
            report = refine(Path((work / "mapped.ok").read_text()), cloud, records, candidate / "0")
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
