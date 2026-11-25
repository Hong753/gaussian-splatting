"""Matched Orange4F vanilla 3DGS runs with fewer training views and two camera solutions."""
import argparse
import errno
import fcntl
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
from types import SimpleNamespace

from scripts.keyframes import natural_key, registered_names
from scripts.orange4f_spirula_3dgs import evaluate, link, lock, native_matcher, prepare, train_prepared
from scripts.orange4f_stock_triangle import OUTPUT, REPO, SOURCE as CAPTURE
from scripts.view_comparisons import atomic_json
from utils.checkpoint_snapshots import atomic_write, digest


BINARIES = ("cameras.bin", "images.bin", "points3D.bin")


def read(path):
    return json.loads(path.read_text())


def thin(names, count):
    names = sorted(names, key=natural_key)
    if len(names) != len(set(names)) or not 2 <= count <= len(names):
        raise ValueError(f"Need {count} distinct keyframes, have {len(set(names))}")
    return [names[round(i*(len(names)-1)/(count-1))] for i in range(count)]


def baseline_run(args):
    if args.baseline:
        path = args.baseline.expanduser().resolve()
        if not (path / "protocol.json").is_file():
            raise ValueError(f"Expected the current stride-3 GS run containing protocol.json at {path}")
        return path
    candidates = [p for p in OUTPUT.glob("*_3dgs_stride3_w1600/protocol.json")
                  if (p.parent / "scene/sparse/0/images.bin").is_file() and "sfm_sha256" in read(p)]
    if candidates:
        return max(candidates, key=lambda p: (p.stat().st_mtime_ns, p)).parent
    # A fresh machine may have completed SfM without having trained the full baseline.
    models = [p for p in OUTPUT.glob("spirula_sfm*_w1600/sparse/0/images.bin")
              if (p.parents[2] / "spirula-input.txt").is_file()]
    if not models:
        raise ValueError(f"No completed Orange4F sparse model or staged stride-3 run under {OUTPUT}")
    source = max(models, key=lambda p: p.stat().st_mtime_ns).parents[2]
    native = native_matcher(source)
    backend = native or ("superglue_lidar" if (source/"sparse/superglue_lidar.json").exists() else "spirula")
    path = OUTPUT / f"{backend}_3dgs_stride3_w1600"
    with (OUTPUT/".keyframe-baseline.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if not (path/"protocol.json").exists():
            with lock(path), (source/".sparse-activation.lock").open("a") as guard:
                fcntl.flock(guard, fcntl.LOCK_SH)
                prepare(SimpleNamespace(source=source, run=path, iterations=30000, expected_frames=5009,
                                        mapper_model=None, native=native, keyframes=None))
    return path


def pin(src, dst, expected):
    """Pin even across a later atomic replacement of the source sparse directory."""
    if not dst.exists():
        def copy(tmp):
            try:
                os.link(src.resolve(strict=True), tmp)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                shutil.copyfile(src, tmp)
        atomic_write(dst, copy)
    if digest(dst) != expected:
        raise ValueError(f"Stored sparse file no longer matches its protocol: {dst}")


def remap(args, run, capture, names, train):
    folder = run / "keyframe_input/images"
    folder.mkdir(parents=True, exist_ok=True)
    for name in train:
        target = folder/name
        target.parent.mkdir(parents=True, exist_ok=True)
        link(run/"scene/images"/name, target)
    if {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()} != set(train):
        raise ValueError("Keyframe SfM input contains images outside the requested selection")
    work = run / "keyframe_sfm"
    cmd = ["bash", str(REPO/"scripts/orange4f_spirula_sfm.sh"), "--"+args.matcher, str(folder.parent), str(work)]
    proc = subprocess.run(cmd, env={**os.environ, "SPIRULA_DEVICE": os.environ.get("SPIRULA_DEVICE", args.gpu)})
    if proc.returncode not in (0, 3, 4):
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    sparse = work / "sparse/0"
    from scripts.orange4f_query_poses import localize
    posed = localize(sparse, run/"scene", capture, names, train, run)
    link(sparse/"points3D.bin", run/"scene/sparse/0/points3D.bin")
    return posed


def stage(args, arm, run, baseline, base, train, capture):
    sparse = run / "scene/sparse/0"
    sparse.mkdir(parents=True, exist_ok=True)
    link(baseline/"scene/images", run/"scene/images")
    names = set(base["train"] + base["test"] + base.get("unregistered", []))
    if set(train) - names or set(base["train"]) & set(base["test"]):
        raise ValueError("Inconsistent source train/test split")
    provenance = base["pose_provenance"] if arm == "fixed" else (
        f"Spirula {args.matcher} on selected RGB only; additional cameras use frozen-map optical flow + "
        "PnP with supplied capture intrinsics; all poses fixed during GS")
    settings = dict(arm=arm, train=train, iterations=args.iterations, pose_provenance=provenance,
                    baseline=str(baseline), reference_protocol_sha256=digest(baseline/"protocol.json"),
                    capture=str(capture), matcher=args.matcher if arm == "remap" else None)
    config = run / "settings.json"
    if config.is_file() and read(config) != settings:
        raise ValueError(f"Different experiment already occupies {run}")
    if not config.exists():
        atomic_json(config, settings)
    protocol = run / "protocol.json"
    if protocol.exists():
        saved = read(protocol)
        if any(digest(sparse/f) != saved["sfm_sha256"][f] for f in BINARIES):
            raise ValueError(f"Prepared sparse model changed in {run}")
        if read(run/"model/keyframes.json")["selected_images"] != train:
            raise ValueError(f"Prepared training split changed in {run}")
        return
    if arm == "fixed":
        for f in BINARIES:
            pin(baseline/"scene/sparse/0"/f, sparse/f, base["sfm_sha256"][f])
        posed = set(registered_names(run/"scene"))
        if posed != set(base["train"]+base["test"]):
            raise ValueError("Fixed-camera scene disagrees with its saved baseline")
        seeds = base["seed_provenance"]
    else:
        posed = remap(args, run, capture, names, train)
        seeds = str(run/"keyframe_sfm/sparse/0/points3D.bin")
    test = sorted(posed - set(train), key=natural_key)
    common = sorted(posed.intersection(base["test"]), key=natural_key)
    if not common:
        raise ValueError(f"No original held-out frame registered in {run}")
    atomic_json(run/"model/keyframes.json", dict(selected_images=train, selected_count=len(train),
                registered_count=len(posed), image_directory="images", source_path=str(run/"scene"),
                preview_heldout_images=common))
    with (sparse/"points3D.bin").open("rb") as stream:
        point_count, = struct.unpack("<Q", stream.read(8))
    atomic_json(protocol, dict(**settings, revision=subprocess.check_output(["git", "rev-parse", "HEAD"],
                cwd=REPO, text=True).strip(), source=str(run/"scene"), sfm_sha256={f:digest(sparse/f) for f in BINARIES},
                seed_points=point_count, seed_provenance=seeds, test=test, capture_frames=len(names),
                unregistered=sorted(names-posed, key=natural_key), width=1600, antialiasing=False))
    print(f"{arm}: {len(train)} train, {len(test)} held-out, {len(names-posed)} unregistered; "
          f"{point_count:,} seed points", flush=True)


def mean_db(rows, names):
    if not names:
        return None
    value = sum(float(rows[n]["psnr_db"]) for n in names) / len(names)
    return "inf" if value == float("inf") else value


def score_coverage(run, base, step):
    output = run/f"evaluation/iteration_{step}"
    result = read(output/"psnr.json")
    protocol = read(run/"protocol.json")
    rows = read(output/"eval/model_0/frames.json")
    original = sorted(set(base["test"]).intersection(rows), key=natural_key)
    result.update(capture_frames=protocol["capture_frames"], unregistered=protocol["unregistered"],
                  original_stride3_heldout=dict(scored=len(original), expected=len(base["test"]),
                                                mean_psnr_db=mean_db(rows, original)))
    atomic_json(output/"psnr.json", result)


def compare(root, baseline, base, step):
    paths = {arm: root/arm/f"evaluation/iteration_{step}" for arm in ("fixed", "remap")}
    if not all((p/"psnr.json").is_file() for p in paths.values()):
        return
    rows = {arm: read(p/"eval/model_0/frames.json") for arm, p in paths.items()}
    names = set.intersection(*(set(v) for v in rows.values()))
    stride3 = set(base["test"])
    selected = read(root/"fixed/protocol.json")["train"]
    remap_train = read(root/"remap/protocol.json")["train"]
    if selected != remap_train:
        raise ValueError("Cannot compare different training keyframes")
    baseline_dir = baseline/f"evaluation/iteration_{step}"
    baseline_rows = read(baseline_dir/"eval/model_0/frames.json") if (baseline_dir/"psnr.json").is_file() else {}
    total = len(base["train"]) + len(stride3) + len(base.get("unregistered", []))
    report = {}
    for label, group, expected in (("all_mutually_posed_heldout", names, total-len(selected)),
                                   ("original_stride3_heldout", names & stride3, len(stride3))):
        ordered = sorted(group, key=natural_key)
        scores = {k: mean_db(v, ordered) for k, v in rows.items()}
        if ordered and label == "original_stride3_heldout" and baseline_rows and group <= baseline_rows.keys():
            scores["stride3_baseline"] = mean_db(baseline_rows, ordered)
        report[label] = dict(scored=len(group), expected=expected, mean_psnr_db=scores, images=ordered)
    target = root / f"paired_psnr_{step}.json"
    atomic_json(target, dict(iteration=step, groups=report))
    for label, result in report.items():
        print(f"{label} ({result['scored']}/{result['expected']}): {result['mean_psnr_db']}", flush=True)
    print(target, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("both","fixed","remap"), default="both")
    parser.add_argument("--baseline", type=Path, help="Existing stride-3 run with the desired current poses")
    parser.add_argument("--matcher", choices=("flexglue","loma-b"), help="Remap frontend; default matches native baseline, else LoMa-B")
    parser.add_argument("--views", type=int, default=240)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES","0"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--evaluate-iteration", type=int)
    args = parser.parse_args()
    if not args.gpu or "," in args.gpu or args.iterations < 1:
        parser.error("Select one GPU and a positive training length")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    baseline = baseline_run(args)
    base = read(baseline/"protocol.json")
    train = thin(base["train"], args.views)
    source = Path(base["source"])
    args.matcher = args.matcher or native_matcher(source) or "loma-b"
    input_marker = source/"spirula-input.txt"
    capture = Path(input_marker.read_text().strip()).parent if input_marker.is_file() else CAPTURE
    root = baseline.parent/f"{baseline.name}_{args.views}kf_{args.matcher}"
    print(f"Reference poses: {baseline}\nRemap frontend: {args.matcher}\nOutput: {root}", flush=True)
    for arm in (("fixed", "remap") if args.arm == "both" else (args.arm,)):
        args.run = root / arm
        print(f"{arm} live previews: {args.run/'model/debug_previews/index.html'}", flush=True)
        if args.evaluate_iteration is not None:
            evaluate(args, args.evaluate_iteration, False)
        else:
            with lock(args.run):
                stage(args, arm, args.run, baseline, base, train, capture)
                if not args.prepare_only:
                    train_prepared(args)
        if not args.prepare_only:
            score_coverage(args.run, base, args.evaluate_iteration or args.iterations)
    if not args.prepare_only:
        compare(root, baseline, base, args.evaluate_iteration or args.iterations)


if __name__ == "__main__":
    main()
