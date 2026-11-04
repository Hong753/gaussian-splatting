"""Select sharp COLMAP-registered training frames without modifying the dataset."""

import hashlib
import json
import os
from pathlib import Path
import re
import struct
from utils.colmap_paths import colmap_model_dir


def _read_exact(file, count):
    value = file.read(count)
    if len(value) != count:
        raise ValueError("Truncated COLMAP images.bin")
    return value


def registered_names(source):
    """Read only image names; skip the potentially huge COLMAP observation arrays."""
    model = colmap_model_dir(source)
    binary = model / "images.bin"
    text = model / "images.txt"
    names = []
    if binary.is_file() and (model / "cameras.bin").is_file():
        file_size = binary.stat().st_size
        with binary.open("rb") as file:
            count = struct.unpack("<Q", _read_exact(file, 8))[0]
            for _ in range(count):
                _read_exact(file, 64)  # image id, pose quaternion, translation, camera id
                name = bytearray()
                while True:
                    char = _read_exact(file, 1)
                    if char == b"\0":
                        break
                    name.extend(char)
                names.append(name.decode("utf-8"))
                npoints = struct.unpack("<Q", _read_exact(file, 8))[0]
                if npoints > (file_size - file.tell()) // 24:
                    raise ValueError("Truncated COLMAP observation data")
                file.seek(npoints * 24, os.SEEK_CUR)
    elif text.is_file() and (model / "cameras.txt").is_file():
        with text.open(encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split(maxsplit=9)
                if len(fields) != 10:
                    raise ValueError(f"Invalid COLMAP image record in {text}")
                names.append(fields[9])
                if file.readline() == "":
                    raise ValueError(f"Missing COLMAP image observations in {text}")
    else:
        raise FileNotFoundError(f"Expected COLMAP images and cameras (.bin or .txt) in {model}")
    if not names or len(set(names)) != len(names):
        raise ValueError("COLMAP model has no registered images or contains duplicate names")
    return names


def natural_key(name):
    parts = re.split(r"(\d+)", name)
    return tuple((0, int(part)) if part.isdigit() else (1, part.casefold()) for part in parts), name


def select_keyframes(source, images, output, group_size, use_eval):
    # The native loader uses a lexicographic 1-in-8 test split. Apply it first,
    # then group only training images in natural filename order.
    names = registered_names(source)
    lexical_names = sorted(names)
    held_out = set(lexical_names[::8]) if use_eval else set()
    candidates = sorted((name for name in names if name not in held_out), key=natural_key)
    if not candidates:
        raise ValueError("No registered training frames remain after the test split")
    fingerprint = hashlib.sha256(json.dumps(lexical_names, ensure_ascii=False).encode("utf-8")).hexdigest()
    manifest_path = output / "keyframes.json"
    metadata = {
        "version": 1,
        "source_path": str(source),
        "image_directory": images,
        "group_size": group_size,
        "eval": use_eval,
        "registered_names_sha256": fingerprint,
        "ordering": "natural sort of COLMAP image names",
        "metric": "variance of Laplacian on full-resolution grayscale images",
        "registered_count": len(names),
        "test_count": len(held_out),
        "candidate_count": len(candidates),
    }
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as file:
            existing = json.load(file)
        if any(existing.get(key) != value for key, value in metadata.items()):
            raise ValueError(f"Existing {manifest_path} does not match these settings or COLMAP images; use a new run name")
        expected_groups = (len(candidates) + group_size - 1) // group_size
        chosen = existing.get("selected_images", [])
        if (len(chosen) != expected_groups or len(set(chosen)) != len(chosen) or
                any(name not in candidates[index * group_size:(index + 1) * group_size]
                    for index, name in enumerate(chosen))):
            raise ValueError(f"Invalid existing keyframe manifest: {manifest_path}")
        print(f"Reusing {len(chosen)} keyframes from {manifest_path}", flush=True)
        return manifest_path

    import cv2

    print(f"Scoring {len(candidates)} registered training images in groups of {group_size}...", flush=True)
    groups = []
    chosen = []
    for start in range(0, len(candidates), group_size):
        group = candidates[start:start + group_size]
        scores = []
        for name in group:
            image_path = source / images / name
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Cannot read registered image: {image_path}")
            score = float(cv2.Laplacian(image, cv2.CV_64F).var())
            scores.append(score)
        winner = max(range(len(group)), key=lambda index: scores[index])
        chosen.append(group[winner])
        groups.append({"images": group, "sharpness": scores, "selected": group[winner]})
        done = min(start + group_size, len(candidates))
        if done == len(candidates) or done // 300 != start // 300:
            print(f"  Scored {done}/{len(candidates)}", flush=True)
    manifest = {**metadata, "selected_count": len(chosen), "selected_images": chosen, "groups": groups}
    output.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
    temporary.replace(manifest_path)
    print(f"Selected {len(chosen)}/{len(candidates)} training images; saved {manifest_path}", flush=True)
    return manifest_path
