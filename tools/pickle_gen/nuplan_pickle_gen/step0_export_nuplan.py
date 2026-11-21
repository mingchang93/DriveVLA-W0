#!/usr/bin/env python3
"""Export nuPlan sqlite logs into the JSON + image layout consumed by step1.

nuPlan ships one sqlite database per log (``<log>.db``) plus the decoded camera
frames under ``sensor_blobs/<log>/<channel>/<filename>.jpg``.  The rest of this
pipeline (step1..step4) expects, per log:

    <json_dir>/<log>.json   {"log", "channel", "images": [...], "poses": [...]}
    <image_dir>/<log>/<channel>/<filename>.jpg

``poses[i]`` is the relative transform between consecutive frames,
``inv(ego_i) @ ego_{i+1}``, and the last entry is identity.  step2 and step3b
multiply consecutive entries to integrate motion, so absolute ego poses would
produce nonsense there.  Pose targets come from the 10 Hz ``image`` rows (not
the 20 Hz ego_pose stream) so images and poses stay 1:1.

Usage:
    python step0_export_nuplan.py --db-root ~/nuplan/dataset/nuplan-v1.1/splits/mini \
        --blob-root ~/nuplan/dataset/nuplan-v1.1/sensor_blobs

Paths default to $NUPLAN_JSON_DIR / $NUPLAN_IMAGES_DIR (same variables that
steps 1-4 read), which default to <repo>/data/nuplan/{json,images}.
"""

import argparse
import json
import os
import sqlite3
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
NUPLAN_ROOT = os.environ.get("NUPLAN_ROOT", os.path.join(REPO_ROOT, "data/nuplan"))

QUERY = """
SELECT i.filename_jpg, e.x, e.y, e.z, e.qw, e.qx, e.qy, e.qz, i.timestamp
FROM image i
JOIN camera c ON i.camera_token = c.token
JOIN ego_pose e ON i.ego_pose_token = e.token
WHERE c.channel = ?
ORDER BY i.timestamp, i.filename_jpg
"""


def pose_matrix(x, y, z, qw, qx, qy, qz):
    """4x4 homogeneous matrix from an ego pose (quaternion is w, x, y, z)."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0:
        raise ValueError("zero-norm quaternion")
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    rot = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = (x, y, z)
    return mat


def relative_poses(poses):
    """inv(T_i) @ T_{i+1} for every frame; identity on the last frame."""
    out = np.tile(np.eye(4, dtype=np.float64), (len(poses), 1, 1))
    for i in range(len(poses) - 1):
        out[i] = np.linalg.inv(poses[i]) @ poses[i + 1]
    return out


def place_image(src, dst, mode):
    if dst.exists():
        return "exists"
    if not src.exists():
        return "missing"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        import shutil

        shutil.copy2(src, dst)
    return "placed"


def export_log(db_path, blob_root, json_dir, image_dir, channel, link_mode):
    log = db_path.stem
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(QUERY, (channel,)).fetchall()
    finally:
        conn.close()

    if len(rows) < 2:
        return {"log": log, "frames": len(rows), "placed": 0, "missing": 0, "skipped": True}

    images, poses = [], []
    placed = missing = 0
    for filename, x, y, z, qw, qx, qy, qz, _ts in rows:
        # nuPlan stores the full relative path in filename_jpg
        # ("<log>/CAM_F0/<file>.jpg"); older exports used a bare basename.
        if "/" in filename:
            rel = filename
        else:
            if not filename.endswith(".jpg"):
                filename = f"{filename}.jpg"
            rel = f"{log}/{channel}/{filename}"
        # Keep images and poses 1:1: step1 drops rows whose file is absent, and
        # step2 indexes poses by the compacted frame list.
        status = place_image(blob_root / rel, image_dir / rel, link_mode)
        if status == "missing":
            missing += 1
            continue
        if status == "placed":
            placed += 1
        images.append(rel)
        poses.append(pose_matrix(x, y, z, qw, qx, qy, qz))

    poses = relative_poses(poses)
    if len(images) < 2:
        return {"log": log, "frames": len(images), "placed": placed, "missing": missing, "skipped": True}
    payload = {
        "log": log,
        "channel": channel,
        "images": images,
        "poses": [p.tolist() for p in poses],
    }
    out_path = json_dir / f"{log}.json"
    out_path.write_text(json.dumps(payload, separators=(",", ":")))

    return {"log": log, "frames": len(images), "placed": placed, "missing": missing, "skipped": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--db-root",
        default=os.environ.get("NUPLAN_DB_ROOT", ""),
        help="Directory searched recursively for <log>.db files",
    )
    parser.add_argument(
        "--blob-root",
        default=os.environ.get("NUPLAN_BLOB_ROOT", ""),
        help="sensor_blobs root containing <log>/<channel>/*.jpg",
    )
    parser.add_argument(
        "--json-dir", default=os.environ.get("NUPLAN_JSON_DIR", os.path.join(NUPLAN_ROOT, "json"))
    )
    parser.add_argument(
        "--image-dir", default=os.environ.get("NUPLAN_IMAGES_DIR", os.path.join(NUPLAN_ROOT, "images"))
    )
    parser.add_argument("--channel", default="CAM_F0")
    parser.add_argument("--link", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--limit", type=int, default=0, help="Export at most N logs (0 = all)")
    args = parser.parse_args()

    if not args.db_root or not args.blob_root:
        parser.error("--db-root and --blob-root are required (or set NUPLAN_DB_ROOT / NUPLAN_BLOB_ROOT)")

    db_root = Path(args.db_root).expanduser()
    blob_root = Path(args.blob_root).expanduser()
    json_dir = Path(args.json_dir).expanduser()
    image_dir = Path(args.image_dir).expanduser()
    json_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    dbs = sorted(db_root.rglob("*.db"))
    if args.limit:
        dbs = dbs[: args.limit]
    if not dbs:
        raise SystemExit(f"no .db files under {db_root}")

    totals = {"logs": 0, "frames": 0, "missing": 0, "skipped": 0}
    for i, db_path in enumerate(dbs, 1):
        stats = export_log(db_path, blob_root, json_dir, image_dir, args.channel, args.link)
        totals["logs"] += 1
        totals["frames"] += stats["frames"]
        totals["missing"] += stats["missing"]
        totals["skipped"] += int(stats["skipped"])
        if stats["skipped"] or stats["missing"]:
            print(f"[{i}/{len(dbs)}] {stats['log']}: frames={stats['frames']} "
                  f"missing_images={stats['missing']} skipped={stats['skipped']}")
        elif i % 25 == 0:
            print(f"[{i}/{len(dbs)}] {stats['log']}: frames={stats['frames']}")

    print(f"EXPORTED logs={totals['logs']} frames={totals['frames']} "
          f"missing_images={totals['missing']} skipped={totals['skipped']}")
    print(f"json_dir={json_dir} image_dir={image_dir}")


if __name__ == "__main__":
    main()
