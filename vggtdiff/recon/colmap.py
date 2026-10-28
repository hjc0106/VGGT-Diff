"""Write an OpenCV COLMAP dataset from generated frames, masks, and cameras."""

from __future__ import annotations

import shutil
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def rotation_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """COLMAP's ``qw qx qy qz`` from a 3x3 rotation."""
    rotation = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def write_colmap(dataset: Path, rows: list[dict], init_ply: Path) -> None:
    images = dataset / "images"
    sparse = dataset / "sparse" / "0"
    images.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)
    width, height = Image.open(rows[0]["image"]).size
    camera_lines, image_lines = [], []
    camera_bin = (sparse / "cameras.bin").open("wb")
    image_bin = (sparse / "images.bin").open("wb")
    with camera_bin as cameras, image_bin as records:
        cameras.write(struct.pack("<Q", len(rows)))
        records.write(struct.pack("<Q", len(rows)))
        for camera_id, row in enumerate(rows, 1):
            intrinsics = np.asarray(row["k"], dtype=np.float64)
            w2c = np.asarray(row["w2c"], dtype=np.float64)
            quat = rotation_to_quat_wxyz(w2c[:3, :3])
            params = [
                intrinsics[0, 0],
                intrinsics[1, 1],
                intrinsics[0, 2],
                intrinsics[1, 2],
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            cameras.write(struct.pack("<iiQQ8d", camera_id, 4, width, height, *params))
            records.write(struct.pack("<i7di", camera_id, *quat, *w2c[:3, 3], camera_id))
            name = row["name"].encode() + b"\0"
            records.write(name)
            records.write(struct.pack("<Q", 0))
            camera_lines.append(" ".join(map(str, [camera_id, "OPENCV", width, height, *params])))
            image_lines.append(
                " ".join(map(str, [camera_id, *quat, *w2c[:3, 3], camera_id, row["name"]])) + "\n"
            )
            shutil.copyfile(row["image"], images / row["name"])
            shutil.copyfile(row["mask"], images / f"{Path(row['name']).stem}_mask.png")
    (sparse / "cameras.txt").write_text("\n".join(camera_lines) + "\n")
    (sparse / "images.txt").write_text("\n".join(image_lines) + "\n")
    (sparse / "points3D.txt").write_text("# Explicit fused initialization\n")
    shutil.copyfile(init_ply, dataset / "point_cloud.ply")
