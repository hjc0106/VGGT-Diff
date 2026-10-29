"""Run COLMAP from the command line and read a text model."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from gen_pro.config import Config


def run_colmap(image_dir: Path, work: Path, cfg: Config) -> dict:
    database = work / "database.db"
    sparse = work / "sparse"
    if database.exists():
        database.unlink()
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)
    log = work / "colmap.log"
    common = [cfg.colmap_bin]
    _run(
        common
        + [
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(image_dir),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            "OPENCV",
            "--SiftExtraction.max_image_size",
            str(cfg.sfm_max_image_size),
            "--SiftExtraction.use_gpu",
            "0",
            "--SiftExtraction.num_threads",
            "16",
        ],
        log,
    )
    _run(
        common
        + [
            "sequential_matcher",
            "--database_path",
            str(database),
            "--SequentialMatching.overlap",
            str(cfg.sfm_overlap),
            "--SequentialMatching.quadratic_overlap",
            "1",
            "--SiftMatching.use_gpu",
            "0",
            "--SiftMatching.num_threads",
            "16",
        ],
        log,
    )
    _run(
        common
        + [
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse),
        ],
        log,
    )
    model = _largest_model(sparse)
    _run(
        common
        + [
            "model_converter",
            "--input_path",
            str(model),
            "--output_path",
            str(model),
            "--output_type",
            "TXT",
        ],
        log,
    )
    return read_text_model(model)


def register_new_images(image_dir: Path, work: Path, names: list[str], cfg: Config) -> dict:
    """Add named frames to an existing reconstruction without moving the current cameras."""
    database = work / "database.db"
    listing = work / "extra_images.txt"
    listing.write_text("\n".join(names) + "\n")
    log = work / "colmap.log"
    model = _model_dir(work / "sparse")
    _run(
        [
            cfg.colmap_bin,
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(image_dir),
            "--image_list_path",
            str(listing),
            "--ImageReader.existing_camera_id",
            "1",
            "--SiftExtraction.max_image_size",
            str(cfg.sfm_max_image_size),
            "--SiftExtraction.use_gpu",
            "0",
            "--SiftExtraction.num_threads",
            "16",
        ],
        log,
    )
    _run(
        [
            cfg.colmap_bin,
            "sequential_matcher",
            "--database_path",
            str(database),
            "--SequentialMatching.overlap",
            str(cfg.sfm_overlap),
            "--SequentialMatching.quadratic_overlap",
            "1",
            "--SiftMatching.use_gpu",
            "0",
            "--SiftMatching.num_threads",
            "16",
        ],
        log,
    )
    output = work / "sparse_extra"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir()
    _run(
        [
            cfg.colmap_bin,
            "image_registrator",
            "--database_path",
            str(database),
            "--input_path",
            str(model),
            "--output_path",
            str(output),
            "--Mapper.fix_existing_images",
            "1",
        ],
        log,
    )
    registered = _model_dir(output)
    _run(
        [
            cfg.colmap_bin,
            "model_converter",
            "--input_path",
            str(registered),
            "--output_path",
            str(registered),
            "--output_type",
            "TXT",
        ],
        log,
    )
    return read_text_model(registered)


def read_text_model(model: Path) -> dict:
    cameras = _read_cameras(model / "cameras.txt")
    images = _read_images(model / "images.txt")
    points = _read_points(model / "points3D.txt")
    return dict(cameras=cameras, images=images, points=int(len(points)), model=str(model))


def _model_dir(path: Path) -> Path:
    if (path / "images.bin").exists() or (path / "images.txt").exists():
        return path
    return _largest_model(path)


def _largest_model(sparse: Path) -> Path:
    models = [path for path in sparse.iterdir() if path.is_dir()]
    if not models:
        raise RuntimeError(f"COLMAP produced no model in {sparse}")
    return max(models, key=lambda path: (path / "images.bin").stat().st_size if (path / "images.bin").exists() else 0)


def _run(command: list[str], log: Path) -> None:
    with log.open("a") as handle:
        handle.write("\n$ " + " ".join(command) + "\n")
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {command[1]}. See {log}")


def _read_cameras(path: Path) -> dict:
    cameras = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        params = [float(value) for value in parts[4:]]
        cameras[int(parts[0])] = dict(
            model=parts[1],
            width=int(parts[2]),
            height=int(parts[3]),
            params=params,
        )
    return cameras


def _read_images(path: Path) -> list[dict]:
    lines = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    images = []
    for line in lines[::2]:
        parts = line.split()
        quat = np.array([float(value) for value in parts[1:5]])
        translation = [float(value) for value in parts[5:8]]
        rotation = _quat_to_matrix(quat)
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        images.append(
            dict(
                image_id=int(parts[0]),
                camera_id=int(parts[8]),
                name=parts[9],
                w2c=pose.tolist(),
            )
        )
    return images


def _read_points(path: Path) -> list:
    if not path.exists():
        return []
    return [line for line in path.read_text().splitlines() if line and not line.startswith("#")]


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
