"""Run COLMAP from the command line and read a text model."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from gen_pro.colmap_gpu import resolve_device
from gen_pro.config import Config
from gen_pro.sfm_profiles import profile_settings


@lru_cache(maxsize=8)
def _probe_colmap(binary: str, python: str | None) -> dict:
    if python:
        command = [python, str(Path(__file__).with_name("colmap_gpu.py")), "--probe"]
    else:
        command = [binary, "-h"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"Cannot inspect COLMAP: {result.stderr.strip() or result.stdout.strip()}")
    if python:
        return json.loads(result.stdout)
    text = result.stdout + result.stderr
    version = re.search(r"COLMAP\s+(\d+)\.(\d+)", text)
    modern = bool(version and tuple(map(int, version.groups())) >= (4, 2))
    return dict(has_cuda="with CUDA" in text and "without CUDA" not in text, modern_options=modern)


def _cli_groups(cfg: Config) -> tuple[str, str, str]:
    modern = _probe_colmap(cfg.colmap_bin, None).get("modern_options", False)
    return (("FeatureExtraction", "FeatureMatching", "fix_existing_frames") if modern
            else ("SiftExtraction", "SiftMatching", "fix_existing_images"))


def _device(cfg: Config) -> str:
    if cfg.sfm_num_threads < 1:
        raise ValueError("sfm_num_threads must be positive")
    profile_settings(cfg.sfm_profile, cfg.sfm_final_ba_iterations)
    if cfg.sfm_keyframes < 2 or not 0 < cfg.sfm_keyframe_min_register_ratio <= 1:
        raise ValueError("Invalid keyframe count or registration ratio")
    if not cfg.colmap_python and (cfg.sfm_profile != "reference" or cfg.sfm_final_ba_iterations
                                  or cfg.sfm_ba_device != "cpu"):
        raise ValueError("SfM profiles and GPU BA require --colmap-python")
    info = _probe_colmap(cfg.colmap_bin, cfg.colmap_python)
    if cfg.sfm_ba_device == "cuda" and not info.get("has_ceres_cuda", False):
        raise RuntimeError("This runtime has no Ceres CUDA support; use --sfm-ba-device cpu")
    device = resolve_device(cfg.sfm_device, info["has_cuda"])
    if cfg.sfm_ba_backend == "caspar" and (not cfg.colmap_python or device != "cuda"):
        raise ValueError("Caspar requires --colmap-python and a CUDA-enabled PyCOLMAP runtime")
    if cfg.sfm_ba_backend == "caspar" and not info.get("has_caspar", False):
        raise RuntimeError("This PyCOLMAP build has no Caspar GPU BA; use --sfm-ba-backend ceres")
    if cfg.sfm_ba_backend == "caspar":
        raise ValueError("Caspar does not support the OPENCV camera model used by this video pipeline")
    print("COLMAP_DEVICE", device, "BA", cfg.sfm_ba_backend, flush=True)
    return device


def _run_pycolmap(image_dir: Path, work: Path, cfg: Config, input_model=None, names=None) -> dict:
    device = _device(cfg)
    frames_path = work / "frames.json"
    rows = json.loads(frames_path.read_text())["frames"] if frames_path.exists() else []
    required = [row["name"] for row in rows if row.get("video_frame") in cfg.manual_frames]
    payload = dict(
        images=str(image_dir.resolve()), work=str(work.resolve()), device=device,
        gpu_index=cfg.sfm_gpu_index, num_threads=cfg.sfm_num_threads,
        max_image_size=cfg.sfm_max_image_size, overlap=cfg.sfm_overlap,
        ba_backend=cfg.sfm_ba_backend,
        input_model=str(input_model.resolve()) if input_model else None,
        image_names=names or [],
        profile=cfg.sfm_profile, keyframes=cfg.sfm_keyframes,
        keyframe_min_register_ratio=cfg.sfm_keyframe_min_register_ratio,
        min_register_ratio=cfg.sfm_min_register_ratio,
        final_ba_iterations=cfg.sfm_final_ba_iterations, ba_device=cfg.sfm_ba_device,
        candidate_frames=rows, required_names=required,
    )
    path = work / ("colmap_register_config.json" if input_model else "colmap_config.json")
    path.write_text(json.dumps(payload, indent=2) + "\n")
    _run([cfg.colmap_python, str(Path(__file__).with_name("colmap_gpu.py")), "--config", str(path)],
         work / "colmap.log")
    result = json.loads((work / "colmap_result.json").read_text())
    return read_text_model(Path(result["model_path"]))


def run_colmap(image_dir: Path, work: Path, cfg: Config) -> dict:
    if cfg.colmap_python:
        return _run_pycolmap(image_dir, work, cfg)
    device = _device(cfg)
    extraction, matching, _ = _cli_groups(cfg)
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
            f"--{extraction}.max_image_size",
            str(cfg.sfm_max_image_size),
            f"--{extraction}.use_gpu",
            str(int(device == "cuda")),
            f"--{extraction}.gpu_index",
            cfg.sfm_gpu_index,
            f"--{extraction}.num_threads",
            str(cfg.sfm_num_threads),
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
            f"--{matching}.use_gpu",
            str(int(device == "cuda")),
            f"--{matching}.gpu_index",
            cfg.sfm_gpu_index,
            f"--{matching}.num_threads",
            str(cfg.sfm_num_threads),
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
            "--Mapper.num_threads",
            str(cfg.sfm_num_threads),
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
    if cfg.colmap_python:
        result_path = work / "colmap_result.json"
        model = (Path(json.loads(result_path.read_text())["model_path"])
                 if result_path.exists() else _model_dir(work / "sparse"))
        return _run_pycolmap(image_dir, work, cfg, model, names)
    device = _device(cfg)
    extraction, matching, fixed_frames = _cli_groups(cfg)
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
            f"--{extraction}.max_image_size",
            str(cfg.sfm_max_image_size),
            f"--{extraction}.use_gpu",
            str(int(device == "cuda")),
            f"--{extraction}.gpu_index",
            cfg.sfm_gpu_index,
            f"--{extraction}.num_threads",
            str(cfg.sfm_num_threads),
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
            f"--{matching}.use_gpu",
            str(int(device == "cuda")),
            f"--{matching}.gpu_index",
            cfg.sfm_gpu_index,
            f"--{matching}.num_threads",
            str(cfg.sfm_num_threads),
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
            f"--Mapper.{fixed_frames}",
            "1",
            "--Mapper.num_threads",
            str(cfg.sfm_num_threads),
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
    start = time.monotonic()
    with log.open("a") as handle:
        handle.write("\n$ " + shlex.join(command) + "\n")
        handle.flush()
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        seconds = time.monotonic() - start
        handle.write(f"\nCOMMAND_DONE seconds={seconds:.3f} returncode={result.returncode}\n")
    path = log.with_name("colmap_commands.json")
    records = json.loads(path.read_text()) if path.exists() else []
    records.append(dict(command=command, seconds=seconds, returncode=result.returncode))
    path.write_text(json.dumps(records, indent=2) + "\n")
    print("COLMAP_COMMAND_DONE", command[1], round(seconds, 3), "seconds", flush=True)
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
    lines = [line for line in path.read_text().splitlines() if not line.startswith("#")]
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
