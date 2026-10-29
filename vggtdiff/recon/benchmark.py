"""Portable, content-checked reproduction of the frozen white-car E1 benchmark."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import e1


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def verify_example(example: Path) -> dict:
    """Check every frozen input before allowing cached outputs or training to run."""
    example = example.resolve()
    manifest = read(example / "benchmark.json")
    if manifest.get("schema_version") != 1 or manifest.get("name") != "white_car_old_e1":
        raise ValueError("Expected the white_car_old_e1 benchmark manifest")
    required = {"trajectory.json", "initial_points.ply", "initialization.json", "evaluation/cameras.json"}
    required |= {f"source_views/{i:02d}.png" for i in range(6)}
    required |= {f"{folder}/{i:02d}.png" for folder in ("frames", "masks") for i in range(80)}
    if not required.issubset(manifest["files"]):
        raise ValueError("Benchmark manifest is missing required frozen inputs")
    for relative, expected in manifest["files"].items():
        path = (example / relative).resolve()
        if not path.is_relative_to(example):
            raise ValueError(f"Benchmark path escapes the example: {relative}")
        if not path.is_file() or e1.sha256(path) != expected:
            raise ValueError(f"Frozen benchmark input is missing or changed: {relative}")
    trajectory = read(example / "trajectory.json")
    if trajectory["source_frames"] != manifest["source_frames"]:
        raise ValueError("The frozen source-frame order changed")
    e1.training_indices(np.asarray(trajectory["target_w2c"]))
    return manifest


def snapshot_example(example: Path, output: Path, manifest: dict) -> Path:
    """Keep each run's references and evaluation inputs independent of shared workspaces."""
    destination = output / "fixture"
    if (destination / "benchmark.json").exists():
        if read(destination / "benchmark.json") != manifest:
            raise ValueError("Output belongs to a different benchmark; choose a new output directory")
        verify_example(destination)
        return destination
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Partial fixture snapshot exists: {destination}; choose a new output directory")
    for relative in manifest["files"]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(example / relative, target)
    shutil.copyfile(example / "benchmark.json", destination / "benchmark.json")
    verify_example(destination)
    return destination


def compare_metrics(actual: dict, manifest: dict) -> dict:
    checks = []
    for category in ("training", "real"):
        for metric, expected in manifest["reference"][category].items():
            value = float(actual[category]["median"][metric])
            tolerance = float(manifest["tolerances"][metric])
            difference = value - expected
            checks.append(dict(
                category=category, metric=metric, reference=expected, actual=value,
                difference=difference, tolerance=tolerance,
                passed=bool(np.isfinite(value) and abs(difference) <= tolerance),
            ))
    return dict(passed=all(row["passed"] for row in checks), checks=checks)


def compare_generation(example: Path, generated: Path) -> dict:
    """Require identical decoded pixels before reusing the frozen segmentation masks."""
    records = []
    for index in range(80):
        reference = np.asarray(Image.open(example / "frames" / f"{index:02d}.png").convert("RGB"))
        path = generated / f"{index:02d}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Generated orbit is incomplete: {path}")
        actual = np.asarray(Image.open(path).convert("RGB"))
        if actual.shape != reference.shape:
            raise ValueError(f"Generated frame size differs from the benchmark: {path}")
        equal = bool(np.array_equal(reference, actual))
        difference = (actual.astype(np.float64) - reference.astype(np.float64)) / 255.0
        mse = float(np.mean(difference ** 2))
        records.append(dict(index=index, pixels_equal=equal, sha256=e1.sha256(path),
                            psnr=float(-10 * np.log10(max(mse, 1e-12))),
                            mae=float(np.abs(difference).mean())))
    return dict(passed=all(row["pixels_equal"] for row in records),
                median_psnr=float(np.median([row["psnr"] for row in records])), frames=records)


def _validate_training_copy(output: Path, fixture: Path) -> list[dict]:
    rows = read(output / "inputs.json")["frames"]
    if [row["orbit_index"] for row in rows] != list(range(79)):
        raise ValueError("Output does not contain the 79 frozen training views")
    trajectory = read(fixture / "trajectory.json")
    for row in rows:
        index = row["orbit_index"]
        image = output / "dataset/images" / f"{index:02d}.png"
        mask = image.with_name(f"{index:02d}_mask.png")
        if e1.sha256(image) != e1.sha256(fixture / "frames" / image.name):
            raise ValueError(f"Training image {index} differs from the frozen benchmark")
        if e1.sha256(mask) != e1.sha256(fixture / "masks" / image.name):
            raise ValueError(f"Training mask {index} differs from the frozen benchmark")
        if not np.array_equal(row["w2c"], trajectory["target_w2c"][index]):
            raise ValueError(f"Training pose {index} differs from the frozen benchmark")
        if not np.array_equal(row["k"], trajectory["target_intrinsics"][index]):
            raise ValueError(f"Training intrinsics {index} differ from the frozen benchmark")
        row.update(image=str(image), mask=str(mask))
    if e1.sha256(output / "shared/initial_points.ply") != e1.sha256(fixture / "initial_points.ply"):
        raise ValueError("Training initialization differs from the frozen benchmark")
    initialization = e1.initialization_values(fixture / "initial_points.ply")
    saved_init = read(output / "shared/initialization.json")
    for key in ("real_surface_points", "volume_seed_points", "seed_opacity", "bounds_world"):
        if saved_init.get(key) != initialization[key]:
            raise ValueError(f"Training initialization metadata changed: {key}")
    experiment = read(output / "experiment.json")
    defaults = e1.load_defaults()
    for key in ("training_seed", "updates_per_image", "outside_weight", "anisotropy_weight", "gaussian_cap", "schedule"):
        if experiment[key] != defaults[key]:
            raise ValueError(f"Training recipe changed: {key}")
    if experiment["scene_extent"] != initialization["scene_extent"]:
        raise ValueError("Training scene extent differs from the frozen benchmark")
    return rows


def evaluate_benchmark(output: Path, fixture: Path, threedgrut: Path, manifest: dict) -> dict:
    import imageio.v2 as imageio
    import torch

    e1.use_threedgrut(threedgrut)
    from threedgrut.model.model import MixtureOfGaussians

    run = output / "e1"
    rows = _validate_training_copy(run, fixture)
    result = read(run / "training/result.json")
    if result["step"] != e1.load_defaults()["schedule"]["position_max_steps"]:
        raise ValueError("E1 did not complete all 49,375 training steps")
    if e1.sha256(run / "model.ply") != result["ply_sha256"]:
        raise ValueError("The exported model differs from its training record")
    checkpoint = run / "training/E1/dataset" / f"ours_{result['step']}" / f"ckpt_{result['step']}.pt"
    model = MixtureOfGaussians(e1._load_conf(run))
    model.init_from_checkpoint(torch.load(checkpoint, map_location="cuda", weights_only=False), setup_optimizer=False)
    model.eval()
    if model.num_gaussians != result["gaussians"]:
        raise ValueError("Checkpoint Gaussian count differs from the training record")
    cameras = read(fixture / "evaluation/cameras.json")
    real = [dict(row, image=str(fixture / row["image"]), mask=str(fixture / row["mask"]))
            for row in cameras["heldout_real"]]
    actual = dict(training=e1.score_views(model, rows), real=e1.score_views(model, real),
                  gaussians=result["gaussians"], step=result["step"])
    report = dict(benchmark=manifest["name"], **actual, **compare_metrics(actual, manifest))
    evaluation = output / "evaluation"
    evaluation.mkdir(exist_ok=True)
    # Use the original independent 160-view orbit, not the training cameras.
    with imageio.get_writer(evaluation / "orbit360.mp4", fps=12, codec="libx264", quality=8,
                            macro_block_size=None) as writer:
        selected = []
        for index, row in enumerate(cameras["orbit360"]):
            rgb, _alpha = e1.render_view(model, dict(row, image=rows[0]["image"]))
            frame = Image.fromarray((rgb * 255).astype(np.uint8))
            writer.append_data(np.asarray(frame))
            if index % 20 == 0:
                tile = Image.new("RGB", (416, 264), "#181818")
                tile.paste(frame.resize((416, 240)), (0, 24))
                ImageDraw.Draw(tile).text((8, 5), f"E1 orbit {index:03d}", fill="white")
                selected.append(tile)
    sheet = Image.new("RGB", (416 * 4, 264 * 2), "black")
    for index, tile in enumerate(selected):
        sheet.paste(tile, ((index % 4) * 416, (index // 4) * 264))
    sheet.save(evaluation / "orbit_contact.jpg", quality=93)
    (evaluation / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    medians = {category: actual[category]["median"] for category in ("training", "real")}
    print("BENCHMARK", "PASS" if report["passed"] else "FAIL", json.dumps(medians), flush=True)
    return report


def reproduce(example: Path, output: Path, threedgrut: Path) -> dict:
    example, output = example.resolve(), output.resolve()
    manifest = verify_example(example)
    output.mkdir(parents=True, exist_ok=True)
    fixture = snapshot_example(example, output, manifest)
    run = output / "e1"
    if (run / "experiment.json").exists():
        _validate_training_copy(run, fixture)
    e1.reconstruct(run, fixture / "frames", fixture / "trajectory.json", fixture / "masks",
                   fixture / "initial_points.ply", threedgrut)
    report = evaluate_benchmark(output, fixture, threedgrut, manifest)
    if not report["passed"]:
        raise RuntimeError(f"E1 benchmark is outside the recorded tolerances: {output / 'evaluation/benchmark.json'}")
    return report
