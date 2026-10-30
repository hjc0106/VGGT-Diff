"""Isolated PyCOLMAP worker; only its interpreter needs pycolmap-cuda12."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

if __package__:
    from .sfm_profiles import keyframe_budgets, profile_settings, select_keyframes
else:
    from sfm_profiles import keyframe_budgets, profile_settings, select_keyframes


def resolve_device(requested: str, has_cuda: bool) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unknown SfM device: {requested}")
    if requested == "cuda" and not has_cuda:
        raise RuntimeError(
            "CUDA was requested, but this COLMAP build has no CUDA support. "
            "Use --colmap-python with an interpreter containing pycolmap-cuda12."
        )
    return "cuda" if requested != "cpu" and has_cuda else "cpu"


def check_ba_backend(pycolmap, name: str) -> None:
    options = pycolmap.BundleAdjustmentOptions()
    options.backend = getattr(pycolmap.BundleAdjustmentBackend, name.upper())
    # The enum exists even in wheels built without Caspar. Test the factory.
    pycolmap.create_default_bundle_adjuster(
        options, pycolmap.BundleAdjustmentConfig(), pycolmap.Reconstruction(),
    )


def has_ceres_cuda(pycolmap) -> bool:
    solver = pycolmap.CeresBundleAdjustmentOptions().solver_options
    solver.num_threads = 1
    solver.linear_solver_type = type(solver.linear_solver_type).__members__["DENSE_QR"]
    enum = type(solver.dense_linear_algebra_library_type)
    solver.dense_linear_algebra_library_type = enum.__members__["CUDA"]
    valid, _message = solver.IsValid()
    return valid


def final_bundle_adjustment(pycolmap, model, config, reference=None):
    options = pycolmap.BundleAdjustmentOptions()
    options.ceres.use_gpu = config.get("ba_device", "cpu") == "cuda"
    options.ceres.gpu_index = config["gpu_index"]
    options.ceres.solver_options.num_threads = config["num_threads"]
    options.ceres.solver_options.max_num_iterations = config["final_ba_iterations"]
    if reference is None:
        pycolmap.bundle_adjustment(model, options)
        return
    fixed = pycolmap.BundleAdjustmentConfig()
    for image_id, image in model.images.items():
        if image.has_pose:
            fixed.add_image(image_id)
    for image in reference.images.values():
        if image.has_pose:
            fixed.set_constant_rig_from_world_pose(image.frame_id)
    for camera_id in reference.cameras:
        fixed.set_constant_cam_intrinsics(camera_id)
    pycolmap.create_default_bundle_adjuster(options, fixed, model).solve()


def reconstruct(config: dict) -> dict:
    import pycolmap

    settings = profile_settings(config.get("profile", "reference"), config.get("final_ba_iterations"))
    if config.get("ba_device", "cpu") == "cuda" and not has_ceres_cuda(pycolmap):
        raise RuntimeError("GPU BA requires a Ceres build with CUDA support; this runtime is CPU-only")
    device_name = resolve_device(config["device"], pycolmap.has_cuda)
    use_gpu = device_name == "cuda"
    if (use_gpu or config.get("ba_device", "cpu") == "cuda") and pycolmap.get_num_cuda_devices() == 0:
        raise RuntimeError("COLMAP has CUDA support, but no CUDA device is visible")
    if config["ba_backend"] == "caspar" and not use_gpu:
        raise ValueError("The Caspar BA backend requires --sfm-device cuda")
    if config["ba_backend"] == "caspar":
        raise ValueError("Caspar does not support the OPENCV camera model used by this video pipeline")
    check_ba_backend(pycolmap, config["ba_backend"])
    device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu
    work, images = Path(config["work"]), Path(config["images"])
    database = work / "database.db"
    existing = config.get("input_model")
    sparse = work / "sparse_extra" / str(time.time_ns()) if existing else work / "sparse"
    if not existing:
        for path in (database, Path(str(database) + "-wal"), Path(str(database) + "-shm")):
            path.unlink(missing_ok=True)
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)
    timing_path = work / ("colmap_register_timings.json" if existing else "colmap_timings.json")
    timings = dict(
        backend="pycolmap", version=pycolmap.__version__, device=device_name,
        gpu_index=config["gpu_index"], ba_backend=config["ba_backend"],
        ba_device=config.get("ba_device", "cpu"), profile=settings["profile"],
        profile_settings=settings, stages=[],
    )

    @contextmanager
    def stage(name):
        print("COLMAP_STAGE_START", name, flush=True)
        start = time.monotonic()
        success = False
        try:
            yield
            success = True
        finally:
            seconds = time.monotonic() - start
            timings["stages"].append(dict(stage=name, seconds=seconds, success=success))
            timing_path.write_text(json.dumps(timings, indent=2) + "\n")
            print("COLMAP_STAGE_DONE", name, round(seconds, 3), "seconds", flush=True)

    print("COLMAP_RUNTIME", json.dumps({k: v for k, v in timings.items() if k != "stages"}), flush=True)
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = config["max_image_size"]
    extraction.num_threads = config["num_threads"]
    extraction.use_gpu = use_gpu
    extraction.gpu_index = config["gpu_index"]
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"
    if existing:
        reference = pycolmap.Reconstruction(existing)
        if len(reference.cameras) != 1:
            raise ValueError("Video registration requires one shared camera")
        reader.existing_camera_id = next(iter(reference.cameras))
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = config["num_threads"]
    matching.use_gpu = use_gpu
    matching.gpu_index = config["gpu_index"]
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = config["overlap"]
    pairing.quadratic_overlap = True
    rows = config.get("candidate_frames") or [
        dict(name=path.name, video_frame=i)
        for i, path in enumerate(sorted(images.iterdir()))
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    required = set(config.get("required_names", []))
    if existing:
        registered_names = {image.name for image in reference.images.values() if image.has_pose}
        selected = sorted(registered_names | set(config.get("image_names", [])))
        budgets = [len(selected)]
        required = set(config.get("image_names", []))
    else:
        selected = []
        budgets = keyframe_budgets(len(rows), config.get("keyframes", 60)) if settings["profile"] == "keyframe" else [len(rows)]
    attempts = []
    for attempt, budget in enumerate(budgets):
        attempt_settings = dict(settings)
        if not existing:
            selected = select_keyframes(rows, budget, required | set(selected))
            if settings["profile"] == "keyframe" and budget == len(rows) and attempt > 0:
                attempt_settings = profile_settings("reference", settings["final_ba_iterations"])
        extract_names = config.get("image_names", []) if existing else selected
        suffix = f"_{budget}" if len(budgets) > 1 else ""
        with stage("feature_extractor" + suffix):
            pycolmap.extract_features(
                database, images, image_names=extract_names,
                camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                extraction_options=extraction, device=device,
            )
        with stage("sequential_matcher" + suffix):
            pycolmap.match_sequential(database, matching, pairing, device=device)
        options = pycolmap.IncrementalPipelineOptions()
        options.num_threads = config["num_threads"]
        options.random_seed = 0
        options.image_names = selected
        for key in ("ba_global_frames_ratio", "ba_global_points_ratio", "ba_global_max_refinements",
                    "ba_local_max_num_iterations", "ba_global_max_num_iterations"):
            setattr(options, key, attempt_settings[key])
        options.ba_use_gpu = config.get("ba_device", "cpu") == "cuda"
        options.ba_gpu_index = config["gpu_index"]
        if existing:
            options.fix_existing_frames = True
            options.mapper.fix_existing_frames = True
            options.constant_cameras = set(reference.cameras)
            options.constant_rigs = set(reference.rigs)
        attempt_path = work / "mapping_attempts" / f"{attempt}_{budget}"
        if attempt_path.exists():
            shutil.rmtree(attempt_path)
        attempt_path.mkdir(parents=True)
        with stage(("image_registrator" if existing else "mapper") + suffix):
            models = pycolmap.incremental_mapping(
                database, images, attempt_path, options=options, input_path=existing or "",
            )
        if models:
            index, model = max(models.items(), key=lambda item: item[1].num_reg_images())
            names = {image.name for image in model.images.values() if image.has_pose}
        else:
            names = set()
        ratio = len(names) / max(len(selected), 1)
        threshold = config.get("keyframe_min_register_ratio", 0.95) if budget < len(rows) and not existing else config.get("min_register_ratio", 0.5)
        accepted = bool(models) and ratio >= threshold and required.issubset(names)
        attempts.append(dict(selected=len(selected), registered=len(names), register_ratio=ratio,
                             required_missing=sorted(required - names), accepted=accepted,
                             settings=attempt_settings))
        print("COLMAP_MAPPING_ATTEMPT", json.dumps(attempts[-1]), flush=True)
        if accepted:
            break
    else:
        raise RuntimeError(f"COLMAP could not register enough frames or required references: {attempts}")
    if settings["final_ba_iterations"]:
        with stage("final_bundle_adjustment"):
            final_bundle_adjustment(
                pycolmap, model,
                dict(config, final_ba_iterations=settings["final_ba_iterations"]),
                reference if existing else None,
            )
    timings["mapping_attempts"] = attempts
    timings["candidate_frames"] = len(rows)
    timings["mapping_frames"] = len(selected)
    timings["effective_profile"] = attempt_settings["profile"]
    model_path = sparse / str(index)
    model_path.mkdir(exist_ok=True)
    with stage("model_converter"):
        model.write(model_path)
        model.write_text(model_path)
    result = dict(
        model_path=str(model_path), registered=model.num_reg_images(),
        points=model.num_points3D(), mean_reprojection_error=model.compute_mean_reprojection_error(),
        **{k: v for k, v in timings.items() if k != "stages"},
    )
    (work / "colmap_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("COLMAP_RESULT", json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.probe:
        import pycolmap

        try:
            check_ba_backend(pycolmap, "caspar")
            has_caspar = True
        except ValueError:
            has_caspar = False
        print(json.dumps(dict(
            version=pycolmap.__version__, has_cuda=pycolmap.has_cuda, has_caspar=has_caspar,
            has_ceres_cuda=has_ceres_cuda(pycolmap),
        )))
    elif args.config:
        reconstruct(json.loads(args.config.read_text()))
    else:
        parser.error("Pass --probe or --config")


if __name__ == "__main__":
    main()
