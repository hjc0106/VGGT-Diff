"""Prepare and train the frozen E1 object reconstruction.

The recipe is 79 orbit views, 625 updates each, outside-opacity weight 0.5, and the
A2 densify/prune schedule. An 80-frame closed orbit holds out the duplicate last frame.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from vggtdiff.recon.colmap import write_colmap

DEFAULTS_PATH = Path(__file__).with_name("reconstruction_defaults.json")
TEMPLATE = Path(__file__).with_name("reconstruction_config.yaml")
UPDATES_PER_IMAGE = 625
TRAINING_VIEWS = 79
POSE_REFINE = dict(
    rotation_max_deg=1.0,
    translation_max_ratio=0.02,
    start_step=1000,
    lr_init=1e-2,
    lr_final=1e-4,
    prior_weight=1e-3,
    fixed_view=0,
)


def load_defaults() -> dict:
    return json.loads(DEFAULTS_PATH.read_text())


def _camera_center(w2c) -> np.ndarray:
    pose = np.asarray(w2c, dtype=np.float64)
    return -pose[:3, :3].T @ pose[:3, 3]


def _bound_vector(raw, limit):
    """Scale a vector so its length is ``limit * tanh(||raw||)``. Zero stays zero and keeps a gradient."""
    import torch

    norm = torch.linalg.vector_norm(raw)
    # Both branches stay finite. Dividing by the raw norm at zero is NaN and poisons the backward.
    safe = torch.where(norm > 1e-8, norm, torch.ones((), device=raw.device, dtype=raw.dtype))
    unit_scale = torch.where(norm > 1e-8, torch.tanh(norm) / safe, torch.ones_like(norm))
    return limit * unit_scale * raw


class PoseCorrection:
    """One bounded camera-frame correction per training view. View 0 stays fixed."""

    def __init__(self, count: int, rotation_max: float, translation_max: float, device):
        import torch

        self.raw = torch.nn.Parameter(torch.zeros(count, 6, device=device))
        self.rotation_max = float(rotation_max)
        self.translation_max = float(translation_max)
        self.m = torch.zeros(count, 6, device=device)
        self.v = torch.zeros(count, 6, device=device)
        self.steps = torch.zeros(count, dtype=torch.long, device=device)

    def motion(self, index: int):
        omega = _bound_vector(self.raw[index, :3], self.rotation_max)
        translation = _bound_vector(self.raw[index, 3:], self.translation_max)
        return omega, translation

    def delta(self, index: int):
        import torch

        omega, translation = self.motion(index)
        wx, wy, wz = omega.unbind()
        zero = omega.new_zeros(())
        skew = torch.stack((
            torch.stack((zero, -wz, wy)),
            torch.stack((wz, zero, -wx)),
            torch.stack((-wy, wx, zero)),
        ))
        rotation = torch.linalg.matrix_exp(skew)
        bottom = omega.new_tensor([0.0, 0.0, 0.0, 1.0])
        return torch.cat((torch.cat((rotation, translation[:, None]), dim=1), bottom[None]), dim=0)

    def penalty(self, index: int):
        import torch

        normalized = torch.tanh(self.raw[index])
        return (normalized ** 2).sum()

    def step(self, index: int, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        import torch

        grad = self.raw.grad
        if grad is None or index == self.fixed_index():
            self.raw.grad = None
            return
        row = grad[index].detach().clone()
        self.raw.grad = None
        with torch.no_grad():
            self.steps[index] += 1
            n = int(self.steps[index])
            self.m[index] = beta1 * self.m[index] + (1 - beta1) * row
            self.v[index] = beta2 * self.v[index] + (1 - beta2) * row * row
            m_hat = self.m[index] / (1 - beta1 ** n)
            v_hat = self.v[index] / (1 - beta2 ** n)
            self.raw[index] -= lr * m_hat / (v_hat.sqrt() + eps)

    @staticmethod
    def fixed_index() -> int:
        return int(POSE_REFINE["fixed_view"])


def pose_refine_settings(rows: list[dict], bounds, enabled: bool) -> dict | None:
    if not enabled:
        return None
    low, high = np.asarray(bounds, dtype=np.float64)
    center = (low + high) / 2
    centers = np.stack([_camera_center(row["w2c"]) for row in rows])
    distances = np.linalg.norm(centers - center, axis=1)
    radius = round(float(np.median(distances)), 8)
    if radius <= 0:
        raise ValueError("Pose refinement needs a positive orbit radius")
    return dict(POSE_REFINE, orbit_radius=radius)


def training_indices(target_w2c: np.ndarray) -> list[int]:
    """Use 79 unique orbit cameras. Drop the closure frame when the orbit repeats it."""
    poses = np.asarray(target_w2c, dtype=np.float64)
    count = len(poses)
    if count == 80 and np.allclose(poses[0], poses[-1]):
        return list(range(TRAINING_VIEWS))
    if count == TRAINING_VIEWS:
        return list(range(TRAINING_VIEWS))
    raise ValueError(
        "E1 trains 79 unique orbit views. Pass 79 cameras, or 80 cameras whose "
        f"first and last poses match. Got {count}."
    )


def frame_file(directory: Path, index: int) -> Path:
    candidates = [directory / f"{index:02d}.png", directory / f"frame_{index:03d}.png"]
    found = [path for path in candidates if path.is_file()]
    if not found:
        raise FileNotFoundError(f"Missing frame {index} in {directory} ({candidates[0].name})")
    return found[0]


def mask_file(directory: Path, index: int) -> Path:
    candidates = [
        directory / f"{index:02d}.png",
        directory / f"{index:02d}_mask.png",
        directory / f"frame_{index:03d}_mask.png",
    ]
    found = [path for path in candidates if path.is_file()]
    if not found:
        raise FileNotFoundError(f"Missing mask {index} in {directory} ({candidates[0].name})")
    return found[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"{path} has no vertex element")


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def initialization_values(init_ply: Path) -> dict:
    """Read the cloud sidecar when it exists, otherwise the frozen white-car defaults."""
    defaults = load_defaults()
    sidecar = Path(init_ply).with_name("initialization.json")
    supplied = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    return dict(
        real_surface_points=int(supplied.get("real_surface_points", defaults["real_surface_points"])),
        volume_seed_points=int(supplied.get("volume_seed_points", defaults["volume_seed_points"])),
        bounds_world=supplied.get("bounds_world", defaults["bounds_world"]),
        scene_extent=float(supplied.get("scene_extent", defaults["scene_extent"])),
        seed_opacity=float(supplied.get("seed_opacity", defaults["seed_opacity"])),
    )


def prepare(
    output: Path,
    frames: Path,
    trajectory_path: Path,
    masks: Path,
    init_ply: Path,
    pose_refine: bool = False,
) -> dict:
    """Write the dataset, or reuse it only when the supplied inputs still match."""
    output = output.resolve()
    experiment_path = output / "experiment.json"
    defaults = load_defaults()
    supplied = initialization_values(init_ply)
    surface_points = supplied["real_surface_points"]
    seed_points = supplied["volume_seed_points"]
    bounds = supplied["bounds_world"]
    scene_extent = supplied["scene_extent"]
    seed_opacity = supplied["seed_opacity"]
    trajectory = json.loads(trajectory_path.read_text())
    target_w2c = np.asarray(trajectory["target_w2c"], dtype=np.float64)
    target_k = np.asarray(trajectory["target_intrinsics"], dtype=np.float64)
    indices = training_indices(target_w2c)
    expected_size = tuple(defaults["image_size"])
    angles = trajectory.get("target_angles_deg")
    unwrapped = trajectory.get("unwrapped_angles_deg")
    rows = []
    for index in indices:
        image = frame_file(frames, index)
        mask = mask_file(masks, index)
        image_size = Image.open(image).size
        if image_size != Image.open(mask).size:
            raise ValueError(f"Frame and mask sizes differ for view {index}")
        if image_size != expected_size:
            raise ValueError(
                f"E1 images are {expected_size[0]}x{expected_size[1]}, "
                f"view {index} is {image_size[0]}x{image_size[1]}"
            )
        row = dict(
            orbit_index=index,
            name=f"{index:02d}.png",
            image=str(image.resolve()),
            mask=str(mask.resolve()),
            sha256=sha256(image),
            mask_sha256=sha256(mask),
            w2c=target_w2c[index].tolist(),
            k=target_k[index].tolist(),
            angle=float(angles[index]) if angles is not None else None,
            unwrapped_angle=float(unwrapped[index]) if unwrapped is not None else None,
        )
        rows.append(row)

    vertices = ply_vertex_count(init_ply)
    expected_vertices = surface_points + seed_points
    if vertices != expected_vertices:
        raise ValueError(
            f"Initialization ply has {vertices} vertices; expected "
            f"{surface_points} surface points plus {seed_points} volume seeds."
        )
    steps = len(rows) * UPDATES_PER_IMAGE
    if steps != defaults["schedule"]["position_max_steps"]:
        raise ValueError(f"E1 runs {defaults['schedule']['position_max_steps']} steps, got {steps}")
    refine = pose_refine_settings(rows, bounds, pose_refine)

    if experiment_path.exists():
        previous = json.loads(experiment_path.read_text())
        saved = json.loads((output / "inputs.json").read_text())["frames"]
        keys = ("orbit_index", "sha256", "mask_sha256", "w2c", "k")
        if len(saved) != len(rows) or any(
            old.get(key) != new[key] for old, new in zip(saved, rows) for key in keys
        ):
            raise ValueError("Prepared E1 inputs changed; choose a new output directory")
        initialization = json.loads((output / "shared/initialization.json").read_text())
        expected_init = dict(real_surface_points=surface_points, volume_seed_points=seed_points,
                             seed_opacity=seed_opacity, bounds_world=bounds)
        if (any(initialization.get(key) != value for key, value in expected_init.items())
                or previous["scene_extent"] != scene_extent
                or sha256(output / "shared/initial_points.ply") != sha256(init_ply)):
            raise ValueError("Prepared E1 initialization changed; choose a new output directory")
        if previous.get("pose_refine") != refine:
            raise ValueError("Prepared E1 pose refinement changed; choose a new output directory")
        print("ALREADY_PREPARED", output, flush=True)
        return previous

    shared = output / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(init_ply, shared / "initial_points.ply")
    _save(
        shared / "initialization.json",
        dict(
            real_surface_points=surface_points,
            volume_seed_points=seed_points,
            seed_opacity=seed_opacity,
            bounds_world=bounds,
            scene_extent=scene_extent,
            ply_sha256=sha256(init_ply),
        ),
    )
    write_colmap(output / "dataset", rows, shared / "initial_points.ply")
    rng = np.random.default_rng(defaults["training_seed"])
    frame_ids = np.array([row["orbit_index"] for row in rows])
    stems = np.concatenate(
        [rng.permutation(frame_ids) for _ in range(UPDATES_PER_IMAGE)]
    ).tolist()
    if Counter(stems) != {int(item): UPDATES_PER_IMAGE for item in frame_ids}:
        raise RuntimeError("Sampling plan does not visit every view the same number of times")
    _save(
        output / "inputs.json",
        dict(
            frames=rows,
            width=expected_size[0],
            height=expected_size[1],
            camera_convention="OpenCV W2C",
        ),
    )
    _save(
        output / "sampling_plan.json",
        dict(
            seed=defaults["training_seed"],
            steps=steps,
            exposures_per_view=UPDATES_PER_IMAGE,
            stems=stems,
        ),
    )
    experiment = dict(
        arm="E1",
        frames=[row["orbit_index"] for row in rows],
        images=len(rows),
        training_seed=defaults["training_seed"],
        steps=steps,
        updates_per_image=UPDATES_PER_IMAGE,
        outside_weight=defaults["outside_weight"],
        anisotropy_weight=defaults["anisotropy_weight"],
        gaussian_cap=defaults["gaussian_cap"],
        schedule=defaults["schedule"],
        scene_extent=scene_extent,
        loss="0.8 L1 + 0.2 SSIM + 0.5 outside opacity + 0.03 anisotropy"
        + (" + 0.001 pose prior" if refine else ""),
        pose_refine=refine,
    )
    _save(experiment_path, experiment)
    _save(output / "status.json", dict(state="prepared", steps=steps))
    print("PREPARED", output, "views", len(rows), "steps", steps, flush=True)
    return experiment


def use_threedgrut(root: Path) -> None:
    root = root.resolve()
    if not (root / "threedgrut").is_dir():
        raise FileNotFoundError(
            f"Expected a 3DGRUT checkout containing threedgrut/ at {root}"
        )
    sys.path.insert(0, str(root))
    from vggtdiff.recon.gsplat_backend import install

    install()
    import threedgrut.utils.misc as threedgrut_misc

    if not hasattr(threedgrut_misc, "to_np"):
        raise RuntimeError("3DGRUT did not register its config helpers")


def write_training_config(output: Path) -> None:
    from omegaconf import OmegaConf

    experiment = json.loads((output / "experiment.json").read_text())
    schedule = experiment["schedule"]
    if not OmegaConf.has_resolver("int_list"):
        OmegaConf.register_new_resolver(
            "int_list", lambda values: [int(value) for value in values]
        )
    conf = OmegaConf.load(TEMPLATE)
    conf.path = str(output / "dataset")
    conf.out_dir = str(output / "training")
    conf.experiment_name = "E1"
    conf.n_iterations = experiment["steps"]
    conf.export_ply.path = str(output / "model.ply")
    conf.initialization.fused_point_cloud_path = str(output / "shared" / "initial_points.ply")
    conf.checkpoint.iterations = list(schedule["checkpoints"])
    conf.model.progressive_training.increase_frequency = schedule["increase_frequency"]
    conf.scheduler.positions.max_steps = experiment["steps"]
    conf.strategy.densify.frequency = schedule["densify_frequency"]
    conf.strategy.densify.start_iteration = schedule["densify_start"]
    conf.strategy.densify.end_iteration = schedule["densify_end"]
    conf.strategy.prune.frequency = schedule["prune_frequency"]
    conf.strategy.prune.start_iteration = schedule["prune_start"]
    conf.strategy.prune.end_iteration = schedule["prune_end"]
    conf.writer.log_image_views = [0, 13, 27, 43, experiment["images"] - 1]
    OmegaConf.resolve(conf)
    OmegaConf.save(conf, output / "training_config.yaml")


def _trainer_class():
    import torch
    from threedgrut.datasets.utils import MultiEpochsDataLoader
    from threedgrut.model.losses import ssim
    from threedgrut.trainer import Trainer3DGRUT

    from vggtdiff.recon.loss import object_losses

    class ReconstructionTrainer(Trainer3DGRUT):
        def __init__(self, conf, output: Path):
            self.output = output
            self.experiment = json.loads((output / "experiment.json").read_text())
            init_path = output / "shared" / "initialization.json"
            self.initialization = json.loads(init_path.read_text())
            self.plan = json.loads((output / "sampling_plan.json").read_text())
            self.sample_counts = Counter()
            self.pose_correction = None
            self.current_view = 0
            super().__init__(conf)

        def init_dataloaders(self, conf):
            Trainer3DGRUT.init_dataloaders(self, conf)
            names = [Path(path).name for path in self.train_dataset.image_paths]
            lookup = {int(Path(name).stem): i for i, name in enumerate(names)}
            indices = self.experiment["frames"]
            if sorted(lookup) != indices:
                raise RuntimeError(f"Dataset stems {sorted(lookup)} do not match {indices}")
            schedule = [lookup[i] for i in self.plan["stems"]]
            if len(schedule) != self.experiment["steps"]:
                raise RuntimeError("Sampling plan length does not match the E1 step count")

            class FixedSampler(torch.utils.data.Sampler):
                def __iter__(self):
                    return iter(schedule)

                def __len__(self):
                    return len(schedule)

            self.train_dataloader = MultiEpochsDataLoader(
                self.train_dataset,
                batch_size=1,
                num_workers=0,
                sampler=FixedSampler(),
                pin_memory=False,
            )
            if self.experiment.get("pose_refine"):
                original = self.train_dataset.get_gpu_batch_with_intrinsics

                def corrected(batch):
                    gpu_batch = original(batch)
                    name = gpu_batch.image_name
                    if isinstance(name, (list, tuple)):
                        name = name[0]
                    self.current_view = int(Path(str(name)).stem)
                    delta = self.pose_correction.delta(self.current_view)
                    gpu_batch.T_to_world = gpu_batch.T_to_world @ delta
                    return gpu_batch

                self.train_dataset.get_gpu_batch_with_intrinsics = corrected

        def init_scene_extents(self, dataset):
            del dataset
            self.scene_extent = float(self.experiment["scene_extent"])
            bounds = self.initialization["bounds_world"]
            self.scene_bbox = tuple(
                torch.tensor(value, device=self.device, dtype=torch.float32) for value in bounds
            )

        def init_densification_and_pruning_strategy(self, conf):
            super().init_densification_and_pruning_strategy(conf)
            original = self.strategy.densify_gaussians
            cap = int(self.experiment["gaussian_cap"])

            def limited(scene_extent):
                if self.model.num_gaussians < cap:
                    original(scene_extent)
                    if self.model.num_gaussians > cap:
                        with torch.no_grad():
                            keep = torch.zeros(
                                self.model.num_gaussians, device=self.device, dtype=torch.bool
                            )
                            density = self.model.get_density().flatten()
                            keep[density.topk(cap).indices] = True
                            strategy = self.strategy
                            strategy._update_param_with_optimizer(
                                lambda _name, param: torch.nn.Parameter(
                                    param[keep], requires_grad=param.requires_grad
                                ),
                                lambda _key, value: value[keep],
                            )
                            strategy._filter_local_weights(keep)
                            strategy.prune_densification_buffers(keep)

            self.strategy.densify_gaussians = limited

        def setup_training(self, conf, model, train_dataset):
            Trainer3DGRUT.setup_training(self, conf, model, train_dataset)
            count = int(self.initialization["real_surface_points"])
            opacity = float(self.initialization["seed_opacity"])
            with torch.no_grad():
                model.density[count:] = float(np.log(opacity / (1.0 - opacity)))
            settings = self.experiment.get("pose_refine")
            if not settings:
                return
            self.pose_correction = PoseCorrection(
                self.experiment["images"],
                math.radians(settings["rotation_max_deg"]),
                settings["translation_max_ratio"] * settings["orbit_radius"],
                self.device,
            )
            schedule = model.scheduler_step

            def stepped(step):
                schedule(step)
                if step >= settings["start_step"]:
                    progress = (step - settings["start_step"]) / max(
                        self.experiment["steps"] - settings["start_step"], 1
                    )
                    lr = settings["lr_init"] * (settings["lr_final"] / settings["lr_init"]) ** progress
                    self.pose_correction.step(self.current_view, lr)
                else:
                    self.pose_correction.raw.grad = None

            model.scheduler_step = stepped

        def get_losses(self, batch, outputs):
            name = batch.image_name
            if isinstance(name, (list, tuple)):
                name = name[0]
            self.sample_counts[Path(str(name)).name] += 1
            l1, structural, alpha = object_losses(
                outputs["pred_rgb"],
                batch.rgb_gt,
                outputs["pred_opacity"],
                batch.mask,
                ssim_fn=ssim,
                dilate=3,
            )
            scales = self.model.get_scale()
            opacity = self.model.get_density().reshape(-1)
            ratio = scales.amax(-1) / scales.amin(-1).clamp_min(1e-8)
            anisotropy = (torch.relu(torch.log(ratio) - math.log(16)) * opacity).sum()
            anisotropy = anisotropy / opacity.sum().clamp_min(1e-6)
            total = (
                0.8 * l1
                + 0.2 * structural
                + float(self.experiment["outside_weight"]) * alpha
                + float(self.experiment["anisotropy_weight"]) * anisotropy
            )
            settings = self.experiment.get("pose_refine")
            if settings:
                total = total + float(settings["prior_weight"]) * self.pose_correction.penalty(self.current_view)
            zero = total.new_zeros(())
            return dict(
                total_loss=total,
                l1_loss=l1,
                l2_loss=zero,
                ssim_loss=structural,
                opacity_loss=alpha,
                scale_loss=anisotropy,
                lpips_loss=zero,
            )

        def save_checkpoint(self, last_checkpoint: bool = False):
            Trainer3DGRUT.save_checkpoint(self, last_checkpoint)
            if self.pose_correction is not None:
                self._write_corrections()

        def _write_corrections(self) -> None:
            import torch

            rows = json.loads((self.output / "inputs.json").read_text())["frames"]
            radius = float(self.experiment["pose_refine"]["orbit_radius"])
            views = []
            with torch.no_grad():
                for row in rows:
                    index = int(row["orbit_index"])
                    omega, translation = self.pose_correction.motion(index)
                    delta = self.pose_correction.delta(index).detach().float().cpu().numpy()
                    corrected = np.linalg.inv(delta) @ np.asarray(row["w2c"], dtype=np.float64)
                    omega_np = omega.detach().float().cpu().numpy()
                    translation_np = translation.detach().float().cpu().numpy()
                    raw = self.pose_correction.raw[index].detach().float().cpu().numpy()
                    views.append(dict(
                        orbit_index=index,
                        raw=raw.tolist(),
                        rotation_deg=np.degrees(omega_np).tolist(),
                        rotation_angle_deg=float(np.degrees(np.linalg.norm(omega_np))),
                        translation=translation_np.tolist(),
                        translation_ratio=float(np.linalg.norm(translation_np) / radius),
                        at_bound=bool(np.any(np.abs(np.tanh(np.linalg.norm(raw[:3]))) > 0.95)
                                      or np.any(np.abs(np.tanh(np.linalg.norm(raw[3:]))) > 0.95)),
                        w2c=corrected.tolist(),
                    ))
            _save(self.output / "training" / "camera_corrections.json", dict(views=views))

    return ReconstructionTrainer


def _load_conf(output: Path):
    from omegaconf import OmegaConf

    return OmegaConf.load(output / "training_config.yaml")


def train(output: Path, threedgrut: Path) -> dict:
    import torch

    result_path = output / "training" / "result.json"
    if result_path.exists():
        print("ALREADY_TRAINED", output, flush=True)
        return json.loads(result_path.read_text())
    if (output / "training" / "initial_state.json").exists():
        raise FileExistsError(f"{output} training already started; not overwriting a partial run")
    use_threedgrut(threedgrut)
    if not (output / "training_config.yaml").exists():
        write_training_config(output)
    experiment = json.loads((output / "experiment.json").read_text())
    torch.set_num_threads(4)
    torch.manual_seed(experiment["training_seed"])
    np.random.seed(experiment["training_seed"])
    trainer = _trainer_class()(_load_conf(output), output)
    trainer.n_epochs = 1
    initial = json.loads((output / "shared" / "initialization.json").read_text())
    expected = int(initial["real_surface_points"]) + int(initial["volume_seed_points"])
    if trainer.model.num_gaussians != expected:
        raise RuntimeError(
            f"Expected {expected} initial Gaussians, found {trainer.model.num_gaussians}"
        )
    _save(
        output / "training" / "initial_state.json",
        dict(gaussians=trainer.model.num_gaussians),
    )
    _save(output / "status.json", dict(state="training", steps=experiment["steps"]))
    start = time.time()
    trainer.run_training()
    trainer.save_checkpoint()
    if trainer.global_step != experiment["steps"]:
        raise RuntimeError(f"Stopped at step {trainer.global_step}, expected {experiment['steps']}")
    if len(trainer.sample_counts) != experiment["images"]:
        raise RuntimeError("Not every training view was sampled")
    if set(trainer.sample_counts.values()) != {UPDATES_PER_IMAGE}:
        raise RuntimeError("Views were not sampled for the frozen update count")
    checkpoint = (
        Path(trainer.tracking.output_dir)
        / f"ours_{experiment['steps']}"
        / f"ckpt_{experiment['steps']}.pt"
    )
    if not checkpoint.exists() or not (output / "model.ply").exists():
        raise RuntimeError("Training finished without the checkpoint or model.ply")
    result = dict(
        step=experiment["steps"],
        checkpoint=str(checkpoint),
        gaussians=int(trainer.model.num_gaussians),
        elapsed_seconds=time.time() - start,
        ply_sha256=sha256(output / "model.ply"),
    )
    _save(result_path, result)
    _save(output / "status.json", dict(state="trained", steps=experiment["steps"]))
    print("TRAINED", checkpoint, flush=True)
    return result


def render_view(model, row: dict):
    import torch
    from threedgrut.datasets.protocols import Batch

    image = Image.open(row["image"])
    width, height = image.size
    intrinsics = np.asarray(row["k"], dtype=np.float64)
    blank = torch.zeros((1, height, width, 3), device="cuda")
    camera_to_world = np.linalg.inv(np.asarray(row["w2c"]))
    camera = torch.tensor(camera_to_world, device="cuda", dtype=torch.float32)[None]
    batch = Batch(
        blank,
        blank,
        camera,
        intrinsics_OpenCVPinholeCameraModelParameters=dict(
            focal_length=[intrinsics[0, 0], intrinsics[1, 1]],
            principal_point=[intrinsics[0, 2], intrinsics[1, 2]],
        ),
    )
    with torch.inference_mode():
        result = model(batch, train=False)
    rgb = result["pred_rgb"][0].clamp(0, 1).cpu().numpy()
    alpha = result["pred_opacity"][0, :, :, 0].cpu().numpy()
    return rgb, alpha


def score_views(model, rows: list[dict]) -> dict:
    records = []
    for row in rows:
        rgb, alpha = render_view(model, row)
        gt = np.asarray(Image.open(row["image"]).convert("RGB"), dtype=np.float64) / 255.0
        mask = np.asarray(Image.open(row["mask"]).convert("L")) > 127
        error = (rgb - gt)[mask]
        if not mask.any() or not np.isfinite(rgb).all() or not np.isfinite(alpha).all():
            raise ValueError(f"Invalid render or empty foreground mask for view {row['orbit_index']}")
        mse = float(np.mean(error**2))
        silhouette = alpha > 0.5
        records.append(
            dict(
                orbit_index=row["orbit_index"],
                foreground_psnr=-10 * math.log10(max(mse, 1e-12)),
                silhouette_iou=float((silhouette & mask).sum() / max((silhouette | mask).sum(), 1)),
                outside_opacity_over_mask_area=float(alpha[~mask].sum() / mask.sum()),
            )
        )
    return dict(
        views=records,
        median=dict(
            foreground_psnr=float(np.median([row["foreground_psnr"] for row in records])),
            silhouette_iou=float(np.median([row["silhouette_iou"] for row in records])),
            outside_opacity_over_mask_area=float(np.median([
                row["outside_opacity_over_mask_area"] for row in records
            ])),
        ),
    )


def evaluate(output: Path, threedgrut: Path) -> dict:
    import torch

    metrics_path = output / "evaluation" / "metrics.json"
    if metrics_path.exists():
        print("ALREADY_EVALUATED", output, flush=True)
        return json.loads(metrics_path.read_text())
    use_threedgrut(threedgrut)
    from threedgrut.model.model import MixtureOfGaussians

    result = json.loads((output / "training" / "result.json").read_text())
    if sha256(output / "model.ply") != result["ply_sha256"]:
        raise RuntimeError("model.ply does not match the training result")
    model = MixtureOfGaussians(_load_conf(output))
    model.init_from_checkpoint(
        torch.load(result["checkpoint"], map_location="cuda", weights_only=False),
        setup_optimizer=False,
    )
    model.eval()
    rows = json.loads((output / "inputs.json").read_text())["frames"]
    # Score the same frozen copies the trainer used, even if the source directory moved.
    for row in rows:
        image = output / "dataset/images" / row["name"]
        row.update(image=str(image), mask=str(image.with_name(f"{image.stem}_mask.png")))
    metrics = score_views(model, rows)
    corrections = output / "training" / "camera_corrections.json"
    if corrections.exists():
        saved = {row["orbit_index"]: row["w2c"] for row in json.loads(corrections.read_text())["views"]}
        refined = [dict(row, w2c=saved[row["orbit_index"]]) for row in rows]
        metrics["refined"] = score_views(model, refined)
    _save(metrics_path, metrics)
    print(
        "EVALUATED",
        "psnr",
        round(metrics["median"]["foreground_psnr"], 2),
        "iou",
        round(metrics["median"]["silhouette_iou"], 3),
        flush=True,
    )
    if "refined" in metrics:
        print(
            "EVALUATED_REFINED",
            "psnr",
            round(metrics["refined"]["median"]["foreground_psnr"], 2),
            "iou",
            round(metrics["refined"]["median"]["silhouette_iou"], 3),
            flush=True,
        )
    return metrics


def reconstruct(
    output: Path,
    frames: Path,
    trajectory: Path,
    masks: Path,
    init_ply: Path,
    threedgrut: Path,
    pose_refine: bool = False,
) -> None:
    prepare(output, frames, trajectory, masks, init_ply, pose_refine=pose_refine)
    train(output, threedgrut)
    evaluate(output, threedgrut)
