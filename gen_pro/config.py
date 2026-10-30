"""Defaults for the six preprocessing steps."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Config:
    video: str | None = None
    prompt: str = "white car"
    omega_checkpoint: str | None = None
    colmap_bin: str = "colmap"
    colmap_python: str | None = None
    sfm_device: str = "auto"
    sfm_gpu_index: str = "0"
    sfm_num_threads: int = 16
    sfm_ba_backend: str = "ceres"
    sfm_profile: str = "reference"
    sfm_keyframes: int = 60
    sfm_keyframe_min_register_ratio: float = 0.95
    sfm_final_ba_iterations: int | None = None
    sfm_ba_device: str = "cpu"
    sam3_checkpoint: str | None = None
    sam3_root: str | None = None
    device: str = "cuda"
    width: int = 832
    height: int = 480
    candidate_count: int = 120
    sharpness_window: int = 2
    jpeg_quality: int = 95
    sfm_max_image_size: int = 1600
    sfm_overlap: int = 15
    sfm_min_register_ratio: float = 0.5
    reference_count: int = 6
    sharpness_quantile: float = 0.4
    manual_frames: tuple[int, ...] = ()
    mask_min_area: float = 0.005
    mask_max_area: float = 0.90
    sam3_confidence: float = 0.15
    crop_margin: float = 0.15
    source_focal_px: float = 965.0
    confidence_quantile: float = 0.4
    voxel_divisions: int = 180
    outlier_drop: float = 0.02
    max_surface_points: int = 25000
    seed_points: int = 4000
    seed_opacity: float = 0.01
    cloud_seed: int = 20261021
    orbit_frames: int = 80
    framing_margin_px: float = 40.0
    radius_growth: float = 1.05
    framing_attempts: int = 30
    align_warn_translation: float = 0.05
    align_warn_rotation_deg: float = 5.0
    projection_min_hit: float = 0.70
    aim_refine_ratio: float = 0.02
    generated_frames: str | None = None

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["manual_frames"] = list(self.manual_frames)
        return payload

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Config":
        raw = json.loads(path.read_text())
        known = {item.name for item in fields(cls)}
        values = {key: value for key, value in raw.items() if key in known}
        if "manual_frames" in values:
            values["manual_frames"] = tuple(int(item) for item in values["manual_frames"])
        return cls(**values)
