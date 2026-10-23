"""Portable defaults for migrated experiments; explicit CLI paths still win."""
import os
from pathlib import Path


def model_root():
    return Path(os.environ.get('VGGTDIFF_MODEL_ROOT',Path(__file__).resolve().parents[1]/'models')).expanduser()


def checkpoint_path():
    return Path(os.environ.get('VGGTDIFF_CHECKPOINT',model_root()/'vggt-diff/vggt-diff.safetensors')).expanduser()


def omega_path():
    return Path(os.environ.get('VGGTDIFF_OMEGA',model_root()/'vggt-omega/vggt_omega_1b_512.pt')).expanduser()


def base_model_path():
    return Path(os.environ.get('VGGTDIFF_BASE_MODEL',model_root()/'Wan2.1-I2V-14B-480P')).expanduser()
