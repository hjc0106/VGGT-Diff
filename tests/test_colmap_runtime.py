"""GPU selection, command wiring, and preservation of existing SfM outputs."""

import json
from pathlib import Path

import pytest

from gen_pro import colmap_io
from gen_pro.colmap_gpu import resolve_device
from gen_pro.config import Config


def test_device_selection_requires_cuda_when_requested():
    assert resolve_device("auto", True) == "cuda"
    assert resolve_device("auto", False) == "cpu"
    assert resolve_device("cpu", True) == "cpu"
    with pytest.raises(RuntimeError, match="no CUDA support"):
        resolve_device("cuda", False)


def test_unsupported_cuda_does_not_delete_existing_reconstruction(tmp_path, monkeypatch):
    database = tmp_path / "database.db"
    database.write_bytes(b"existing database")
    model = tmp_path / "sparse/0/images.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"existing cameras")
    monkeypatch.setattr(colmap_io, "_probe_colmap", lambda *_: dict(has_cuda=False))
    with pytest.raises(RuntimeError, match="no CUDA support"):
        colmap_io.run_colmap(tmp_path / "frames", tmp_path, Config(sfm_device="cuda"))
    assert database.read_bytes() == b"existing database"
    assert model.read_bytes() == b"existing cameras"


@pytest.mark.parametrize("modern", [False, True])
def test_cli_gpu_settings_apply_to_initial_and_added_frames(tmp_path, monkeypatch, modern):
    monkeypatch.setattr(colmap_io, "_probe_colmap", lambda *_: dict(has_cuda=True, modern_options=modern))
    monkeypatch.setattr(colmap_io, "read_text_model", lambda _: {})
    commands = []

    def run(command, _log):
        commands.append(command)
        if "--output_path" in command:
            output = Path(command[command.index("--output_path") + 1])
            if command[1] == "mapper":
                output = output / "0"
            output.mkdir(parents=True, exist_ok=True)
            (output / "images.bin").write_bytes(b"model")

    monkeypatch.setattr(colmap_io, "_run", run)
    cfg = Config(sfm_device="cuda", sfm_gpu_index="2", sfm_num_threads=8)
    colmap_io.run_colmap(tmp_path / "frames", tmp_path, cfg)
    colmap_io.register_new_images(tmp_path / "frames", tmp_path, ["extra.jpg"], cfg)
    feature_commands = [c for c in commands if c[1] in {"feature_extractor", "sequential_matcher"}]
    assert len(feature_commands) == 4
    for command in feature_commands:
        group = (("FeatureExtraction" if modern else "SiftExtraction") if command[1] == "feature_extractor"
                 else ("FeatureMatching" if modern else "SiftMatching"))
        for option, value in [("use_gpu", "1"), ("gpu_index", "2"), ("num_threads", "8")]:
            assert command[command.index(f"--{group}.{option}") + 1] == value


def test_pycolmap_registration_receives_frozen_model_and_image_names(tmp_path, monkeypatch):
    model = tmp_path / "sparse/0"
    model.mkdir(parents=True)
    (model / "images.bin").touch()
    monkeypatch.setattr(colmap_io, "_probe_colmap", lambda *_: dict(has_cuda=True))
    monkeypatch.setattr(colmap_io, "read_text_model", lambda path: dict(model=str(path)))

    def run(command, _log):
        assert command[0] == "/isolated/python"
        config = json.loads(Path(command[-1]).read_text())
        assert config["device"] == "cuda"
        assert config["input_model"] == str(model)
        assert config["image_names"] == ["extra.jpg"]
        (tmp_path / "colmap_result.json").write_text(json.dumps(dict(model_path=str(model))))

    monkeypatch.setattr(colmap_io, "_run", run)
    cfg = Config(colmap_python="/isolated/python", sfm_device="cuda")
    colmap_io.register_new_images(tmp_path / "frames", tmp_path, ["extra.jpg"], cfg)


def test_runtime_options_round_trip_in_saved_configuration(tmp_path):
    cfg = Config(colmap_python="/isolated/python", sfm_device="cuda", sfm_gpu_index="1")
    path = tmp_path / "config.json"
    cfg.save(path)
    assert Config.load(path) == cfg


def test_unavailable_gpu_ba_fails_before_starting_sfm(monkeypatch):
    monkeypatch.setattr(
        colmap_io, "_probe_colmap", lambda *_: dict(has_cuda=True, has_caspar=False),
    )
    cfg = Config(colmap_python="/isolated/python", sfm_device="cuda", sfm_ba_backend="caspar")
    with pytest.raises(RuntimeError, match="no Caspar GPU BA"):
        colmap_io._device(cfg)


def test_caspar_is_rejected_for_the_opencv_pipeline_even_if_compiled(monkeypatch):
    monkeypatch.setattr(
        colmap_io, "_probe_colmap", lambda *_: dict(has_cuda=True, has_caspar=True),
    )
    cfg = Config(colmap_python="/isolated/python", sfm_device="cuda", sfm_ba_backend="caspar")
    with pytest.raises(ValueError, match="OPENCV camera model"):
        colmap_io._device(cfg)


def test_cuda_sift_does_not_imply_cuda_bundle_adjustment(tmp_path, monkeypatch):
    database = tmp_path / "database.db"
    database.write_bytes(b"keep existing features")
    monkeypatch.setattr(
        colmap_io, "_probe_colmap", lambda *_: dict(has_cuda=True, has_ceres_cuda=False),
    )
    cfg = Config(colmap_python="/isolated/python", sfm_ba_device="cuda")
    with pytest.raises(RuntimeError, match="no Ceres CUDA support"):
        colmap_io.run_colmap(tmp_path / "frames", tmp_path, cfg)
    assert database.read_bytes() == b"keep existing features"


def test_repeated_registration_uses_latest_successful_model(tmp_path, monkeypatch):
    latest = tmp_path / "sparse_extra/previous/0"
    (tmp_path / "colmap_result.json").write_text(json.dumps(dict(model_path=str(latest))))
    received = []
    monkeypatch.setattr(
        colmap_io, "_run_pycolmap",
        lambda images, work, cfg, model, names: received.append((model, names)),
    )
    colmap_io.register_new_images(tmp_path / "frames", tmp_path, ["next.jpg"], Config(colmap_python="/python"))
    assert received == [(latest, ["next.jpg"])]
