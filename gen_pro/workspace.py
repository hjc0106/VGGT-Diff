"""Work directory, per-step status, and downstream invalidation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

STEPS = ("video", "select", "sources", "cameras", "orbit", "frame_masks")
STEP_DIRS = {
    "video": "1_video",
    "select": "2_select",
    "sources": "3_sources",
    "cameras": "4_cameras",
    "orbit": "5_orbit",
    "frame_masks": "6_frame_masks",
}


def _digest(payload) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def step_dir(self, name: str) -> Path:
        path = self.root / STEP_DIRS[name]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def read_json(self, path: Path) -> dict:
        return json.loads(Path(path).read_text())

    def write_json(self, path: Path, payload: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")

    def overrides(self, name: str) -> dict:
        path = self.step_dir(name) / "overrides.json"
        if not path.exists():
            return {}
        return self.read_json(path)

    def write_overrides(self, name: str, payload: dict) -> None:
        self.write_json(self.step_dir(name) / "overrides.json", payload)

    def record(self, name: str) -> dict | None:
        path = self.step_dir(name) / "step.json"
        if not path.exists():
            return None
        return self.read_json(path)

    def upstream_hash(self, name: str) -> str | None:
        index = STEPS.index(name)
        if index == 0 or name == "frame_masks":
            return None
        previous = self.record(STEPS[index - 1])
        if previous is None:
            raise FileNotFoundError(f"Step {STEPS[index - 1]} has not been run")
        return _digest(previous)

    def fresh(self, name: str, params: dict) -> bool:
        record = self.record(name)
        if record is None or record.get("status") != "done":
            return False
        if record.get("params") != params:
            return False
        if record.get("upstream") != self.upstream_hash(name):
            return False
        directory = self.step_dir(name)
        return all((directory / item).exists() for item in record.get("outputs", []))

    def complete(self, name: str, params: dict, outputs: list[str], metrics: dict) -> dict:
        record = dict(
            step=name,
            status="done",
            params=params,
            upstream=self.upstream_hash(name),
            outputs=outputs,
            metrics=metrics,
        )
        self.write_json(self.step_dir(name) / "step.json", record)
        self.invalidate_after(name)
        return record

    def fail(self, name: str, params: dict, outputs: list[str], metrics: dict) -> dict:
        record = dict(
            step=name,
            status="failed",
            params=params,
            upstream=self.upstream_hash(name),
            outputs=outputs,
            metrics=metrics,
        )
        self.write_json(self.step_dir(name) / "step.json", record)
        self.invalidate_after(name)
        return record

    def invalidate_after(self, name: str) -> None:
        if name not in STEPS:
            raise KeyError(name)
        start = STEPS.index(name) + 1
        for later in STEPS[start:]:
            path = self.step_dir(later) / "step.json"
            if not path.exists():
                continue
            record = self.read_json(path)
            if record.get("status") == "done":
                record["status"] = "stale"
                self.write_json(path, record)

    def status(self) -> list[dict]:
        rows = []
        for name in STEPS:
            record = self.record(name)
            rows.append(
                dict(
                    step=name,
                    status="missing" if record is None else record.get("status"),
                    metrics=None if record is None else record.get("metrics"),
                )
            )
        return rows
