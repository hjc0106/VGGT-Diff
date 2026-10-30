"""SfM schedules and deterministic, time-spread keyframe selection."""

from __future__ import annotations


def profile_settings(profile: str, final_ba_iterations: int | None = None) -> dict:
    if profile not in {"reference", "balanced", "keyframe"}:
        raise ValueError(f"Unknown SfM profile: {profile}")
    balanced = profile != "reference"
    final = (100 if balanced else 0) if final_ba_iterations is None else final_ba_iterations
    if final < 0:
        raise ValueError("Final BA iterations cannot be negative")
    return dict(
        profile=profile,
        ba_global_frames_ratio=1.3 if balanced else 1.1,
        ba_global_points_ratio=1.3 if balanced else 1.1,
        ba_global_max_refinements=2 if balanced else 5,
        ba_local_max_num_iterations=25,
        ba_global_max_num_iterations=50,
        final_ba_iterations=final,
    )


def select_keyframes(rows: list[dict], count: int, required_names=()) -> list[str]:
    """Keep mandatory images/endpoints, then fill the largest temporal gaps."""
    if count < 2:
        raise ValueError("At least two keyframes are required")
    ordered = sorted(rows, key=lambda row: (row.get("video_frame", 0), row["name"]))
    names = [row["name"] for row in ordered]
    if len(set(names)) != len(names):
        raise ValueError("Candidate image names must be unique")
    required = set(required_names)
    missing = required.difference(names)
    if missing:
        raise ValueError(f"Required images are missing: {sorted(missing)}")
    if len(names) <= count:
        return names
    selected = {i for i, name in enumerate(names) if name in required}
    selected.update((0, len(names) - 1))
    if len(selected) > count:
        raise ValueError("Keyframe budget is smaller than mandatory images and endpoints")
    positions = [float(row.get("video_frame", i)) for i, row in enumerate(ordered)]
    while len(selected) < count:
        best = max(
            (i for i in range(len(names)) if i not in selected),
            key=lambda i: (
                min(abs(positions[i] - positions[j]) for j in selected),
                float(ordered[i].get("sharpness", 0)), -i,
            ),
        )
        selected.add(best)
    return [names[i] for i in sorted(selected)]


def keyframe_budgets(total: int, initial: int) -> list[int]:
    if initial < 2:
        raise ValueError("At least two keyframes are required")
    return sorted({min(total, initial), min(total, max(90, initial)), total})
