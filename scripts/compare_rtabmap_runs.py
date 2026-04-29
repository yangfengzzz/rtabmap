#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sqlite3
import statistics
import struct
import sys
import zlib
from pathlib import Path


STAT_KEYS = [
    "Keypoint/Current_frame/words",
    "Loop/Visual_matches",
    "Loop/Visual_inliers",
    "Loop/Visual_inliers_ratio",
    "Timing/Total/ms",
    "Timing/Memory_update/ms",
    "Timing/Hypotheses_creation/ms",
    "Timing/Hypotheses_validation/ms",
    "TimingMem/Keypoints_detection/ms",
    "TimingMem/Descriptors_extraction/ms",
]


def decompress_stat_blob(blob: bytes | None) -> str:
    if not blob:
        return ""
    if len(blob) < 12:
        return ""
    rows, cols, cv_type = struct.unpack("<iii", blob[-12:])
    if cv_type != 1 or rows != 1 or cols <= 0:
        return ""
    raw = zlib.decompress(blob[:-12])
    return raw.decode("utf-8", errors="replace").rstrip("\x00")


def parse_stat_text(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in text.split(";"):
      if not item or ":" not in item:
        continue
      key, value = item.split(":", 1)
      try:
        result[key] = float(value)
      except ValueError:
        continue
    return result


def safe_mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def safe_max(values: list[float]) -> float | None:
    return max(values) if values else None


def load_summary(path: Path) -> dict[str, object]:
    con = sqlite3.connect(path)
    try:
        node_count, stamp_min, stamp_max = con.execute(
            "SELECT COUNT(*), MIN(stamp), MAX(stamp) FROM Node;"
        ).fetchone()
        stats_count = con.execute("SELECT COUNT(*) FROM Statistics;").fetchone()[0]
        file_size_mb = path.stat().st_size / (1024.0 * 1024.0)

        link_counts = dict(
            con.execute("SELECT type, COUNT(*) FROM Link GROUP BY type;").fetchall()
        )
        admin = con.execute(
            "SELECT opt_map_x_min, opt_map_y_min, opt_map_resolution FROM Admin LIMIT 1;"
        ).fetchone()

        stat_values: dict[str, list[float]] = {key: [] for key in STAT_KEYS}
        for (blob,) in con.execute("SELECT data FROM Statistics WHERE data IS NOT NULL;"):
            parsed = parse_stat_text(decompress_stat_blob(blob))
            for key in STAT_KEYS:
                value = parsed.get(key)
                if value is not None and math.isfinite(value):
                    stat_values[key].append(value)
    finally:
        con.close()

    duration_s = (stamp_max - stamp_min) if stamp_min is not None and stamp_max is not None else None
    return {
        "label": path.stem,
        "path": str(path),
        "file_size_mb": file_size_mb,
        "nodes": int(node_count or 0),
        "stats_rows": int(stats_count or 0),
        "duration_s": duration_s,
        "neighbor_links": int(link_counts.get(0, 0)),
        "loop_links": int(link_counts.get(1, 0)),
        "child_links": int(link_counts.get(2, 0)),
        "landmark_links": int(link_counts.get(3, 0)),
        "map_x_min": admin[0] if admin else None,
        "map_y_min": admin[1] if admin else None,
        "map_resolution": admin[2] if admin else None,
        "avg_words": safe_mean(stat_values["Keypoint/Current_frame/words"]),
        "avg_visual_matches": safe_mean(stat_values["Loop/Visual_matches"]),
        "avg_visual_inliers": safe_mean(stat_values["Loop/Visual_inliers"]),
        "avg_inlier_ratio": safe_mean(stat_values["Loop/Visual_inliers_ratio"]),
        "avg_total_ms": safe_mean(stat_values["Timing/Total/ms"]),
        "max_total_ms": safe_max(stat_values["Timing/Total/ms"]),
        "avg_memory_update_ms": safe_mean(stat_values["Timing/Memory_update/ms"]),
        "avg_hyp_creation_ms": safe_mean(stat_values["Timing/Hypotheses_creation/ms"]),
        "avg_hyp_validation_ms": safe_mean(stat_values["Timing/Hypotheses_validation/ms"]),
        "avg_keypoint_detection_ms": safe_mean(stat_values["TimingMem/Keypoints_detection/ms"]),
        "avg_descriptor_ms": safe_mean(stat_values["TimingMem/Descriptors_extraction/ms"]),
    }


def fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_metric(name: str, left: dict[str, object], right: dict[str, object], key: str, digits: int = 2) -> None:
    print(f"{name:28} {fmt(left.get(key), digits):>14}  {fmt(right.get(key), digits):>14}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two RTAB-Map databases numerically."
    )
    parser.add_argument("baseline_db", help="Baseline RTAB-Map database path")
    parser.add_argument("comparison_db", help="Comparison RTAB-Map database path")
    args = parser.parse_args()

    left_path = Path(args.baseline_db).expanduser().resolve()
    right_path = Path(args.comparison_db).expanduser().resolve()

    for path in (left_path, right_path):
        if not path.is_file():
            print(f"Database not found: {path}", file=sys.stderr)
            return 1

    left = load_summary(left_path)
    right = load_summary(right_path)

    print(f"Baseline:   {left['path']}")
    print(f"Comparison: {right['path']}")
    print()
    print(f"{'Metric':28} {'Baseline':>14}  {'Comparison':>14}")
    print(f"{'-'*28} {'-'*14}  {'-'*14}")
    print_metric("DB file size (MB)", left, right, "file_size_mb")
    print_metric("Nodes", left, right, "nodes", 0)
    print_metric("Statistics rows", left, right, "stats_rows", 0)
    print_metric("Duration (s)", left, right, "duration_s")
    print_metric("Neighbor links", left, right, "neighbor_links", 0)
    print_metric("Loop links", left, right, "loop_links", 0)
    print_metric("Child links", left, right, "child_links", 0)
    print_metric("Landmark links", left, right, "landmark_links", 0)
    print_metric("Map resolution", left, right, "map_resolution", 4)
    print_metric("Avg words/frame", left, right, "avg_words")
    print_metric("Avg visual matches", left, right, "avg_visual_matches")
    print_metric("Avg visual inliers", left, right, "avg_visual_inliers")
    print_metric("Avg inlier ratio", left, right, "avg_inlier_ratio", 3)
    print_metric("Avg total time (ms)", left, right, "avg_total_ms")
    print_metric("Max total time (ms)", left, right, "max_total_ms")
    print_metric("Avg memory update", left, right, "avg_memory_update_ms")
    print_metric("Avg hyp creation", left, right, "avg_hyp_creation_ms")
    print_metric("Avg hyp validation", left, right, "avg_hyp_validation_ms")
    print_metric("Avg keypoint detect", left, right, "avg_keypoint_detection_ms")
    print_metric("Avg descriptor time", left, right, "avg_descriptor_ms")
    print()
    print("Notes:")
    print("- Lower timing numbers are faster.")
    print("- Fewer loop links can still be fine; stability matters more than raw count.")
    print("- For a fair test, scan the same route at similar speed and compare closed-loop runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
