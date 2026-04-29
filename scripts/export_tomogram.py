#!/usr/bin/env python3
"""
Offline RTAB-Map DB -> PCT-compatible tomogram exporter.

This script uses RTAB-Map's existing cloud export path to generate an
intermediate mesh-less cloud, converts it to PCD, then applies an adapted copy
of PCT_planner's CuPy tomogram pipeline to produce a drop-in compatible pickle.
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
EXPORT_BIN = REPO_DIR / "build" / "bin" / "rtabmap-export"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def die(message: str, code: int = 1) -> int:
    print(message, file=sys.stderr)
    return code


def parse_scalar(value: str):
    value = value.strip()
    if not value:
        return ""
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_simple_yaml(path: Path) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if "\t" in raw_line[: len(raw_line) - len(raw_line.lstrip())]:
            raise ValueError(f"Tabs are not supported in YAML config ({path}:{lineno})")
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Expected key/value pair in {path}:{lineno}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid indentation in {path}:{lineno}")
        parent = stack[-1][1]
        if value == "":
            node: dict = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = parse_scalar(value)
    return root


def to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in value.items()})
    return value


def require_python_module(name: str, install_hint: str):
    try:
        return __import__(name)
    except ModuleNotFoundError:
        raise RuntimeError(
            f"Missing Python dependency '{name}'. Install it first ({install_hint})."
        ) from None


def resolve_output_dir(config, cli_output_dir: str | None) -> Path:
    if cli_output_dir:
        return Path(cli_output_dir).expanduser().resolve()
    export_dir = getattr(config.map, "export_dir", "data/tomography/exports")
    out_path = Path(export_dir)
    if not out_path.is_absolute():
        out_path = REPO_DIR / out_path
    return out_path.resolve()


def export_cloud_from_db(db_path: Path, output_dir: Path, base_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(EXPORT_BIN),
        "--cloud",
        "--output",
        base_name,
        "--output_dir",
        str(output_dir),
        str(db_path),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    ply_path = output_dir / f"{base_name}_cloud.ply"
    if not ply_path.exists():
        raise RuntimeError(f"Expected RTAB-Map cloud export was not generated: {ply_path}")
    return ply_path


def compute_mapping_metadata(points: np.ndarray, cfg):
    points_max = np.max(points, axis=0)
    points_min = np.min(points, axis=0)
    points_min[-1] = cfg.map.ground_h
    map_dim_x = int(np.ceil((points_max[0] - points_min[0]) / cfg.map.resolution)) + 4
    map_dim_y = int(np.ceil((points_max[1] - points_min[1]) / cfg.map.resolution)) + 4
    n_slice_init = int(np.ceil((points_max[2] - points_min[2]) / cfg.map.slice_dh))
    center = (points_max[:2] + points_min[:2]) / 2
    slice_h0 = points_min[-1] + cfg.map.slice_dh
    return center.astype(np.float32), map_dim_x, map_dim_y, n_slice_init, float(slice_h0)


def export_tomogram_pickle(
    cfg,
    points: np.ndarray,
    pickle_path: Path,
) -> None:
    from pct_tomography.tomogram import Tomogram

    center, map_dim_x, map_dim_y, n_slice_init, slice_h0 = compute_mapping_metadata(points, cfg)
    print(f"Map center: [{center[0]:.2f}, {center[1]:.2f}]")
    print(f"Map dims: x={map_dim_x}, y={map_dim_y}")
    print(f"Initial slice count: {n_slice_init}")
    print(f"slice_h0: {slice_h0:.3f}")

    repeats = int(getattr(getattr(cfg, "runtime", SimpleNamespace(benchmark_repeats=0)), "benchmark_repeats", 0))
    tomogram = Tomogram(cfg)
    tomogram.initMappingEnv(center, map_dim_x, map_dim_y, n_slice_init, slice_h0)

    benchmark_passes = max(0, repeats)
    t_map = 0.0
    t_trav = 0.0
    t_simp = 0.0
    t_all = 0.0
    result = None
    for i in range(benchmark_passes + 1):
        start = time.time()
        result = tomogram.point2map(points)
        if i > 0:
            gpu_times = result[-1]
            t_map += gpu_times["t_map"]
            t_trav += gpu_times["t_trav"]
            t_simp += gpu_times["t_simp"]
            t_all += (time.time() - start) * 1e3
    assert result is not None

    layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c, _ = result
    print(f"Simplified slice count: {layers_g.shape[0]}")
    if benchmark_passes > 0:
        print(f"avg t_map  (ms): {t_map / benchmark_passes:.3f}")
        print(f"avg t_trav (ms): {t_trav / benchmark_passes:.3f}")
        print(f"avg t_simp (ms): {t_simp / benchmark_passes:.3f}")
        print(f"avg t_all  (ms): {t_all / benchmark_passes:.3f}")

    payload = {
        "data": np.stack((layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c)).astype(np.float16),
        "resolution": cfg.map.resolution,
        "center": center,
        "slice_h0": slice_h0,
        "slice_dh": cfg.map.slice_dh,
    }
    with pickle_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tomogram exported: {pickle_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a PCT-compatible tomogram from an RTAB-Map database.")
    parser.add_argument("--db", required=True, help="Path to the RTAB-Map .db file.")
    parser.add_argument("--config", required=True, help="Path to tomography YAML config.")
    parser.add_argument("--output-dir", help="Override output directory.")
    parser.add_argument("--pcd-path", help="Override final intermediate PCD path.")
    parser.add_argument("--keep-pcd", dest="keep_pcd", action="store_true", default=True, help="Keep the generated PCD (default).")
    parser.add_argument("--no-keep-pcd", dest="keep_pcd", action="store_false", help="Delete the generated PCD after pickle export.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()

    if not db_path.is_file():
        return die(f"Database not found: {db_path}")
    if not config_path.is_file():
        return die(f"Config not found: {config_path}")
    if not EXPORT_BIN.is_file():
        return die(f"Missing required binary: {EXPORT_BIN}")

    try:
        config = to_namespace(load_simple_yaml(config_path))
    except Exception as exc:
        return die(f"Failed to load config {config_path}: {exc}")

    output_dir = resolve_output_dir(config, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = db_path.stem
    final_pcd_path = (
        Path(args.pcd_path).expanduser().resolve()
        if args.pcd_path
        else output_dir / f"{base_name}.pcd"
    )
    if final_pcd_path.suffix.lower() != ".pcd":
        return die(f"--pcd-path must end with .pcd: {final_pcd_path}")
    pickle_path = output_dir / f"{base_name}.pickle"

    try:
        open3d = require_python_module("open3d", "pip install open3d")
        require_python_module("cupy", "install a CuPy wheel matching your CUDA toolkit")
    except RuntimeError as exc:
        return die(str(exc))

    try:
        ply_path = export_cloud_from_db(db_path, output_dir, base_name)

        point_cloud = open3d.io.read_point_cloud(str(ply_path))
        if point_cloud.is_empty():
            return die(f"Exported cloud is empty: {ply_path}")

        points = np.asarray(point_cloud.points, dtype=np.float32)
        print(f"Loaded cloud points: {points.shape[0]}")
        if points.ndim != 2 or points.shape[1] < 3:
            return die(f"Unexpected point cloud shape from {ply_path}: {points.shape}")
        points = points[:, :3]

        final_pcd_path.parent.mkdir(parents=True, exist_ok=True)
        if not open3d.io.write_point_cloud(str(final_pcd_path), point_cloud):
            return die(f"Failed to write PCD: {final_pcd_path}")
        print(f"Intermediate PCD: {final_pcd_path}")

        export_tomogram_pickle(config, points, pickle_path)
    except subprocess.CalledProcessError as exc:
        return die(f"rtabmap-export failed with exit code {exc.returncode}")
    except Exception as exc:  # noqa: BLE001
        return die(f"Tomogram export failed: {exc}")
    finally:
        raw_ply = output_dir / f"{base_name}_cloud.ply"
        if raw_ply.exists():
            raw_ply.unlink()
        if not args.keep_pcd and final_pcd_path.exists():
            final_pcd_path.unlink()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
