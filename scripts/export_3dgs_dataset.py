#!/usr/bin/env python3
"""Export an RTAB-Map database to a 3DGS-ready COLMAP/Nerfstudio dataset."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


def find_binary(repo_root: pathlib.Path) -> pathlib.Path:
    binary = repo_root / "build" / "bin" / "rtabmap-export3dgs_dataset"
    if not binary.exists():
        raise FileNotFoundError(
            f"Missing exporter binary: {binary}. Build target 'export3dgs_dataset' first."
        )
    return binary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export an RTAB-Map .db to a 3DGS-ready COLMAP + Nerfstudio dataset."
    )
    parser.add_argument("--db", required=True, help="Input RTAB-Map database path")
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: <db_basename>_3dgs beside the database)",
    )
    parser.add_argument(
        "--image-format",
        default="png",
        choices=("png", "jpg", "jpeg"),
        help="Exported undistorted image format",
    )
    parser.add_argument(
        "--max-linear-speed",
        type=float,
        default=0.0,
        help="Ignore frames faster than this linear speed in m/s",
    )
    parser.add_argument(
        "--max-angular-speed",
        type=float,
        default=0.0,
        help="Ignore frames faster than this angular speed in rad/s",
    )
    parser.add_argument(
        "--laplacian-threshold",
        type=float,
        default=0.0,
        help="Ignore frames below this Laplacian variance blur threshold",
    )
    parser.add_argument(
        "--export-nerfstudio",
        type=str,
        default="true",
        choices=("true", "false"),
        help="Whether to write transforms.json",
    )
    parser.add_argument(
        "--export-colmap",
        type=str,
        default="true",
        choices=("true", "false"),
        help="Whether to write COLMAP sparse text files",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    binary = find_binary(repo_root)

    db_path = pathlib.Path(args.db).expanduser().resolve()
    if not db_path.exists():
        parser.error(f"Database does not exist: {db_path}")

    cmd = [
        str(binary),
        "--db",
        str(db_path),
        "--image_format",
        args.image_format,
        "--max_linear_speed",
        str(args.max_linear_speed),
        "--max_angular_speed",
        str(args.max_angular_speed),
        "--laplacian_threshold",
        str(args.laplacian_threshold),
        "--export_colmap",
        args.export_colmap,
        "--export_nerfstudio",
        args.export_nerfstudio,
    ]
    if args.output_dir:
        cmd.extend(["--output_dir", str(pathlib.Path(args.output_dir).expanduser().resolve())])

    completed = subprocess.run(cmd, cwd=repo_root)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
