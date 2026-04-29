#!/usr/bin/env python3
"""
Convert an upstream LightGlue checkpoint into a plain-dict archive that
RTAB-Map's native C++ LightGlue loader can read directly.
"""

import argparse
from collections import OrderedDict
from pathlib import Path
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a LightGlue checkpoint saved as an OrderedDict into a "
            "plain-dict archive for RTAB-Map's native C++ loader."
        )
    )
    parser.add_argument("input", help="Path to upstream LightGlue checkpoint (*.pth)")
    parser.add_argument(
        "output",
        nargs="?",
        help="Output path for converted weights. Defaults to <input>_native.pth",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_native{input_path.suffix or '.pth'}")
    )

    if not input_path.is_file():
        print(f"Input checkpoint not found: {input_path}", file=sys.stderr)
        return 1

    state = torch.load(input_path, map_location="cpu")
    if not isinstance(state, (dict, OrderedDict)):
        print(
            f"Expected a dict-like checkpoint, got {type(state).__name__}",
            file=sys.stderr,
        )
        return 1

    converted = dict(state)
    if not converted:
        print("Checkpoint is empty after conversion.", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
