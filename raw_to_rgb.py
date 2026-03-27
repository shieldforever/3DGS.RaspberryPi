#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_cameras(camera_json: Path) -> list[dict]:
    with camera_json.open("r", encoding="utf-8") as f:
        cameras = json.load(f)
    if not isinstance(cameras, list) or len(cameras) == 0:
        raise ValueError(f"No cameras found in {camera_json}")
    return cameras


def convert_one(raw_path: Path, out_path: Path, width: int, height: int) -> None:
    data = np.fromfile(raw_path, dtype=np.uint8)
    expected = width * height * 4
    if data.size < expected:
        raise ValueError(f"{raw_path}: size {data.size} is smaller than expected {expected}")
    if data.size > expected:
        data = data[:expected]

    rgba = data.reshape((height, width, 4))
    rgb = rgba[:, :, :3]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path, rgb)
    print(f"Saved RGB image to {out_path} ({width}x{height})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RGBA8 raw output(s) to RGB image(s).")
    parser.add_argument("--raw", default="", help="Single raw input file path")
    parser.add_argument("--out", default="", help="Single output image path (.png)")
    parser.add_argument("--raw-dir", default="output", help="Raw directory for batch mode")
    parser.add_argument("--out-dir", default="output", help="Output directory for batch mode")
    parser.add_argument("--width", type=int, default=0, help="Image width")
    parser.add_argument("--height", type=int, default=0, help="Image height")
    parser.add_argument("--camera-json", default="cameras.json", help="cameras.json path")
    parser.add_argument("--camera-index", type=int, default=-1, help="Single camera index in cameras.json")
    parser.add_argument("--all-cameras", action="store_true", help="Convert all camera views in cameras.json")
    args = parser.parse_args()

    cameras = load_cameras(Path(args.camera_json))
    run_all = args.all_cameras or args.camera_index < 0

    if run_all:
        raw_dir = Path(args.raw_dir)
        out_dir = Path(args.out_dir)
        for i, camera in enumerate(cameras):
            stem = Path(str(camera.get("img_name", f"camera_{i}"))).stem
            raw_path = raw_dir / f"{stem}.raw"
            out_path = out_dir / f"{stem}.png"
            width = int(camera["width"])
            height = int(camera["height"])
            convert_one(raw_path, out_path, width, height)
    else:
        if args.camera_index >= len(cameras):
            raise ValueError(f"camera_index={args.camera_index} is out of range")
        camera = cameras[args.camera_index]
        raw_path = Path(args.raw) if args.raw else Path("output/output.raw")
        out_path = Path(args.out) if args.out else Path("output/output.png")
        if args.width > 0 and args.height > 0:
            width, height = args.width, args.height
        else:
            width = int(camera["width"])
            height = int(camera["height"])
        convert_one(raw_path, out_path, width, height)


if __name__ == "__main__":
    main()
