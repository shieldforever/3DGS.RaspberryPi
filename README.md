# 3DGS.RaspberryPi

## Introduction

`3DGS.Raspberry` is an efficient implementation of 3D Gaussian Splatting on a Raspberry Pi 5 board.

The project is developed based on [3DGS.cpp](https://github.com/shg8/3DGS.cpp), a cross-platform, high performance renderer for Gaussian Splatting using Vulkan Compute.

The project focuses on improving end-to-end throughput on Raspberry Pi by introducing a CPU-GPU collaboration pipeline:

- CPU: `preprocess`, `prefix_sum`, `preprocess_sort`, `sort`, `tile_boundary`
- GPU: `render` only
- Cross-frame overlap with double buffering (`FRAMES_IN_FLIGHT=2`)

The main target is improving end-to-end throughput and reducing visible stalls when rendering multiple camera views.

## Detailed Design

**Function Support.**
The V3DV Mesa Vulkan driver in Raspberry Pi 5 does not support some standard Vulkan features, such as subgroups and shaderInt64, so we rewrite core functions like preprocess, sort, and rasterization.

**A CPU-GPU Collaboration Pipeline.**
The functions except rasterization is executed on CPU and the rasterization is executed on GPU.
Therefore, we can maximize the utilization of computing resouce on the board and achieving overlapping between two image rendering.

## Main Results

We evaluate a 3DGS model trained on the "Train" scene from "Tank & Temples" dataset.
We adopt the optimization techniques from [SpeedySplat](https://github.com/j-alex-hanson/speedy-splat) during training, including intersection test and Gaussian pruning.
3DGS.RaspberryPi can achieve **~3.6 FPS** on the Raspberry Pi 5 board.

## Build

### Linux (recommended)

Dependencies (typical Ubuntu names):

- `vulkan-headers`
- `vulkan-validationlayers`
- `glslang-dev`
- `libglfw3-dev`
- `libglm-dev`
- CMake + C++ compiler

Build:

```bash
git clone <your-repo-url> 3DGS.Raspberry
cd 3DGS.Raspberry
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

Binary:

- `build/apps/viewer/3dgs_viewer`

## Run

### 1) Default camera (no camera file)

If `--camera-json` is not passed, the renderer uses the built-in initial camera pose.

```bash
./build/apps/viewer/3dgs_viewer ./point_cloud.ply
```

### 2) Render with camera views from `cameras.json`

```bash
./build/apps/viewer/3dgs_viewer \
  --camera-json ./cameras.json \
  ./point_cloud-moe.ply
```

Useful options:

- `--camera-index <idx>`: render one specific view
- `--max-views <n>`: render first `n` views
- `--output-raw <path-or-dir>`: output raw path/directory
- `--device <id>`: choose Vulkan physical device
- `--width <w> --height <h>`: override window/swapchain size

## Timing & Benchmark Logs

The renderer prints stage timing and pipeline summary logs:

- `[TIMING][ms] ...` per-sample stage breakdown
- `[PIPELINE][avg_ms] ...` average CPU pre, GPU render, serial estimate, pipeline estimate, overlap gain
- `[E2E][ms] ...` aggregate latency/fps summary

Example benchmark command:

```bash
./build/apps/viewer/3dgs_viewer --camera-json ./cameras.json ./point_cloud.ply 2>&1 | tee run.log
```

## Camera Notes

- `cameras.json` is expected to match the Python export format used by 3DGS tooling.
- This project aligns rotation handling with JSON semantics (`C2W` in JSON, converted to `W2C` when building view matrix).
- Intrinsics (`fx/fy`) are interpreted with camera entry resolution to avoid FoV distortion.

## Output Files

- When batch rendering views from `cameras.json`, outputs are written per-view (or per-frame when configured).
- Raw outputs can be converted/visualized with helper scripts such as `raw_to_rgb.py`.

## Contact

- Bowen Zhu: zhubowen@sjtu.edu.cn
- Haomin Li: haominli@sjtu.edu.cn
- Fangxin Liu: liufangxin@sjtu.edu.cn
- Li Jiang: ljiang_cs@sjtu.edu.cn

## Acknowledgements

- [Official Code of 3DGS](https://github.com/graphdeco-inria/gaussian-splatting)
- [shg8/3DGS.cpp](https://github.com/shg8/3DGS.cpp)
- [j-alex-hanson/speedy-splat](https://github.com/j-alex-hanson/speedy-splat)
