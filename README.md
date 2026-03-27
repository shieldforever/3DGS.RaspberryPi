# 3DGS.RaspberryPi

## Introduction

3DGS.RaspberryPi is an efficient implementation of 3D Gaussian Splatting on a Raspberry Pi 5 board.

Our project is developed based on [3DGS.cpp](https://github.com/shg8/3DGS.cpp), a cross-platform, high performance renderer for Gaussian Splatting using Vulkan Compute.

## Detailed Design

**Function Support.**
The V3DV Mesa Vulkan driver in Raspberry Pi 5 does not support some standard Vulkan features, such as subgroups and shaderInt64, so we rewrite core functions like preprocess, sort, and rasterization.

**A CPU-GPU Collaboration Pipeline.**
The functions except rasterization is executed on CPU and the rasterization is executed on GPU.
Therefore, we can maximize the utilization of computing resouce on the board and achieving overlapping between two image rendering.

## Validation

We evaluate a 3DGS model trained on the "Train" scene from "Tank & Temples" dataset.
We adopt the optimization techniques from [SpeedySplat](https://github.com/j-alex-hanson/speedy-splat) during training, including intersection test and Gaussian pruning.
3DGS.RaspberryPi can achieve **~3.6 FPS** on the Raspberry Pi 5 board.

## Contact

- Bowen Zhu: zhubowen@sjtu.edu.cn
- Haomin Li: haominli@sjtu.edu.cn
- Fangxin Liu: liufangxin@sjtu.edu.cn
- Li Jiang: ljiang_cs@sjtu.edu.cn
