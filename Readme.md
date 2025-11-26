# CUDA Accelerated Image Processing Pipeline

## Project Description
This project implements a high-performance image processing tool using C++ and NVIDIA CUDA. The application leverages the massive parallelism of the GPU to perform pixel-wise operations on Portable Pixel Map (PPM) images.

The pipeline performs two distinct stages:
1.  **Grayscale Conversion:** Transforms RGB channels into a single luminance channel using standard colorimetric weights.
2.  **Gaussian Blur:** Applies a variable-radius box blur using a 2D stencil operation.

## Prerequisites
* NVIDIA GPU with Compute Capability 3.0 or higher.
* CUDA Toolkit (10.0+).
* GCC/G++ Compiler.
* Linux Environment (or WSL on Windows).

## Project Structure
* `blur_project.cu`: Single-source file containing host logic, GPU kernels, and memory management.
* `Makefile`: Build script for compiling the project using `nvcc`.

## Building the Project
To compile the software, navigate to the project directory and run:

```bash
make build
