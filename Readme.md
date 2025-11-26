# CUDA Capstone: Parallel Image Processing

## Project Description
This project implements a high-performance image processing pipeline using C++ and CUDA. The goal is to demonstrate GPU acceleration for stencil operations by implementing a parallel Grayscale converter and a Gaussian Blur filter.

## Prerequisites
* NVIDIA GPU (Compute Capability 3.0+)
* CUDA Toolkit (10.0+)
* GCC/G++ Compiler
* Make

## Compilation
To build the project, simply run the make command in the project directory:

```bash
make build
```

This will generate the executable `blur_tool`.

## Usage
The application accepts command-line arguments to specify input/output files and processing parameters.

### Syntax
```bash
./blur_tool -i <input_file> -o <output_file> -b <blur_radius>
```

### Parameters
* `-i`: Path to the input image (PPM format).
* `-o`: Path where the processed image will be saved.
* `-b`: Integer value for blur radius (e.g., 1, 3, 5). Higher values result in more blur.

### Example Run
```bash
./blur_tool -i input.ppm -o output.ppm -b 5
```

## Implementation Details
* **Kernel Design:** Uses 2D thread blocks (16x16) to align with image geometry and ensure coalesced memory access on the GPU.
* **Memory Management:** Explicit `cudaMalloc` and `cudaMemcpy` are used to manage VRAM usage efficiently, ensuring data is transferred correctly between Host and Device.
* **Boundary Handling:** The blur kernel implements boundary checking to handle edge pixels without memory access violations.
