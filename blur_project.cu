#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cstdlib>

// ----------------- Structures -----------------

struct Pixel {
    unsigned char r, g, b;
};

struct Image {
    int width, height, maxVal;
    std::vector<Pixel> data;
};

// ----------------- GPU Kernels -----------------

__global__ void grayscaleKernel(Pixel* input, Pixel* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        unsigned char gray = (unsigned char)(0.299f * input[idx].r + 0.587f * input[idx].g + 0.114f * input[idx].b);
        output[idx] = {gray, gray, gray};
    }
}

__global__ void blurKernel(Pixel* input, Pixel* output, int width, int height, int blurSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixValR = 0;
        int pixValG = 0;
        int pixValB = 0;
        int pixels = 0;

        for (int blurX = -blurSize; blurX <= blurSize; blurX++) {
            for (int blurY = -blurSize; blurY <= blurSize; blurY++) {
                int curX = x + blurX;
                int curY = y + blurY;

                if (curX >= 0 && curX < width && curY >= 0 && curY < height) {
                    int idx = curY * width + curX;
                    pixValR += input[idx].r;
                    pixValG += input[idx].g;
                    pixValB += input[idx].b;
                    pixels++;
                }
            }
        }
        int outIdx = y * width + x;
        output[outIdx].r = (unsigned char)(pixValR / pixels);
        output[outIdx].g = (unsigned char)(pixValG / pixels);
        output[outIdx].b = (unsigned char)(pixValB / pixels);
    }
}

// ----------------- Host Helper Functions -----------------

void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

Image readPPM(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening input file." << std::endl;
        exit(1);
    }

    std::string format;
    int w, h, max;
    file >> format >> w >> h >> max;
    
    // Consume newline
    file.ignore(256, '\n');

    Image img;
    img.width = w;
    img.height = h;
    img.maxVal = max;
    img.data.resize(w * h);

    file.read(reinterpret_cast<char*>(img.data.data()), w * h * sizeof(Pixel));
    return img;
}

void writePPM(const std::string& filename, const Image& img) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening output file." << std::endl;
        exit(1);
    }

    file << "P6\n" << img.width << " " << img.height << "\n" << img.maxVal << "\n";
    file.write(reinterpret_cast<const char*>(img.data.data()), img.width * img.height * sizeof(Pixel));
}

// ----------------- Main Driver -----------------

int main(int argc, char** argv) {
    std::string inputFile = "input.ppm";
    std::string outputFile = "output.ppm";
    int blurSize = 1;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) inputFile = argv[++i];
        else if (arg == "-o" && i + 1 < argc) outputFile = argv[++i];
        else if (arg == "-b" && i + 1 < argc) blurSize = std::atoi(argv[++i]);
    }

    Image h_input = readPPM(inputFile);
    size_t imgSize = h_input.width * h_input.height * sizeof(Pixel);

    Pixel *d_input, *d_temp, *d_output;
    checkCuda(cudaMalloc(&d_input, imgSize), "Alloc Input");
    checkCuda(cudaMalloc(&d_temp, imgSize), "Alloc Temp");
    checkCuda(cudaMalloc(&d_output, imgSize), "Alloc Output");

    checkCuda(cudaMemcpy(d_input, h_input.data.data(), imgSize, cudaMemcpyHostToDevice), "Copy H2D");

    dim3 blockSize(16, 16);
    dim3 gridSize((h_input.width + blockSize.x - 1) / blockSize.x, (h_input.height + blockSize.y - 1) / blockSize.y);

    grayscaleKernel<<<gridSize, blockSize>>>(d_input, d_temp, h_input.width, h_input.height);
    checkCuda(cudaDeviceSynchronize(), "Grayscale Kernel");

    blurKernel<<<gridSize, blockSize>>>(d_temp, d_output, h_input.width, h_input.height, blurSize);
    checkCuda(cudaDeviceSynchronize(), "Blur Kernel");

    Image h_output = h_input;
    checkCuda(cudaMemcpy(h_output.data.data(), d_output, imgSize, cudaMemcpyDeviceToHost), "Copy D2H");

    writePPM(outputFile, h_output);

    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);

    std::cout << "Processed " << h_input.width << "x" << h_input.height << " image. Saved to " << outputFile << std::endl;
    return 0;
}
