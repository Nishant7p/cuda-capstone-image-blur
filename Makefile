nvcc = /usr/local/cuda/bin/nvcc
cc = g++

all: clean build

build: blur_project.cu
	$(nvcc) -o blur_tool blur_project.cu

run:
	./blur_tool -i input.ppm -o output.ppm -b 2

clean:
	rm -f blur_tool output.ppm
