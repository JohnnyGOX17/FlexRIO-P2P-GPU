# Compilation options for GNU Make

# Location of CUDA Toolkit
CUDA_PATH 	?= "/usr/local/cuda-8.0"
HOST_CC 	?= g++
NVCC		:= $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_CC)
NVCCFLAGS	:= -m64
LIBRARIES 	= -ldl -lcufft -lculibos

build: GPU_FFT

GPU_FFT: GPU_FFT.cu NiFpga.c
	$(NVCC) $(NVCCFLAGS) GPU_FFT.cu NiFpga.c -o GPU_FFT $(LIBRARIES)

clean:
	rm GPU_FFT 
	rm ./plot/SimSignal.dat
	rm ./plot/spectrum.png
