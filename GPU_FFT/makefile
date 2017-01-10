CC=nvcc
CFLAGS=-ldl

test: GPU_FFT.cu NiFpga.c
	$(CC) -o GPU_FFT GPU_FFT.cu NiFpga.c -ldl

clean:
	rm GPU_FFT 
