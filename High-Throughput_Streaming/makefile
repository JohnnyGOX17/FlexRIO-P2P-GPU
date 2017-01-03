CC=nvcc
CFLAGS=-ldl

test: high-throughput_streaming.cu NiFpga.c
	$(CC) -o high-throughput_streaming high-throughput_streaming.cu NiFpga.c -ldl

clean:
	rm high-throughput_streaming 
