CC=nvcc
CFLAGS=-ldl

test: throughput_test.cu NiFpga.c
	$(CC) -o throughput_test throughput_test.cu NiFpga.c -ldl

clean:
	rm throughput_test 
