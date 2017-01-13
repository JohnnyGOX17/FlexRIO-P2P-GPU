/**
 * GPU FFT Header File
 *
 */

#ifndef GPU_FFT_H_
#define GPU_FFT_H_

#include "NiFpga_FPGA_Main.h"   // NI FPGA C API Generated .h file for bitfile

#define SAMPLES         10000   // number of samples to get
#define NX              SAMPLES // FFT transform size
#define BATCH_SIZE      10      // # of batches to run
#define ITERATIONS      100
#define COMPLEX_SIZE    (SAMPLES/2 + 1)
 
// use inline definition for error checking to allow easy app exit
#define CHECKSTAT(stat) if (stat != 0) { printf("%d: Error: %d\n", __LINE__, stat); return 1; }

#define checkCudaErrors(val) __checkCudaErrors__ ( (val), #val, __FILE__, __LINE__ )
 
// keep datatypes uniform between FIFOs and operations
typedef int32_t         fifotype;
// define data type as complex as we are doing an in-place real-to-complex FFT
typedef cuComplex       cufftComplex;

/*
 * CUDA Error Checking
 */
template <typename T>
inline void __checkCudaErrors__(T code, const char *func, const char *file, int line) 
{
  if (code) {
    fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line, (unsigned int)code, func);
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}

__device__ __host__ inline cufftComplex ComplexMul(cufftComplex a, cufftComplex b)
{
  cufftComplex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}

#define HELP_STRING "Usage: GPU_FFT [OPTIONS]\nGPU FFT Example with NI FlexRIO device\n\n\t-H,\tTransfer data from FPGA to host memory before transferring to GPU\n\t-l,\tPass simulated signal through digital Low-Pass FIR Filter on FPGA\n\t-a,\tAdd White Gaussian Noise to simulated signal on FPGA\n\t-t,\tWrite generated time-domain signal from FlexRIO to file (must be used with -H option)\n"

#endif // GPU_FFT_H_
