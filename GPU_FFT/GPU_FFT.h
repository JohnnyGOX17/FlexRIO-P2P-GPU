/**
 * GPU FFT Header File
 *
 */

#ifndef GPU_FFT_H_
#define GPU_FFT_H_

#include "NiFpga_FPGA_Main.h"   // NI FPGA C API Generated .h file for bitfile

#define NX              256     // FFT transform size
#define BATCH_SIZE      1000    // # of batches to run
#define SAMPLES         1000    // number of samples to get
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

#endif // GPU_FFT_H_
