/**
* GPU FFT Example
*
* This example showcases the FFT capabilities of a P2P NVIDIA GPU with a
* FlexRIO device. In this case, the FlexRIO module is creating a simulated CW
* tone and can perform Additive White Gaussian Noise (AWGN) and/or a Finite
* Impulse Response (FIR) Low-Pass Filter (LPF) on the generated signal. The
* signal is then sent to the GPU where a parallelized FFT takes place using
* NVIDIA's CUFFT library and some logarithmic conversion to calculate the
* power spectrum of the signal. This resulting signal is then written to a file
* and/or a GNUplot host application for data logging and plotting.
*
* For more information on NI FPGA functions, see the NI FPGA Interface C API
* Help. For more information on NVIDIA CUDA functions and operation, see the
* help files included with the NVIDIA CUDA Driver.
*
* Date:         1/10/2016
* Author:       John Gentile
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <sys/times.h>
#include <ctype.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include "NiFpga_FPGA_Main.h"   // NI FPGA C API Generated .h file for bitfile

#define HELP_STRING "Usage: GPU_FFT [OPTIONS]\nGPU FFT Example with NI FlexRIO device\n\n\t-H,\tTransfer data from FPGA to host memory before transferring to GPU\n\t-l,\tPass simulated signal through digital Low-Pass FIR Filter on FPGA\n\t-a,\tAdd White Gaussian Noise to simulated signal on FPGA\n\t-t,\tWrite generated time-domain signal from FlexRIO to file (must be used with -H option)\n"

// Keep samples as power of 2 in case using multiple GPUs
#define SAMPLES         1048576
#define COMPLEX_SIZE    (SAMPLES*2 + 1)

// use inline definition for error checking to allow easy app exit
#define CHECKSTAT(stat) if (stat != 0) { printf("%d: Error: %d\n", __LINE__, stat); return 1; }

#define checkCudaErrors(val) __checkCudaErrors__ ( (val), #val, __FILE__, __LINE__ )

// keep datatypes uniform between FIFOs and operations
typedef int32_t         fifotype;
// define data type as complex as we are doing an in-place real-to-complex FFT
typedef cuComplex       cufftComplex;

template <typename T> // Templated to allow for different CUDA error types
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

// Used to convert int32_t data input from FlexRIO to cufftReal
// Other scaling can occur here as well
__global__ void ConvertInputToComplex(
    const int32_t * __restrict__ dataIn,
    cufftReal * __restrict__ dataOut)
{
  const int numThreads = blockDim.x * gridDim.x;
  const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t offset = threadId; offset < SAMPLES; offset += numThreads)
    dataOut[offset] = (cufftReal)((float)dataIn[offset]/127.0f);
}

/*
* Main Program Execution
*/
int main(int argc, char **argv)
{
  /*
  * Initialization of program values, NI FPGA and CUDA
  */
  int viahostflag = 0;  // -H, do transfers and operations via host
  int lpfflag = 0;      // -l, pass signal through Low-Pass FIR on FlexRIO
  int awgnflag = 0;     // -a, add white gaussian noise to signal on FlexRIO
  int timeflag =0;      // -t, write time-domain signal (only if -H is set)
  int c;

  int32_t * h_data = NULL; // ptr for when transferring data to host first

  // Process command line arguments and set above flags
  while ((c = getopt(argc, argv, "Hlath")) != -1)
  {
    switch (c)
    {
      case 'H':
        viahostflag = 1;
        break;
      case 'l':
        lpfflag = 1;
        break;
      case 'a':
        awgnflag = 1;
        break;
      case 't':
        timeflag = 1;
        break;
      case 'h':
        printf(HELP_STRING);
        exit(0);
      default:
        abort();
    }
  }


  // initialize NI FPGA interfaces; use status variable for error handling
  printf("Initializing NI FPGA: ");
  CHECKSTAT(NiFpga_Initialize());
  NiFpga_Session session;

  // Download bitfile to target; get path to bitfile
  // TODO: change to full path to bitfile as necessary
  printf("Downloading bitfile ");
  CHECKSTAT(NiFpga_Open("/home/nitest/FlexRIO-P2P-GPU/GPU_FFT/NiFpga_FPGA_Main.lvbitx",
        NiFpga_FPGA_Main_Signature, "RIO0", 0, &session));
  printf("DONE\n");

  struct cudaDeviceProp d_props;
  int device = 0; //TODO: modify to take stdin for multiple GPUs
  checkCudaErrors(cudaGetDevice(&device));
  checkCudaErrors(cudaGetDeviceProperties(&d_props, device));
  if (d_props.major < 2)
  {
    printf("CUDA Error: This example requires a CUDA device with architecture SM2.0 or higher\n");
    exit(EXIT_FAILURE);
  }

  /*
   * Allocate CUDA Memory for FFT and log scaling operations
   * As well, allocate complex CUDA Memory for R2C result
  */
  printf("Allocating CUDA and Host Memory: \n");
  int32_t *init_signal;         // initial storage for non-scaled data input
  cufftReal *tmp_result1;       // temp storage for scaling data input
  cufftComplex *gpu_result;     // Where CUFFT will be stored
  cufftComplex *hcomp_data;     // Where host memory will recieve complex data result

  checkCudaErrors(cudaMalloc((void **)&init_signal, sizeof(*init_signal)*SAMPLES));
  checkCudaErrors(cudaMalloc((void **)&tmp_result1, sizeof(cufftReal)*SAMPLES));
  checkCudaErrors(cudaMalloc((void **)&gpu_result, sizeof(cufftComplex)*COMPLEX_SIZE));

  hcomp_data = (cufftComplex*) malloc(sizeof(cufftComplex)*COMPLEX_SIZE);
  if (hcomp_data == NULL)
  {
    printf("Host Error: Host failed to allocate memory\n");
    return -1;
  }

  /*
   * Make CUFFT plan for 1D Real-to-Complex FFT
   * also link data path to CUDA device
   */
  cufftHandle plan;
  checkCudaErrors(cufftCreate(&plan));
  checkCudaErrors(cufftPlan1d(&plan, SAMPLES, CUFFT_R2C, 1));

  // Configure P2P FIFO between FlexRIO and GPU using NVIDIA GPU Direct
  if (viahostflag == 1) /* Host transfer */
  {
    CHECKSTAT(NiFpga_ConfigureFifo(session, NiFpga_FPGA_Main_TargetToHostFifoI32_T2HDMAFIFO,
          (size_t)SAMPLES));
  }
  else /* P2P via RDMA */
  {
    CHECKSTAT(NiFpga_ConfigureFifoBuffer(session, NiFpga_FPGA_Main_TargetToHostFifoI32_T2HDMAFIFO,
          (uint64_t)init_signal, SAMPLES, NULL, NiFpga_DmaBufferType_NvidiaGpuDirectRdma));
  }

  /*
   * Set NI FPGA Control/Indicator Values
   */
  // Set RMS Noise value of AWGN Algorithm on FlexRIO (out of i16 full scale, here set as 2048)
  NiFpga_WriteU16(session, NiFpga_FPGA_Main_ControlU16_RMSNoise, 2048);

  // Reset FPGA algorithms and clear FIFOs
  NiFpga_WriteBool(session, NiFpga_FPGA_Main_ControlBool_aReset, NiFpga_True);
  NiFpga_WriteBool(session, NiFpga_FPGA_Main_ControlBool_aReset, NiFpga_False);

  if (lpfflag == 1) {
    printf("Enabling Low-Pass FIR Filter on FlexRIO\n");
    NiFpga_WriteBool(session, NiFpga_FPGA_Main_ControlBool_LPFEnable, NiFpga_True);
  }
  else
    NiFpga_WriteBool(session, NiFpga_FPGA_Main_ControlBool_LPFEnable, NiFpga_False);

  if (awgnflag == 1) {
    printf("Adding White Gaussian Noise to Signal\n");
    NiFpga_WriteBool(session, NiFpga_FPGA_Main_ControlBool_AWGNEnable, NiFpga_True);
  }
  else
    NiFpga_WriteBool(session, NiFpga_FPGA_Main_ControlBool_AWGNEnable, NiFpga_False);

  /*
   * DMA (or copy from host) signal to GPU and execute FFT plan
   */
  if (viahostflag == 1)
  {
    printf("Copy host memory signal to CUDA Device\n");
    h_data = (int32_t *)malloc(SAMPLES * sizeof(int32_t));
    CHECKSTAT(NiFpga_ReadFifoI32(session, NiFpga_FPGA_Main_TargetToHostFifoI32_T2HDMAFIFO,
          h_data, (size_t)SAMPLES, 5000, NULL));

    if (timeflag == 1)
    {
      printf("Writing time-domain signal to TimeSignal.dat\n");
      FILE *f = fopen("TimeSignal.dat", "w");
      for(int i=0; i<SAMPLES; i++)
      {
        if (i==0)
          fprintf(f, "# X Y\n");
        // Write real component to file
        fprintf(f, "%d %d\n", i, h_data[i]);
      }
      fclose(f);
    }
    if (cudaMemcpy(init_signal, h_data, (SAMPLES*sizeof(int32_t)), cudaMemcpyHostToDevice) != cudaSuccess)
    {
      printf("CUDA Error: Device failed to copy host memory to device\n");
      return -1;
    }
  }
  else
  {
    printf("DMA'ing FlexRIO data to GPU\n");
    size_t elemsAcquired, elemsRemaining;
    CHECKSTAT(NiFpga_AcquireFifoReadElementsI32(session,
          NiFpga_FPGA_Main_TargetToHostFifoI32_T2HDMAFIFO, &init_signal,
          SAMPLES, 5000, &elemsAcquired, &elemsRemaining));
    printf("%d samples acquired with %d elements remaining in FIFO/n", elemsAcquired, elemsRemaining);
  }

  /*
   * Start FFT on GPU
  */
  printf("Executing CUFFT Plan...\n");

  // create timers
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  float elapsedTime=0;

  checkCudaErrors(cudaEventRecord(start, 0));

  // Transform input data using our customer kernel
  ConvertInputToComplex<<<32, 128>>>(init_signal, tmp_result1);
  checkCudaErrors(cudaGetLastError());

  // Execute FFT on data
  checkCudaErrors(cufftExecR2C(plan, tmp_result1, gpu_result));
  checkCudaErrors(cudaGetLastError());

  // Stop and record time taken
  checkCudaErrors(cudaEventRecord(end, 0));
  // Sync and wait for completion
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventSynchronize(end));
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, end));
  printf("FFT took %fms to complete on GPU\n", elapsedTime);

  // Copy FFT result back to host)
  printf("Copying CUFFT result back to host memory...\n");
  checkCudaErrors(cudaMemcpy(hcomp_data, gpu_result, SAMPLES, cudaMemcpyDeviceToHost));


  /*
   * Write resulting data to file
   */
  printf("Writing signal to SimSignal.dat\n");
  FILE *f = fopen("SimSignal.dat", "w");
  for(int32_t i=0; i<COMPLEX_SIZE; i++)
  {
    if (i==0)
      fprintf(f, "# X Y\n");
    // Write real component to file
    // TODO: Implement power conversion as CUFFT Callback
    fprintf(f, "%d %f\n", i, 20.0f*log(hcomp_data[i].y));
  }
  fclose(f);

  // Close out CUFFT plan(s) and free CUDA memory
  printf("Closing CUFFT and freeing CUDA memory...\n");
  checkCudaErrors(cufftDestroy(plan));

  // Commented out- since we destroyed link to CUDA, we need not stop the FIFO
  //CHECKSTAT(NiFpga_ReleaseFifoElements(session, NiFpga_FPGA_Main_TargetToHostFifoI32_T2HDMAFIFO, SAMPLES));
  //CHECKSTAT(NiFpga_StopFifo(session, NiFpga_FPGA_Main_TargetToHostFifoI32_T2HDMAFIFO));

  checkCudaErrors(cudaFree(init_signal));
  checkCudaErrors(cudaFree(tmp_result1));
  checkCudaErrors(cudaFree(gpu_result));
  free(hcomp_data);
  free(h_data);

  // Clean up NVIDIA Driver state
  cudaDeviceReset();

  // Close NI FPGA References; must be last NiFpga calls
  printf("Stopping NI FPGA...\n");
  CHECKSTAT(NiFpga_Close(session, 0));
  CHECKSTAT(NiFpga_Finalize());

  return 0;
}
