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

#define HELP_STRING \
  "Usage: GPU_FFT [OPTIONS]\n" \
  "GPU FFT Example with NI FlexRIO device\n\n" \
  "\t-H,\tTransfer data from FPGA to host memory before transferring to GPU\n" \
  "\t-l,\tPass simulated signal through digital Low-Pass FIR Filter on FPGA\n" \
  "\t-a,\tAdd White Gaussian Noise to simulated signal on FPGA\n" \
  "\t-t,\tWrite generated time-domain signal from FlexRIO to file (must be used with -H option)\n" \
  "\t-b [./bitfile],\tPath to *.lvbitx bitfile\n" \
  "\t-s [signature],\tSignature of the bitfile\n" \
  "\t-r [RIO0],\tRIO resource string to open (e.g. RIO0 or rio://mysystem/RIO)\n"

#define SAMPLES         1048576*4
#define COMPLEX_SIZE    (SAMPLES*2 + 1)
#define MAX_STR_LEN     256

// use inline method for error checking to allow easy app exit
#define checkStatus(val) checkStatus__ ( (val), #val, __FILE__, __LINE__ )
template <typename T> // Templated to allow for different CUDA/NiFpga error types
inline void checkStatus__(T code, const char *func, const char *file, int line)
{
  if (code) {
    fprintf(stderr, "Error at %s:%d code=%d \"%s\" \n", file, line, (unsigned int)code, func);
    cudaDeviceReset();
    NiFpga_Finalize();
    exit(EXIT_FAILURE);
  }
}

__device__ __host__ inline cuComplex ComplexMul(cuComplex a, cuComplex b)
{
  cuComplex c;
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
  int viahostflag = 0,          // -H, do transfers and operations via host
      lpfflag = 0,              // -l, pass signal through Low-Pass FIR on FlexRIO
      awgnflag = 0,             // -a, add white gaussian noise to signal on FlexRIO
      timeflag = 0,             // -t, write time-domain signal (only if -H is set)
      opt;
  char bitPath [MAX_STR_LEN],   // -b [./bitfile], path to *.lvbitx bitfile
       bitSig [MAX_STR_LEN],    // -s [signature], signature of the bitfile
       rioDev [MAX_STR_LEN];    // -r [RIO0], RIO resource string to open (e.g. RIO0 or rio://mysystem/RIO)

  // Process command line arguments and set above flags
  while ((opt = getopt(argc, argv, "Hlathb:s:r:")) != -1)
  {
    switch (opt)
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
        fprintf(stderr, HELP_STRING);
        exit(EXIT_SUCCESS);
      case 'b':
        strcpy(bitPath, optarg);
        break;
      case 's':
        strcpy(bitSig, optarg);
        break;
      case 'r':
        strcpy(rioDev, optarg);
        break;
      default:
        abort();
    }
  }


  // initialize NI FPGA interfaces; use status variable for error handling
  fprintf(stderr, "Initializing NI FPGA: ");
  checkStatus(NiFpga_Initialize());
  NiFpga_Session session;

  // Download bitfile to target; get path to bitfile
  // TODO: change to full path to bitfile as necessary
  fprintf(stderr, "Downloading bitfile ");
  checkStatus(NiFpga_Open(bitPath, bitSig, rioDev, 0, &session));
  fprintf(stderr, "DONE\n");

  struct cudaDeviceProp d_props;
  int device = 0; // device specifies which GPU should be used. Change this to the index of the desired GPU, if necessary
  checkStatus(cudaGetDevice(&device));
  checkStatus(cudaGetDeviceProperties(&d_props, device));
  if (d_props.major < 2) {
    fprintf(stderr, "CUDA Error: This example requires a CUDA device with architecture SM2.0 or higher\n");
    exit(EXIT_FAILURE);
  }

  /*
   * Allocate CUDA Memory for FFT and log scaling operations
   * As well, allocate complex CUDA Memory for R2C result
  */
  fprintf(stderr, "Allocating CUDA and Host Memory: \n");
  int32_t *init_signal;         // initial storage for non-scaled data input
  cufftReal *tmp_result1;       // temp storage for scaling data input
  cuComplex *gpu_result;     // Where CUFFT will be stored
  cuComplex *hcomp_data;     // Where host memory will recieve complex data result

  checkStatus(cudaMalloc((void **)&init_signal, sizeof(*init_signal)*(16*SAMPLES)));
  checkStatus(cudaMalloc((void **)&tmp_result1, sizeof(cufftReal)*SAMPLES));
  checkStatus(cudaMalloc((void **)&gpu_result, sizeof(cuComplex)*COMPLEX_SIZE));

  hcomp_data = (cuComplex*) malloc(sizeof(cuComplex)*COMPLEX_SIZE);
  if (hcomp_data == NULL) {
    fprintf(stderr, "Host Error: Host failed to allocate memory\n");
    return -1;
  }

  /*
   * Make CUFFT plan for 1D Real-to-Complex FFT
   * also link data path to CUDA device
   */
  cufftHandle plan;
  checkStatus(cufftCreate(&plan));
  checkStatus(cufftPlan1d(&plan, SAMPLES, CUFFT_R2C, 1));

  // Configure P2P FIFO between FlexRIO and GPU using NVIDIA GPU Direct
  if (viahostflag == 1) { /* Host transfer */
    checkStatus(NiFpga_ConfigureFifo(session, NiFpga_FPGA_Main_TargetToHostFifoI32_T2HDMAFIFO,
          (size_t)SAMPLES));
  }
  else { /* P2P via RDMA */
    checkStatus(NiFpga_ConfigureFifoBuffer(session, NiFpga_FPGA_Main_TargetToHostFifoI32_T2HDMAFIFO,
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
    fprintf(stderr, "Enabling Low-Pass FIR Filter on FlexRIO\n");
    NiFpga_WriteBool(session, NiFpga_FPGA_Main_ControlBool_LPFEnable, NiFpga_True);
  }
  else
    NiFpga_WriteBool(session, NiFpga_FPGA_Main_ControlBool_LPFEnable, NiFpga_False);

  if (awgnflag == 1) {
    fprintf(stderr, "Adding White Gaussian Noise to Signal\n");
    NiFpga_WriteBool(session, NiFpga_FPGA_Main_ControlBool_AWGNEnable, NiFpga_True);
  }
  else
    NiFpga_WriteBool(session, NiFpga_FPGA_Main_ControlBool_AWGNEnable, NiFpga_False);

  /*
   * DMA (or copy from host) signal to GPU and execute FFT plan
   */
  if (viahostflag == 1) {
    int32_t * h_data = NULL; // ptr for when transferring data to host first
    fprintf(stderr, "Copy host memory signal to CUDA Device\n");
    h_data = (int32_t *)malloc(SAMPLES * sizeof(int32_t));
    checkStatus(NiFpga_ReadFifoI32(session, NiFpga_FPGA_Main_TargetToHostFifoI32_T2HDMAFIFO,
          h_data, (size_t)SAMPLES, 5000, NULL));

    if (timeflag == 1) {
      fprintf(stderr, "Writing time-domain signal to TimeSignal.dat\n");
      FILE *f = fopen("TimeSignal.dat", "w");
      for(int i = 0; i < SAMPLES; i++)
      {
        if (i == 0)
          fprintf(f, "# X Y\n");
        // Write real component to file
        fprintf(f, "%d %d\n", i, h_data[i]);
      }
      fclose(f);
    }
    if (cudaMemcpy(init_signal, h_data, (SAMPLES*sizeof(int32_t)), cudaMemcpyHostToDevice) != cudaSuccess) {
      fprintf(stderr, "CUDA Error: Device failed to copy host memory to device\n");
      return -1;
    }
    free(h_data);
  }
  else {
    fprintf(stderr, "DMA'ing FlexRIO data to GPU\n");
    size_t elemsAcquired, elemsRemaining;
    checkStatus(NiFpga_AcquireFifoReadElementsI32(session,
          NiFpga_FPGA_Main_TargetToHostFifoI32_T2HDMAFIFO, &init_signal,
          SAMPLES, 5000, &elemsAcquired, &elemsRemaining));
    fprintf(stderr, "%d samples acquired with %d elements remaining in FIFO\n", elemsAcquired, elemsRemaining);
  }

  /*
   * Start FFT on GPU
  */
  fprintf(stderr, "Executing CUFFT Plan...\n");

  // create timers
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  float elapsedTime=0;

  checkStatus(cudaEventRecord(start, 0));

  // Convert input signal from real to complex
  ConvertInputToComplex<<<32, 128>>>(init_signal, tmp_result1);
  checkStatus(cudaGetLastError());

  // Execute FFT on data
  checkStatus(cufftExecR2C(plan, tmp_result1, gpu_result));
  checkStatus(cudaGetLastError());

  // Stop and record time taken
  checkStatus(cudaEventRecord(end, 0));
  // Sync and wait for completion
  checkStatus(cudaDeviceSynchronize());
  checkStatus(cudaEventSynchronize(end));
  checkStatus(cudaEventElapsedTime(&elapsedTime, start, end));
  fprintf(stderr, "FFT took %fms to complete on GPU\n", elapsedTime);

  // Copy FFT result back to host)
  fprintf(stderr, "Copying CUFFT result back to host memory...\n");
  checkStatus(cudaMemcpy(hcomp_data, gpu_result, SAMPLES, cudaMemcpyDeviceToHost));

  /*
   * Write resulting data to file
   */
  fprintf(stderr, "Writing signal to stdout:\n\n");
  for(int32_t i = 0; i < COMPLEX_SIZE; i++)
  {
    if (i == 0)
      fprintf(stdout, "# X Y\n");
    // Write real component to file
    // TODO: Implement power conversion as CUFFT Callback
    fprintf(stdout, "%d %f\n", i, 20.0f*log(hcomp_data[i].y));
  }

  // Close out CUFFT plan(s) and free CUDA memory
  fprintf(stderr, "Closing CUFFT and freeing CUDA memory...\n");
  checkStatus(cufftDestroy(plan));
  checkStatus(cudaFree(init_signal));
  checkStatus(cudaFree(tmp_result1));
  checkStatus(cudaFree(gpu_result));
  free(hcomp_data);

  // Clean up NVIDIA Driver state
  cudaDeviceReset();

  // Close NI FPGA References; must be last NiFpga calls
  fprintf(stderr, "Stopping NI FPGA...\n");
  checkStatus(NiFpga_Close(session, 0));
  checkStatus(NiFpga_Finalize());

  return EXIT_SUCCESS;
}
