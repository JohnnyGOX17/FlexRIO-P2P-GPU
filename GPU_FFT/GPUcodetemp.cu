/**
* GPU FFT Example 
* 
* This example showcases the FFT capabilities of a P2P NVIDIA GPU with a 
* FlexRIO device. In this case, the FlexRIO module is creating a simulated CW
* tone and can perform Additive White Gaussian Noise (AWGN) and/or a Finite
* Impulse Response (FIR) Low-Pass Filter (LPF) on the generated signal. The
* signal is then sent to the GPU where a parallelized FFT takes place using 
* NVIDIA's CUFFT library and some logarithmic conversion to calculate the 
* power spectrum of the signal. This resulting signal is then piped to a file
* and/or a GNUplot host application for data logging and plotting.
*
* For more information on NI FPGA functions, see the NI FPGA Interface C API 
* Help. For more information on NVIDIA CUDA functions and operation, see the
* help files included with the NVIDIA CUDA Driver.
*
* Date:         1/10/2016
* Author:       John Gentile
*/

// includes- system
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <sys/times.h>
#include <ctype.h>
#include <unistd.h>

// includes- project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include "GPU_FFT.h"

#define TILE_DIM        32
#define BLOCK_ROWS      8
/*
 * Custom Kernel Implementations
 */

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

// Function for Convolving Output of FFT
__global__ void ConvolveAndTranspose(
    const cufftComplex * __restrict__ dataIn, 
    cufftComplex * __restrict__ dataOut, 
    const cufftComplex * __restrict__ filter)
{

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int yBase = blockIdx.y * TILE_DIM + threadIdx.y;

  if(x < COMPLEX_SIZE)
  {
    for(int j=0; j < TILE_DIM; j+= BLOCK_ROWS)
    {
      int y = yBase + j;
      if (y >= BATCH_SIZE) break;
      cufftComplex value = ComplexMul(dataIn[y * COMPLEX_SIZE + x], filter[x]);
      dataOut[x*BATCH_SIZE + y] = value;
    }
  }
}


/*
 * Main Program Execution
 */
int main(int argc, char **argv)
{

  /*
   * Initialization of program values, NI FPGA and CUDA
  */

  // program flag declrarations
  int viahostflag = 0;  // -H, do transfers and operations via host
  int lpfflag = 0;      // -l, pass signal through Low-Pass FIR on FlexRIO
  int awgnflag = 0;     // -a, add white gaussian noise to signal on FlexRIO
  int timeflag =0;      // -t, write time-domain signal (only if -H is set)
  int c;

  // Process command line arguments and set above flags
  while ((c = getopt(argc, argv, "Hlat")) != -1)
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
  CHECKSTAT(NiFpga_Open("/home/nitest/FlexRIO-P2P-GPU/GPU_FFT/NiFpga_FPGA_Main.lvbitx", NiFpga_FPGA_Main_Signature, "RIO0", 0, &session));
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
  cufftComplex *tmp_result2;    // temp storage for convolution of FFT data
  cufftComplex *gpu_result;     // Where CUFFT will be stored
  cufftComplex *hcomp_data;     // Where host memory will recieve complex data result
  cufftComplex *rfilter;        // Filter memory space

  checkCudaErrors(cudaMalloc((void **)&init_signal, sizeof(int32_t)*SAMPLES));
  checkCudaErrors(cudaMalloc((void **)&tmp_result1, sizeof(cufftReal)*SAMPLES));
  checkCudaErrors(cudaMalloc((void **)&tmp_result2, sizeof(cufftComplex)*SAMPLES));
  checkCudaErrors(cudaMalloc((void **)&gpu_result, sizeof(cufftComplex)*COMPLEX_SIZE));
  checkCudaErrors(cudaMalloc((void **)&rfilter, sizeof(cufftComplex)*COMPLEX_SIZE));
  cufftComplex * tempfilter = (cufftComplex*) malloc(sizeof(cufftComplex)*COMPLEX_SIZE);

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
  checkCudaErrors(cufftPlan1d(&plan, NX, CUFFT_R2C, 1));

  // Configure P2P FIFO between FlexRIO and GPU using NVIDIA GPU Direct
  // CHECKSTAT(NiFpga_ConfigureFifoBuffer(session, NiFpga_FPGA_Main_TargetToHostFifoI32_T2HDMAFIFO, (uint64_t)gpu_mem, NX*NFRAMES, NULL, NiFpga_DmaBufferType_NvidiaGpuDirectRdma)); 
  
  if (viahostflag == 1)
    CHECKSTAT(NiFpga_ConfigureFifo(session, NiFpga_FPGA_Main_TargetToHostFifoI32_T2HDMAFIFO, (size_t)10*SAMPLES));
  
  /*
   * Set NI FPGA Control/Indicator Values
   */
  // Set RMS Noise value of AWGN Algorithm on FlexRIO (out of i16 full scale)
  NiFpga_WriteU16(session, NiFpga_FPGA_Main_ControlU16_RMSNoise, 2048);

  // Reset FPGA algorithms and clear FIFOs 
  NiFpga_WriteBool(session, NiFpga_FPGA_Main_ControlBool_aReset, NiFpga_True);
  NiFpga_WriteBool(session, NiFpga_FPGA_Main_ControlBool_aReset, NiFpga_False);

  if (lpfflag == 1)
  {
    printf("Enabling Low-Pass FIR Filter on FlexRIO\n");
    NiFpga_WriteBool(session, NiFpga_FPGA_Main_ControlBool_LPFEnable, NiFpga_True);
  }
  if (awgnflag == 1)
  {
    printf("Adding White Gaussian Noise to Signal\n");
    NiFpga_WriteBool(session, NiFpga_FPGA_Main_ControlBool_AWGNEnable, NiFpga_True);
  }

  
  /*
   * DMA (or copy from host) signal to GPU and execute FFT plan
   */
  if (viahostflag == 1)
  {
    printf("Copy host memory signal to CUDA Device\n");
    int32_t h_data [SAMPLES];
    CHECKSTAT(NiFpga_ReadFifoI32(session, NiFpga_FPGA_Main_TargetToHostFifoI32_T2HDMAFIFO, h_data, (size_t)SAMPLES, 5000, NULL));

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
    if (cudaMemcpy(init_signal, h_data, SAMPLES, cudaMemcpyHostToDevice) != cudaSuccess)
    {
      printf("CUDA Error: Device failed to copy host memory to device\n");
      return -1;
    }
  }
  
  dim3 block(TILE_DIM, BLOCK_ROWS);
  dim3 grid((COMPLEX_SIZE + block.x - 1)/block.x, (BATCH_SIZE + block.y -1)/block.y);

  // Make Rand() based filter for convolution
  for (size_t i=0; i<COMPLEX_SIZE; i++)
  {
    srand(42);
    tempfilter[i].x = rand() / (float)RAND_MAX;
    tempfilter[i].y = 0.5f;
  }
  checkCudaErrors(cudaMemcpy(rfilter, tempfilter, COMPLEX_SIZE, cudaMemcpyHostToDevice));

  // Transform input data using our customer kernel
  ConvertInputToComplex<<<32, 128>>>(init_signal, tmp_result1);
  checkCudaErrors(cudaGetLastError());

  // Execute FFT
  printf("Executing CUFFT Plan...\n");
  checkCudaErrors(cufftExecR2C(plan, tmp_result1, tmp_result2));
  checkCudaErrors(cudaGetLastError());

  // Convolve FFT output data
  ConvolveAndTranspose<<<grid, block>>>(tmp_result2, gpu_result, rfilter);
  checkCudaErrors(cudaGetLastError());

  // Sync and wait for completion
  checkCudaErrors(cudaDeviceSynchronize());

  // Copy FFT result back to host)
  printf("Copying CUFFT result back to host memory...\n");
  checkCudaErrors(cudaMemcpy(hcomp_data, gpu_result, SAMPLES, cudaMemcpyDeviceToHost));


  /* 
   * Write resulting data to file
   */
  printf("Writing signal to SimSignal.dat\n");
  FILE *f = fopen("SimSignal.dat", "w");
  for(int i=0; i<COMPLEX_SIZE; i++)
  {
    if (i==0)
      fprintf(f, "# X Y\n");
    // Write real component to file
    fprintf(f, "%d %d\n", i, 20*log(hcomp_data[i].y));
  }
  fclose(f);

  // Close out CUFFT plan(s) and free CUDA memory
  printf("Closing CUFFT and freeing CUDA memory...\n");
  checkCudaErrors(cufftDestroy(plan));

  checkCudaErrors(cudaFree(init_signal));
  checkCudaErrors(cudaFree(tmp_result1));
  checkCudaErrors(cudaFree(tmp_result2));
  checkCudaErrors(cudaFree(gpu_result));
  checkCudaErrors(cudaFree(rfilter));
  free(hcomp_data);

  // Clean up NVIDIA Driver state
  cudaDeviceReset();

  // Close NI FPGA References; must be last NiFpga calls
  printf("Stopping NI FPGA...\n");
  CHECKSTAT(NiFpga_StopFifo(session, NiFpga_FPGA_Main_TargetToHostFifoI32_T2HDMAFIFO));
  CHECKSTAT(NiFpga_Close(session, 0));
  CHECKSTAT(NiFpga_Finalize());

  return 0;

}