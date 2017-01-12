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


int main(int argc, char **argv)
{

  /*
   * Initialization of program values, NI FPGA and CUDA
  */

  // program flag declrarations
  int viahostflag = 0;  // -H, do transfers and operations via host
  int lpfflag = 0;      // -l, pass signal through Low-Pass FIR on FlexRIO
  int awgnflag = 0;     // -a, add white gaussian noise to signal on FlexRIO
  int c;

  // Process command line arguments and set above flags
  while ((c = getopt(argc, argv, "Hla")) != -1)
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

  // Allocate CUDA Memory for FFT and log scaling operations
  // As well, allocate complex CUDA Memory for R2C result
  printf("Allocating CUDA and Host Memory: \n");
  cufftComplex *gpu_mem;        // ptr for CUDA Device Memory
  cufftComplex *hcomp_data;     // ptr for host memory to recieve complex data
  if (cudaMalloc((void**)&gpu_mem, sizeof(cufftComplex)*NX) != cudaSuccess)
  {
    printf("CUDA Error: Failed to allocate memory on device\n");
    return -1;
  }
  hcomp_data = (cufftComplex*) malloc(sizeof(cufftComplex)*SAMPLES);
  if (hcomp_data == NULL)
  {
    printf("Host Error: Host failed to allocate memory\n");
    return -1;
  }

  // Make CUFFT plan for 1D Real-to-Complex FFT
  cufftHandle plan;
  if (cufftPlan1d(&plan, NX, CUFFT_R2C, 1) != CUFFT_SUCCESS)
  {
    printf("CUDA Error: CUFFT Plan creation failed\n");
    return -1;
  }

  // Configure P2P FIFO between FlexRIO and GPU using NVIDIA GPU Direct
  // CHECKSTAT(NiFpga_ConfigureFifoBuffer(session, NiFpga_FPGA_Main_TargetToHostFifoI32_T2HDMAFIFO, (uint64_t)gpu_mem, NX*NFRAMES, NULL, NiFpga_DmaBufferType_NvidiaGpuDirectRdma)); 
  
  if (viahostflag == 1)
    CHECKSTAT(NiFpga_ConfigureFifo(session, NiFpga_FPGA_Main_TargetToHostFifoI32_T2HDMAFIFO, (size_t)10*SAMPLES));
  
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

    if (cudaMemcpy((cufftReal*)gpu_mem, h_data, SAMPLES, cudaMemcpyHostToDevice) != cudaSuccess)
    {
      printf("CUDA Error: Device failed to copy host memory to device\n");
      return -1;
    }
  }
  
  // Execute FFT
  printf("Executing CUFFT Plan...\n");
  if (cufftExecR2C(plan, (cufftReal*)gpu_mem, gpu_mem) != CUFFT_SUCCESS) 
  {
    printf("CUFFT Error: Execution of FFT Plan failed\n");
    return -1;
  }

  // Sync and wait for completion
  if (cudaDeviceSynchronize() != cudaSuccess)
  {
    printf("CUDA Error: Device failed to synchronize\n");
    return -1;
  }

  // Copy FFT result back to host)
  printf("Copying CUFFT result back to host memory...\n");
  if (cudaMemcpy(hcomp_data, gpu_mem, SAMPLES, cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    printf("CUDA Error: Device failed to copy data back to host memory\n");
    return -1;
  }


  /* 
   * Write resulting data to file
   */
  printf("Writing signal to SimSignal.dat\n");
  FILE *f = fopen("SimSignal.dat", "w");
  for(int i=0; i<SAMPLES; i++)
  {
    if (i==0)
      fprintf(f, "# X Y\n");
    // Write real component to file
    fprintf(f, "%d %d\n", i, hcomp_data[i].x);
  }
  fclose(f);

  // Close out CUFFT plan(s) and free CUDA memory
  printf("Closing CUFFT and freeing CUDA memory...\n");
  cufftDestroy(plan);
  cudaFree(gpu_mem);

  // Close NI FPGA References; must be last NiFpga calls
  printf("Stopping NI FPGA...\n");
  CHECKSTAT(NiFpga_StopFifo(session, NiFpga_FPGA_Main_TargetToHostFifoI32_T2HDMAFIFO));
  CHECKSTAT(NiFpga_Close(session, 0));
  CHECKSTAT(NiFpga_Finalize());

  return 0;

}
