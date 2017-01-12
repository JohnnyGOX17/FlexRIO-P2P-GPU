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

// includes- project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include "NiFpga_FPGA_Main.h"

#define NX              (128*8192) //4096 (64k pages)
#define NFRAMES         3
#define BATCH_NFRAMES   1000
#define CUDA_NTHREADS   1024
#define CUDA_NBLOCKS    (NX/CUDA_NTHREADS)

// use inline definition for error checking to allow easy app exit
#define CHECKSTAT(stat) if (stat != 0) { printf("%d: Error: %d\n", __LINE__, stat); return 1; }

// keep datatypes uniform between FIFOs and operations
typedef uint64_t fifotype;


int main(int argc, char **argv)
{

  // initialize NI FPGA interfaces; use status variable for error handling
  printf("Initializing NI FPGA...\n");
  CHECKSTAT(NiFpga_Initialize());
  NiFpga_Session session;
  
  // Download bitfile to target; get path to bitfile
  // TODO: change to full path to bitfile as necessary
  CHECKSTAT(NiFpga_Open("/home/nitest/FlexRIO-P2P-GPU/GPU_FFT/NiFpga_FPGA_Main.lvbitx", NiFpga_FPGA_Main_Signature, "RIO0", 0, &session));
  

  // Configure P2P FIFO between FlexRIO and GPU using NVIDIA GPU Direct
  // CHECKSTAT(NiFpga_ConfigureFifoBuffer(session, NiFpga_FPGA_Main_TargetToHostFifoI32_T2HDMAFIFO, (uint64_t)gpu_mem, NX*NFRAMES, NULL, NiFpga_DmaBufferType_NvidiaGpuDirectRdma)); 
  
  CHECKSTAT(NiFpga_ConfigureFifo(session, NiFpga_FPGA_Main_TargetToHostFifoI32_T2HDMAFIFO, (size_t)10000));
  
  // Set RMS Noise value of AWGN Algorithm on FlexRIO (out of i16 full scale)
  NiFpga_WriteU16(session, NiFpga_FPGA_Main_ControlU16_RMSNoise, 2048);

  // Reset FPGA algorithms and clear FIFOs 
  NiFpga_WriteBool(session, NiFpga_FPGA_Main_ControlBool_aReset, NiFpga_True);
  NiFpga_WriteBool(session, NiFpga_FPGA_Main_ControlBool_aReset, NiFpga_False);

  int32_t data [1000];
  CHECKSTAT(NiFpga_ReadFifoI32(session, NiFpga_FPGA_Main_TargetToHostFifoI32_T2HDMAFIFO, data, (size_t)1000, 5000, NULL));

  FILE *f = fopen("SimSignal.dat", "w");
  for(int i=0; i<1000; i++)
  {
    if (i==0)
      fprintf(f, "# X Y\n");
    fprintf(f, "%d %d\n", i, data[i]);
  }
  fclose(f);

  // Close NI FPGA References; must be last NiFpga calls
  printf("Stopping NI FPGA...\n");
  CHECKSTAT(NiFpga_StopFifo(session, NiFpga_FPGA_Main_TargetToHostFifoI32_T2HDMAFIFO));
  CHECKSTAT(NiFpga_Close(session, 0));
  CHECKSTAT(NiFpga_Finalize());

  return 0;

}