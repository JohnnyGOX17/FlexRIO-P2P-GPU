/**
* High-Throughput Streaming Example 
* 
* This example shows a basic example of Peer-to-Peer DMA transfers between
* an NI FlexRIO module and a NVIDIA GPU that is capable of NVIDIA GPU Direct
* technology. This example will measure throughput between the two devices and
* report the measurement back to the user.
*
* For more information on NI FPGA functions, see the NI FPGA Interface C API 
* Help. For more information on NVIDIA CUDA functions and operation, see the
* help files included with the NVIDIA CUDA Driver.
*
* Date:         12/14/2016
* Author:       John Gentile
*/

#include <stdio.h>
#include <sys/time.h>
#include <sys/times.h>
#include "NiFpga_HighThroughputStreamingFPGAPXIe797xR.h"

#define NX              (128*8192) //4096 (64k pages)
#define NFRAMES         3

// use inline definition for error checking to allow easy app exit
#define CHECKSTAT(stat) if (stat != 0) { printf("%d: Error: %d\n", __LINE__, stat); return 1; }

// keep datatypes uniform between FIFOs and operations
typedef int16_t fifotype;


int main(int argc, char **argv)
{
  // initialize NI FPGA interfaces; use status variable for error handling
  printf("Initializing NI FPGA...\n");
  CHECKSTAT(NiFpga_Initialize());
  NiFpga_Session session;
  
  // Download bitfile to target; get path to bitfile
  // TODO: change to full path to bitfile as necessary
  CHECKSTAT(NiFpga_Open("/home/nitest/FlexRIO-P2P-GPU/High-Throughput_Streaming/NiFpga_HighThroughputStreamingFPGAPXIe797xR.lvbitx", NiFpga_HighThroughputStreamingFPGAPXIe797xR_Signature, "RIO0", 0, &session));
  
  NiFpga_WriteBool(session, NiFpga_HighThroughputStreamingFPGAPXIe797xR_ControlBool_inputStart, NiFpga_False);
  NiFpga_WriteBool(session, NiFpga_HighThroughputStreamingFPGAPXIe797xR_ControlBool_outputStart, NiFpga_False);
  // Commented out as FPGA VI is usually already running
  // CHECKSTAT(NiFpga_Run(session, 0));

  uint16_t maxIn, maxOut;
  NiFpga_ReadU16(session, NiFpga_HighThroughputStreamingFPGAPXIe797xR_IndicatorU16_inputMaxThroughput, &maxIn);
  NiFpga_ReadU16(session, NiFpga_HighThroughputStreamingFPGAPXIe797xR_IndicatorU16_outputMaxThroughput, &maxOut);
  printf("Max Input Throughput: %u MB/s\nMax Output Throughput: %u MB/s\n", maxIn, maxOut);


  // Allocate CUDA memory; the CUDA device will operate on frames of samples so
  // we willl allocate two frames worth of space 
  printf("Allocating CUDA Memory: ");
  fifotype *gpu_mem;
  cudaError_t cuerr = cudaMalloc(&gpu_mem, sizeof(fifotype)*NX*NFRAMES);
  printf("%p\n", gpu_mem);

  // Configure P2P FIFO between FlexRIO and GPU using NVIDIA GPU Direct
  CHECKSTAT(NiFpga_ConfigureFifoBuffer(session, NiFpga_HighThroughputStreamingFPGAPXIe797xR_TargetToHostFifoI16_InputFIFO, (uint64_t)gpu_mem, NX*NFRAMES, NULL, NiFpga_DmaBufferType_NvidiaGpuDirectRdma)); 






  
  CHECKSTAT(NiFpga_StartFifo(session, NiFpga_HighThroughputStreamingFPGAPXIe797xR_TargetToHostFifoI16_InputFIFO));
  
  
  // Close NI FPGA References; must be last NiFpga calls
  printf("Stopping NI FPGA...\n");
  CHECKSTAT(NiFpga_StopFifo(session, NiFpga_HighThroughputStreamingFPGAPXIe797xR_TargetToHostFifoI16_InputFIFO));
  CHECKSTAT(NiFpga_Close(session, 0));
  CHECKSTAT(NiFpga_Finalize());

  return 0;

}
