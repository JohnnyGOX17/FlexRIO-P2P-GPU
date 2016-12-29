/*
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
* Date: 12/14/2016
*/

#include <stdio.h>
#include <sys/time.h>
#include <sys/times.h>
#include "NiFpga_HighThroughputStreamingFPGAPXIe797xR.h"

#define NX (128*8192) //4096 (64k pages)

// use inline definition for error checking to allow easy app exit
#define CHECKSTAT(stat) if (stat != 0) { printf("%d: Error: %d\n", __LINE__, stat); return 1; }

// #TODO path to bitfile for FlexRIO target; change path as necessary
#define PATH_TO_BITFILE "/home/nitest/FlexRIO_P2P_GPU/High-Throughput Streaming/NiFpga_HighThroughputStreamingFPGAPXIe797XR.lvbitx"

int main(int argc, char **argv)
{
  // initialize NI FPGA interfaces; use status variable for error handling
  printf("Initializing NI FPGA...\n");
  CHECKSTAT(NiFpga_Initialize());
  NiFpga_Session session = NULL;
  
  // Download bitfile to target; get path to bitfile
  CHECKSTAT(NiFpga_Open(PATH_TO_BITFILE, NiFpga_HighThroughputStreamingFPGAPXIe797xR_Signature, "RIO0", 0, &session));
   
  // Allocate CUDA memory 

  
  // Close NI FPGA References; must be last NiFpga calls
  CHECKSTAT(NiFpga_Close(session, 0));
  CHECKSTAT(NiFpga_Finalize());

  return 0;

}
