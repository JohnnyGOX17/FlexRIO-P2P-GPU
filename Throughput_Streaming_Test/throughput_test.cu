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
#include "NiFpga_FPGA_main.h"

#define NX              (128*8192) //4096 (64k pages)
#define NFRAMES         3
#define BATCH_NFRAMES   1000
#define CUDA_NTHREADS   1024
#define CUDA_NBLOCKS    (NX/CUDA_NTHREADS)

// use inline definition for error checking to allow easy app exit
#define CHECKSTAT(stat) if (stat != 0) { printf("%d: Error: %d\n", __LINE__, stat); return 1; }

// keep datatypes uniform between FIFOs and operations
typedef uint64_t fifotype;


__global__ void inplace_add(unsigned long long *in)
{
  extern __shared__ fifotype sdata[];

  // each thread copies an element into shared memory
  unsigned int tid = threadIdx.x;
  sdata[tid] = in[tid + blockIdx.x*blockDim.x];
  __syncthreads();

  // Now each block figures out the sum of shared memory
  int stride;
  for (stride=1; stride < blockDim.x; stride *= 2)
  {
    int idx = 2*stride*tid;
    if (idx+stride < blockDim.x)
      sdata[idx] += sdata[idx+stride];

    __syncthreads();
  }

  // each block writes out result
  if (tid == 0)
    in[blockIdx.x] = sdata[0];
}


int main(int argc, char **argv)
{

  // initialize NI FPGA interfaces; use status variable for error handling
  printf("Initializing NI FPGA...\n");
  CHECKSTAT(NiFpga_Initialize());
  NiFpga_Session session;
  
  // Download bitfile to target; get path to bitfile
  // TODO: change to full path to bitfile as necessary
  CHECKSTAT(NiFpga_Open("/home/nitest/FlexRIO-P2P-GPU/Throughput_Streaming_Test/NiFpga_FPGA_main.lvbitx", NiFpga_FPGA_main_Signature, "RIO0", 0, &session));
  
  // Allocate CUDA memory; the CUDA device will operate on frames of samples so
  // we willl allocate two frames worth of space 
  printf("Allocating CUDA Memory: ");
  fifotype *gpu_mem;
  cudaError_t cuerr = cudaMalloc(&gpu_mem, sizeof(fifotype)*NX*NFRAMES);
  printf("%p\n", gpu_mem);

  // Configure P2P FIFO between FlexRIO and GPU using NVIDIA GPU Direct
  CHECKSTAT(NiFpga_ConfigureFifoBuffer(session, NiFpga_FPGA_main_TargetToHostFifoU64_FlexRIO_FIFO, (uint64_t)gpu_mem, NX*NFRAMES, NULL, NiFpga_DmaBufferType_NvidiaGpuDirectRdma)); 
  
  //CHECKSTAT(NiFpga_StartFifo(session, NiFpga_FPGA_main_TargetToHostFifoU64_FlexRIO_FIFO));
  
  // Setup batch size to be very large
  NiFpga_WriteU64(session, NiFpga_FPGA_main_ControlU64_BatchSize, (long long)NX*(long long)BATCH_NFRAMES+1000);

  // Start transferring data
  printf("Transferring Data");
  // create rising edge start signal
  NiFpga_WriteBool(session, NiFpga_FPGA_main_ControlBool_StartTransfer, NiFpga_False);
  NiFpga_WriteBool(session, NiFpga_FPGA_main_ControlBool_StartTransfer, NiFpga_True);


  size_t tot_bytes = 0; // total bytes transferred
  struct timeval start_time;
  gettimeofday(&start_time, NULL);

  uint64_t running_count = 0;
  fifotype *datap;
  size_t elems_acquired, elems_remaining = NX;

  // Start recording the user/system times
  struct tms start_tms, end_tms;
  times(&start_tms);

  // Process batch frames
  int i;
  for (i=0; i<BATCH_NFRAMES; i++)
  {
    // Acquire data from FlexRIO -> GPU
    tot_bytes += sizeof(fifotype)*(NX);
    CHECKSTAT(NiFpga_AcquireFifoReadElementsU64(session, NiFpga_FPGA_main_TargetToHostFifoU64_FlexRIO_FIFO, &datap, NX, 3000, &elems_acquired, &elems_remaining));
    
    // Add frame to running count
    {

      inplace_add<<<CUDA_NBLOCKS,CUDA_NTHREADS,sizeof(fifotype)*CUDA_NTHREADS>>>((unsigned long long *)datap);

      int nresults = CUDA_NBLOCKS;
      while (nresults > 1)
      {
        int n_threads_to_run = (nresults > 1024)?1024:nresults;
        int n_blocks_to_run  = (nresults > 1024)?nresults/1024:1;
        inplace_add<<<n_blocks_to_run,n_threads_to_run,sizeof(fifotype)*n_threads_to_run>>>((unsigned long long *)datap);
        nresults /= 1024;
      }

      // Find result of add
      uint64_t framecnt = 0;
      cudaMemcpy(&framecnt, datap, sizeof(uint64_t), cudaMemcpyDeviceToHost);
      running_count += framecnt;
    }

    // Release this frame so FlexRIO can fill it while we process next frame
    NiFpga_ReleaseFifoElements(session, NiFpga_FPGA_main_TargetToHostFifoU64_FlexRIO_FIFO, elems_acquired);
  
  }

  printf("Final count (on GPU) is %llu\n", running_count);

  // calculate time taken and data transfer rates
  struct timeval stop_time;
  gettimeofday(&stop_time, NULL);
  int dt = (stop_time.tv_sec - start_time.tv_sec)*1000 + (stop_time.tv_usec - start_time.tv_usec)/1000;
  if (stop_time.tv_usec < start_time.tv_usec)
    dt += 1000;

  printf("Read %llu bytes in %d ms (%.2fMB/s)\n", tot_bytes, dt, ((float)tot_bytes)/(float)dt / 1000.0);

  times(&end_tms);
  int cpu_us = (end_tms.tms_stime + end_tms.tms_cstime - start_tms.tms_stime - start_tms.tms_cstime + end_tms.tms_utime + end_tms.tms_cutime - start_tms.tms_utime - start_tms.tms_cutime)*1000000/CLOCKS_PER_SEC;

  printf("%d usertime to %.2f us/frame\n", cpu_us, (float)cpu_us / BATCH_NFRAMES);

  // Recompute final sum count locally to verify data integrity
  uint64_t checksum = 0;
  uint64_t j = 0;
  for (j=0; j<(long long)NX*(long long)BATCH_NFRAMES; j++)
  {
    checksum +=j;
  }

  printf("Final checksum on CPU: %llu\n", checksum);
  if (checksum != running_count)
    printf("*** WARNING: FINAL COUNT DOES NOT MATCH CHECKSUM.\n");
  
  cudaFree(gpu_mem);

  // Close NI FPGA References; must be last NiFpga calls
  printf("Stopping NI FPGA...\n");
  CHECKSTAT(NiFpga_StopFifo(session, NiFpga_FPGA_main_TargetToHostFifoU64_FlexRIO_FIFO));
  CHECKSTAT(NiFpga_Close(session, 0));
  CHECKSTAT(NiFpga_Finalize());

  return 0;

}
