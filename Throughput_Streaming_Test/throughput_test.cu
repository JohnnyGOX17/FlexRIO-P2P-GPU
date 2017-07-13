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
#include <unistd.h>
#include <sys/time.h>
#include <sys/times.h>
#include "NiFpga_FPGA_main.h"

#define MAX_STR_LEN     256
#define HELP_STRING \
  "Usage: throughput_test [OPTIONS]\n\n" \
  "\t-h,\tDisplay help string\n" \
  "\t-b [./bitfile],\tPath to *.lvbitx bitfile\n" \
  "\t-s [signature],\tSignature of the bitfile\n" \
  "\t-r [RIO0],\tRIO resource string to open (e.g. RIO0 or rio://mysystem/RIO)\n"

// use inline method for error checking to allow easy app exit
inline void CHECKSTAT(int32_t stat)
{
  if (stat != 0) {
    printf("%d: Error: %d\n", __LINE__, stat);
    exit(EXIT_FAILURE);
  }
  return;
}
// keep datatypes uniform between FIFOs and operations
typedef uint64_t fifotype;

uint64_t SAMPLES = 8192*128; // number of samples per batch to use for checksum calc
uint32_t CUDA_NTHREADS = 1024;
uint32_t CUDA_NBLOCKS = SAMPLES / CUDA_NTHREADS;
uint32_t BATCH_NFRAMES = 1000; // SAMPLES*BATCH_NFRAMES represents the total number of elements to transfer

/*
 * Simple kernel function to sum all of shared memory buffer across
 * CUDA_NTHREADS to calculate a checksum on the data transferred from FlexRIO
 * to CUDA GPU
 */
__global__ void inplace_add(uint64_t * in)
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

/* Spinner for command-line prompt */
char spin()
{
  static const char sp[] = "/-\\|";
  static int spi = 0;

  if (spi >= 3)
    spi=0;
  else
    spi++;
  return sp[spi];
}

int main(int argc, char **argv)
{
  /*
  * Initialization of values passed to NI FPGA
  */
  int opt;
  char bitPath [MAX_STR_LEN],   // -b [./bitfile], path to *.lvbitx bitfile
       bitSig [MAX_STR_LEN],    // -s [signature], signature of the bitfile
       rioDev [MAX_STR_LEN];    // -r [RIO0], RIO resource string to open (e.g. RIO0 or rio://mysystem/RIO)

  // Process command line arguments and set above flags
  while ((opt = getopt(argc, argv, "hb:s:r:")) != -1)
  {
    switch (opt)
    {
      case 'h':
        printf(HELP_STRING);
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
        break;
    }
  }

  // initialize NI FPGA interfaces; use status variable for error handling
  printf("Initializing NI FPGA: Downloading bitfile");
  CHECKSTAT(NiFpga_Initialize());
  NiFpga_Session session;

  // Download bitfile to target; get path to bitfile
  // TODO: change to full path to bitfile as necessary
  CHECKSTAT(NiFpga_Open(bitPath, bitSig, rioDev, 0, &session));
  printf(" DONE\n");

  // Allocate CUDA memory
  printf("Allocating CUDA Memory ");
  fifotype *gpu_mem;
  if (cudaMalloc(&gpu_mem, sizeof(fifotype)*SAMPLES*2) != cudaSuccess) {
    printf("Error allocating memory in CUDA Device!\n");
    return 1;
  }
  printf("at %p\n", gpu_mem);

  // Configure P2P FIFO between FlexRIO and GPU using NVIDIA GPU Direct
  CHECKSTAT(NiFpga_ConfigureFifoBuffer(session, NiFpga_FPGA_main_TargetToHostFifoU64_FlexRIO_FIFO,
        (uint64_t)gpu_mem, SAMPLES*2, NULL, NiFpga_DmaBufferType_NvidiaGpuDirectRdma));

  // Setup batch size to be very large
  NiFpga_WriteU64(session, NiFpga_FPGA_main_ControlU64_BatchSize, (uint64_t)SAMPLES*BATCH_NFRAMES);

  // Start transferring data by creating rising edge start signal
  NiFpga_WriteBool(session, NiFpga_FPGA_main_ControlBool_StartTransfer, NiFpga_False);
  NiFpga_WriteBool(session, NiFpga_FPGA_main_ControlBool_StartTransfer, NiFpga_True);


  struct timeval start_time;
  gettimeofday(&start_time, NULL);

  size_t tot_bytes = 0; // total bytes transferred
  uint64_t running_count = 0;
  size_t elems_acquired, elems_remaining = 0;
  fifotype *datap;

  // Start recording the user/system times
  struct tms start_tms, end_tms;
  times(&start_tms);

  // Process batch frames
  for (int i=0; i<BATCH_NFRAMES; i++)
  {
    //output spinner
    printf("\rTransferring Data %c", spin());
    // Acquire data from FlexRIO -> GPU
    tot_bytes += sizeof(fifotype)*SAMPLES;
    CHECKSTAT(NiFpga_AcquireFifoReadElementsU64(session, NiFpga_FPGA_main_TargetToHostFifoU64_FlexRIO_FIFO, &datap, SAMPLES, 3000, &elems_acquired, &elems_remaining));

    // Add frame to running count
    {
      inplace_add<<<CUDA_NBLOCKS,CUDA_NTHREADS,sizeof(fifotype)*CUDA_NTHREADS>>>(datap);

      int nresults = CUDA_NBLOCKS;
      while (nresults > 1)
      {
        int n_threads_to_run = (nresults > 1024)?1024:nresults;
        int n_blocks_to_run  = (nresults > 1024)?nresults/1024:1;
        inplace_add<<<n_blocks_to_run,n_threads_to_run,sizeof(fifotype)*n_threads_to_run>>>(datap);
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

  printf("\b\nFinal count (on GPU) is %llu:\n", running_count);

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
  for (uint64_t j=0; j < SAMPLES*BATCH_NFRAMES; j++)
    checksum +=j;

  printf("Final checksum on CPU: %llu\n", checksum);
  if (checksum != running_count)
    printf("*** WARNING: FINAL COUNT DOES NOT MATCH CHECKSUM.\n");
  else
    printf("Success! Checksums on CPU and GPU match\n");

  cudaFree(gpu_mem);
  cudaDeviceReset();

  // Close NI FPGA References; must be last NiFpga calls
  printf("Stopping NI FPGA...\n");
  NiFpga_StopFifo(session, NiFpga_FPGA_main_TargetToHostFifoU64_FlexRIO_FIFO);
  NiFpga_Close(session, 0);
  NiFpga_Finalize();

  return 0;
}
