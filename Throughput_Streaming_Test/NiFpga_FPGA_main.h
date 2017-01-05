/*
 * Generated with the FPGA Interface C API Generator 16.0.0
 * for NI-RIO 16.0.0 or later.
 */

#ifndef __NiFpga_FPGA_main_h__
#define __NiFpga_FPGA_main_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1600
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_FPGA_main_Bitfile;
 */
#define NiFpga_FPGA_main_Bitfile "NiFpga_FPGA_main.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_FPGA_main_Signature = "A4984845D7D04C0954EBAEAE86FA557F";

typedef enum
{
   NiFpga_FPGA_main_IndicatorI32_Iteration = 0x1000C,
} NiFpga_FPGA_main_IndicatorI32;

typedef enum
{
   NiFpga_FPGA_main_IndicatorU64_Count = 0x10008,
} NiFpga_FPGA_main_IndicatorU64;

typedef enum
{
   NiFpga_FPGA_main_ControlBool_StartTransfer = 0x10002,
} NiFpga_FPGA_main_ControlBool;

typedef enum
{
   NiFpga_FPGA_main_ControlU64_BatchSize = 0x10004,
} NiFpga_FPGA_main_ControlU64;

typedef enum
{
   NiFpga_FPGA_main_TargetToHostFifoU64_FlexRIO_FIFO = 0,
} NiFpga_FPGA_main_TargetToHostFifoU64;

#endif
