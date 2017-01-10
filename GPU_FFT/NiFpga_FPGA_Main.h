/*
 * Generated with the FPGA Interface C API Generator 16.0.0
 * for NI-RIO 16.0.0 or later.
 */

#ifndef __NiFpga_FPGA_Main_h__
#define __NiFpga_FPGA_Main_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1600
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_FPGA_Main_Bitfile;
 */
#define NiFpga_FPGA_Main_Bitfile "NiFpga_FPGA_Main.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_FPGA_Main_Signature = "DAA4B54616BF18D27170CFDD9178EF17";

typedef enum
{
   NiFpga_FPGA_Main_ControlBool_AWGNEnable = 0x10012,
   NiFpga_FPGA_Main_ControlBool_LPFEnable = 0x10006,
   NiFpga_FPGA_Main_ControlBool_aReset = 0x10002,
} NiFpga_FPGA_Main_ControlBool;

typedef enum
{
   NiFpga_FPGA_Main_ControlU16_RMSNoise = 0x1000A,
} NiFpga_FPGA_Main_ControlU16;

typedef enum
{
   NiFpga_FPGA_Main_TargetToHostFifoI32_T2HDMAFIFO = 0,
} NiFpga_FPGA_Main_TargetToHostFifoI32;

#endif
