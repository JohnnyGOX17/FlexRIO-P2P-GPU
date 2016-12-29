/*
 * Generated with the FPGA Interface C API Generator 16.0.0
 * for NI-RIO 16.0.0 or later.
 */

#ifndef __NiFpga_HighThroughputStreamingFPGAPXIe797xR_h__
#define __NiFpga_HighThroughputStreamingFPGAPXIe797xR_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1600
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_HighThroughputStreamingFPGAPXIe797xR_Bitfile;
 */
#define NiFpga_HighThroughputStreamingFPGAPXIe797xR_Bitfile "NiFpga_HighThroughputStreamingFPGAPXIe797xR.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_HighThroughputStreamingFPGAPXIe797xR_Signature = "EC4463FFB1485764E1C683A488CE2417";

typedef enum
{
   NiFpga_HighThroughputStreamingFPGAPXIe797xR_IndicatorBool_inputEverTimedOut = 0x10056,
   NiFpga_HighThroughputStreamingFPGAPXIe797xR_IndicatorBool_outputEverTimedOut = 0x10032,
} NiFpga_HighThroughputStreamingFPGAPXIe797xR_IndicatorBool;

typedef enum
{
   NiFpga_HighThroughputStreamingFPGAPXIe797xR_IndicatorU16_inputMaxThroughput = 0x10006,
   NiFpga_HighThroughputStreamingFPGAPXIe797xR_IndicatorU16_outputMaxThroughput = 0x1000A,
} NiFpga_HighThroughputStreamingFPGAPXIe797xR_IndicatorU16;

typedef enum
{
   NiFpga_HighThroughputStreamingFPGAPXIe797xR_IndicatorU32_inputFIFOSize = 0x10014,
   NiFpga_HighThroughputStreamingFPGAPXIe797xR_IndicatorU32_inputMinimumNumberofElementstoWrite = 0x10048,
   NiFpga_HighThroughputStreamingFPGAPXIe797xR_IndicatorU32_inputNumberofElementstoWrite = 0x1004C,
   NiFpga_HighThroughputStreamingFPGAPXIe797xR_IndicatorU32_outputFIFOSize = 0x1000C,
   NiFpga_HighThroughputStreamingFPGAPXIe797xR_IndicatorU32_outputMinimumNumberofElementstoRead = 0x1003C,
   NiFpga_HighThroughputStreamingFPGAPXIe797xR_IndicatorU32_outputNumberofElementstoRead = 0x10038,
} NiFpga_HighThroughputStreamingFPGAPXIe797xR_IndicatorU32;

typedef enum
{
   NiFpga_HighThroughputStreamingFPGAPXIe797xR_ControlBool_inputReset = 0x10052,
   NiFpga_HighThroughputStreamingFPGAPXIe797xR_ControlBool_inputStart = 0x10012,
   NiFpga_HighThroughputStreamingFPGAPXIe797xR_ControlBool_inputStop = 0x10046,
   NiFpga_HighThroughputStreamingFPGAPXIe797xR_ControlBool_outputReset = 0x10036,
   NiFpga_HighThroughputStreamingFPGAPXIe797xR_ControlBool_outputStart = 0x10002,
   NiFpga_HighThroughputStreamingFPGAPXIe797xR_ControlBool_outputStop = 0x10042,
} NiFpga_HighThroughputStreamingFPGAPXIe797xR_ControlBool;

typedef enum
{
   NiFpga_HighThroughputStreamingFPGAPXIe797xR_ControlU16_inputRequestedThroughput = 0x10022,
   NiFpga_HighThroughputStreamingFPGAPXIe797xR_ControlU16_inputRequestedThroughputDenominator = 0x1001E,
   NiFpga_HighThroughputStreamingFPGAPXIe797xR_ControlU16_outputRequestedThroughput = 0x1002A,
   NiFpga_HighThroughputStreamingFPGAPXIe797xR_ControlU16_outputRequestedThroughputDenominator = 0x1001A,
} NiFpga_HighThroughputStreamingFPGAPXIe797xR_ControlU16;

typedef enum
{
   NiFpga_HighThroughputStreamingFPGAPXIe797xR_TargetToHostFifoI16_InputFIFO = 1,
} NiFpga_HighThroughputStreamingFPGAPXIe797xR_TargetToHostFifoI16;

typedef enum
{
   NiFpga_HighThroughputStreamingFPGAPXIe797xR_HostToTargetFifoI16_OutputFIFO = 0,
} NiFpga_HighThroughputStreamingFPGAPXIe797xR_HostToTargetFifoI16;

#endif
