#!/bin/bash
scp nitest@flexlinux:/home/nitest/FlexRIO-P2P-GPU/GPU_FFT/TimeSignal.dat /home/jgentile/Downloads/
gnuplot -p tlt 
