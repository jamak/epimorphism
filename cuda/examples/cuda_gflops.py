#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08
from ctypes import *
from time import time, clock

from cuda.cuda_defs import *
from cuda.cuda_api import *

from cpuFunctions import cpuGFLOPS,checkError
from gpuFunctions import gpuGFLOPS

BLOCK_SIZE = 128
GRID_SIZE  = 64
ITERATIONS = 512
S4 = sizeof(c_float)

# This is SUBSTANTIALLY slower than cu_gflops.py. Why?
# Looping about 50 times almost as fast as cu_gflops.py.

def main(loops = 1):

    blockDim  = dim3(BLOCK_SIZE,1,1)
    gridDim   = dim3(GRID_SIZE,1,1)

    t0 = time()
    cudaThreadSynchronize()
    for i in range(loops):
        cudaConfigureCall(gridDim,blockDim,0,0)
        gpuGFLOPS()
    cudaThreadSynchronize()
    t0 = time()-t0
    cudaThreadExit()

    flopsc = 4096.*ITERATIONS*BLOCK_SIZE
    flopsg = flopsc*GRID_SIZE
    flopsc *= 1.e-9*float(loops)
    flopsg *= 1.e-9*float(loops)

    t1 = time()
    for i in range(loops):
        cpuGFLOPS()
    t1 = time()-t1
    peakg = 4.*8.*2.*1.458 # 2MP*8SP/MP*2flops/SP/clock*clock[GHz]
    print "%8.3f%8.2f%8.3f%8.2f [%.2f]" % (
        t1,flopsc/t1,t0,flopsg/t0,peakg)

if __name__ == "__main__":
    import sys

    cudaSetDevice(0)

    loops = 1
    if len(sys.argv) > 1:
        loops = int(sys.argv[1])
    print "%5d" % (loops),
    main(loops)
