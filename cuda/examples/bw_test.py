#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08

from ctypes import *
from time import time, clock

from cuda.cuda_api import *
from cuda.cuda_defs import *
from cuda.cuda_utils import mallocHost
from ctypes_array import convert
from numpy import abs,max

PINNED = 1
MEMCOPY_ITERATIONS = 1000

def compare(a,b):
    a1 = convert(a)
    b1 = convert(b)
    diff = max(abs(a1-b1))
    return diff

#///////////////////////////////////////////////////////////////////////////////
#//  test the bandwidth of a device to host memcopy of a specific size
#///////////////////////////////////////////////////////////////////////////////
def testDeviceToHostTransfer(size,mode):
    dtype = c_int
    memSize = size*sizeof(dtype)
    amountCopied = memSize*MEMCOPY_ITERATIONS
    d_idata = c_void_p()

    h_idata = mallocHost(size,dtype,mode)
    h_odata = mallocHost(size,dtype,mode)
    for i in range(size):
        h_idata[i] = dtype(size-i)
        h_odata[i] = dtype(123)

    cudaMalloc(byref(d_idata),memSize)
    cudaMemcpy(d_idata,h_idata,memSize,cudaMemcpyHostToDevice)
    t0 = time()
    for i in range(MEMCOPY_ITERATIONS):
        cudaMemcpy(h_odata,d_idata,memSize,cudaMemcpyDeviceToHost)
    t1 = time()-t0
    diff = compare(h_idata,h_odata)
    print "Max abs difference = %3s" % diff,
    bandwidthInMBs = amountCopied/(t1*float((1 << 20)))

    if mode == PINNED:
        cudaFreeHost(h_idata)
        cudaFreeHost(h_odata)
    cudaFree(d_idata)
    print "Device To Host  : %5.0f MB/s" % bandwidthInMBs
    return bandwidthInMBs

#///////////////////////////////////////////////////////////////////////////////
#//! test the bandwidth of a host to device memcopy of a specific size
#///////////////////////////////////////////////////////////////////////////////
def testHostToDeviceTransfer(size,mode):
    dtype = c_float
    memSize = size*sizeof(dtype)
    amountCopied = memSize*MEMCOPY_ITERATIONS
    d_idata = c_void_p()

    h_idata = mallocHost(size,dtype,mode)
    h_odata = mallocHost(size,dtype,mode)
    for i in range(size):
        h_idata[i] = dtype(size-i)
        h_odata[i] = dtype(456)

    cudaMalloc(byref(d_idata),memSize)
    t0 = time()
    for i in range(MEMCOPY_ITERATIONS):
        cudaMemcpy(d_idata,h_idata,memSize,cudaMemcpyHostToDevice)
    cudaMemcpy(h_odata,d_idata,memSize,cudaMemcpyDeviceToHost)
    t1 = time()-t0
    diff = compare(h_idata,h_odata)
    print "Max abs difference = %3s" % diff,
    bandwidthInMBs = amountCopied/(t1*float((1 << 20)))

    if mode == PINNED:
        cudaFreeHost(h_idata)
        cudaFreeHost(h_odata)
    cudaFree(d_idata)
    print "Host To Device  : %5.0f MB/s" % bandwidthInMBs
    return bandwidthInMBs

#///////////////////////////////////////////////////////////////////////////////
#//! test the bandwidth of a device to device memcopy of a specific size
#///////////////////////////////////////////////////////////////////////////////
def testDeviceToDeviceTransfer(size,mode):
    dtype = c_double
    memSize = size*sizeof(dtype)
    amountCopied = memSize*MEMCOPY_ITERATIONS
    d_idata = c_void_p()
    d_odata = c_void_p()

    h_idata = mallocHost(size,dtype,mode)
    h_odata = mallocHost(size,dtype,mode)
    for i in range(size):
        h_idata[i] = dtype(size-i)
        h_odata[i] = dtype(789)

    cudaMalloc(byref(d_idata),memSize)
    cudaMalloc(byref(d_odata),memSize)
    cudaMemcpy(d_idata,h_idata,memSize,cudaMemcpyHostToDevice)
    t0 = time()
    for i in range(MEMCOPY_ITERATIONS):
        cudaMemcpy(d_odata,d_idata,memSize,cudaMemcpyDeviceToDevice)
    cudaThreadSynchronize() 
    t1 = time()-t0
    cudaMemcpy(h_odata,d_odata,memSize,cudaMemcpyDeviceToHost)
    diff = compare(h_idata,h_odata)
    print "Max abs difference = %3s" % diff,
    bandwidthInMBs = (2.*amountCopied)/(t1*float((1 << 20)))

    if mode == PINNED:
        cudaFreeHost(h_idata)
        cudaFreeHost(h_odata)
    cudaFree(d_idata)
    cudaFree(d_odata)
    print "Device To Device: %5.0f MB/s" % bandwidthInMBs
    return bandwidthInMBs

if __name__ == "__main__":
    import sys
    size = 1024*1024
    mode = PINNED
    if len(sys.argv) > 1:
        mode = 0

    memtype = {0:"pageable   ",PINNED:"page-locked"}
    print "+-------------------------+"
    print "| Bandwidth transfer test |"
    print "| using CUDA runtime API  |"
    print "| %s memory      |" % memtype[mode]
    print "+-------------------------+\n"

    testDeviceToHostTransfer(size,mode)
    testHostToDeviceTransfer(size,mode)
    testDeviceToDeviceTransfer(size,mode)
