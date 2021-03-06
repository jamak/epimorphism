#!/bin/env python
# coding:utf-8: © Arno Pähler, 2007-08

from ctypes import *
from time import time

from cuda.cublas_api import *
from cuda.cublas_defs import *
from cuda.cuda_api import cudaThreadSynchronize

from cpuFunctions import vectorInit,cpuSAXPY,checkError
from ctypes_array import *

useSciPy = True
if useSciPy:
    from scipy.lib.blas.fblas import saxpy as _saxpy
    
    def saxpy(a,x,y):
        nx = convert(x)
        ny = convert(y)
        nz = _saxpy(nx,ny,a=a.value)
        convert(nz,None,None,y)
        return y
else:
    saxpy = cpuSAXPY

def main(vlength = 128,loops = 1):
    print "+-----------------------+"
    print "|   CUBLAS SAXPY Test   |"
    print "|   using CUBLAS API    |"
    print "+-----------------------+\n"
    print "Parameters: %d %d\n" % (vlength,loops)
    runTest(vlength,loops)

def runTest(vlength = 128,loops = 1):
    n2 = vlength*vlength
    alfa = c_float(.5)

    cublasInit()

    h_X = (c_float*n2)()
    h_Y = (c_float*n2)()
    g_Y = (c_float*n2)()
    vectorInit(h_X)
    vectorInit(h_Y)

    d_X = c_void_p()
    d_Y = c_void_p()
    cublasAlloc(n2, sizeof(c_float), byref(d_X))
    cublasAlloc(n2, sizeof(c_float), byref(d_Y))
 
    cublasSetVector(n2, sizeof(c_float), h_X, 1, d_X, 1)
    cublasSetVector(n2, sizeof(c_float), h_Y, 1, d_Y, 1)

    flops = (2.e-9*n2)*float(loops)
    t0 = time()
    for i in range(loops):
        cublasSaxpy(n2, alfa, d_X, 1, d_Y, 1)
    cudaThreadSynchronize()
    t0 = time()-t0

    print "Processing time: %.3g sec" % t0
    print "Gigaflops GPU: %.2f (%d)" % (flops/t0,n2)

    t1 = time()
    for i in range(loops):
        cpuSAXPY(alfa,h_X,h_Y)
    t1 = time()-t1
    
    print "\nProcessing time: %.3g sec" % t1
    print "Gigaflops CPU: %.2f" % (flops/t1)
    print "GPU vs. CPU  : %.2f" % (t1/t0)

    cublasGetVector(n2, sizeof(c_float), d_Y, 1, g_Y, 1)
    err,mxe = checkError(h_Y,g_Y)
    print "\nAvg and max rel error = %.2e %.2e" % (err,mxe)

    cublasFree(d_X)
    cublasFree(d_Y)

    cublasShutdown()

if __name__ == "__main__":
    import sys

##  square root of vector length
    vlength,loops = 1024,10
    if len(sys.argv) > 1:
        vlength = int(sys.argv[1])
    if len(sys.argv) > 2:
        loops = int(sys.argv[2])
    main(vlength,loops)
