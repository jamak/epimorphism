import os
import re
from ctypes import *

libnum = 0

def loadKernel(state):

    global libnum
    
    # open file & read contents
    file = open("aer/kernel.ecu")
    contents = file.read()
    file.close()

    # replace variables
    for key in state.__dict__:
        contents = re.compile('\%' + key + '\%').sub(str(state.__dict__[key]), contents)

    # write file contents
    file = open("aer/__kernel.cu", 'w')
    file.write(contents)
    file.close()

    # compile
    os.system("rm aer/kernel" + str(libnum) + ".so")
    os.system("rm aer/kernel" + str(libnum - 1) + ".so")
    os.system("/usr/local/cuda/bin/nvcc -o aer/kernel" + str(libnum) + ".so  --shared --ptxas-options=-v aer/__kernel.cu")

    # interface    
    lib = CDLL('aer/kernel' + str(libnum) + '.so', RTLD_LOCAL)
    kernel = lib.__device_stub_kernel_fb
    kernel.restype = None
    kernel.argtypes = [ c_void_p, c_ulong, c_void_p, c_int, c_float, c_float, c_float, c_float ]

    libnum+=1

    return kernel


