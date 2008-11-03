import os
import re
from ctypes import *

def loadKernel(state):
    
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
    os.system("rm aer/kernel.so")
    os.system("aer/make_kernel")

    # interface
    lib = CDLL("aer/kernel.so")
    kernel = lib.__device_stub_kernel_fb
    kernel.restype = None
    kernel.argtypes = [ c_void_p, c_ulong, c_void_p, c_int ]
    return kernel


