import os
import re
from ctypes import *

libnum = 0

import time

def render_file(name, state):
    # open file & read contents
    file = open("aer/" + name + ".ecu")
    contents = file.read()
    file.close()

    par_name_str = ""

    for par_name in state.par_names:
        par_name_str += "#define " + par_name + " par[" + str(state.par_names[par_name]) + "]\n"

    contents = re.compile('\%PAR_NAMES\%').sub(par_name_str, contents)

    # replace variables
    for key in state.__dict__:
        contents = re.compile('\%' + key + '\%').sub(str(state.__dict__[key]), contents)

    # write file contents
    file = open("aer/__" + name + ".cu", 'w')
    file.write(contents)
    file.close()

def loadKernel(state):

    global libnum

    render_file("seed", state)
    render_file("kernel", state)

    # compile
    os.system("rm lib/kernel" + str(libnum) + ".so")
    os.system("rm lib/kernel" + str(libnum - 1) + ".so")
    os.system("/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -o lib/kernel" + str(libnum) + ".so  --shared --ptxas-options=-v aer/__kernel.cu")

    # via ctypes interface
    lib = CDLL('lib/kernel' + str(libnum) + '.so', RTLD_LOCAL)
    kernel = lib.__device_stub_kernel_fb
    kernel.restype = None
    kernel.argtypes = [ c_void_p, c_ulong, c_void_p, c_int, c_float, c_float, c_float, c_float ]


    libnum+=1

    return kernel


