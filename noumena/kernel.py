import os
import re
from ctypes import *

libnum = 0

import time

import threading

def loadKernel(engine, state):
    compiler = Compiler(engine, state)
    compiler.run()


def resume(engine):
    print "resume"
    global libnum
    # via ctypes interface
    lib = CDLL('common/lib/kernel' + str(libnum) + '.so', RTLD_LOCAL)
    kernel = lib.__device_stub_kernel_fb
    kernel.restype = None
    kernel.argtypes = [ c_void_p, c_ulong, c_void_p, c_int, c_float, c_float, c_float, c_float ]


    libnum+=1

    engine.bind_kernel(kernel)


class Compiler(threading.Thread):

    def __init__(self, engine, state):
        self.engine, self.state = engine, state
        threading.Thread.__init__(self)

    def render_file(self, name):
        # open file & read contents
        file = open("aer/" + name + ".ecu")
        contents = file.read()
        file.close()

        par_name_str = ""

        for i in xrange(len(self.state.par_names)):
            par_name_str += "#define " + self.state.par_names[i] + " par[" + str(i) + "]\n"

        contents = re.compile('\%PAR_NAMES\%').sub(par_name_str, contents)

        # replace variables
        for key in self.state.__dict__:
            contents = re.compile('\%' + key + '\%').sub(str(self.state.__dict__[key]), contents)

        # write file contents
        file = open("aer/__" + name + ".cu", 'w')
        file.write(contents)
        file.close()

    def load(self):
        global libnum

        self.render_file("seed")
        self.render_file("kernel")

        # compile
        os.system("rm lib/kernel" + str(libnum) + ".so")
        os.system("rm lib/kernel" + str(libnum - 1) + ".so")
        os.system("/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -o common/lib/kernel" + str(libnum) + ".so  --shared --ptxas-options=-v aer/__kernel.cu")
        os.system("rm __kernel.linkinfo")
        resume(self.engine)


    def run(self):
        self.load()
