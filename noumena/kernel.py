import os
import re
from ctypes import *

libnum = 0

import threading



def compile_kernel(engine):
    compiler = Compiler(engine)
    compiler.start()


def bind_kernel(name):
    print "bind kernel"

    # via ctypes interface
    try:
        lib = CDLL('common/lib/' + name, RTLD_LOCAL)
    except:
        print "kernel compilation error.  exiting."
        exit()
    kernel = lib.__device_stub_kernel_fb
    kernel.restype = None
    kernel.argtypes = [ c_void_p, c_ulong, c_void_p, c_int, c_float, c_float, c_float, c_float ]

    return kernel



class Compiler(threading.Thread):

    def __init__(self, engine, callback):
        self.engine, self.state, self.callback = engine, engine.state, callback
        threading.Thread.__init__(self)


    def render_file(self, name):
        # open file & read contents
        file = open("aer/" + name + ".ecu")
        contents = file.read()
        file.close()

        if(name == "kernel"):
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


    def run(self):
        global libnum

        self.render_file("seed")
        self.render_file("kernel")

        name = "kernel" + str(libnum) + ".so"

        libnum += 1

        os.system("rm common/lib/" + name)
        os.system("/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -o common/lib/" + name + " --shared --ptxas-options=-v aer/__kernel.cu")
        os.system("rm __kernel.linkinfo")
        self.callback(name)

