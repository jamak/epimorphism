import os
import re
import sys
from ctypes import *

import threading
import StringIO

libnum = 0


def bind_kernel(name):
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

    def __init__(self, data, callback):
        self.data, self.callback = data, callback
        threading.Thread.__init__(self)
        self.update_vars = {}
        self.update_vars.update(data)

    def update(self, new_vars):
        self.update_vars.update(new_vars)
        self.do_update = True

    def render_file(self, name):
        # open file & read contents
        file = open("aer/" + name + ".ecu")
        contents = file.read()
        file.close()

        if(name == "kernel"):
            par_name_str = ""

            for i in xrange(len(self.data["par_names"])):
                par_name_str += "#define " + self.data["par_names"][i] + " par[" + str(i) + "]\n"

            contents = re.compile('\%PAR_NAMES\%').sub(par_name_str, contents)

        # replace variables
        for key in self.update_vars:
            contents = re.compile('\%' + key + '\%').sub(str(self.data[key]), contents)

        # write file contents
        file = open("aer/__" + name + ".cu", 'w')
        file.write(contents)
        file.close()


    def run(self):

        global libnum

        self.do_update = True

        while(self.do_update):
            self.do_update = False
            self.render_file("seed")
            self.render_file("kernel")

            name = "kernel" + str(libnum) + ".so"

            libnum += 1

            os.system("rm common/lib/" + name)
            os.system("/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -o common/lib/%s --shared  aer/__kernel.cu" % name)
            os.system("rm __kernel.linkinfo")

        self.callback(name)

