import os
import re
import sys
from ctypes import *

import threading
import StringIO

libnum = 0


def bind_kernel(name):

    # attempt to load kernel
    try:
        lib = CDLL("tmp/" + name, RTLD_LOCAL)
        os.system("rm tmp/" + name)
    except:
        print "kernel not found.  exiting."
        exit()

    # extract function
    kernel = lib.__device_stub_kernel_fb
    kernel.restype = None
    kernel.argtypes = [ c_void_p, c_ulong, c_void_p, c_int, c_float, c_float, c_float, c_float ]

    return kernel


class Compiler(threading.Thread):
    ''' A Compiler object if responsible for asynchronously calling nvcc.
        The compilation can be restarted by a call to update. '''

    def __init__(self, data, callback):

        self.data, self.callback = data, callback

        # init update_vars
        self.update_vars = {}
        self.update_vars.update(data)

        # init thread
        threading.Thread.__init__(self)


    def update(self, new_vars):

        # set update info
        self.update_vars.update(new_vars)
        self.do_update = True


    def render_file(self, name):

        # open file & read contents
        file = open("aer/" + name + ".ecu")
        contents = file.read()
        file.close()

        # bind PAR_NAMES
        par_name_str = ""

        for i in xrange(len(self.data["par_names"])):
            par_name_str += "#define %s par[%d]\n" % (self.data["par_names"][i], i)

        contents = re.compile('\%PAR_NAMES\%').sub(par_name_str, contents)

        # replace variables
        for key in self.update_vars:
            contents = re.compile("\%" + key + "\%").sub(str(self.data[key]), contents)

        # write file contents
        file = open("aer/__%s.cu" % name, 'w')
        file.write(contents)
        file.close()


    def run(self):

        global libnum

        # begin update loop
        self.do_update = True
        while(self.do_update):
            self.do_update = False

            # render files
            self.render_file("seed")
            self.render_file("kernel")

            # get name
            name = "kernel%d.so" % libnum
            libnum += 1

            # compile
            os.system("/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -o tmp/%s --shared  aer/__kernel.cu" % name)
            if(os.path.exists("__kernel.linkinfo")) : os.system("rm __kernel.linkinfo")

        # execute callback
        self.callback(name)

