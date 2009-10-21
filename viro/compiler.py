import os
import re
import hashlib
import threading
import time

from ctypes import *

from common.log import *
set_log("COMPILER")

def get_functions(name):
    ''' Creates & returns ctypes interfaces to the kernel .so '''
    debug("Getting functions from: %s" % name)

    # attempt to load kernel
    try:
        lib = cdll.LoadLibrary("kernels/%s.so" % name)#, RTLD_LOCAL)
    except:
        critical("Kernel not found")
        os._exit()

    # extract function - this could probably be done more smartly
    kernel = lib.__device_stub__Z9kernel_fbP6float4mP6uchar4iffff
    kernel.restype = None
    kernel.argtypes = [ c_void_p, c_ulong, c_void_p, c_int, c_float, c_float, c_float, c_float ]

    reset = lib.__device_stub__Z5resetP6float4m
    reset.restype = None
    reset.argtypes = [ c_void_p, c_ulong ]

    return (kernel, reset)


class Compiler(threading.Thread):
    ''' A Compiler object if responsible for asynchronously calling nvcc.
        The compilation can be restarted by a call to update. '''

    def __init__(self, callback, config):
        debug("Initializing Compiler")

        self.callback, self.config = callback, config

        self.substitutions = {}

        # init thread
        threading.Thread.__init__(self)


    def splice_components(self):
        ''' This method dynamicly generates the interpolated component switch
            statements that are spliced into the kernels '''
        debug("Splicing components")

        for component_name in self.config['datamanager'].component_names:
            component_list = self.config['datamanager'].components[component_name]

            idx = self.config['datamanager'].component_names.index(component_name)

            clause1 = "switch(component_idx[%d][0]){\n" % idx
            for component in component_list:
                name = component[0]
                clause1 += "case %d: %s0 = %s;break;\n" % (component_list.index(component), component_name.lower(), name)
            clause1 += "}\n"

            clause2 = "switch(component_idx[%d][1]){\n" % idx
            for component in component_list:
                name = component[0]
                clause2 += "case %d: %s1 = %s;break;\n" % (component_list.index(component), component_name.lower(), name)
            clause2 += "}\n"

            interp = "if(internal[%d] != 0){" % idx
            sub = "min((_clock - internal[%d]) / switch_time, 1.0f)" % (idx)
            interp += "%s\n%s = ((1.0f - %s) * (%s0) + %s * (%s1));" % (clause2,  component_name.lower(), sub, component_name.lower(), sub, component_name.lower())
            interp += "}else{\n%s = %s0;\n}" % (component_name.lower(), component_name.lower())

            self.substitutions[component_name] = clause1 + interp

        return self


    def render_file(self, name):
        ''' Substitues escape sequences in a .ecu file with dynamic content '''
        debug("Rendering: %s", name)

        # open file & read contents
        file = open("aeon/" + name)
        contents = file.read()
        file.close()

        # splice components
        self.splice_components()

        # bind PAR_NAMES
        par_name_str = ""

        for i in xrange(len(self.config["par_names"])):
            par_name_str += "#define %s par[%d]\n" % (self.config["par_names"][i], i)

        self.substitutions["PAR_NAMES"] = par_name_str

        # replace variables
        for key in self.substitutions:
            contents = re.compile("\%" + key + "\%").sub(str(self.substitutions[key]), contents)

        # write file contents
        file = open("aeon/__%s" % (name.replace(".ecu", ".cu")), 'w')
        file.write(contents)
        file.close()


    def run(self):
        ''' Executes the main Compiler sequence '''
        debug("Executing")

        # render ecu files
        files = [file for file in os.listdir("aeon") if re.search("\.ecu$", file)]

        for file in files:
            self.render_file(file)

        # hash files
        files = [file for file in os.listdir("aeon") if re.search("\.cu$", file)]

        contents = ""
        for file in files:
            contents += open("aeon/" + file).read()

        hash = hashlib.sha1(contents).hexdigest()

        # make name
        name = "kernel-%s" % hash

        # compile if library doesn't exist
        if(not os.path.exists("kernels/%s.so" % name)):
            info("Compiling kernel - %s" % name)

            os.system("/usr/local/cuda/bin/nvcc  --host-compilation=c -Xcompiler -fPIC -o kernels/%s.so --shared %s aeon/__kernel.cu" % (name, self.config['ptxas_stats'] and "--ptxas-options=-v" or ""))

            # remove tmp files
            files = [file for file in os.listdir("aeon") if re.search("\.ecu$", file)]
            for file in files:
                os.system("rm aeon/__%s" % (file.replace(".ecu", ".cu")))
            if(os.path.exists("__kernel.linkinfo")) : os.system("rm __kernel.linkinfo")

        else:
            time.sleep(1)

        # execute callback
        self.callback(name)

