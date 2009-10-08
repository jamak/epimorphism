from phenom.datamanager import *

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
    debug("Binding kernel: %s" % name)

    # attempt to load kernel
    try:
        lib = cdll.LoadLibrary("tmp/%s.so" % name)#, RTLD_LOCAL)
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

    def __init__(self, data, callback, context):
        debug("Initializing Compiler")

        self.callback, self.context = callback, context

        self.data = data.copy()

        # init update_vars
        self.update_vars = {}
        self.update_vars.update(data)

        # start datamanager & manage components
        self.datamanager = DataManager()

        self.splice_components()

        # init thread
        threading.Thread.__init__(self)


    def splice_components(self):
        ''' This method dynamicly generates the interpolated component switch
            statements that are spliced into the kernels '''
        debug("Splicing components")

        var = self.data
        for component_name in self.datamanager.components:
            component_list = getattr(self.datamanager, component_name)

            idx = self.datamanager.components.index(component_name)

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

            self.data[component_name] = clause1 + interp

        return self


    def render_file(self, name):
        ''' Substitues escape sequences in a .ecu file with dynamic content '''
        debug("Rendering: %s", name)

        # open file & read contents
        file = open("aeon/" + name)
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
        if(not os.path.exists("tmp/%s.so" % name)):
            info("Compiling kernel - %s" % name)

            os.system("/usr/local/cuda/bin/nvcc  --host-compilation=c -Xcompiler -fPIC -o tmp/%s.so --shared %s aeon/__kernel.cu" % (name, self.context.ptxas_stats and "--ptxas-options=-v" or ""))

            # remove tmp files
            files = [file for file in os.listdir("aeon") if re.search("\.ecu$", file)]
            for file in files:
                os.system("rm aeon/__%s" % (file.replace(".ecu", ".cu")))
            if(os.path.exists("__kernel.linkinfo")) : os.system("rm __kernel.linkinfo")

        else:
            time.sleep(1)

        # execute callback
        self.callback(name)

