from ctypes import *
from cuda.cuda_defs import *
from cuda.cuda_api import *

from threading import Thread

import time

from logger import *
logger = Logger("log.txt", "en:  ")
log = logger.log

lib = CDLL("./kernel.so")

kernel1 = lib.__device_stub_kernel1
kernel1.restype = None
kernel1.argtypes = [ c_void_p ]

class Engine(Thread):
    def __init__(self, profile, state):
        Thread.__init__(self)

        self.profile, self.state = profile, state
        log("initializing")
        
        self.exit = False

    def register_buffers(self):
        status = cudaGLRegisterBufferObject(self.pbo0)
        status = cudaGLRegisterBufferObject(self.pbo1)

        self.pbo0_ptr = c_void_p()
        self.pbo1_ptr = c_void_p()               
        

    def run(self):
        log("starting")
        while(not self.exit):
            self.pbo0_ptr = c_void_p()
            self.pbo1_ptr = c_void_p()

            status = cudaGLMapBufferObject(byref(self.pbo0_ptr), self.pbo0)
            #status = cudaGLMapBufferObject(byref(self.pbo1_ptr), self.pbo1)

            block = dim3(10, 10, 1)
            grid = dim3(50, 50 ,1)
            status = cudaConfigureCall(grid, block, 0, 0)

            kernel1(self.pbo0_ptr)            

            status = cudaGLUnmapBufferObject(self.pbo0)
            #status = cudaGLUnmapBufferObject(self.pbo1)
            #vptr = c_void_p()
            #status = cudaGLMapBufferObject(byref(vptr),vbo)        

            #status = cudaGLUnmapBufferObject(vbo)
