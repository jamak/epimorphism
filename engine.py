from ctypes import *
from cuda.cuda_utils import *
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
kernel1.argtypes = [ c_void_p, c_ulong, c_float ]

kernel2 = lib.__device_stub_kernel2
kernel2.restype = None
kernel2.argtypes = [ c_void_p, c_void_p, c_ulong, c_float ]

kernel_copy = lib.__device_stub_kernel_copy
kernel_copy.restype = None
kernel_copy.argtypes = [ c_void_p, c_ulong, c_void_p, c_int ]


class Engine(Thread):

    offset = 0

    def __init__(self, profile, state):
        Thread.__init__(self)

        self.profile, self.state = profile, state
        log("initializing")
        
        self.exit = False
        self.d = self.c = 0

        self.cuda_device = c_int()
        cudaGetDevice(byref(self.cuda_device))

        self.cuda_properties = cudaDeviceProp()
        cudaGetDeviceProperties(self.cuda_properties, self.cuda_device)

        print str(self.cuda_properties)

        self.time_accumulator = 0

        #channel_desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned)
        #self.array = cudaArray_p()

        #cudaMallocArray(byref(self.array), channel_desc, self.profile.kernel_dim, self.profile.kernel_dim)

        self.output_array = c_void_p()
        self.output_pitch = c_ulong()
        status = cudaMallocPitch(byref(self.output_array), byref(self.output_pitch), self.profile.kernel_dim * sizeof(float4), self.profile.kernel_dim)
        if status != cudaSuccess:
            raise GPUException(
            "Failed to allocate memory")

        status = cudaMemset2D(self.output_array, self.output_pitch, 0, self.profile.kernel_dim * sizeof(float4), self.profile.kernel_dim)
        if status != cudaSuccess:
            raise GPUException(
            "Failed to set memory")

        self.render_request = False

    def register_buffer(self):
        self.pbo0_ptr = c_void_p()
        status = cudaGLRegisterBufferObject(self.pbo0)

    def cleanup(self):
        #cudaFreeArray(self.array)
        cudaFree(self.output_array)
        status = cudaGLUnregisterBufferObject(self.pbo0)

    def render_to_buffer(self):
        self.render_request = True
        #while(self.render_request):            
        #    pass

    def run(self):
        log("starting")
        while(not self.exit):
            # print 'a'
            self.do()

    def do(self):
        #events = [cudaEvent_t(), cudaEvent_t(), cudaEvent_t(), cudaEvent_t(), cudaEvent_t()]
        #[cudaEventCreate(byref(event)) for event in events]    

        self.c += 1            
        self.offset += 1.0 / 1000.0
        self.offset = self.offset % 1                

        #if(self.render_request):

        #cudaEventRecord(events[0], 0)

        block = dim3(10, 10, 1)
        grid = dim3(100, 100, 1)
        status = cudaConfigureCall(grid, block, 0, 0)

        cudaGLMapBufferObject(byref(self.pbo0_ptr), self.pbo0)

        #print 1
        # kernel1(self.output_array, self.output_pitch, self.offset)            
        kernel2(self.output_array, self.pbo0_ptr, self.output_pitch, self.offset)            
        #print 2

        #cudaEventRecord(events[1], 0)        

        #cudaEventRecord(events[2], 0)

        #cudaMemcpy2D(self.pbo0_ptr, self.profile.kernel_dim * sizeof(uchar4), self.output_array, self.output_pitch, self.profile.kernel_dim * sizeof(uchar4), 
        #             self.profile.kernel_dim, cudaMemcpyDeviceToDevice)

        #block = dim3(10, 10, 1)
        #grid = dim3(100, 100, 1)
        #status = cudaConfigureCall(grid, block, 0, 0)

        # kernel_copy(self.output_array, self.output_pitch, self.pbo0_ptr, self.profile.kernel_dim)            

        #cudaEventRecord(events[3], 0)
    
        cudaGLUnmapBufferObject(self.pbo0)

        #cudaEventRecord(events[4], 0)

        #cudaEventSynchronize(events[4])

        #elapsed_time = c_float()
        #cudaEventElapsedTime(byref(elapsed_time), events[0], events[4])

        #self.time_accumulator += elapsed_time.value

        #if(self.c % 1000 == 0):
        #    time = self.time_accumulator / 1000.0
        #    self.time_accumulator = 0
        #    print "time = " + str(time)
            #print "fps = " + str(1.0 / time)

        #[cudaEventDestroy(event) for event in events]    
        
        render_request = False


