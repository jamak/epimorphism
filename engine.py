from ctypes import *
from cuda.cuda_utils import *
from cuda.cuda_defs import *
from cuda.cuda_api import *

import time

from logger import *
logger = Logger("log.txt", "en:  ")
log = logger.log

# load kernels
lib = CDLL("./kernel.so")

kernel2 = lib.__device_stub_kernel2
kernel2.restype = None
kernel2.argtypes = [ c_void_p, c_void_p, c_ulong, c_float ]

class Engine:

    offset = 0

    def __init__(self, profile, state):

        self.profile, self.state = profile, state

        log("initializing")
        
        self.exit = False        

        # get device
        self.cuda_device = c_int()
        cudaGetDevice(byref(self.cuda_device))

        # get/print properties
        self.cuda_properties = cudaDeviceProp()
        cudaGetDeviceProperties(self.cuda_properties, self.cuda_device)
        print str(self.cuda_properties)

        # create 
        channel_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat)

        self.array = cudaArray_p()
        cudaMallocArray(byref(self.array), channel_desc, self.profile.kernel_dim, self.profile.kernel_dim)

        # bind texture
        self.texture_ref = textureReference()
        cudaBindTextureToArray(byref(self.texture_ref), self.array, channel_desc)

        # create
        self.output_array = c_void_p()
        self.output_pitch = c_ulong()

        cudaMallocPitch(byref(self.output_array), byref(self.output_pitch), self.profile.kernel_dim * sizeof(float4), self.profile.kernel_dim)    
        cudaMemset2D(self.output_array, self.output_pitch, 0, self.profile.kernel_dim * sizeof(float4), self.profile.kernel_dim)    

        # initialize timing info
        self.time_events = True
        self.frame_count = 0.0
        self.events = [cudaEvent_t() for i in range(4)]
        [cudaEventCreate(byref(event)) for event in self.events]    
        self.event_accum_tmp = [0 for i in range(len(self.events) - 1)]
        self.event_accum = [0 for i in range(len(self.events) - 1)]
        self.time_accum = 0


    def register(self):
        self.pbo0_ptr = c_void_p()

        status = cudaGLRegisterBufferObject(self.pbo0)        
        cudaGLMapBufferObject(byref(self.pbo0_ptr), self.pbo0)        


    def cleanup(self):
        cudaFreeArray(self.array)
        cudaFree(self.output_array)
        [cudaEventDestroy(event) for event in self.events]    
        status = cudaGLUnregisterBufferObject(self.pbo0)


    def record(self, idx):
        if(self.time_events):
            cudaEventRecord(self.events[idx], 0)


    def do(self):
        self.frame_count += 1            

        self.record(0)

        block = dim3(10, 10, 1)
        grid = dim3(100, 100, 1)
        status = cudaConfigureCall(grid, block, 0, 0)

        self.offset += 1.0 / 1000.0
        self.offset = self.offset % 1                

        kernel2(self.output_array, self.pbo0_ptr, c_ulong(self.output_pitch.value / sizeof(float4)), self.offset)            
        # kernel3(self.output_array, self.pbo0_ptr, self.array, self.output_pitch, self.offset)            

        self.record(1)

        cudaMemcpy2DToArray(self.array, self.profile.kernel_dim, self.profile.kernel_dim, self.output_array, self.output_pitch, self.profile.kernel_dim, self.profile.kernel_dim, cudaMemcpyDeviceToDevice)

        self.record(2)
    
        cudaGLUnmapBufferObject(self.pbo0)

        self.record(3)

        if(self.time_events):
            cudaEventSynchronize(self.events[-1])        
            
            times = [c_float() for i in range(len(self.events) - 1)]        
            [cudaEventElapsedTime(byref(times[i]), self.events[i], self.events[i+1]) for i in range(len(times))]

            self.event_accum_tmp = [self.event_accum_tmp[i] + times[i].value for i in range(len(times))]
            self.event_accum = [self.event_accum[i] + times[i].value for i in range(len(times))]

            if(self.frame_count % 1000 == 0):
                for i in range(len(times)):
                    print "event" + str(i) + "-" + str(i + 1) + ": " + str(self.event_accum_tmp[i] / 1000.0) + "ms"
                    print "event" + str(i) + "-" + str(i + 1) + "~ " + str(self.event_accum[i] / self.frame_count) + "ms"
                print "total: " + str(sum(self.event_accum_tmp) / 1000.0) + "ms"
                print "total~ " + str(sum(self.event_accum) / self.frame_count) + "ms"
                self.event_accum_tmp = [0 for i in range(len(self.events) - 1)]

        cudaThreadSynchronize()
