from ctypes import *
from cuda.cuda_utils import *
from cuda.cuda_defs import *
from cuda.cuda_api import *

from Image import *

import time

from logger import *
logger = Logger("log.txt", "en:  ")
log = logger.log

# load kernels
lib = CDLL("./kernel.so")

kernel_fb = lib.__device_stub_kernel_fb
kernel_fb.restype = None
kernel_fb.argtypes = [ c_void_p, c_ulong, c_void_p, c_float, c_int ]

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

        # create input_array 
        channel_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat)

        self.input_array = cudaArray_p()
        cudaMallocArray(byref(self.input_array), byref(channel_desc), self.profile.kernel_dim, self.profile.kernel_dim)

        # initialize array
        empty = (c_ubyte * (sizeof(float4) * self.profile.kernel_dim ** 2))()
        ## image_str = open("image189.png").tostring("raw", "RGBA", 0, -1)          
        cudaMemcpyToArray(self.input_array, 0, 0, empty, sizeof(float4) * self.profile.kernel_dim ** 2, cudaMemcpyHostToDevice)

        # bind texture
        self.tex_ref = textureReference_p()
        cudaGetTextureReference(byref(self.tex_ref), "input_texture")
        cudaBindTextureToArray(self.tex_ref, self.input_array, byref(channel_desc))

        # create output_2D
        self.output_2D = c_void_p()
        self.output_2D_pitch = c_ulong()

        cudaMallocPitch(byref(self.output_2D), byref(self.output_2D_pitch), self.profile.kernel_dim * sizeof(float4), self.profile.kernel_dim)    
        cudaMemset2D(self.output_2D, self.output_2D_pitch, 0, self.profile.kernel_dim * sizeof(float4), self.profile.kernel_dim)    

        # initialize timing info
        self.time_events = True
        self.frame_count = 0.0
        self.events = [cudaEvent_t() for i in range(4)]
        [cudaEventCreate(byref(event)) for event in self.events]    
        self.event_accum_tmp = [0 for i in range(len(self.events) - 1)]
        self.event_accum = [0 for i in range(len(self.events) - 1)]
        self.time_accum = 0


    def register(self):
        self.pbo_ptr = c_void_p()
        status = cudaGLRegisterBufferObject(self.pbo)        
        cudaGLMapBufferObject(byref(self.pbo_ptr), self.pbo)        


    def cleanup(self):
        cudaFreeArray(self.input_array)
        cudaFree(self.output_2D)
        [cudaEventDestroy(event) for event in self.events]    
        status = cudaGLUnregisterBufferObject(self.pbo)


    def record(self, idx):
        if(self.time_events):
            cudaEventRecord(self.events[idx], 0)


    def do(self):
        self.record(0)

        self.frame_count += 1            

        block = dim3(16, 16, 1)
        grid = dim3(self.profile.kernel_dim / 16, self.profile.kernel_dim / 16, 1)
        status = cudaConfigureCall(grid, block, 0, 0)

        kernel_fb(self.output_2D, c_ulong(self.output_2D_pitch.value / sizeof(float4)), self.pbo_ptr, self.offset, self.profile.kernel_dim)            
        self.record(1)

        cudaMemcpy2DToArray(self.input_array, 0, 0, self.output_2D, self.output_2D_pitch, self.profile.kernel_dim * sizeof(float4), self.profile.kernel_dim, cudaMemcpyDeviceToDevice)
        self.record(2)
    
        cudaGLUnmapBufferObject(self.pbo)
        self.record(3)

        # compute and print timings
        if(self.time_events):
            cudaEventSynchronize(self.events[-1])        
            
            times = [c_float() for i in range(len(self.events) - 1)]        
            [cudaEventElapsedTime(byref(times[i]), self.events[i], self.events[i+1]) for i in range(len(times))]

            self.event_accum_tmp = [self.event_accum_tmp[i] + times[i].value for i in range(len(times))]
            self.event_accum = [self.event_accum[i] + times[i].value for i in range(len(times))]

            if(self.frame_count % self.profile.debug_freq == 0):
                for i in range(len(times)):
                    print "event" + str(i) + "-" + str(i + 1) + ": " + str(self.event_accum_tmp[i] / self.profile.debug_freq) + "ms"
                    print "event" + str(i) + "-" + str(i + 1) + "~ " + str(self.event_accum[i] / self.frame_count) + "ms"
                print "total: " + str(sum(self.event_accum_tmp) / self.profile.debug_freq) + "ms"
                print "total~ " + str(sum(self.event_accum) / self.frame_count) + "ms"
                self.event_accum_tmp = [0 for i in range(len(self.events) - 1)]

        cudaThreadSynchronize()
