from ctypes import *
from cuda.cuda_utils import *
from cuda.cuda_defs import *
from cuda.cuda_api import *

from noumena.logger import *

from noumena.kernel import *

class Engine:    

    def __init__(self, profile, state):

        log("en: initializing")    
        self.profile, self.state = profile, state

        # get device
        self.cuda_device = c_int()
        cudaGetDevice(byref(self.cuda_device))

        # get/print properties
        self.cuda_properties = cudaDeviceProp()
        cudaGetDeviceProperties(self.cuda_properties, self.cuda_device)
        print str(self.cuda_properties)

        # create frame buffer 
        self.channel_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat)
        self.fb = cudaArray_p()
        cudaMallocArray(byref(self.fb), byref(self.channel_desc), self.profile.kernel_dim, self.profile.kernel_dim)

        # initialize frame buffer
        empty = (c_ubyte * (sizeof(float4) * self.profile.kernel_dim ** 2))()
        cudaMemcpyToArray(self.fb, 0, 0, empty, sizeof(float4) * self.profile.kernel_dim ** 2, cudaMemcpyHostToDevice)

        # create output_2D
        self.output_2D, self.output_2D_pitch = c_void_p(), c_ulong()
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

        # set block & grid size
        self.block = dim3(8, 8, 1)
        self.grid = dim3(self.profile.kernel_dim / 8, self.profile.kernel_dim / 8, 1)

        # compile the kernel
        self.compile_kernel()

        # misc variables

        self.next_frame = False


    def __del__(self):
        print "del!!!"
        cudaFreeArray(self.fb)
        cudaFree(self.output_2D)
        cudaGLUnregisterBufferObject(self.pbo)

        [cudaEventDestroy(event) for event in self.events]    


    def register_pbo(self, pbo):

        self.pbo, self.pbo_ptr = pbo, c_void_p()
        status = cudaGLRegisterBufferObject(self.pbo)        
        cudaGLMapBufferObject(byref(self.pbo_ptr), self.pbo)        


    def record_event(self, idx):

        if(self.time_events):
            cudaEventRecord(self.events[idx], 0)


    def print_timings(self):

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


    def compile_kernel(self):

        # compute kernel
        self.kernel = loadKernel(self.state)

        # bind texture
        self.tex_ref = textureReference_p()

        cudaGetTextureReference(byref(self.tex_ref), "input_texture")

        self.tex_ref.contents.normalized = True
        self.tex_ref.contents.filterMode = cudaFilterModeLinear
        self.tex_ref.contents.addressMode[0] = cudaAddressModeWrap
        self.tex_ref.contents.addressMode[1] = cudaAddressModeWrap

        cudaBindTextureToArray(self.tex_ref, self.fb, byref(self.channel_desc))


    def get_fb(self):

        pass


    def set_fb(data):
        
        pass


    def do(self, messages):

        # check manual_iter
        if(self.state.manual_iter and not self.next_frame):
            return
        self.next_frame = False

        # if necessary, recompileza
        if("recompile" in messages):
            self.compile_kernel()

        # begin
        self.record_event(0)
        self.frame_count += 1            

        # upload par & zn     
        par = (c_float * len(self.state.par))(*[p for p in self.state.par])
        cudaMemcpyToSymbol("par", byref(par), len(par), 0, cudaMemcpyHostToDevice)
        zn = (float2 * len(self.state.zn))(*[(z.real, z.imag) for z in self.state.zn])
        cudaMemcpyToSymbol("zn", byref(zn), sizeof(zn), 0, cudaMemcpyHostToDevice)

        # call kernel
        cudaConfigureCall(self.grid, self.block, 0, 0)
        self.kernel(self.output_2D, c_ulong(self.output_2D_pitch.value / sizeof(float4)), self.pbo_ptr, 
                    self.profile.kernel_dim, 1.0 / self.profile.kernel_dim, 1.0001 / self.profile.kernel_dim, 
                    1.0 / self.state.FRACT ** 2, 2.0 / (self.profile.kernel_dim * (self.state.FRACT - 1.0)))            
        self.record_event(1)

        # copy data to input_array
        cudaMemcpy2DToArray(self.fb, 0, 0, self.output_2D, self.output_2D_pitch, self.profile.kernel_dim * sizeof(float4), 
                            self.profile.kernel_dim, cudaMemcpyDeviceToDevice)
        self.record_event(2)
    
        # unmap pbo
        cudaGLUnmapBufferObject(self.pbo)
        self.record_event(3)

        # compute and print timings
        self.print_timings()

