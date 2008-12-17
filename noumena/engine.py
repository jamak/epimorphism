from ctypes import *
from cuda.cuda_defs import *
from cuda.cuda_api import *

from noumena.compiler import *

import time


class Engine(object):
    ''' The engine object is the applications interface, via cuda, to the graphics hardware.
        It is responsible for the setup and maintenence of the cuda environment and the graphics kernel.
        It communicates to the renderer via pbo  '''

    def __init__(self, profile, state, pbo):

        self.profile, self.state = profile, state

        # get device
        self.cuda_device = c_int()
        cudaGetDevice(byref(self.cuda_device))

        # DEACTIVATED - throws an error on close
        # get/print properties
        # self.cuda_properties = cudaDeviceProp()
        # cudaGetDeviceProperties(self.cuda_properties, self.cuda_device)
        # print str(self.cuda_properties)

        # create frame buffer
        self.channel_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat)
        self.fb = cudaArray_p()
        cudaMallocArray(byref(self.fb), byref(self.channel_desc), self.profile.kernel_dim, self.profile.kernel_dim)

        # initialize frame buffer
        empty = (c_ubyte * (sizeof(float4) * self.profile.kernel_dim ** 2))()
        cudaMemcpyToArray(self.fb, 0, 0, empty, sizeof(float4) * self.profile.kernel_dim ** 2, cudaMemcpyHostToDevice)

        # create output_2D
        self.output_2D, self.output_2D_pitch = c_void_p(), c_uint()
        cudaMallocPitch(byref(self.output_2D), byref(self.output_2D_pitch),
                        self.profile.kernel_dim * sizeof(float4), self.profile.kernel_dim)
        cudaMemset2D(self.output_2D, self.output_2D_pitch, 0, self.profile.kernel_dim * sizeof(float4),
                     self.profile.kernel_dim)

        # initialize timing info
        self.time_events = False
        self.frame_count = 0.0
        self.events = [cudaEvent_t() for i in range(4)]
        [cudaEventCreate(byref(event)) for event in self.events]
        self.event_accum_tmp = [0 for i in range(len(self.events) - 1)]
        self.event_accum = [0 for i in range(len(self.events) - 1)]
        self.time_accum = 0

        # set block & grid size
        self.block = dim3(8, 8, 1)
        self.grid = dim3(self.profile.kernel_dim / 8, self.profile.kernel_dim / 8, 1)

        # compile kernel
        self.kernel = None
        Compiler(self.state.__dict__, self.set_new_kernel).start()

        # register_pbo
        self.pbo, self.pbo_ptr = pbo, c_void_p()
        status = cudaGLRegisterBufferObject(self.pbo)
        cudaGLMapBufferObject(byref(self.pbo_ptr), self.pbo)

        # malloc host array
        self.host_array = c_void_p()
        cudaMallocHost(byref(self.host_array), 4 * (self.profile.kernel_dim ** 2) * sizeof(c_ubyte))

        # flag to bind texture
        self.new_kernel = None


    # for some reason this isn't called
    def __del__(self):

        # clear cuda memory
        cudaFreeArray(self.fb)
        cudaFree(self.output_2D)
        cudaFree(self.host_array)

        # unregister pbo
        cudaGLUnregisterBufferObject(self.pbo)

        # delete events
        [cudaEventDestroy(event) for event in self.events]


    def record_event(self, idx):

        # record an event
        if(self.time_events):
            cudaEventRecord(self.events[idx], 0)


    def print_timings(self):

        if(self.time_events):

            # synchronize
            cudaEventSynchronize(self.events[-1])

            # get times
            times = [c_float() for i in range(len(self.events) - 1)]
            [cudaEventElapsedTime(byref(times[i]), self.events[i], self.events[i+1]) for i in range(len(times))]

            # set accumulators
            self.event_accum_tmp = [self.event_accum_tmp[i] + times[i].value for i in range(len(times))]
            self.event_accum = [self.event_accum[i] + times[i].value for i in range(len(times))]

            if(self.frame_count % self.profile.debug_freq == 0):

                # print times
                for i in range(len(times)):

                    print "event" + str(i) + "-" + str(i + 1) + ": " + str(self.event_accum_tmp[i] / self.profile.debug_freq) + "ms"
                    print "event" + str(i) + "-" + str(i + 1) + "~ " + str(self.event_accum[i] / self.frame_count) + "ms"

                # print totals
                print "total: " + str(sum(self.event_accum_tmp) / self.profile.debug_freq) + "ms"
                print "total~ " + str(sum(self.event_accum) / self.frame_count) + "ms"

                # reset tmp accumulator
                self.event_accum_tmp = [0 for i in range(len(self.events) - 1)]


    def set_new_kernel(self, name):
        self.new_kernel = name


    def switch_kernel(self):
        # start clock if necessary
        if(not self.kernel) : self.t_start = time.clock()

        # bind kernel
        self.kernel = bind_kernel(self.new_kernel)
        self.new_kernel = None

        # create texture reference
        self.tex_ref = textureReference_p()
        cudaGetTextureReference(byref(self.tex_ref), "input_texture")

        # set texture parameters
        self.tex_ref.contents.normalized = True
        self.tex_ref.contents.filterMode = cudaFilterModeLinear
        self.tex_ref.contents.addressMode[0] = cudaAddressModeClamp
        self.tex_ref.contents.addressMode[1] = cudaAddressModeClamp

        # bind tex_ref to fb. copy output_2D to fb
        cudaBindTextureToArray(self.tex_ref, self.fb, byref(self.channel_desc))
        cudaMemcpy2DToArray(self.fb, 0, 0, self.output_2D, self.output_2D_pitch, self.profile.kernel_dim * sizeof(float4),
                            self.profile.kernel_dim, cudaMemcpyDeviceToDevice)


    def do(self):

        # idle until kernel found
        while(not self.kernel and not self.new_kernel): time.sleep(0.01)

        # switch kernel if necessary
        if(self.new_kernel) : self.switch_kernel()

        # begin
        self.record_event(0)
        self.frame_count += 1

        # upload par & zn & internal
        par = (c_float * len(self.state.par))(*[p for p in self.state.par])
        cudaMemcpyToSymbol("par", byref(par), sizeof(par), 0, cudaMemcpyHostToDevice)

        internal = (c_float * len(self.state.internal))(*[p for p in self.state.internal])
        cudaMemcpyToSymbol("internal", byref(internal), sizeof(internal), 0, cudaMemcpyHostToDevice)

        zn = (float2 * len(self.state.zn))(*[(z.real, z.imag) for z in self.state.zn])
        cudaMemcpyToSymbol("zn", byref(zn), sizeof(zn), 0, cudaMemcpyHostToDevice)

        # upload clock
        clock = c_float(time.clock() - self.t_start)
        cudaMemcpyToSymbol("count", byref(clock), sizeof(clock), 0, cudaMemcpyHostToDevice)

        # call kernel
        cudaConfigureCall(self.grid, self.block, 0, 0)
        self.kernel(self.output_2D, c_ulong(self.output_2D_pitch.value / sizeof(float4)), self.pbo_ptr,
                    self.profile.kernel_dim, 1.0 / self.profile.kernel_dim, 1.0001 / self.profile.kernel_dim,
                    1.0 / self.state.FRACT ** 2, 2.0 / (self.profile.kernel_dim * (self.state.FRACT - 1.0)))
        self.record_event(1)

        # copy data to output_2D
        cudaMemcpy2DToArray(self.fb, 0, 0, self.output_2D, self.output_2D_pitch, self.profile.kernel_dim * sizeof(float4),
                            self.profile.kernel_dim, cudaMemcpyDeviceToDevice)
        self.record_event(2)

        # unmap pbo
        cudaGLUnmapBufferObject(self.pbo)
        self.record_event(3)

        # compute and print timings
        self.print_timings()


    def get_fb(self):
        ''' This function returns an copy of the the current pbo.
            The return value is a dim ** 2 array of 4 * c_ubyte '''

        # map buffer
        cudaGLMapBufferObject(byref(self.pbo_ptr), self.pbo)

        # copy pbo to host
        res = cudaMemcpy2D(self.host_array, self.profile.kernel_dim * sizeof(c_ubyte) * 4, self.pbo_ptr,
                           self.profile.kernel_dim * sizeof(c_ubyte) * 4, self.profile.kernel_dim * sizeof(c_ubyte) * 4,
                           self.profile.kernel_dim, cudaMemcpyDeviceToHost)

        # return c_ubyte array
        return (c_ubyte * (4 * (self.profile.kernel_dim ** 2))).from_address(self.host_array.value)


    def set_fb(self, data):
        ''' This manually sets the framebuffer.
            data is a dim ** 2 array of float4 '''

        # copy data to fb
        cudaMemcpy2DToArray(self.fb, 0, 0, data, self.profile.kernel_dim * sizeof(float4),
                            self.profile.kernel_dim * sizeof(float4), self.profile.kernel_dim,
                            cudaMemcpyHostToDevice)


    def reset_fb(self):
        ''' This funcion resets the framebuffer to solid black '''

        # set_fb with empty buffer
        self.set_fb((float4 * (self.profile.kernel_dim ** 2))())
