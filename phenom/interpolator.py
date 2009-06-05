from noumena.compiler import *

import time

COMPILE_TIME = 1.9

class Interpolator(object):

    def __init__(self, cmdcenter, state, renderer, engine, context):
        self.cmdcenter, self.state, self.renderer, self.engine, self.context = cmdcenter, state, renderer, engine, context

        # get new kernel settings
        self.new_kernel = dict([(data, [None, None]) for data in self.cmdcenter.datamanager.components])

        # animation settings
        self.animating = dict([(data, [None, None, None]) for data in self.cmdcenter.datamanager.components])


    def set_new_kernel(self, data, idx, name):

        # compiler callback
        self.new_kernel[data][idx] = name


    def interpolate(self, data, idx_idx, o_val, val, callback):
        self.renderer.echo_string = "switching %s to: %s" % (data, val)

        sub = "min((_clock - internal[%d]) / %ff, 1.0f)" % (idx_idx, self.context.component_switch_time)

        intrp = "((1.0f - %s) * (%s) + %s * (%s))" % (sub, o_val, sub, val)

        setattr(self.state, data, intrp)

        self.new_kernel[data][0] = None

        var = self.state.__dict__

        # check for multi-compile
        for key in self.cmdcenter.datamanager.components:
            if(self.animating[key][0] and time.clock() + COMPILE_TIME < self.animating[key][0] and key != data):
                var.update({key : self.animating[key][1]})
                self.animating[key][2].update({key : self.animating[key][1]})

        Compiler(var, (lambda name: self.set_new_kernel(data, 0, name)), self.context).start()

        while(not self.new_kernel[data][0] and not self.context.exit) : time.sleep(0.1)
        if(self.context.exit) : exit()

        # phase 1
        self.animating[data] = [time.clock() + self.context.component_switch_time, getattr(self.state, data), None]

        self.engine.new_kernel = self.new_kernel[data][0]

        self.new_kernel[data][1] = None

        self.state.internal[idx_idx] = time.clock() - self.engine.t_start

        setattr(self.state, data, val)

        compiler = Compiler(self.state.__dict__, (lambda name: self.set_new_kernel(data, 1, name)), self.context)

        self.animating[data][2] = compiler

        # compiler.start()

        while((time.clock() < self.animating[data][0] or not self.new_kernel[data][1]) and not self.context.exit) : time.sleep(0.01)
        if(self.context.exit) : exit()

        # complete
        self.animating[data] = [None, None, None]
        self.engine.new_kernel = self.new_kernel[data][1]
        self.new_kernel[data][1] = None

        print "done switching %s" % data
        self.renderer.echo_string = None

        callback()


