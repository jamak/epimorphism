from noumena.compiler import *

import time

COMPILE_TIME = 1.9

class Interpolator(object):

    def __init__(self, cmdcenter, state, renderer, engine, context):
        self.cmdcenter, self.state, self.renderer, self.engine, self.context = cmdcenter, state, renderer, engine, context

        self.new_kernel = None

    def set_new_kernel(self, name):
        # compiler callback
        self.new_kernel = name


    def interpolate_splice(self, idx_idx, val_idx, callback):

        self.state.component_idx[2 * idx_idx + 1] = val_idx
        self.state.internal[idx_idx] = time.clock() - self.engine.t_start

        while(time.clock() - self.engine.t_start - self.state.internal[idx_idx] < self.context.component_switch_time):
            #print time.clock() - self.engine.t_start - self.state.internal[idx_idx], self.context.component_switch_time, time.clock(), self.engine.t_start, self.state.internal[idx_idx]
            time.sleep(0.1)
        self.state.internal[idx_idx] = 0
        self.state.component_idx[2 * idx_idx] = val_idx
       # callback()


    def interpolate(self, data, idx_idx, val, callback):

        print "start switching %s" % data


        self.renderer.echo_string = "switching %s to: %s" % (data, val)

        var = self.state.__dict__

        Compiler(var, self.set_new_kernel, self.context).start()

        while(not self.new_kernel and not self.context.exit) : time.sleep(0.1)

        if(self.context.exit) : exit()

        setattr(self.state, data, val)

        self.new_kernel = None

        print "done switching %s" % data
        self.renderer.echo_string = None

        # callback()


