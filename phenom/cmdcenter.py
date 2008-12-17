from phenom.animator import *
from phenom.console import *
from phenom.keyboard import *
from phenom.mouse import *
from phenom.server import *
from phenom.midi import *
from phenom.video import *
from phenom.setter import *

from aer.datamanager import *

from common.default import *
from common.complex import *
from common.compiler import *

import StringIO
from copy import *
import sys
import time

import Image

COMPILE_TIME = 2.5

class CmdEnv(dict):

    def __init__(self, data, funcs):
        self.data, self.funcs = data, funcs


    def __getitem__(self, key):
        for d in self.data:
            if d.has_key(key):
                return d[key]
        return self.funcs[key]


    def __setitem__(self, key, value):
        for d in self.data:
            if d.has_key(key):
                d[key] = value


class CmdCenter(Setter):

    def __init__(self, state, renderer, engine, context):

        self.state, self.renderer, self.engine, self.context = state, renderer, engine, context
        self.animator = Animator()
        self.video_renderer = VideoRenderer(self)
        mouse_handler = MouseHandler(self, renderer.profile)
        keyboard_handler = KeyboardHandler(self)
        console = Console(self)
        self.renderer.register_callbacks(keyboard_handler.keyboard, mouse_handler.mouse, mouse_handler.motion,
                                         console.render_console, console.console_keyboard)

        # start datamanager
        self.datamanager = DataManager(self.state)

        # start server
        if(self.context.server):
            self.server = Server(self)
            self.server.start()

        # start midi
        if(self.context.midi):
            self.midi = MidiHandler(self)
            self.midi.start()

        # start video_renderer
        if(self.context.render_video):
            self.video_renderer.video_start()

        # generate cmd exec environment
        func_blacklist = ['do', '__del__', '__init__', 'kernel', 'print_timings', 'record_event', 'start', 'switch_kernel',
                          'keyboard', 'console_keyboard', 'register_callbacks', 'render_console', 'capture', 'render_fps',
                          'video_time', 'set_inner_loop'] + dir(object)

        def get_funcs(obj):
            return dict([(attr, getattr(obj, attr)) for attr in dir(obj) if callable(getattr(obj, attr)) and attr not in func_blacklist])

        funcs = {'bindings' : self.bindings, 'funcs' : self.funcs, 'save' : self.save, 'data_bindings' : self.data_bindings}
        funcs.update(get_funcs(self.renderer))
        funcs.update(get_funcs(self.animator))
        funcs.update(get_funcs(self.video_renderer))
        funcs.update(get_funcs(self.engine))

        funcs.update(default_funcs)

        self.env = CmdEnv([self.state.__dict__, self.context.__dict__], funcs)

        # init indices
        self.indices = [0 for i in xrange(len(self.datamanager.__dict__))]

        # animation settings
        self.animating = dict([(data, [None, None, None]) for data in self.datamanager.__dict__.keys()])

        self.new_kernel = dict([(data, [None, None]) for data in self.datamanager.__dict__.keys()])


    def set_new_kernel(self, data, idx, name):
        self.new_kernel[data][idx] = name


    def inc_data(self, data, idx):

        idx_idx = self.datamanager.__dict__.keys().index(data)
        self.indices[idx_idx] += idx
        self.indices[idx_idx] %= len(eval("self.datamanager." + data))
        val = eval("self.datamanager." + data)[self.indices[idx_idx]]

        for line in val[1]:
            exec(line, self.env)

        self.blend_to_data(data, val[0])


    def blend_to_data(self, data, val):

        # phase 0
        print "blending %s to: %s" % (data, val)

        idx_idx = self.datamanager.__dict__.keys().index(data)
        intrp = "((1.0f - (count - internal[%d]) / %ff) * (%s) + (count - internal[%d]) / %ff * (%s))" % (idx_idx, self.context.switch_time, eval("self.state." + data),
                                                                                                          idx_idx, self.context.switch_time, val)
        setattr(self.state, data, intrp)

        self.new_kernel[data][0] = None

        var = self.state.__dict__

        # check for multi-compile
        for key in self.datamanager.__dict__.keys():
            if(self.animating[key][0] and time.clock() + COMPILE_TIME < self.animating[key][0] and key != data):
                new_var = {}
                new_var.update(var)
                new_var.update({key : self.animating[key][1]})
                var = new_var
                self.animating[key][2].update({key : self.animating[key][1]})



        Compiler(var, (lambda name: self.set_new_kernel(data, 0, name))).start()

        while(not self.new_kernel[data][0]) : time.sleep(0.1)

        # phase 1
        self.animating[data] = [time.clock() + self.context.switch_time, getattr(self.state, data), None]

        self.engine.new_kernel = self.new_kernel[data][0]

        self.new_kernel[data][1] = None

        self.state.internal[idx_idx] = time.clock() - self.engine.t_start

        setattr(self.state, data, val)

        compiler = Compiler(self.state.__dict__, (lambda name: self.set_new_kernel(data, 1, name)))

        self.animating[data][2] = compiler

        compiler.start()

        while(time.clock() < self.animating[data][0] or not self.new_kernel[data][1]) : time.sleep(0.01)

        # complete
        self.animating[data] = [None, None, None]
        self.engine.new_kernel = self.new_kernel[data][1]
        self.new_kernel[data][1] = None


    def grab_image(self):
        return Image.frombuffer("RGBA", (self.engine.profile.kernel_dim, self.engine.profile.kernel_dim), self.engine.get_fb(), "raw", "RGBA", 0, -1).convert("RGB")


    def bindings(self):
        for i in xrange(len(self.state.par_names)):
            print self.state.par_names[i], ':', i


    def funcs(self):
        for key in self.env.funcs.keys() : print key


    def data_bindings(self):
        keys = self.datamanager.__dict__.keys()
        keys.sort()
        for i in xrange(len(keys)) : print i+1, ":", keys[i]


    def save(self):
        print "saving"
        StateManager().save_state(self.state, self.grab_image())


    def cmd(self, code):

        out = StringIO.StringIO()
        sys.stdout = out

        err = ""

        try:
            exec(code, self.env)
        except:
            err = traceback.format_exc().split("\n")[-2]

        sys.stdout = sys.__stdout__

        res = [out.getvalue(), err]

        out.close()

        return res


    def do(self):

        self.animator.do()

        if(self.context.render_video):
            self.video_renderer.capture()

