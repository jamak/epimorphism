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

        funcs = {'bindings' : self.bindings, 'funcs' : self.funcs, 'save' : self.save}
        funcs.update(get_funcs(self.renderer))
        funcs.update(get_funcs(self.animator))
        funcs.update(get_funcs(self.video_renderer))
        funcs.update(get_funcs(self.engine))

        funcs.update(default_funcs)

        self.env = CmdEnv([self.state.__dict__, self.context.__dict__], funcs)

        # init indices
        self.T_idx = 0
        self.T_SEED_idx = 0
        self.SEED_idx = 0
        self.SEED_W_idx = 0
        self.SEED_C_idx = 0
        self.SEED_A_idx = 0

        # animation settings
        self.animating = {}


    def inc_data(self, data, idx):

        exec("self." + data + "_idx += idx")
        exec("self." + data + "_idx %= len(self.datamanager." + data + ")")
        exec("val = self.datamanager." + data + "[self." + data + "_idx]")
        intrp = "((1.0f - (count - internal[0]) / %ff) * (%s) + (count - internal[0]) / %ff * (%s))" % (5.0, eval("self.state." + data), 5.0, val[0])
        #intrp = "(1.0f - par[15]) * (%s) + par[15] * (%s)" % (eval("self.state." + data), val[0])
        exec("self.state." + data + " = intrp")

        for line in val[1]:
            exec(line, self.env)

        self.animating[data] = [val[0], None]

        self.new_kernel = None

        Compiler(self.state.__dict__, self).start()

        while(not self.new_kernel): time.sleep(0.01)

        self.engine.new_kernel = self.new_kernel
        self.new_kernel = None

        self.state.internal[0] = time.clock() - self.engine.t_start
        self.animating["T"][1] = time.clock() + 5
        self.state.T = self.animating["T"][0]

        Compiler(self.state.__dict__, self).start()

        while(time.clock() < self.animating["T"][1] or not self.new_kernel) : time.sleep(0.01)

        self.engine.new_kernel = self.new_kernel
        self.new_kernel = None



    def grab_image(self):
        return Image.frombuffer("RGBA", (self.engine.profile.kernel_dim, self.engine.profile.kernel_dim), self.engine.get_fb(), "raw", "RGBA", 0, 1).convert("RGB")


    def bindings(self):
        for i in xrange(len(self.state.par_names)):
            print self.state.par_names[i], ':', i


    def funcs(self):
        for key in self.env.funcs.keys() : print key


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

