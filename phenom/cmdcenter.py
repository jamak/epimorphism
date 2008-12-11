from phenom.animator import *
from phenom.console import *
from phenom.keyboard import *
from phenom.mouse import *
from phenom.server import *
from phenom.midi import *

from aer.datamanager import *

import StringIO

import sys

class CmdCenter(object):

    def __init__(self, state, renderer, engine, context):
        self.state, self.renderer, self.engine, self.context = state, renderer, engine, context
        self.animator = Animator()
        mouse_handler = MouseHandler(self, renderer.profile)
        keyboard_handler = KeyboardHandler(self)
        console = Console(self)
        self.renderer.register_callbacks(keyboard_handler.keyboard, mouse_handler.mouse, mouse_handler.motion, console.render_console, console.console_keyboard)

        # start datamanager
        self.datamanager = DataManager()

        # start server
        if(context.server):
            self.server = Server(self)
            self.server.start()

        # start midi
        if(context.midi):
            self.midi = MidiHandler(self)
            self.midi.start()

        # generate cmd exec environment
        func_blacklist = ['do', '__del__', '__init__', 'kernel', 'print_timings', 'record_event', 'start',
                          'keyboard', 'console_keyboard', 'register_callbacks', 'render_console']

        def get_funcs(obj):
            return dict([(attr, getattr(obj, attr)) for attr in dir(obj) if callable(getattr(obj, attr)) and attr not in func_blacklist])

        self.env = get_funcs(self.engine)
        self.env.update(get_funcs(self.renderer))
        self.env.update(get_funcs(self.animator))
        self.env.update(self.state.__dict__)
        self.env.update(self.context.__dict__)

        # init indices
        self.T_idx = 0
        self.T_SEED_idx = 0
        self.SEED_idx = 0
        self.SEED_W_idx = 0
        self.SEED_C_idx = 0
        self.SEED_A_idx = 0


    def inc_data(self, data, idx):
        exec("self." + data + "_idx += idx")
        exec("self." + data + "_idx %= len(self.datamanager." + data + ")")
        exec("val = self.datamanager." + data + "[self." + data + "_idx]")
        exec("self.state." + data + " = val[0]")
        for line in val[1]:
            exec(line)
        self.engine.load_kernel()


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

