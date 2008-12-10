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

    def __init__(self, state, renderer, engine):
        self.state, self.renderer, self.engine = state, renderer, engine
        self.animator = Animator()
        mouse_handler = MouseHandler(self, renderer.profile)
        keyboard_handler = KeyboardHandler(self)
        console = Console(self)
        self.renderer.register_callbacks(keyboard_handler.keyboard, mouse_handler.mouse, mouse_handler.motion, console.render_console, console.console_keyboard)

        # start datamanager
        self.datamanager = DataManager()

        # start server
        self.server = Server(self)
        self.server.start()

        # start midi
        self.midi = MidiHandler(self)
        self.midi.start()

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


   # def __getattribute__(self, name):
   #     return object.__getattribute__(self, name)

    def __getattr__(self, name):
        self.__missing_method_name = name
        return getattr(self, "__methodmissing__")

    def __methodmissing__(self, *args, **kwargs):
        print "asdfasdfasd"


    def cmd(self, code):
        out = StringIO.StringIO()
        sys.stdout = out

        err = ""

        try:
            exec code
        except:
            err = traceback.format_exc().split("\n")[-2]


        sys.stdout = sys.__stdout__

        res = [out.getvalue(), err]

        out.close()

        return res


    def do(self):
        self.animator.do()

