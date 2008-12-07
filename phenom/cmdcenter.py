from phenom.animator import *
from phenom.console import *
from phenom.keyboard import *
from phenom.mouse import *

from aer.datamanager import *

import StringIO

import sys

class CmdCenter:

    def __init__(self, state, renderer, engine):
        self.state, self.renderer, self.engine = state, renderer, engine
        self.animator = Animator()
        mouse_handler = MouseHandler(self, renderer.profile)
        keyboard_handler = KeyboardHandler(self)
        console = Console(self)
        self.renderer.register_callbacks(keyboard_handler.keyboard, mouse_handler.mouse, mouse_handler.motion, console.render_console, console.console_keyboard)

        self.datamanager = DataManager(self)    
        self.t_idx = 0
        self.t_seed_idx = 0


    def inc_t(self, idx):
        self.t_idx += idx
        val = self.datamanager.t[self.t_idx]
        self.state.T = val[0]
        for line in val[1]:
            exec(line)
        self.engine.compile_kernel()


    def inc_t_seed(self, idx):
        self.t_seed_idx += idx
        val = self.datamanager.t_seed[self.t_idx]
        self.state.T_SEED = val[0]
        for line in val[1]:
            exec(line)
        self.engine.compile_kernel()


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
    
