from phenom.animator import *
from phenom.console import *
from phenom.keyboard import *
from phenom.mouse import *

import StringIO

import sys

class CmdCenter:

    TPATH = "aer/t.epi"
    TSEEDPATH = "aer/t_seed.epi"

    def __init__(self, state, renderer, engine):
        self.state, self.renderer, self.engine = state, renderer, engine
        self.animator = Animator()
        mouse_handler = MouseHandler(self, renderer.profile)
        keyboard_handler = KeyboardHandler(self)
        console = Console(self)
        self.renderer.register_callbacks(keyboard_handler.keyboard, mouse_handler.mouse, mouse_handler.motion, console.render_console, console.console_keyboard)
        
        # load t
        file = open(self.TPATH)
        self.t = []
        self.t_idx = 0
        for line in file.readlines():
            if(line == "\n"):
                continue
            data = line.split(':')
            data[0] = data[0].strip()
            data[1] = data[1].strip()[1:-1].split(',')
            self.t.append(data)
        file.close()

        # load t_seed
        file = open(self.TSEEDPATH)
        self.t_seed = []
        self.t_seed_idx = 0
        for line in file.readlines():
            if(line == "\n"):
                continue
            data = line.split(':')
            data[0] = data[0].strip()
            data[1] = data[1].strip()[1:-1].split(',')
            self.t_seed.append(data)
        file.close()


    def load_t(self, idx):
        idx = idx % len(self.t)
        for i in range(0, 6):
            print self.t[idx][1][i].replace('i', 'j')
            self.state.zn[i] = complex(self.t[idx][1][i].replace('i', 'j'))
        self.state.T = self.t[idx][0]
        self.engine.compile_kernel()


    def load_t_seed(self, idx):
        idx = idx % len(self.t_seed)
        for i in range(0, 3):
            print self.t_seed[idx][1][i].replace('i', 'j')
            self.state.zn[6 + i] = complex(self.t_seed[idx][1][i].replace('i', 'j'))
        self.state.T_SEED = self.t_seed[idx][0]
        self.engine.compile_kernel()


    def inc_t(self, idx):
        self.t_idx += idx
        self.load_t(self.t_idx)


    def inc_t_seed(self, idx):
        self.t_seed_idx += idx
        self.load_t_seed(self.t_seed_idx)


    def cmd(self, code):
        out = StringIO.StringIO()
        
        sys.stdout = out
 
        res = ""

        try:
            exec code
        except:
            res = traceback.format_exc().split("\n")[-2]


        sys.stdout = sys.__stdout__
    
        res = [out.getvalue(), res]

        out.close()

        return res

    def do(self):
        self.animator.do()
    
