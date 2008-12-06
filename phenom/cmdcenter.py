from phenom.animator import *
from phenom.keyboard import *
from phenom.mouse import *

class CmdCenter:

    TPATH = "aer/t.epi"

    def __init__(self, state, renderer, engine):
        self.state, self.renderer, self.engine = state, renderer, engine
        self.animator = Animator()
        MouseHandler(self, renderer.profile)
        KeyboardHandler(self)

        # load t
        file = open(self.TPATH)
        self.t = []
        for line in file.readlines():
            data = line.split(':')
            data[1] = data[1].replace('(', '').replace(')', '').split(',')
            self.t.append(data)
        file.close()


    def load_t(self, idx):
        idx = idx % len(self.t)
        idx = 0
        for i in range(0, len(self.state.zn)):
            print self.t[idx][1][i].replace('i', 'j')
            self.state.zn[i] = complex(self.t[idx][1][i].replace('i', 'j'))
        self.state.T = self.t[idx][0]
        self.engine.compile_kernel()


    def do(self):
        self.animator.do()
    
