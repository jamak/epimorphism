from phenom.animator import *

class CmdCenter:
    def __init__(self, state, renderer, engine):
        self.state, self.renderer, self.engine = state, renderer, engine
        self.animator = Animator()

    def do(self):
        self.animator.do()
    
