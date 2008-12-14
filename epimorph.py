#! /usr/bin/python

# Epimorphism - v3.0

import sys

from common.state import *
from phenom.cmdcenter import *
from noumena.engine import *
from noumena.renderer import *


# initialize state/profile/context
#  ex. ./epimorph.py ~state:state2 %T:"z + zn[0] * 4.0" ~midi:4==4
#    will load the program with the state: "state2.est", profile: "box1.prf", context: "default.ctx"
#    it will also set state.T = "z + zn[0] * 4.0" and context.midi = True
manager = StateManager()
schema = [("context", '~', 'default', 'ctx'), ("state", '%', 'default', 'est'), ("profile", '@', 'box1', 'prf')]

for scheme in schema:
    var = dict(tuple(map(lambda x: (x[0], eval('"' + x[1] + '"')), (cmd[1:].split(':') for cmd in sys.argv[1:] if cmd[0] == scheme[1]))))
    obj = manager.load_dict(var.setdefault(scheme[0], scheme[2]) + "." + scheme[3], **var)
    exec(scheme[0] + " = obj")

# initialize components
renderer   = Renderer(profile, state)
engine     = Engine(profile, state, renderer.pbo)
cmdcenter  = CmdCenter(state, renderer, engine, context)

# create and set execution loop
def inner_loop():
    cmdcenter.do()
    if(not context.manual_iter or context.next_frame):
        engine.do()
        context.next_frame = False
    renderer.do()

renderer.set_inner_loop(inner_loop)

# start
renderer.start()
