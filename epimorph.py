#! /usr/bin/python

#  EPIMORPHISM - v3.0b
#
#  The project is structured into 4 branches:
#    noumena - the parametric engine which creates & displays
#    phenom  - the control system for noumena
#    aer     - the banks of data/libraries phenom uses to control noumena
#    common  - common functionality across packages & utility libraries
#
#  The project uses 3 data structures for configuration - state/profile/context
#    see common/state.py for reference
#
#
#  COMMAND LINE ARGUMENTS:
#
#  ex. ./epimorph.py ~state:state2 %T:"z + zn[0] * 4.0" ~midi:4==4
#    will load the program with the state: state2.est, profile: box1.prf, context: default.ctx
#    it will also set state.T = "z + zn[0] * 4.0" and context.midi = eval(4==4) = True
#


import sys

from common.state import *
from phenom.cmdcenter import *
from noumena.engine import *
from noumena.renderer import *


# initialize state/profile/context
manager = StateManager()
info_schema = [("context", '~', 'default', 'ctx'), ("state", '%', 'default', 'est'), ("profile", '@', 'box1', 'prf')]

for info in info_schema:
    var = dict(tuple(map(lambda x: (x[0], eval('"' + x[1] + '"')), (cmd[1:].split(':') for cmd in sys.argv[1:] if cmd[0] == info[1]))))
    obj = manager.load_dict(var.setdefault(info[0], info[2]) + "." + info[3], **var)
    exec(info[0] + " = obj")

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
