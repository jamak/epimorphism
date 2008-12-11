#! /usr/bin/python

# Epimorphism - v3.0b

import sys
import datetime

from phenom.cmdcenter import *

from noumena.state import *
from noumena.engine import *
from noumena.renderer import *

from common.logger import *

log("EP: START - " + datetime.date.today().strftime("%m/%d/%y"))

# get variables
print sys.argv
profile_vars = dict(tuple(map(lambda x: (x[0], eval(x[1])), (cmd[1:].split(':') for cmd in sys.argv[1:] if cmd[0] == '@'))))
state_vars   = dict(tuple(map(lambda x: (x[0], eval(x[1])), (cmd[1:].split(':') for cmd in sys.argv[1:] if cmd[0] == '%'))))
other_vars   = dict(tuple(map(lambda x: (x[0], eval(x[1])), (cmd[1:].split(':') for cmd in sys.argv[1:] if cmd[0] == '~'))))

print other_vars

# initialize information
manager = StateManager()

state_name = other_vars.setdefault('state', 'default')
state = manager.load_state(state_name, **state_vars)

profile_name = other_vars.setdefault('profile', 'box1')
profile = manager.load_profile(profile_name, **profile_vars)

context = manager.default_context(**other_vars)

# initialize
renderer   = Renderer(profile, state)
engine     = Engine(profile, state, renderer.pbo)
cmdcenter  = CmdCenter(state, renderer, engine, context)

# create and set execution loop
def inner_loop():
    cmdcenter.do()
    engine.do()
    renderer.do()

renderer.set_inner_loop(inner_loop)

# start
renderer.start()
