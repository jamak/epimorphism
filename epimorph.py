#! /usr/bin/python

# Epimorphism - v3.0b

import sys
import datetime

from noumena.state import *
from noumena.engine import *
from noumena.renderer import *
from noumena.logger import *

from phenom.animator import *


# log("EP: START - " + datetime.date.today().strftime("%m/%d/%y"))

# get variables
profile_vars = dict(tuple(cmd[1:].split(':')) for cmd in sys.argv[1:] if cmd[0] == '@')
state_vars = dict(tuple(cmd[1:].split(':')) for cmd in sys.argv[1:] if cmd[0] == '$')
other_vars = dict(tuple(cmd[1:].split(':')) for cmd in sys.argv[1:] if cmd[0] == '~')

# initialize states
manager = StateManager()

state_name = other_vars.setdefault('state', 'default')
state = manager.load_state(state_name, **state_vars)

profile_name = other_vars.setdefault('profile', 'box1')
profile = manager.load_profile(profile_name, **profile_vars)

# initialize
animator = Animator(state)
engine = Engine(profile, state)
renderer = Renderer(animator, engine)

# create & set execution
def execution():
    messages = animator.do()
    engine.do(messages)
    renderer.do()

renderer.set_execution(execution)

# start state
renderer.start()
