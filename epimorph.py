#! /usr/bin/python

import sys
import os

from config.configmanager import *
from phenom.cmdcenter import *
from noumena.engine import *
from noumena.renderer import *

# TODO: run unclutter to hide mouse

# initialize state/profile/context
manager = ConfigManager()

def parse_args(sym):
    return dict(tuple(map(lambda x: (x[0], eval(x[1])), (cmd[1:].split(':') for cmd in sys.argv[1:] if cmd[0] == sym))))

context_vars = parse_args("~")

context_vars.setdefault("context", "default")

context = manager.load_dict("context", context_vars["context"], **context_vars)
state =   manager.load_dict("state", context.state, **parse_args("%"))
profile = manager.load_dict("profile", context.profile, **parse_args("@"))

# initialize components
renderer   = Renderer(state, profile, context)
engine     = Engine(state, profile, context, renderer.pbo)
cmdcenter  = CmdCenter(state, renderer, engine, context)

# create execution loop
def inner_loop():

    cmdcenter.do()
    engine.do()
    renderer.do()

    if(context.exit):
        renderer.__del__()
        engine.__del__()
        cmdcenter.__del__()
        sys.exit()

# set execution loop
renderer.set_inner_loop(inner_loop)

# define start function
def start():
    renderer.start()

# start
<<<<<<< HEAD:epimorph.py
if(context.auto_start):
=======
if(context.autostart):
>>>>>>> 127da883db27c9a2ddd6d3c85eba550a32b1acac:epimorph.py
    start()
