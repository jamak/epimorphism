#! /usr/bin/python

import sys
import os

from config.configmanager import *
from phenom.cmdcenter import *
from noumena.engine import *
from noumena.renderer import *

# run unclutter to hide mouse


# initialize state/profile/context
manager = ConfigManager()

def parse_args(sym):
    return dict(tuple(map(lambda x: (x[0], eval('"' + x[1] + '"')), (cmd[1:].split(':') for cmd in sys.argv[1:] if cmd[0] == sym))))

context_vars = parse_args("~")

context_vars.setdefault("context", "default")
context_vars.setdefault("state", "default")
context_vars.setdefault("profile", "projector1")

context = manager.load_dict(context_vars["context"] + ".ctx", **context_vars)
state =   manager.load_dict(context_vars["state"] + ".est", **parse_args("%"))
profile = manager.load_dict(context_vars["profile"] + ".prf", **parse_args("@"))

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

# start
renderer.start()
