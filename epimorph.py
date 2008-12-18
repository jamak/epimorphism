#! /usr/bin/python

import sys

from config.configmanager import *
from phenom.cmdcenter import *
from noumena.engine import *
from noumena.renderer import *


# initialize state/profile/context
manager = ConfigManager()
info_schema = [("context", '~', 'default', 'ctx'), ("state", '%', 'default', 'est'), ("profile", '@', 'box1', 'prf')]

for info in info_schema:
    var = dict(tuple(map(lambda x: (x[0], eval('"' + x[1] + '"')), (cmd[1:].split(':') for cmd in sys.argv[1:] if cmd[0] == info[1]))))
    obj = manager.load_dict(var.setdefault(info[0], info[2]) + "." + info[3], **var)
    exec(info[0] + " = obj")

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
