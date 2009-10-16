#! /usr/bin/python

import sys
import os
import atexit

from config.configmanager import *
from noumena.interface import *
from noumena.engine import *
from phenom.cmdcenter import *

from common.runner import *

from common.log import *
set_log("EPIMORPH")
info("Starting Epimorphism")


# define & register exit handler
def exit():
    debug("Exiting program")

    # remove unclutter
    os.system("killall unclutter")

atexit.register(exit)


# run unclutter to remove mouse pointer
os.system("unclutter -idle 0.25 -jitter 1 &")


# initialize state/profile/context
debug("Initializing state/profile/context")

manager = ConfigManager()

def parse_args(sym):
    return dict(tuple(map(lambda x: (x[0], eval(x[1])), (cmd[1:].split(':') for cmd in sys.argv[1:] if cmd[0] == sym))))

if(len(sys.argv[1:]) != 0):
    debug("with args %s" % (str(sys.argv[1:])))

context_vars = parse_args("~")
context_vars.setdefault("context", "default")

context = manager.load_dict("context", context_vars["context"], **context_vars)
state   = manager.load_dict("state", context.state, **parse_args("%"))
profile = manager.load_dict("profile", context.profile, **parse_args("@"))


# encapsulated for asynchronous execution
def main():
    info("Starting main loop")

    # initialize modules
    debug("Initializing modules")

    global interface, engine, cmdcenter
    interface = Interface(context)
    engine    = Engine(state, profile)
    cmdcenter = CmdCenter(state, interface, engine, context)
    interface.sync_cmd(cmdcenter)

    engine.sync(interface.renderer)

    # create execution loop
    def inner_loop():
        cmdcenter.do()
        
        if(not (state.manual_iter and not state.next_frame)): 
            state.next_frame = False
            engine.do()

        interface.do()

        if(context.exit):
           # interface.__del__()
            engine.__del__()
            cmdcenter.__del__()
            sys.exit()
            os._exit()

    # set execution loop & start - this is lame
    interface.renderer.set_inner_loop(inner_loop)
    interface.renderer.start()

    info("Main loop completed")


# define start function
def start():
    async(main)


# start
if(context.autostart):
    start()

