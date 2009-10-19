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

    # interface.__del__()
    # engine.__del__()
    # cmdcenter.__del__()

    # remove unclutter
    os.system("killall unclutter")

atexit.register(exit)

# run unclutter to remove mouse pointer
os.system("unclutter -idle 0.25 -jitter 1 -root&")

# initialize env/state/profile/context
debug("Initializing state/profile/context")

manager = ConfigManager()

def parse_args(sym):
    return dict(tuple(map(lambda x: (x[0], eval(x[1])), (cmd[1:].split(':') for cmd in sys.argv[1:] if cmd[0] == sym))))

if(len(sys.argv[1:]) != 0):
    debug("with args %s" % (str(sys.argv[1:])))

env_vars = parse_args("~")
env_vars.setdefault("env", "default")

env     = manager.load_dict("environment", env_vars["env"], **env_vars)
context = manager.load_dict("context", env.context, **parse_args("@"))
profile = manager.load_dict("profile", env.profile, **parse_args("#"))
state   = manager.load_dict("state", env.state, **parse_args("%"))

# encapsulated for asynchronous execution
def main():
    info("Starting main loop")

    # initialize & sync modules
    debug("Initializing modules")

    global interface, engine, cmdcenter
    interface = Interface(context)
    engine    = Engine(profile)
    cmdcenter = CmdCenter(env, state, interface, engine)

    debug("Syncing modules")
    interface.sync_cmd(cmdcenter)
    engine.sync(interface.renderer)

    # compile engine kernel - this needs to be generalized
    debug("Compiling kernel")
    compiler_config = {'ptxas_stats': profile.ptxas_stats, 'par_names':state.par_names, 'datamanager':cmdcenter.componentmanager.datamanager}
    Compiler(engine.set_new_kernel, compiler_config).start()

    # start main loop
    debug("Starting")
    cmdcenter.start()

    info("Main loop completed")

# define start function
def start():
    async(main)

# start
if(env.autostart):
    start()

