#! /usr/bin/python

import sys
import os
import atexit

import config.configmanager
from noumena.interface import *
from viro.engine import *
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
os.system("unclutter -idle 0.25 -jitter 1 -root&")

# initialize env/state/profile/context/env
debug("Initializing state/profile/context/env")

def parse_args(sym):
    return dict(tuple(map(lambda x: (x[0], eval(x[1])), (cmd[1:].split(':') for cmd in sys.argv[1:] if cmd[0] == sym))))

if(len(sys.argv[1:]) != 0):
    debug("with args %s" % (str(sys.argv[1:])))

env_vars = parse_args("~")
env_vars.setdefault("env", "default")

env     = configmanager.load_dict("environment", env_vars["env"], **env_vars)
context = configmanager.load_dict("context", env.context, **parse_args("@"))
profile = configmanager.load_dict("profile", env.profile, **parse_args("#"))
state   = configmanager.load_dict("state", env.state, **parse_args("%"))

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

    # compile engine - CLEAN THIS
    engine.compile({'ptxas_stats': profile.ptxas_stats, 'par_names':state.par_names, 'datamanager':cmdcenter.componentmanager.datamanager,
                    'splice':env.splice_components, 'state':state, 'cull_enabled':env.cull_enabled})

    # start main loop
    debug("Starting")
    cmdcenter.start()

    info("Main loop completed")

    # clean objects
    interface.__del__()
    engine.__del__()
    cmdcenter.__del__()

# start
def start():
    async(main)

if(env.autostart):
    start()

