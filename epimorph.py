#! /usr/bin/python

import sys
import os
import atexit
import re

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

if(len(sys.argv[1:]) != 0):
    debug("with args %s" % (str(sys.argv[1:])))

# parse command line arguments
args={"application":"default", "app":{}, "env":{}, "context":{}, "profile":{}, "state":{}}
for arg in sys.argv[1:]:
    
    # application
    split = re.compile("=").split(arg)
    if(len(split) == 1): 
        args["application"] = arg
        continue

    # parse val
    val = split[1]
    if(val[0] == '$'): val = eval(val[1:])
    if(val == "True"): val = True
    if(val == "False"): val = False

    # create vars
    split = re.compile("\.").split(split[0])
    if(len(split) == 1): args["app"][split[0]] = val
    else : args[split[0]][split[1]] = val

# create structures
app     = configmanager.load_dict("app", args["application"], **args["app"])
env     = configmanager.load_dict("environment", app.env, **args["env"])
context = configmanager.load_dict("context", app.context, **args["context"])
profile = configmanager.load_dict("profile", app.profile, **args["profile"])
state   = configmanager.load_dict("state", app.state, **args["state"])

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

