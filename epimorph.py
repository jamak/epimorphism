# Epimorphism - v3.0

import sys
import datetime

from state import *
from engine import *
from interface import *

from logger import *
logger = Logger('log.txt', 'EP:  ')
log = logger.log
log("START - " + datetime.date.today().strftime("%m/%d/%y"))

#get variables
app_vars = dict(tuple(cmd.split(':')) for cmd in sys.argv[1:] if cmd[0] != '_')
state_vars = dict(tuple(cmd[1:].split(':')) for cmd in sys.argv[1:] if cmd[0] == '_')

#initialize states
manager = StateManager();
state_name = app_vars.setdefault('state', 'default')
state = manager.load_state(state_name, **state_vars)
profile_name = app_vars.setdefault('profile', 'box1')
profile = manager.load_profile(profile_name, **app_vars)

#initialize
engine = Engine(profile, state)
interface = Interface(engine, profile)

#start state
engine.start()
interface.start()

