from logger import *

logger = Logger('log.txt', 'st:  ')
log = logger.log

class State:
    pass

class Profile:
    pass

class StateManager:
    def load_state(self, state_name, vars):
        log("load state - " + state_name)
        log("with vars - " + str(vars))

    def load_profile(self, profile_name, vars):
        log("load profile - " + profile_name)
        log("with vars - " + str(vars))        
