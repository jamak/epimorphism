import pickle
import os.path

from logger import *
logger = Logger('log.txt', 'st:  ')
log = logger.log

class State:
    def __init__(self, **vars):
        self.__dict__.update(vars)

class Profile:
    def __init__(self, **vars):
        self.__dict__.update(vars)

class StateManager:
    def load_state(self, state_name, **vars):
        log("load state - " + state_name)
        log("with vars - " + str(vars))

        file = open("state/" + state_name + ".epi", "r")
        return pickle.load(file)


    def load_profile(self, profile_name, **vars):
        log("load profile - " + profile_name)
        log("with vars - " + str(vars))

        file = open("profile/" + profile_name + ".prf", "r")
        #return pickle.load(file)

        profile = Profile(name=profile_name, viewport_width=500, viewport_height=500, full_screen=False, viewport_refresh=60, vp_scale=1.0, vp_center_x=0.0, vp_center_y=0.0, kernel_dim=1000)
        return profile


    def save_state(self, state, name=''):
        log("save state")

        if(name == ''):
            i = 0
            while(os.path.exists("state/state_" + i + ".epi")):
                i += 1
            name = "state_" + i

        log("  as " + name)

        file = open("state/" + name + ".epi", "w")
        pickle.dump(state, file)


    def save_profile(self, profile, name):
        log("save profile")
        log("  as " + name)

        file = open("profile/" + name + ".prf", "w")
        pickle.dump(profile, file)

